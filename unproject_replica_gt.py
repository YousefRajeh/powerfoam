"""Build a 3D point-level ground truth for room_0, analogous to ScanNet's official
per-vertex-labeled mesh (`{scene}_vh_clean_2.labels.ply`) used by OpenGaussian's point-cloud
mIoU protocol -- back-projected from our own 2D semantic_class renders + depth + camera poses,
since room_0 (unlike ScanNet) has no official 3D-mesh-with-labels of its own.

Real class NAMES (not just numeric ids) come from replica_semantic_nerf's info_semantic.json
("classes": [{"id":.., "name":..}, ...]) -- the standard Replica/Semantic-NeRF release's own
class list, not a reimplementation/guess.

Camera intrinsics: room_0's transforms_train.json gives `camera_angle_x` (horizontal FOV) for a
640x480 image (verified against rgb/semantic_class/depth file sizes) -- standard NeRF-Blender
convention: fx = fy = 0.5*W / tan(0.5*camera_angle_x), cx=W/2, cy=H/2.

Depth: uint16 PNG in millimeters (verified: depth_1.png range 580-6004 -> 0.58-6.0m, a plausible
indoor room scale matching traj_w_c.txt's camera position scale).

Output is voxel-downsampled (majority-vote label per voxel) both for tractability (900 frames x
640x480 pixels is far more points than needed) and to land at a GT point DENSITY comparable to a
real mesh's vertex density, rather than a raw per-pixel cloud.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOM0_DIR = Path(r"D:\Downloads\powerfoam\data\replica\room_0")
CLASS_NAMES_PATH = Path(r"D:\Downloads\replica_semantic_nerf\semantic_info\semantic_info\room_0\info_semantic.json")


def load_class_names():
    info = json.loads(CLASS_NAMES_PATH.read_text())
    return {c["id"]: c["name"] for c in info["classes"]}


def load_intrinsics():
    meta = json.loads((ROOM0_DIR / "transforms_train.json").read_text())
    camera_angle_x = meta["camera_angle_x"]
    w, h = Image.open(ROOM0_DIR / "rgb" / "rgb_0.png").size
    fx = fy = 0.5 * w / np.tan(0.5 * camera_angle_x)
    cx, cy = w / 2.0, h / 2.0
    return w, h, fx, fy, cx, cy


def backproject_frame(frame_idx, c2w, w, h, fx, fy, cx, cy, pixel_stride, depth_range):
    depth_path = ROOM0_DIR / "depth" / f"depth_{frame_idx}.png"
    label_path = ROOM0_DIR / "semantic_class" / f"semantic_class_{frame_idx}.png"
    if not depth_path.exists() or not label_path.exists():
        return None, None

    depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0  # mm -> m
    labels = np.array(Image.open(label_path))

    ys, xs = np.meshgrid(
        np.arange(0, h, pixel_stride), np.arange(0, w, pixel_stride), indexing="ij"
    )
    ys, xs = ys.ravel(), xs.ravel()
    d = depth[ys, xs]
    lbl = labels[ys, xs]

    valid = (lbl != 0) & (d > depth_range[0]) & (d < depth_range[1])
    if not np.any(valid):
        return None, None
    ys, xs, d, lbl = ys[valid], xs[valid], d[valid], lbl[valid]

    # Pinhole unprojection in OpenCV camera convention (X-right, Y-down, Z-forward into the
    # scene) -- NOT Blender/NeRF convention (Y-up, Z-backward). data_loader/replica.py's own
    # docstring is explicit about this: "traj_w_c.txt... OpenCV convention, already the
    # convention PowerFoam/TorchCamera expects -- no blender2opencv-style conversion needed."
    # Verified independently against info_semantic.json's gravity_dir ([~0,~0,-0.9999], i.e.
    # gravity points -Z / world +Z is up): an earlier NeRF-convention version of this function
    # put ceiling-labeled points at LOWER world Z than floor-labeled points -- backwards -- this
    # convention fix corrects that (no sign flips needed; OpenCV's own axes already match the
    # pixel row/column and forward-looking directions directly).
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=-1)  # (N, 4)

    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32), lbl.astype(np.int64)


def voxel_downsample_majority_label(points, labels, voxel_size):
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    num_voxels = counts.shape[0]
    out_points = np.zeros((num_voxels, 3), dtype=np.float32)
    out_labels = np.zeros(num_voxels, dtype=np.int64)

    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(num_voxels + 1))
    for v in range(num_voxels):
        member_idx = order[boundaries[v]:boundaries[v + 1]]
        out_points[v] = points[member_idx].mean(axis=0)
        vals, cnts = np.unique(labels[member_idx], return_counts=True)
        out_labels[v] = vals[np.argmax(cnts)]

    return out_points, out_labels


def main(frame_stride, pixel_stride, voxel_size, output_path):
    w, h, fx, fy, cx, cy = load_intrinsics()
    print(f"Intrinsics: {w}x{h}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    traj = np.loadtxt(ROOM0_DIR / "traj_w_c.txt").reshape(-1, 4, 4)
    print(f"{traj.shape[0]} camera poses available")

    all_points, all_labels = [], []
    frame_indices = list(range(0, traj.shape[0], frame_stride))
    for frame_idx in tqdm(frame_indices):
        pts, lbl = backproject_frame(
            frame_idx, traj[frame_idx], w, h, fx, fy, cx, cy,
            pixel_stride=pixel_stride, depth_range=(0.1, 8.0),
        )
        if pts is not None:
            all_points.append(pts)
            all_labels.append(lbl)

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Raw back-projected points: {points.shape[0]}")

    points, labels = voxel_downsample_majority_label(points, labels, voxel_size)
    print(f"After {voxel_size}m voxel downsampling: {points.shape[0]} points")

    class_names = load_class_names()
    present_ids = sorted(set(labels.tolist()))
    print(f"Classes present ({len(present_ids)}): "
          f"{[(i, class_names.get(i, '?')) for i in present_ids]}")

    np.savez(
        output_path, points=points, labels=labels,
        class_id_to_name=json.dumps(class_names),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frame-stride", type=int, default=10, help="Use every Nth camera pose.")
    p.add_argument("--pixel-stride", type=int, default=4, help="Sample every Nth pixel per frame.")
    p.add_argument("--voxel-size", type=float, default=0.03, help="Downsampling voxel size, meters.")
    p.add_argument("--output", default=r"D:\Downloads\powerfoam\artifacts\replica_room0\gt_point_cloud_3d.npz")
    args = p.parse_args()
    main(args.frame_stride, args.pixel_stride, args.voxel_size, args.output)
