"""Build a real, depth-derived initial point cloud for the Replica room_0
scene, shared between PowerFoam and gsplat so neither representation gets a
better or worse starting point than the other.

Why this exists: an earlier attempt initialized both from a blind gaussian
scatter around camera positions (no depth used), since Replica ships no
COLMAP/SfM reconstruction. That crashed gsplat's training outright -- most
scattered points landed in empty air, got near-zero gradient, and gsplat's
periodic opacity reset (every reset_every=3000 steps) wiped out nearly the
whole model at once because so few points had ever earned real opacity to
begin with (step 3000: 18,249 Gaussians; step 3400: 0). Every other scene in
this project (garden) initializes from real SfM points precisely to avoid
this failure mode. Replica ships real depth maps, so back-projecting a
sample of train-view pixels into world space gives an equivalent
surface-aligned point cloud without needing SfM.
"""
import numpy as np
import torch
from PIL import Image

DATA_DIR = r"D:\Downloads\powerfoam\data\replica\room_0"
NATIVE_WIDTH, NATIVE_HEIGHT = 640, 480
DEPTH_SCALE = 1.0 / 1000.0  # mm -> m, per the dataset's own readme
POINTS_PER_FRAME = 3000
FRAME_STRIDE = 20  # sample every 20th train frame across the whole trajectory
OUTPUT_PATH = r"D:\Downloads\powerfoam\data\replica\room_0\pointcloud.pt"


def build_split(num_images):
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def main():
    rgb_dir = f"{DATA_DIR}/rgb"
    depth_dir = f"{DATA_DIR}/depth"
    poses = np.loadtxt(f"{DATA_DIR}/traj_w_c.txt").reshape(-1, 4, 4).astype(np.float32)
    num_frames = poses.shape[0]
    train_idx, _ = build_split(num_frames)
    sample_idx = train_idx[::FRAME_STRIDE]
    print(f"[build_replica_pointcloud] sampling {len(sample_idx)} of {len(train_idx)} train frames")

    focal = 0.5 * NATIVE_WIDTH
    cx, cy = NATIVE_WIDTH / 2, NATIVE_HEIGHT / 2
    ys, xs = np.meshgrid(np.arange(NATIVE_HEIGHT), np.arange(NATIVE_WIDTH), indexing="ij")

    rng = np.random.default_rng(0)
    all_points, all_colors = [], []
    for i in sample_idx.tolist():
        depth = np.array(Image.open(f"{depth_dir}/depth_{i}.png")).astype(np.float32) * DEPTH_SCALE
        rgb = np.array(Image.open(f"{rgb_dir}/rgb_{i}.png").convert("RGB"))
        valid = depth > 0
        flat_idx = np.flatnonzero(valid)
        if flat_idx.size == 0:
            continue
        pick = rng.choice(flat_idx, size=min(POINTS_PER_FRAME, flat_idx.size), replace=False)
        py, px = ys.ravel()[pick], xs.ravel()[pick]
        z = depth.ravel()[pick]
        # Pinhole unprojection to camera space (OpenCV convention: x-right,
        # y-down, z-forward -- matches the dataset's own stated convention).
        x_cam = (px - cx) * z / focal
        y_cam = (py - cy) * z / focal
        cam_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)  # (n, 4)
        world_points = cam_points @ poses[i].T  # c2w: camera-space column vec -> world
        all_points.append(world_points[:, :3])
        all_colors.append(rgb.reshape(-1, 3)[pick])

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"[build_replica_pointcloud] {points.shape[0]} points total, "
          f"bounds min={points.min(axis=0)} max={points.max(axis=0)}")

    torch.save({
        "points": torch.from_numpy(points).float(),
        "colors": torch.from_numpy(colors).float() / 255.0,
    }, OUTPUT_PATH)
    print(f"[build_replica_pointcloud] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
