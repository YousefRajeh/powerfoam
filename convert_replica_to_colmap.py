"""Convert Replica room_0's native traj_w_c.txt (flattened 4x4 OpenCV-convention
camera-to-world matrices) + its depth-derived point cloud into a real COLMAP
binary triplet (cameras.bin/images.bin/points3D.bin), so it can be trained
with splat-distiller's gaussian_splatting/simple_trainer.py -- the exact same
gsplat-based 3DGS trainer already used, unmodified, for garden -- instead of
going through the buggy beta_splatting/DBS path.

This is a data-format adapter only: it does not touch any splat-distiller
code. It writes the standard COLMAP binary format (verified against the
pycolmap fork's own reader, submodules-independent -- this is COLMAP's
documented binary layout: https://colmap.github.io/format.html#binary-file-format).
"""
import shutil
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SRC_DIR = Path(r"D:\Downloads\powerfoam\data\replica\room_0")
DST_DIR = Path(r"D:\Downloads\powerfoam\data\replica\room_0_colmap")
NUM_FRAMES = 900
WIDTH, HEIGHT = 640, 480
FOCAL = 0.5 * WIDTH  # hfov = 90 deg, matches PowerFoam's own Replica loader
CX, CY = WIDTH / 2.0, HEIGHT / 2.0


def write_cameras_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", 1))  # num_cameras
        # camera_id, model_id(1=PINHOLE), width, height
        f.write(struct.pack("IiQQ", 1, 1, WIDTH, HEIGHT))
        f.write(struct.pack("dddd", FOCAL, FOCAL, CX, CY))


def write_images_bin(path, camtoworlds):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", NUM_FRAMES))  # num_images
        for i in range(NUM_FRAMES):
            c2w = camtoworlds[i]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            image_id = i + 1
            camera_id = 1
            f.write(struct.pack("<I4d3dI", image_id, qw, qx, qy, qz, t[0], t[1], t[2], camera_id))
            # Zero-padded: gsplat_ext's Parser sorts image_names lexicographically
            # (np.argsort(image_names)) to establish per-image position -- "rgb_0",
            # "rgb_1", "rgb_10", ... would sort out of numeric order and scramble
            # which real frame lands at which split position. Zero-padding keeps
            # sorted position == real frame index, which the train/test split
            # (and downstream evaluation against ground truth by frame index) both
            # depend on.
            f.write(f"rgb_{i:03d}.png".encode("ascii") + b"\x00")
            f.write(struct.pack("Q", 0))  # num_points2D = 0, no tracked keypoints needed


def write_points3D_bin(path, points, colors):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(points)))
        for idx, (xyz, rgb) in enumerate(zip(points, colors)):
            point3D_id = idx + 1
            r, g, b = [int(max(0, min(255, round(c)))) for c in rgb]
            f.write(struct.pack("<Q3d3BdQ", point3D_id, xyz[0], xyz[1], xyz[2], r, g, b, 1.0, 0))


def main():
    poses = np.loadtxt(SRC_DIR / "traj_w_c.txt").reshape(-1, 4, 4).astype(np.float64)
    assert poses.shape[0] == NUM_FRAMES

    import torch
    cloud = torch.load(SRC_DIR / "pointcloud.pt", map_location="cpu", weights_only=True)
    points = cloud["points"].numpy().astype(np.float64)
    colors = (cloud["colors"].numpy().astype(np.float64) * (255.0 if cloud["colors"].max() <= 1.0 else 1.0))

    (DST_DIR / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    images_dir = DST_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_FRAMES):
        src = SRC_DIR / "rgb" / f"rgb_{i}.png"
        dst = images_dir / f"rgb_{i:03d}.png"
        if not dst.exists():
            shutil.copyfile(src, dst)

    write_cameras_bin(DST_DIR / "sparse" / "0" / "cameras.bin")
    write_images_bin(DST_DIR / "sparse" / "0" / "images.bin", poses)
    write_points3D_bin(DST_DIR / "sparse" / "0" / "points3D.bin", points, colors)
    print(f"wrote COLMAP triplet + {NUM_FRAMES} images to {DST_DIR}")
    print(f"points3D: {len(points)} points")


if __name__ == "__main__":
    main()
