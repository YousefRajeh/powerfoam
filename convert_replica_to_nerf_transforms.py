"""Convert Replica room_0's native traj_w_c.txt (flattened 4x4 OpenCV-convention
camera-to-world matrices) into the NeRF-synthetic transforms_train.json /
transforms_test.json format that splat-distiller's beta_splatting/train.py
(via readNerfSyntheticInfo) can consume natively, unmodified.

This is a data-format adapter only -- it does not touch any splat-distiller
code. It supplies Replica's data in one of the two formats their own
sceneLoadTypeCallbacks already know how to read (the other being COLMAP).

Test split matches the convention already used elsewhere in this project for
Replica (idx % 8 == 0 held out as test) so PowerFoam and gsplat/DBS numbers on
this scene stay comparable.
"""
import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(r"D:\Downloads\powerfoam\data\replica\room_0")
NUM_FRAMES = 900
NATIVE_WIDTH = 640
CAMERA_ANGLE_X = math.pi / 2  # hfov = 90 deg, matches PowerFoam's own Replica loader


def opencv_c2w_to_blender_c2w(c2w):
    # readCamerasFromTransforms undoes this via c2w[:3,1:3] *= -1 (Blender->COLMAP
    # axes); the same op is self-inverse, so applying it again converts our
    # OpenCV/COLMAP-convention c2w into the Blender convention it expects on disk.
    out = c2w.copy()
    out[:3, 1:3] *= -1
    return out


def main():
    poses = np.loadtxt(DATA_DIR / "traj_w_c.txt").reshape(-1, 4, 4).astype(np.float64)
    assert poses.shape[0] == NUM_FRAMES, f"expected {NUM_FRAMES} poses, got {poses.shape[0]}"

    indices = np.arange(NUM_FRAMES)
    test_mask = indices % 8 == 0
    train_idx, test_idx = indices[~test_mask], indices[test_mask]

    def build_frames(idx_array):
        frames = []
        for i in idx_array.tolist():
            c2w_blender = opencv_c2w_to_blender_c2w(poses[i])
            frames.append({
                "file_path": f"./rgb/rgb_{i}",
                "transform_matrix": c2w_blender.tolist(),
            })
        return frames

    for split_name, idx_array in [("train", train_idx), ("test", test_idx)]:
        payload = {"camera_angle_x": CAMERA_ANGLE_X, "frames": build_frames(idx_array)}
        out_path = DATA_DIR / f"transforms_{split_name}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path} ({len(idx_array)} frames)")


if __name__ == "__main__":
    main()
