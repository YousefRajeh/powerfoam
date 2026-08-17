"""Export a SparseFeatureOperator-format held-out test operator for the REAL Splat
Feature Solver 3DGS reconstruction on room_0 (room0_splatdistiller/ckpts/ckpt_29999_rank0.pt),
across the SAME 113 test views PowerFoam's artifacts/replica_room0/test_operator.pt already
covers (camera poses reused from artifacts/replica_room0_gsplat/ckpt.pt, which has the exact
K/camtoworlds/test_idx for this scene) -- so the two operators are directly comparable via
feature_foam_lifting.operator's n_eff/entropy diagnostics, for the empirical validation of the
partitioning-bound proof.
"""
import sys
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")

import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from export_gsplat_operator import export_view_operator  # noqa: E402

SPLATDISTILLER_CKPT = r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\ckpts\ckpt_29999_rank0.pt"
CAMERA_SOURCE_CKPT = r"D:\Downloads\powerfoam\artifacts\replica_room0_gsplat\ckpt.pt"
OUTPUT_PATH = r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\test_operator.pt"
DEVICE = "cuda"


def main():
    splats = torch.load(SPLATDISTILLER_CKPT, map_location=DEVICE, weights_only=False)["splats"]
    means = splats["means"].to(DEVICE)
    quats = F.normalize(splats["quats"].to(DEVICE), p=2, dim=-1)
    scales = torch.exp(splats["scales"].to(DEVICE))
    opacities = torch.sigmoid(splats["opacities"].to(DEVICE))
    colors = torch.sigmoid(splats["sh0"].to(DEVICE).squeeze(1))  # DC term only -- placeholder for validation render, not used by n_eff
    num_primitives = means.shape[0]

    cams = torch.load(CAMERA_SOURCE_CKPT, map_location=DEVICE, weights_only=False)
    K, width, height = cams["K"].to(DEVICE), cams["width"], cams["height"]
    camtoworlds = cams["camtoworlds"]
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)
    test_idx = cams["test_idx"]
    print(f"[export_splatdistiller_test_operator] {num_primitives} gaussians, {len(test_idx)} test views, {width}x{height}")

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=DEVICE), torch.arange(width, device=DEVICE), indexing="ij",
    )
    pixel_grid = torch.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], dim=-1)

    all_rows, all_cols, all_vals, all_view_ids, all_pixels = [], [], [], [], []
    t0 = time.time()
    for local_id, view_id in enumerate(test_idx.tolist()):
        row_indices, col_indices, values, _, _ = export_view_operator(
            means, quats, scales, opacities, colors, viewmats_all[view_id], K, width, height,
        )
        all_rows.append((row_indices + local_id * height * width).cpu())
        all_cols.append(col_indices.cpu())
        all_vals.append(values.cpu())
        all_view_ids.append(torch.full((height * width,), local_id, dtype=torch.long).cpu())
        all_pixels.append(pixel_grid.cpu())
        if (local_id + 1) % 10 == 0 or local_id == len(test_idx) - 1:
            print(f"[export_splatdistiller_test_operator] view {local_id + 1}/{len(test_idx)} (global {view_id}) nnz={row_indices.numel()}")

    elapsed = time.time() - t0
    state = {
        "row_indices": torch.cat(all_rows),
        "col_indices": torch.cat(all_cols),
        "values": torch.cat(all_vals),
        "num_rows": len(test_idx) * height * width,
        "num_primitives": num_primitives,
        "row_view_ids": torch.cat(all_view_ids),
        "row_pixels": torch.cat(all_pixels),
    }
    print(f"[export_splatdistiller_test_operator] TIMING elapsed_sec={elapsed:.3f} total_nnz={state['row_indices'].numel()}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUTPUT_PATH)
    print(f"[export_splatdistiller_test_operator] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
