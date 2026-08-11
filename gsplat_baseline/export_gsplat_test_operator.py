"""Export a batch SparseFeatureOperator-format held-out test operator for the
trained gsplat garden checkpoint, across its 24 test views -- the
Gaussian-splat analogue of PowerFoam's test_operator.pt.

Uses LOCAL test-view ids (0..23, in ascending global-index order) as
row_view_ids and row-major (y, x) as row_pixels, at the same 420x648
resolution PowerFoam used -- the exact same row layout as
artifacts/garden/test_operator.pt, so PowerFoam's already-extracted
artifacts/garden/test_observations_v2.pt (the ground-truth OpenCLIP features
for these same 24 views, in this same row order) is directly reusable as
ground truth for this operator too, with no re-extraction needed.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gsplat_operator import export_view_operator  # noqa: E402

CKPT_DIR = os.environ.get("GSPLAT_ARTIFACT_DIR", r"D:\Downloads\powerfoam\artifacts\garden_gsplat")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")
OUTPUT_PATH = os.path.join(CKPT_DIR, "test_operator.pt")
DEVICE = "cuda"


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    means, quats, scales, opacities, colors = (
        ckpt["means"].to(DEVICE), ckpt["quats"].to(DEVICE),
        torch.exp(ckpt["scales"]).to(DEVICE), torch.sigmoid(ckpt["opacities"]).to(DEVICE),
        torch.sigmoid(ckpt["colors"]).to(DEVICE),
    )
    K, width, height = ckpt["K"].to(DEVICE), ckpt["width"], ckpt["height"]
    camtoworlds = ckpt["camtoworlds"]
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)
    test_idx = ckpt["test_idx"]
    num_primitives = means.shape[0]
    print(f"[export_gsplat_test_operator] {num_primitives} gaussians, {len(test_idx)} test views, {width}x{height}")

    # row_view_ids/row_pixels are per-ROW (one entry per unique pixel across
    # all views, length num_rows), NOT per-nonzero -- matches
    # SparseFeatureOperator's convention (see feature_metrics()'s per-view
    # breakdown, which indexes cosine[num_rows] by view_ids[num_rows]).
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=DEVICE), torch.arange(width, device=DEVICE), indexing="ij",
    )
    pixel_grid = torch.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], dim=-1)  # (H*W, 2), row-major

    all_rows, all_cols, all_vals, all_view_ids, all_pixels = [], [], [], [], []
    t0 = time.time()
    for local_id, view_id in enumerate(test_idx.tolist()):
        row_indices, col_indices, values, _, _ = export_view_operator(
            means, quats, scales, opacities, colors, viewmats_all[view_id], K, width, height,
        )
        all_rows.append(row_indices + local_id * height * width)
        all_cols.append(col_indices)
        all_vals.append(values)
        all_view_ids.append(torch.full((height * width,), local_id, dtype=torch.long, device=DEVICE))
        all_pixels.append(pixel_grid)
        print(f"[export_gsplat_test_operator] view {local_id + 1}/{len(test_idx)} (global {view_id}) nnz={row_indices.numel()}")

    elapsed = time.time() - t0
    state = {
        "row_indices": torch.cat(all_rows).cpu(),
        "col_indices": torch.cat(all_cols).cpu(),
        "values": torch.cat(all_vals).cpu(),
        "num_rows": len(test_idx) * height * width,
        "num_primitives": num_primitives,
        "row_view_ids": torch.cat(all_view_ids).cpu(),
        "row_pixels": torch.cat(all_pixels).cpu(),
    }
    print(f"[export_gsplat_test_operator] TIMING elapsed_sec={elapsed:.3f} total_nnz={state['row_indices'].numel()}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUTPUT_PATH)
    print(f"[export_gsplat_test_operator] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
