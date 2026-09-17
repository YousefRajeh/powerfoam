"""Do OUR exported operator weights equal Splat Feature Solver's `vis`?

Every comparison we make between their lift and ours assumes it. We build A ourselves from
`gsplat.rasterize_to_indices_in_range` and recompute the alpha-compositing weight in PyTorch;
their kernel produces `vis` inside CUDA and, per `distill.py:21`, accumulates `vis*vis`. Those
agree only if our recomputation matches their kernel, which has never been checked.

The test is exact and leaves no room for a view-set or intrinsics mismatch: both implementations
are given the SAME camera, taken from THEIR dataset, on the same checkpoint.

  theirs   renderer.inverse_render(K, extrinsic, W, H, features=ones(1 channel))
           -> (features_per_image, weights_per_image, ids)
           `weights_per_image` is what accumulates into `splat_weights`, i.e. their per-primitive
           weight total for this view.
  ours     export_view_operator(...) -> (row, col, val); per-primitive totals are
           sum_i val_ij and sum_i val_ij^2.

If their kernel squares, `weights_per_image` matches OUR sum of val^2 and not our sum of val.
Whichever it matches is then the documented fact, rather than an assumption.

Run inside the splat-distiller env, under vcvars (gsplat JIT-compiles on import):
    cmd /c run_verify_vis.bat
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\splat-distiller")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--feature-folder", default="SAMOpenCLIP_features")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--tikhonov", type=float, default=None)
    a = ap.parse_args()
    dev = "cuda"

    from gsplat_ext import Dataset, Parser, GaussianPrimitive, GaussianRenderer
    from export_gsplat_operator import export_view_operator

    parser = Parser(data_dir=a.dir, factor=1, test_every=8)
    ds = Dataset(parser, split="train", load_features=False)
    data = ds[a.view]
    K = torch.as_tensor(data["K"]).float().to(dev)
    c2w = torch.as_tensor(data["camtoworld"]).float().to(dev)
    img = torch.as_tensor(data["image"])
    H, W = int(img.shape[0]), int(img.shape[1])
    print(f"view {a.view}: {W}x{H}")

    splats = GaussianPrimitive()
    splats.from_file(a.ckpt, tikhonov=a.tikhonov)
    splats.to(dev)
    renderer = GaussianRenderer(splats)

    # THEIRS: one channel of ones, so the returned feature sum is the weight sum under whatever
    # weighting their kernel uses, and `weights_per_image` is their accumulator verbatim.
    feats = torch.ones(1, H, W, 1, device=dev)
    with torch.no_grad():
        f_img, w_img, ids = renderer.inverse_render(
            K=K[None], extrinsic=c2w[None], width=W, height=H, features=feats)
    P = splats.geometry["means"].shape[0]
    theirs = torch.zeros(P, device=dev, dtype=torch.float64)
    theirs[ids] = w_img.reshape(-1).double()
    print(f"theirs: {int((theirs > 0).sum()):,} primitives touched, "
          f"sum {float(theirs.sum()):.6e}, max {float(theirs.max()):.6e}")

    # OURS: the same camera, the same checkpoint
    g = splats.geometry
    viewmat = torch.linalg.inv(c2w.double()).float()
    with torch.no_grad():
        row, col, val, *_ = export_view_operator(
            g["means"], g["quats"], g["scales"], g["opacities"].reshape(-1),
            torch.zeros((P, 1), device=dev), viewmat, K, W, H,
            max_hits_per_pixel=a.cap, transmittance_floor=1e-3)
    col = col.long()
    s1 = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, col, val.double())
    s2 = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, col, val.double() ** 2)
    print(f"ours:   {int((s1 > 0).sum()):,} primitives touched, "
          f"sum(val) {float(s1.sum()):.6e}, sum(val^2) {float(s2.sum()):.6e}")

    def cmp(name, mine):
        m = (theirs > 0) | (mine > 0)
        both = (theirs > 0) & (mine > 0)
        num = float((theirs[m] - mine[m]).abs().sum())
        den = float(theirs[m].abs().sum()) + 1e-30
        rel = num / den
        if int(both.sum()) > 1:
            x, y = theirs[both].cpu().numpy(), mine[both].cpu().numpy()
            r = float(np.corrcoef(x, y)[0, 1])
            ratio = float(np.median(y / np.maximum(x, 1e-30)))
        else:
            r, ratio = float("nan"), float("nan")
        print(f"  {name:<14} rel L1 diff {rel:9.3e}   corr {r:8.5f}   "
              f"median(ours/theirs) {ratio:8.4f}   support overlap "
              f"{int(both.sum()):,}/{int(m.sum()):,}")
        return rel

    print("\ncomparing their per-primitive weight against each of ours:")
    r1 = cmp("sum(val)", s1)
    r2 = cmp("sum(val^2)", s2)
    best = "sum(val^2)  [their kernel SQUARES]" if r2 < r1 else "sum(val)  [their kernel is linear in vis]"
    print(f"\n=> closest match: {best}")
    print(f"   VERDICT: {'MATCH' if min(r1, r2) < 1e-3 else 'MISMATCH -- our weights are NOT their vis'}"
          f" (best rel L1 {min(r1, r2):.3e})")


if __name__ == "__main__":
    main()
