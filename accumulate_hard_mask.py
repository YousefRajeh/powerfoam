"""Per-view HARD mask assignment: give each cell one mask per view, not a blend of several.

THE PROBLEM THIS ATTACKS. The standard lift forms, for cell j in view v, the contribution-weighted
mean of the per-pixel embeddings it touched. When a cell's pixels fall inside several different SAM
masks -- which they routinely do -- that mean blends unit vectors pointing at different objects.
Measured on scene0062_00 (level 3, black-both), the mean per-view feature norm c_intra is 0.790:
about a fifth of every cell's per-view magnitude is destroyed by within-view blending, and c_intra
predicts per-cell correctness far better (decile spread 0.601) than the cross-view term c_inter does
(0.175). Crucially this is NOT a cell-size artefact -- c_intra rises with projected area
(Spearman +0.067, wrong sign for the size story) and still spans 0.000-1.000 inside a narrow size
band.

THE CHANGE. For each (cell, view), accumulate the rendering weight each MASK received from that
cell's pixels, take the arg-max mask, and use that single mask's embedding as the cell's observation
for that view:

    m*(j, v) = argmax_m  sum_{pixels p in mask m} A_pj
    f_j^(v)  = embedding[m*(j, v)]          (unit norm by construction)
    W_j^(v)  = sum_p A_pj                   (unchanged: the cell's total weight in this view)

so c_intra becomes exactly 1 and the only surviving conflict is genuine cross-view disagreement.
This is a decision about WHICH observation a cell gets, not a new regulariser -- the disjoint
partition is what makes "this cell's pixels" a well-defined set to take an arg-max over.

WHAT WOULD MAKE IT FAIL. Arg-max discards the runner-up entirely, so a cell genuinely straddling a
true object boundary now commits to one side instead of hedging. If mIoU falls, that is the reason,
and the fix would be a confidence-gated version (hard-assign only when the winning mask holds a
clear majority) rather than abandoning the idea. `--min-margin` implements exactly that gate.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def load_masks(feature_dir, stem, level, H, W):
    """(n_masks, F) unit-norm embeddings and an (H, W) int32 map of mask ids (-1 = none)."""
    f = np.load(Path(feature_dir) / f"{stem}_f.npy").astype(np.float32)
    s = np.load(Path(feature_dir) / f"{stem}_s.npy")
    if s.ndim == 3:
        # a single-level artifact stores its one level at index 0 (SAM_ONLY_LEVEL extraction);
        # a 4-level artifact stores LangSplat's hierarchy and we want the requested one.
        s = s[level] if s.shape[0] > level else s[0]
    if s.shape != (H, W):
        t = torch.from_numpy(s.astype(np.int32))[None, None].float()
        s = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest")[0, 0].numpy().astype(np.int32)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    f = f / np.maximum(n, 1e-8)
    return torch.from_numpy(f), torch.from_numpy(s.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="hard-assign only when the winning mask holds this share of the cell's "
                         "weight in that view; below it, fall back to the soft mean")
    ap.add_argument("--out", default="artifacts/adaptive/s0062_hardmask.pt")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]

    feat_dir = a.features.format(scene=a.scene)
    images_dir = Path(cargs.data_path) / cargs.scene / "images"
    stems = sorted(p.stem for p in images_dir.iterdir())
    assert len(stems) == len(dh.cameras), f"{len(stems)} images vs {len(dh.cameras)} cameras"

    dev = "cuda"
    F = None
    num_hard = torch.zeros(P, device=dev)
    num_soft = torch.zeros(P, device=dev)
    support = torch.zeros(P, device=dev)
    svw = torch.zeros(P, device=dev)
    intra_hard = torch.zeros(P, device=dev)
    intra_soft = torch.zeros(P, device=dev)
    n_fallback = 0
    n_assigned = 0

    for vi, cam in enumerate(dh.cameras):
        H, W = int(cam.height), int(cam.width)
        f_masks, seg = load_masks(feat_dir, stems[vi], a.level, H, W)
        f_masks = f_masks.to(dev)
        seg = seg.reshape(-1).to(dev)
        if F is None:
            F = f_masks.shape[1]
            num_hard = torch.zeros(P, F, device=dev)
            num_soft = torch.zeros(P, F, device=dev)
        M = f_masks.shape[0]

        op = export_operator_for_views(model, [cam], [vi])
        rows = op.row_indices.to(dev)
        cols = op.col_indices.to(dev)
        vals = op.values.to(dev).float()
        mid = seg[rows]                                  # mask id per nonzero
        ok = mid >= 0
        rows, cols, vals, mid = rows[ok], cols[ok], vals[ok], mid[ok]
        if vals.numel() == 0:
            continue

        # per-view total weight per cell (unchanged by the assignment rule)
        Wv = torch.zeros(P, device=dev).index_add_(0, cols, vals)
        present = Wv > 0

        # weight each MASK received from each cell, as a flat (cell, mask) histogram
        key = cols * M + mid
        hist = torch.zeros(P * M, device=dev).index_add_(0, key, vals).view(P, M)
        best_w, best_m = hist.max(dim=1)
        share = torch.where(Wv > 0, best_w / Wv.clamp_min(1e-12), torch.zeros_like(Wv))

        # soft per-view feature (the standard lift), for the fallback and for comparison
        soft = torch.zeros(P, F, device=dev).index_add_(0, cols, vals[:, None] * f_masks[mid])
        soft = torch.where(present[:, None], soft / Wv.clamp_min(1e-12)[:, None],
                           torch.zeros_like(soft))

        hard = f_masks[best_m]                                   # unit norm by construction
        use_hard = present & (share >= a.min_margin)
        n_assigned += int(use_hard.sum())
        n_fallback += int((present & ~use_hard).sum())
        per_view = torch.where(use_hard[:, None], hard, soft)

        num_hard += Wv[:, None] * per_view
        num_soft += Wv[:, None] * soft
        support += Wv
        svw += Wv ** 2
        intra_hard += Wv * per_view.norm(dim=1)
        intra_soft += Wv * soft.norm(dim=1)
        if vi % 10 == 0:
            print(f"  view {vi}/{len(dh.cameras)}  masks={M}  "
                  f"cells present={int(present.sum()):,}  mean winning share={share[present].mean():.3f}",
                  flush=True)

    valid = support > 0
    out = {
        "numerator": num_hard.cpu(), "numerator_soft": num_soft.cpu(),
        "support": support.cpu(), "sum_view_weight_sq": svw.cpu(),
        "intra_sum": intra_hard.cpu(), "intra_sum_soft": intra_soft.cpu(),
        "level": a.level, "min_margin": a.min_margin, "features": feat_dir,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(out, a.out)

    ci_h = (intra_hard[valid] / support[valid]).mean().item()
    ci_s = (intra_soft[valid] / support[valid]).mean().item()
    print(f"\n{int(valid.sum()):,} cells with support")
    print(f"hard-assigned (cell,view) pairs: {n_assigned:,}   fell back to soft: {n_fallback:,}")
    print(f"c_intra  soft {ci_s:.4f}  ->  hard {ci_h:.4f}   "
          f"(hard should be ~1.000 when min-margin=0)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
