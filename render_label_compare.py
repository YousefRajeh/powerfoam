"""Side-by-side label renders: GT vs plain argmax vs Potts-smoothed, plus a CHANGE map.

WHY. IoU counts points and cannot distinguish a compact object mask from the same number of points
scattered as speckle; the facet-graph metrics (`n_comp`, `lcc_frac`) quantify that but do not show
it. This renders the actual per-point label map so the difference can be judged by eye, which is
how the fragmentation problem was noticed in the first place.

PANELS
  GT              ground-truth class per point
  plain           per-primitive cosine argmax -- the exact optimum of the unary problem
  potts lam=...   + Potts prior on the exact facet graph (the only place geometry can enter)
  CHANGE          what the prior did:   green = changed and now CORRECT
                                        red   = changed and now WRONG
                                        blue  = changed, both wrong (different wrong class)
                                        grey  = unchanged
The change map is the honest view: a smoothing prior always makes masks look tidier, and the
question is whether the tidying is right.

PROJECTION. Orthographic, painter's algorithm (far-to-near), so the nearest surface wins each
pixel. Default view is top-down, which for ScanNet rooms (no ceiling in the GT mesh) shows floor,
furniture tops and wall footprints -- the layout where speckle is most visible. `--axis` switches.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from determinism import enable_determinism
from graphcut import multiclass_potts_icm
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
from diagnose_scannet_miou import (assign_points_to_power_cells, load_foam,
                                   load_scannet_pointcept_gt)

GT_ROOT = r"D:\Downloads\scannet_pointcept"

# A fixed, high-contrast palette so a class has the SAME colour in every panel. Without this the
# panels cannot be compared at all -- matplotlib would recolour per-panel by label frequency.
PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def rasterise(pts, cols, axis=2, res=900, bg=(255, 255, 255), rad=1, crop=None, mark=None):
    """Orthographic splat with painter's algorithm (far first, so the nearest surface wins).

    `rad` draws each point as a (2*rad+1)^2 block -- single pixels make a 10^5-point cloud look
    like faint noise and hide exactly the speckle-vs-compact difference this figure exists to show.
    `crop` is a world-space (lo, hi) box on the two projected axes, for zooming on one object.
    `mark` is a boolean per point drawn LAST at full size, so a handful of changed points stay
    visible against 10^5 unchanged ones.
    """
    keep = [i for i in range(3) if i != axis]
    uv = pts[:, keep]
    if crop is not None:
        lo, hi = crop
    else:
        lo, hi = uv.min(0), uv.max(0)
    span = float(np.max(hi - lo)) or 1.0
    ij = ((uv - lo) / span * (res - 1)).astype(np.int64)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < res) & (ij[:, 1] >= 0) & (ij[:, 1] < res)
    depth = pts[:, axis]
    img = np.full((res, res, 3), bg, dtype=np.uint8)

    def blit(sel, r):
        idx = np.where(sel & inside)[0]
        if idx.size == 0:
            return
        idx = idx[np.argsort(depth[idx])]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = np.clip(res - 1 - ij[idx, 1] + dy, 0, res - 1)
                x = np.clip(ij[idx, 0] + dx, 0, res - 1)
                img[y, x] = cols[idx]

    blit(np.ones(pts.shape[0], dtype=bool) if mark is None else ~mark, rad)
    if mark is not None:
        blit(mark, max(rad, 2))
    return img


def class_box(pts, sel, pad=0.35, axis=2):
    """World-space crop box around the selected points, on the two projected axes."""
    keep = [i for i in range(3) if i != axis]
    uv = pts[sel][:, keep]
    lo, hi = uv.min(0) - pad, uv.max(0) + pad
    c = (lo + hi) / 2
    half = float(np.max(hi - lo)) / 2
    return c - half, c + half


def main():
    enable_determinism()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--feat", default=None)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--axis", type=int, default=2, help="projection axis: 2=top-down")
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rad", type=int, default=1)
    ap.add_argument("--focus", default=None, help="class name to zoom on")
    a = ap.parse_args()
    feat_file = a.feat or f"solved_geometric_median_{a.recon}_ogl3"
    device = "cuda"

    cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", a.scene)) if os.path.isdir(p)]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    centers, radii = load_foam(f"output/scannet_{a.scene}_{a.recon}", device)
    d = torch.load(f"artifacts/scannet/{a.scene}/{feat_file}.pt", map_location=device,
                   weights_only=True)
    feats = d["primitive_features"].to(device).float()
    valid = d["valid_mask"].cpu().numpy()
    g = torch.load(f"artifacts/ablation_cache/{a.scene}_pf_"
                   f"{'tfroz' if a.recon == 'truefrozen' else 'nonfroz'}_delaunay.pt",
                   map_location="cpu", weights_only=False)
    indptr = g["offsets"].numpy().astype(np.int64)
    indices = g["adjacent"].numpy().astype(np.int64)

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
    owned = assigned >= 0
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set]
            if name_to_id[n] in present]
    tids = [i for i, _ in kept]
    names = [n for _, n in kept]
    gt_np = remap_gt_labels(raw_labels, tids)              # 0 = ignore, 1..K
    text = embed_class_names(names, device)
    sim = (F.normalize(feats, dim=-1) @ text.T).cpu().numpy()

    lab_plain = sim.argmax(1)
    lab_potts = multiclass_potts_icm(sim, indptr, indices, lam=a.lam, live=valid.astype(bool))

    def to_points(lab):
        p = np.zeros(gt_points.shape[0], dtype=np.int64)
        p[owned] = lab[assigned[owned]] + 1
        return p

    pred_plain, pred_potts = to_points(lab_plain), to_points(lab_potts)
    scored = gt_np != 0

    def cols_from(lbl):
        c = np.full((gt_points.shape[0], 3), 225, dtype=np.uint8)
        m = lbl > 0
        c[m] = PALETTE[(lbl[m] - 1) % len(PALETTE)]
        return c

    # change map
    changed = (pred_plain != pred_potts) & scored
    ok_before = (pred_plain == gt_np) & scored
    ok_after = (pred_potts == gt_np) & scored
    ch = np.full((gt_points.shape[0], 3), 220, dtype=np.uint8)
    ch[changed & ok_after & ~ok_before] = [40, 170, 60]      # fixed
    ch[changed & ~ok_after & ok_before] = [214, 39, 40]      # broken
    ch[changed & ~ok_after & ~ok_before] = [70, 110, 200]    # wrong -> different wrong
    n_fix = int((changed & ok_after & ~ok_before).sum())
    n_brk = int((changed & ~ok_after & ok_before).sum())
    n_neu = int((changed & ~ok_after & ~ok_before).sum())

    acc0 = float(ok_before.sum()) / max(int(scored.sum()), 1)
    acc1 = float(ok_after.sum()) / max(int(scored.sum()), 1)

    panels = [("ground truth", cols_from(gt_np)),
              (f"plain argmax  (acc {acc0*100:.1f}%)", cols_from(pred_plain)),
              (f"+ Potts lam={a.lam}  (acc {acc1*100:.1f}%)", cols_from(pred_potts)),
              (f"change: fixed {n_fix:,} / broken {n_brk:,} / re-wrong {n_neu:,}", ch)]

    crop = None
    if a.focus:
        assert a.focus in names, f"{a.focus} not present; have {names}"
        k = names.index(a.focus) + 1
        sel = (gt_np == k) | (pred_plain == k) | (pred_potts == k)
        crop = class_box(gt_points, sel, axis=a.axis)
        print(f"focus '{a.focus}': {int(sel.sum()):,} points in view")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8))
    for ax, (title, c) in zip(axes, panels):
        mk = changed if title.startswith("change") else None
        ax.imshow(rasterise(gt_points, c, axis=a.axis, res=a.res, rad=a.rad,
                            crop=crop, mark=mk))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{a.scene} / {a.recon} / {a.class_set} -- "
                 f"net {n_fix - n_brk:+,} points corrected", fontsize=13)
    fig.tight_layout()
    out = a.out or (f"artifacts/scannet/{a.scene}_labels_lam{a.lam}"
                + (f"_{a.focus}" if a.focus else "") + ".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"fixed {n_fix:,}  broken {n_brk:,}  re-wrong {n_neu:,}  net {n_fix-n_brk:+,}")
    print(f"accuracy {acc0*100:.2f}% -> {acc1*100:.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
