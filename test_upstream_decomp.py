"""Verify run_upstream_decomp's arithmetic against a from-scratch recomputation.

The driver never sees raw pixels: it takes each view's RATIO from `sam_label_image` and multiplies
it back by the denominator it believes that ratio used, then pools the reconstructed counts. That is
only correct if every ratio's denominator is exactly what the driver assumes:

    region_purity, sam_ceiling_pixel_acc, clip_pixel_acc   -> labelled AND assigned pixels ("cov")
    frac_unassigned                                        -> labelled pixels
    clip_region_acc                                        -> regions that own >=1 labelled pixel

If any of those is off, the pooled headline is silently wrong in a way no invariant would catch.
So this recomputes all of them directly from the label image and the SAM maps, with no ratios in
the middle, and compares.

Also checks the orderings that must hold by construction:
    clip_pixel_acc <= sam_ceiling_pixel_acc     (naming cannot beat a perfect labeller)
    every rate in [0, 1]
and that pooling by pixel counts differs from a naive per-view mean -- if it does not, the pooling
choice is untested by this data and the claim that it matters needs dropping.
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

from oracle_projected import official_lut, LABEL2D_ROOT
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import GT_ROOT
from upstream_modes import sam_label_image, _load_sam
import run_upstream_decomp as drv

SCENE = sys.argv[1] if len(sys.argv) > 1 else "scene0062_00"
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev = "cuda"

d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", SCENE)) if os.path.isdir(q)][0]
_, raw, names = load_scannet_pointcept_gt(d, "segment20")
n2i = {n: i for i, n in enumerate(names)}
pres = set(np.unique(raw).tolist())
kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
C = len(kept)
lut = official_lut(kept, n2i)
TT = embed_class_names(kept, dev); TT = TT / TT.norm(dim=-1, keepdim=True)
stems = [os.path.splitext(f)[0] for f in
         sorted(os.listdir(f"data/scannet/{SCENE}_colmap/images"))][:NV]

# ---- from scratch: accumulate raw pixel/region counts, no ratios anywhere ----
tot_lab = tot_unass = tot_cov = 0
tot_major = tot_ceil = tot_clip = 0
tot_reg = tot_regok = 0
per_view_clip = []
for stem in stems:
    fp = os.path.join(LABEL2D_ROOT, SCENE, "label", f"{stem}.png")
    if not os.path.exists(fp):
        continue
    a = np.array(Image.open(fp)).astype(np.int64)
    H, W = a.shape
    gt = lut[np.clip(a, 0, lut.shape[0] - 1)].reshape(-1)
    s_flat, f = _load_sam(SCENE, stem, H, W)
    n_reg = f.shape[0]
    s_flat = np.where(s_flat >= n_reg, -1, s_flat)
    lab = gt > 0
    unass = s_flat < 0
    cov = lab & ~unass
    tot_lab += int(lab.sum()); tot_unass += int((lab & unass).sum()); tot_cov += int(cov.sum())

    occ = np.zeros((n_reg, C), np.int64)
    np.add.at(occ, (s_flat[cov], gt[cov] - 1), 1)
    tot_major += int(occ.max(1).sum())
    reg_has = occ.sum(1) > 0
    reg_gt = np.where(reg_has, occ.argmax(1) + 1, 0)
    tot_ceil += int((reg_gt[s_flat[cov]] == gt[cov]).sum())

    ft = torch.from_numpy(f).to(dev); ft = ft / ft.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    reg_cls = (ft @ TT.T).argmax(1).cpu().numpy() + 1
    ok = int((reg_cls[s_flat[cov]] == gt[cov]).sum())
    tot_clip += ok
    per_view_clip.append(ok / max(int(cov.sum()), 1))
    tot_reg += int(reg_has.sum()); tot_regok += int((reg_cls[reg_has] == reg_gt[reg_has]).sum())

ref = {
    "frac_unassigned": tot_unass / tot_lab,
    "region_purity": tot_major / tot_cov,
    "sam_ceiling_pixel_acc": tot_ceil / tot_cov,
    "clip_pixel_acc": tot_clip / tot_cov,
    "clip_region_acc": tot_regok / tot_reg,
}
got = drv.one_scene(SCENE, max_views=NV)

print(f"{SCENE}, {NV} views, {tot_lab:,} labelled px, {tot_reg} regions\n")
print(f"{'quantity':<26}{'from scratch':>14}{'driver':>14}{'abs diff':>12}")
bad = 0
for k, v in ref.items():
    dv = abs(got[k] - v)
    tol = 2e-4 if "acc" in k or "purity" in k else 1e-9   # driver rounds reconstructed counts
    flag = "" if dv < tol else "   <-- MISMATCH"
    bad += dv >= tol
    print(f"{k:<26}{v:>14.6f}{got[k]:>14.6f}{dv:>12.2e}{flag}")

print()
assert 0 <= ref["clip_pixel_acc"] <= 1 and 0 <= ref["region_purity"] <= 1, "rate outside [0,1]"
assert ref["clip_pixel_acc"] <= ref["sam_ceiling_pixel_acc"] + 1e-9, \
    "CLIP beat the perfect-labeller ceiling -- impossible, the ceiling is misdefined"
print("invariants OK: rates in [0,1]; CLIP never beats the SAM-region ceiling")

naive = float(np.mean(per_view_clip))
print(f"\npooling check: pixel-weighted {ref['clip_pixel_acc']:.4f} vs naive per-view mean "
      f"{naive:.4f}  (delta {abs(naive - ref['clip_pixel_acc']):.4f})")
if abs(naive - ref["clip_pixel_acc"]) < 1e-3:
    print("  NOTE: the two agree here, so this sample does not exercise the pooling choice")

print("\nRESULT:", "PASS" if bad == 0 else f"*** {bad} MISMATCH(ES) -- do not run ***")
sys.exit(1 if bad else 0)
