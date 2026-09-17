"""Decompose the upstream error U over all 10 scenes, all views.

The oracle scores ~50 mIoU above the real pipeline. That gap is U, and quoting it as one number
says only "the upstream is bad". This measures the named stages so a bound can carry a MEASURED
upstream term instead of a fitted constant:

    pixel -> SAM region -> CLIP feature of that region -> class

    U_unassigned   SAM produced no region for the pixel (id -1): no evidence is deposited at all
    U_region_geom  the region straddles a label boundary, so one class cannot serve all its pixels
    U_clip         CLIP names the region wrongly

Each is a separate failure and they compose: the SAM ceiling (label every region with its own
majority GT class) bounds what any labeller could reach on those regions, and CLIP's pixel accuracy
is what the real upstream actually reaches. The difference is the labelling stage in isolation.

POOLING. Everything is accumulated as raw PIXEL COUNTS and divided once at the end. Averaging
per-view percentages would weight a view with 200k labelled pixels the same as one with 20k, and
the per-view CLIP accuracy swings 0.06-0.94, so that choice would materially move the headline.
Per-scene rows are kept so any subset can be re-pooled.

Only labelled pixels count: a region straddling labelled and unlabelled ground is charged for the
labelled part alone.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from oracle_projected import official_lut, LABEL2D_ROOT
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT
from upstream_modes import sam_label_image, _load_sam


def one_scene(scene, dev="cuda", max_views=0):
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    _, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    C = len(kept)
    lut = official_lut(kept, n2i)
    TT = embed_class_names(kept, dev)
    TT = TT / TT.norm(dim=-1, keepdim=True)

    stems = [os.path.splitext(f)[0] for f in
             sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))]
    if max_views:
        stems = stems[:max_views]

    # raw counts, summed over views
    n_lab = n_unassigned = 0
    occ_top = occ_tot = 0.0          # region purity: majority-share of labelled pixels
    n_cov = n_ceil_ok = n_clip_ok = 0
    reg_tot = reg_ok = 0
    used = 0
    for stem in stems:
        fp = os.path.join(LABEL2D_ROOT, scene, "label", f"{stem}.png")
        if not os.path.exists(fp):
            continue
        a = np.array(Image.open(fp)).astype(np.int64)
        H, W = a.shape
        gt = lut[np.clip(a, 0, lut.shape[0] - 1)].reshape(-1)
        try:
            _, s1 = sam_label_image(scene, stem, H, W, gt, C, dev, "sam_gt")
            _, s2 = sam_label_image(scene, stem, H, W, gt, C, dev, "sam_clip", TT=TT)
        except FileNotFoundError:
            continue
        lab = int(s1["px_labelled"])
        if lab == 0:
            continue
        used += 1
        n_lab += lab
        n_unassigned += int(s1["px_unassigned_of_labelled"])
        cov = lab - int(s1["px_unassigned_of_labelled"])
        n_cov += cov
        n_ceil_ok += int(round(s1["gt_pixel_acc_ceiling"] * cov))
        n_clip_ok += int(round(s2["clip_pixel_acc"] * cov))
        # region purity is already a ratio over labelled+assigned pixels; recover its numerator
        occ_top += s1["region_purity"] * cov
        occ_tot += cov
        if not np.isnan(s2["clip_region_acc"]):
            reg_tot += int(s1["n_regions_with_gt"])
            reg_ok += int(round(s2["clip_region_acc"] * s1["n_regions_with_gt"]))
    if used == 0:
        raise RuntimeError("no usable views")
    f = lambda a, b: (a / b) if b else float("nan")
    return {
        "scene": scene, "views": used, "C": C,
        "px_labelled": int(n_lab),
        "frac_unassigned": f(n_unassigned, n_lab),
        "region_purity": f(occ_top, occ_tot),
        "sam_ceiling_pixel_acc": f(n_ceil_ok, n_cov),
        "clip_pixel_acc": f(n_clip_ok, n_cov),
        "clip_region_acc": f(reg_ok, reg_tot),
        "n_regions_with_gt": int(reg_tot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--max-views", type=int, default=0, help="0 = all views")
    ap.add_argument("--out", default="artifacts/scannet/upstream_decomp.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["scene"] for r in rows}
    for sc in a.scenes.split(","):
        if sc in done:
            print(f"[{sc}] cached", flush=True); continue
        try:
            r = one_scene(sc, max_views=a.max_views)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}", flush=True); continue
        rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{sc}] views {r['views']:>3}  unassigned {r['frac_unassigned']:.2%}  "
              f"region purity {r['region_purity']:.4f}  SAM ceiling {r['sam_ceiling_pixel_acc']:.4f}  "
              f"CLIP pixel {r['clip_pixel_acc']:.4f}  CLIP region {r['clip_region_acc']:.4f}",
              flush=True)
    if not rows:
        return
    # pool by labelled pixels, not by scene
    W = np.array([r["px_labelled"] for r in rows], float)
    g = lambda k: float(np.average([r[k] for r in rows], weights=W))
    print(f"\nPOOLED over {len(rows)} scenes, {int(W.sum()):,} labelled pixels")
    print(f"  U_unassigned   (SAM gave no region)      {g('frac_unassigned'):.2%}")
    print(f"  region purity  (SAM boundary quality)    {g('region_purity'):.4f}")
    print(f"  SAM ceiling    (perfect labeller)        {g('sam_ceiling_pixel_acc'):.4f}")
    print(f"  CLIP pixel acc (the real upstream)       {g('clip_pixel_acc'):.4f}")
    print(f"  -> labelling stage costs                 "
          f"{g('sam_ceiling_pixel_acc') - g('clip_pixel_acc'):.4f} of pixel accuracy")
    v = [r["clip_pixel_acc"] for r in rows]
    print(f"  CLIP pixel acc per scene: min {min(v):.3f} max {max(v):.3f} "
          f"(spread is the variance term a bound must respect)")


if __name__ == "__main__":
    main()
