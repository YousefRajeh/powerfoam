"""Score Splat Feature Solver's OWN lifted features in OUR protocol, 10 scenes.

`artifacts/sfs_frozen/<scene>/ckpt_29999_rank0_features.pt` is the canonical SFS lift: produced
by their `distill.py`, on a checkpoint bit-identical to `recon_remote/gs_froz/<scene>/ckpt.pt`
(means/opacities/scales all max|diff| = 0). Table~\\ref{tab:3dseg} currently carries SFS only as a
PUBLISHED row (33.33 / 51.35 at 19 classes); our own "3DGS, frozen" row is OUR solver on their
geometry, which is a different thing.

Running their features through our evaluation does two jobs at once:

  1. it checks our harness against a published number we did not produce -- if their features
     score near 33.33 here, our protocol is not quietly advantaging us;
  2. it puts their real pipeline on the same axis as the overlap measurements, so the
     representation gap can be attributed rather than asserted.

No rendering and no solve: this reads the finished feature field and applies per-primitive cosine
argmax, exactly as every other row of that table does.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics,
                                       load_gaussian_means_opacities, apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_nearest_center
from diagnose_holes import SCENES, GT_ROOT

SFS = "artifacts/sfs_frozen"


def one(scene, class_set, opacity_threshold, gt_opacity_mask, dev="cuda"):
    ck = f"{SFS}/{scene}/ckpt_29999_rank0.pt"
    ft = f"{SFS}/{scene}/ckpt_29999_rank0_features.pt"
    means, opac = load_gaussian_means_opacities(ck, dev)
    X = torch.load(ft, map_location=dev, weights_only=False).float()
    assert X.shape[0] == means.shape[0], (X.shape, means.shape)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    text = embed_class_names(kept, dev)

    valid = opac >= opacity_threshold
    assigned = assign_points_to_nearest_center(
        gt_pts, means, valid=None if gt_opacity_mask else valid)
    if gt_opacity_mask:
        gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, opac, opacity_threshold, scene)

    pred = (torch.nn.functional.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    pl = np.zeros(len(gt_pts), np.int64)
    own = assigned >= 0
    pl[own] = pred[assigned[own]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pl).long(), C + 1)
    nz = X.norm(dim=-1)
    return dict(scene=scene, P=int(X.shape[0]), miou=float(miou), macc=float(macc),
                covered=float((nz > 0).float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/sfs_canonical_eval.json")
    a = ap.parse_args()
    res = {}
    PUBLISHED = {"opengaussian19": (33.33, 51.35), "opengaussian15": (36.43, 55.38),
                 "opengaussian10": (44.74, 63.53)}
    for cs in a.class_sets.split(","):
        rows = []
        for sc in a.scenes.split(","):
            try:
                rows.append(one(sc, cs, a.opacity_threshold, not a.no_gt_opacity_mask))
            except Exception as e:
                print(f"[{cs}/{sc}] SKIP {type(e).__name__}: {e}")
                continue
        res[cs] = rows
        if rows:
            mi = st.mean(r["miou"] for r in rows) * 100
            ma = st.mean(r["macc"] for r in rows) * 100
            p = PUBLISHED.get(cs)
            tag = f"   published {p[0]:.2f} / {p[1]:.2f}   delta {mi-p[0]:+.2f} / {ma-p[1]:+.2f}" if p else ""
            print(f"{cs:<16} n={len(rows):2d}  mIoU {mi:6.2f}  mAcc {ma:6.2f}{tag}")
            print("   per-scene mIoU: " + " ".join(f"{r['miou']*100:.1f}" for r in rows))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
