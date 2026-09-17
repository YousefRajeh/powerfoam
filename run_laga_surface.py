"""Score LaGa's per-point labels: point mIoU + semantic surface metrics, 19/15/10 classes.

LaGa bypasses make_baseline_eval_inputs.py because its product is per-point LABELS, not 512-d
features: `get_point_seg` scores each class by LERF-style relevancy against the negatives
("object","things","stuff","texture"), zeroes anything under cosine_thresh=0.21, and argmaxes. So
there is no CLIP argmax left for us to apply -- the labels go straight into the metric.

INDEX IDENTITY, not assignment. LaGa was trained frozen at one Gaussian per GT vertex, and its
scene ply, contrastive ply and the scene's GT vertex count are all equal (verified per scene). So
point i of the label array IS GT point i; no power-cell query or nearest-neighbour step is involved,
and there is no assignment error to report.

TWO PROTOCOL ASYMMETRIES, both stated rather than silently normalised away:

1. QUERY SET. Every other baseline here is scored with a text bank built from the classes PRESENT in
   that scene; LaGa was run with the full 19-name bank (its own convention, cell 43). A larger bank
   is strictly harder, so this disadvantages LaGa. Predictions are therefore kept in the full 19
   label space and `calculate_metrics` averages over the classes present in GT, exactly as
   OpenGaussian's evaluator does -- rather than remapping LaGa's predictions into a present-only
   space, which would silently discard its wrong-but-absent-class predictions.

2. ABSTENTION. The cosine threshold means LaGa can decline to predict; those points keep label 0 and
   score as misses. Coverage is therefore NOT 100% as it is for the closed-set argmax baselines, and
   the per-scene unassigned fraction is recorded next to every score.

The opacity mask is OpenGaussian's rule at the same 0.1 threshold used everywhere else in this
project, applied to the scene Gaussians' sigmoid opacity, which for this frozen arm indexes GT
points directly.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       load_scannet_pointcept_gt, remap_gt_labels)

LABELS = r"D:\Downloads\baselines\LaGa\laga_scannet_labels"
POINTCEPT = r"D:\Downloads\scannet_pointcept"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
OPACITY_THRESH = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--out", default="artifacts/baseline_eval/laga_surface.json")
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args()
    enable_determinism()

    rows = []
    for scene, split in SPLIT.items():
        p = os.path.join(a.labels, f"{scene}.pt")
        if not os.path.exists(p):
            print(f"[miss] {scene}", flush=True)
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pred_full = d["labels"].numpy().astype(np.int64)      # 0 = unassigned, 1..19
        opacity = d["opacity"].numpy().reshape(-1)
        laga_names = list(d["target_names"])

        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, split, scene), "segment20")
        if gt_pts.shape[0] != pred_full.shape[0]:
            print(f"[MISALIGNED] {scene}: {pred_full.shape[0]:,} labels vs "
                  f"{gt_pts.shape[0]:,} GT points -- skipping", flush=True)
            continue
        pts = np.asarray(gt_pts, dtype=np.float64)
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        unassigned = float((pred_full == 0).mean()) * 100

        for cs in a.class_sets.split(","):
            names = list(OPENGAUSSIAN_CLASS_SETS[cs])
            # GT into LaGa's own full-bank label space: class j of `names` -> id j+1, everything
            # else -> 0 (ignored). calculate_metrics then averages over the classes present in GT.
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            # LaGa ran with the 19-name bank; for the 15/10 subsets its predictions that name a
            # class outside the subset cannot be right, so they are mapped to 0 (a miss) rather
            # than being re-argmaxed, which would require re-running its query stage.
            keep = {laga_names.index(n) + 1: i + 1 for i, n in enumerate(names)
                    if n in laga_names}
            pred = np.zeros_like(pred_full)
            for src, dst in keep.items():
                pred[pred_full == src] = dst

            gt_m = gt.copy()
            dropped_pct = 0.0
            if not a.no_mask:
                low = opacity < OPACITY_THRESH
                scored_before = int((gt_m != 0).sum())
                gt_m[low] = 0
                dropped_pct = (scored_before - int((gt_m != 0).sum())) / max(scored_before, 1) * 100

            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_m).long(),
                                                 torch.from_numpy(pred).long(), nc)
            sm = semantic_surface_metrics(GTSurfaceIndex(pts, gt_m, nc), pred)
            rec = {"tag": "laga_frozen", "scene": scene, "class_set": cs,
                   "miou": float(miou) * 100, "macc": float(macc) * 100,
                   "unassigned_pct": unassigned, "dropped_pct": dropped_pct,
                   "n_classes_present": len(set(np.unique(gt_m).tolist()) - {0})}
            rec.update({k: float(sm[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(sm.get(k), (int, float))})
            rows.append(rec)
            print(f"  laga {scene} {cs:15s} mIoU={rec['miou']:5.2f} mAcc={rec['macc']:5.2f} "
                  f"scd={rec['scd']:.4f} bF1={rec['boundary_f1']:.3f} "
                  f"unassigned={unassigned:.2f}% dropped={dropped_pct:.2f}%", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")

    for cs in a.class_sets.split(","):
        v = [r for r in rows if r["class_set"] == cs]
        if v:
            m = lambda k: float(np.mean([x[k] for x in v]))
            print(f"{cs:16s} n={len(v):>2}  mIoU {m('miou'):6.2f}  mAcc {m('macc'):6.2f}  "
                  f"SCD {m('scd'):.4f}  HD95 {m('hd95'):.4f}  BF1 {m('boundary_f1'):.3f}  "
                  f"unassigned {m('unassigned_pct'):.2f}%")


if __name__ == "__main__":
    main()
