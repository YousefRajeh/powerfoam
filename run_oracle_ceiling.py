"""How much of the ScanNet++ shortfall is ASSIGNMENT and how much is FEATURES?

THE MEASUREMENT. Label every cell with the majority GT label of the points it owns, then score
normally. No features are involved, so this is the best score ANY per-cell method could achieve on
this assignment -- the quantisation ceiling Phi(a, S, L) of Theorem 1. Comparing it across datasets
splits the loss cleanly:

  * ceiling LOW   -> the assignment/discretisation is the bottleneck. More cells, better cells, or
                    a different correspondence would be required; no feature work can pass it.
  * ceiling HIGH  -> the assignment is fine and the entire gap is the features/classifier, so
                    reconstructing from more views would not help.

Reported alongside:
  * `purity` -- the volume-weighted fraction of GT points whose label equals their cell's majority
    label. This is the same quantity as the ceiling but read per point instead of per class.
  * `n_gt_per_cell` -- how many GT points share one cell. If several GT points with DIFFERENT
    labels fall in one cell, the cell cannot satisfy all of them however good its feature is.

Run on BOTH datasets with identical code so the comparison means something.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_cluster_classify_eval import SCENES as SN_SCENES
from run_overnight import SPP, RECON
from run_spp_eval import benchmark_map, load_gt, coverage_filter


def ceiling(assigned, owned, gt_lab, n_cells, n_cls):
    """Majority GT label per cell, then score. gt_lab is 1-based with 0 = ignore."""
    a = assigned[owned]; g = gt_lab[owned]
    keep = g > 0
    a, g = a[keep], g[keep]
    if a.size == 0:
        return 0.0, 0.0
    votes = np.zeros((n_cells, n_cls + 1), dtype=np.int32)
    np.add.at(votes, (a, g), 1)
    best = votes.argmax(1)
    pred = np.zeros(gt_lab.shape[0], dtype=np.int64)
    pred[owned] = best[assigned[owned]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pred).long(), n_cls + 1)
    purity = float((pred[owned][keep] == g).mean())
    return float(miou) * 100, purity


def main():
    enable_determinism()
    res = {"scannet": {}, "scannetpp": {}}

    print("=== ScanNet (19cls) ===")
    print(f"{'scene':<15}{'cells':>10}{'gt':>10}{'gt/cell':>9}{'ceiling':>9}{'purity':>8}")
    for scene in list(SN_SCENES):
        ck = f"output/scannet_{scene}_nonfrozen"
        sp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            continue
        c, r = load_points_radii(ck)
        vm = torch.load(sp, map_location="cpu", weights_only=True)["valid_mask"].numpy()
        gt, rawl, names = load_scannet_pointcept_gt(
            os.path.join(r"D:\Downloads\scannet_pointcept", SN_SCENES[scene], scene), "segment20")
        a = assign_points_to_power_cells(gt, c, r, valid=vm, k=64)
        own = a >= 0
        n2i = {n: i for i, n in enumerate(names)}
        pres = set(np.unique(rawl).tolist())
        kept = [n2i[n] for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
        g = remap_gt_labels(rawl, kept)
        ceil_, pur = ceiling(a, own, g, len(c), len(kept))
        occ = np.bincount(a[own], minlength=len(c))
        gpc = float(occ[occ > 0].mean())
        res["scannet"][scene] = {"ceiling": ceil_, "purity": pur, "gt_per_cell": gpc,
                                 "cells": len(c), "gt": int(gt.shape[0])}
        print(f"{scene:<15}{len(c):>10,}{gt.shape[0]:>10,}{gpc:>9.2f}{ceil_:>9.2f}{pur:>8.3f}")

    print("\n=== ScanNet++ (top100) ===")
    print(f"{'scene':<15}{'cells':>10}{'gt':>10}{'gt/cell':>9}{'ceiling':>9}{'purity':>8}")
    top, r2b = benchmark_map()
    for scene in SPP:
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"artifacts/scannetpp/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            continue
        c, r = load_points_radii(ck)
        vm = torch.load(sp, map_location="cpu", weights_only=True)["valid_mask"].numpy()
        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        a = assign_points_to_power_cells(gt_pts, c, r, valid=vm, k=64)
        own = a >= 0
        keepc, _, _ = coverage_filter(gt_pts, a, c, vm, 20.0)
        lab = np.where(keepc, lab0, -1)
        pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
        g = remap_gt_labels(lab, pres)
        ceil_, pur = ceiling(a, own, g, len(c), len(pres))
        occ = np.bincount(a[own], minlength=len(c))
        gpc = float(occ[occ > 0].mean())
        res["scannetpp"][scene] = {"ceiling": ceil_, "purity": pur, "gt_per_cell": gpc,
                                   "cells": len(c), "gt": int(gt_pts.shape[0]),
                                   "n_classes": len(pres)}
        print(f"{scene:<15}{len(c):>10,}{gt_pts.shape[0]:>10,}{gpc:>9.2f}{ceil_:>9.2f}{pur:>8.3f}")

    for k in ("scannet", "scannetpp"):
        d = res[k]
        if not d: continue
        print(f"\n{k}: ceiling {np.mean([v['ceiling'] for v in d.values()]):.2f}  "
              f"purity {np.mean([v['purity'] for v in d.values()]):.3f}  "
              f"gt/cell {np.mean([v['gt_per_cell'] for v in d.values()]):.2f}")
    json.dump(res, open("artifacts/oracle_ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
