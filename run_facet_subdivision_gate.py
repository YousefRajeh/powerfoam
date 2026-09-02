"""Gate test: would per-FACET features have the capacity to fix multi-GT-per-cell errors?

THE IDEA (user's). A power cell is a convex polyhedron whose boundary is exactly one planar facet
per adjacent cell -- that is `adjacency_true_facet`, mean degree 15.3. Instead of one feature per
cell, give each facet its own feature, and attribute a ray to facets using its entry/exit points.
Gaussians cannot do this: no facets, no bounded disjoint partition, no exact dual.

WHAT THIS SCRIPT MEASURES, AND WHAT IT DOES NOT. Full per-facet solving needs the traversal kernel
to record entry/exit facet IDs, which `export_feature_operator` does not return -- a real renderer
change. Before paying for that, test the NECESSARY condition: inside cells that contain more than one
GT class, does the facet partition ALIGN with the label boundary? If the classes are scrambled across
facet-sides, per-facet features cannot separate them however well they are solved, and the renderer
work is wasted. If they align, the capacity is there and the renderer change is worth doing.

A point's facet-side is its SECOND-nearest cell in power distance: the facet it is closest to is the
one shared with that neighbour. So the sub-partition is computable exactly, with no new geometry.

THE BOUND THIS SITS UNDER. The oracle ceiling is 91.92 on ScanNet++ (cell purity 0.976) against
26.59 achieved, so the ENTIRE multi-GT-per-cell problem is worth <=8 mIoU. This test can only tell
us how much of that 8 is reachable -- it cannot address the 65-point gap, which the per-class
breakdown attributes to CLIP not separating adjacent same-appearance surfaces
(kitchen counter 0.00, kitchen cabinet 3.17, refrigerator 2.34).
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
from scipy.spatial import cKDTree

from determinism import enable_determinism
from build_true_facet_graph import load_points_radii
from point_cloud_query import assign_points_to_power_cells
from run_overnight import RECON
from run_spp_eval import benchmark_map, load_gt, coverage_filter

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494"]


def second_nearest_power(pts, centers, radii, first, k=24, chunk=200_000):
    """Cell whose facet the point lies closest to = 2nd smallest power distance."""
    tree = cKDTree(centers)
    out = np.full(pts.shape[0], -1, dtype=np.int64)
    r2 = radii ** 2
    for s in range(0, pts.shape[0], chunk):
        e = min(s + chunk, pts.shape[0])
        _, idx = tree.query(pts[s:e], k=min(k, centers.shape[0]), workers=-1)
        idx = np.atleast_2d(idx)
        d2 = ((pts[s:e, None, :] - centers[idx]) ** 2).sum(-1) - r2[idx]
        # mask out the owner, then take the argmin of what remains
        own = first[s:e][:, None]
        d2 = np.where(idx == own, np.inf, d2)
        out[s:e] = idx[np.arange(e - s), d2.argmin(1)]
    return out


def purity(groups, labels):
    """Fraction of points whose label equals their group's majority label."""
    order = np.argsort(groups, kind="stable")
    g, l = groups[order], labels[order]
    bnd = np.flatnonzero(np.diff(g)) + 1
    hit = 0
    for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(g)]):
        seg = l[a:b]
        if seg.size:
            hit += np.bincount(seg).max()
    return hit / max(len(g), 1)


def main():
    enable_determinism()
    top, r2b = benchmark_map()
    res = {}
    print(f"{'scene':<13}{'mixed cells':>12}{'pts in them':>12}"
          f"{'cell purity':>13}{'facet purity':>14}{'gain':>8}")
    for scene in SCENES:
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"artifacts/scannetpp/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            continue
        centers, radii = load_points_radii(ck)
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        vm = torch.load(sp, map_location="cpu", weights_only=True)["valid_mask"].numpy()
        gt, lab0, _ = load_gt(scene, top, r2b)
        a = assign_points_to_power_cells(gt, centers, radii, valid=vm, k=64)
        keepc, _, _ = coverage_filter(gt, a, centers, vm, 20.0)
        ok = (a >= 0) & keepc & (lab0 >= 0)
        pts, own, lab = gt[ok].astype(np.float64), a[ok], lab0[ok]

        # cells holding more than one GT class -- the only ones subdivision could ever help
        order = np.argsort(own, kind="stable")
        o_s, l_s = own[order], lab[order]
        bnd = np.flatnonzero(np.diff(o_s)) + 1
        mixed_cells = set()
        for s_, e_ in zip(np.r_[0, bnd], np.r_[bnd, len(o_s)]):
            if e_ - s_ >= 2 and np.unique(l_s[s_:e_]).size > 1:
                mixed_cells.add(int(o_s[s_]))
        sel = np.fromiter((c in mixed_cells for c in own), bool, len(own))
        if sel.sum() < 50:
            print(f"{scene:<13}{'(too few mixed cells)':>40}"); continue

        snd = second_nearest_power(pts[sel], centers, radii, own[sel])
        # facet sub-cell identity = (owner, neighbour-across-the-nearest-facet)
        sub = own[sel].astype(np.int64) * (centers.shape[0] + 1) + snd
        _, sub = np.unique(sub, return_inverse=True)
        p_cell = purity(own[sel], lab[sel])
        p_facet = purity(sub, lab[sel])
        res[scene] = {"mixed_cells": len(mixed_cells), "pts": int(sel.sum()),
                      "cell_purity": p_cell, "facet_purity": p_facet,
                      "gain": p_facet - p_cell,
                      "mean_subcells": float(len(np.unique(sub)) / max(len(mixed_cells), 1))}
        print(f"{scene:<13}{len(mixed_cells):>12,}{int(sel.sum()):>12,}"
              f"{p_cell:>13.4f}{p_facet:>14.4f}{p_facet - p_cell:>+8.4f}", flush=True)
    if res:
        g = np.mean([v["gain"] for v in res.values()])
        sc = np.mean([v["mean_subcells"] for v in res.values()])
        print(f"\nmean purity gain inside mixed cells: {g:+.4f}"
              f"   mean sub-cells per mixed cell: {sc:.2f}")
        print("Interpretation: this is the CAPACITY of a facet partition to separate the classes "
              "that share a cell. It is a necessary condition for per-facet features to help, not "
              "a sufficient one -- the 2D observations must also differ.")
    json.dump(res, open("artifacts/scannetpp/facet_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
