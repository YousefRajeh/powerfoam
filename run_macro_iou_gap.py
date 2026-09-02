"""How much mIoU is left on the table by MAJORITY labelling? -- the last decision-rule avenue.

WHY THIS EXISTS. `test_metric_audit.py` established that labelling each cell with its majority GT
class maximises ACCURACY, not macro-IoU: under mIoU every class counts equally regardless of size,
so a rare class can be worth more in a cell where it is a minority. The 91.92 "ceiling" is therefore
the majority-label score, a LOWER bound on what a per-cell method can reach.

This measures the gap. It is the one part of the decision rule not ruled out by
[[CSLS-paper-ideas-2026-08-31]], because every arm tested there attacked HUBNESS, whereas this
attacks the mismatch between argmax and the macro-IoU objective (Nowozin CVPR 2014; Koyejo et al.
NeurIPS 2014). If the gap is ~2 points the `iou_plugin` family is dead and the decision rule is
closed for good; if it is large, it is worth real effort.

THE SEARCH IS SMALL, NOT COMBINATORIAL. Assigning cell c a class m that does not occur in c leaves
I_m unchanged while increasing P_m (hurting m) and removes n_{c,l} from I_l (hurting l) -- strictly
worse on both terms. So an optimal label is always a class PRESENT in the cell, and a pure cell has
exactly one candidate. Cells are 97.6% pure, so coordinate ascent runs over ~2.4% of them with a
handful of candidates each.

WHAT IS REPORTED. Coordinate ascent gives a LOWER bound on the true optimum (it can stall in a local
maximum), so the reported gap is conservative: the real ceiling is at least this high.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from determinism import enable_determinism
from evaluate_point_cloud_miou import remap_gt_labels
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_overnight import RECON, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


def cell_histograms(assigned, gt, n_cells, n_cls):
    """H[c,k] = number of scored GT points in cell c whose label is k+1 (k is 0-based class id).

    Only OWNED points with a real label enter H. Unowned points still count towards the per-class
    GT totals (they are scored as pred=0, i.e. a false negative), which is why T is computed
    separately over all scored points.
    """
    lab = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
    scored = lab > 0
    T = np.bincount(lab[scored] - 1, minlength=n_cls).astype(np.int64)
    own = scored & (assigned >= 0)
    flat = assigned[own].astype(np.int64) * n_cls + (lab[own] - 1)
    H = np.bincount(flat, minlength=n_cells * n_cls).reshape(n_cells, n_cls).astype(np.int64)
    return H, T


def miou_from_counts(I, T, P, present):
    """mIoU from per-class intersection / gt-total / predicted-total, averaged over present classes.

    IoU_k = I_k / (T_k + P_k - I_k). Mirrors calculate_metrics, which averages only over classes
    occurring in the GT of this scene.
    """
    U = T + P - I
    iou = np.where(U > 0, I / np.maximum(U, 1), 0.0)
    return float(iou[present].mean()) * 100.0


def coordinate_ascent(H, T, present, labels, max_sweeps=50):
    """Greedily relabel cells to maximise mIoU. Only classes present in a cell are candidates."""
    n_cells, n_cls = H.shape
    N = H.sum(1)                                   # scored points owned by each cell
    I = np.zeros(n_cls, dtype=np.int64)
    P = np.zeros(n_cls, dtype=np.int64)
    np.add.at(I, labels, H[np.arange(n_cells), labels])
    np.add.at(P, labels, N)

    # Build candidate lists ONLY for impure cells. Doing it for all 700k would be a 700k-iteration
    # Python loop for no purpose: a pure cell has a single candidate and can never move.
    nnz = (H > 0).sum(1)
    movable = np.flatnonzero(nnz > 1)
    cand = {int(c): np.flatnonzero(H[c]) for c in movable}
    log(f"    {len(movable):,} impure cells of {n_cells:,} ({len(movable)/max(n_cells,1)*100:.2f}%)")

    def iou_of(k, Ik, Pk):
        u = T[k] + Pk - Ik
        return Ik / u if u > 0 else 0.0

    changed_total = 0
    for sweep in range(max_sweeps):
        changed = 0
        for c in movable:
            cur = labels[c]
            base_cur = iou_of(cur, I[cur], P[cur])
            # remove the cell's contribution from its current class
            I_cur_wo, P_cur_wo = I[cur] - H[c, cur], P[cur] - N[c]
            best_gain, best_m = 0.0, cur
            for m in cand[int(c)]:
                if m == cur:
                    continue
                base_m = iou_of(m, I[m], P[m])
                new_cur = iou_of(cur, I_cur_wo, P_cur_wo)
                new_m = iou_of(m, I[m] + H[c, m], P[m] + N[c])
                gain = (new_cur - base_cur) + (new_m - base_m)
                if gain > best_gain + 1e-15:
                    best_gain, best_m = gain, m
            if best_m != cur:
                I[cur], P[cur] = I_cur_wo, P_cur_wo
                I[best_m] += H[c, best_m]; P[best_m] += N[c]
                labels[c] = best_m
                changed += 1
        changed_total += changed
        if changed == 0:
            break
    return labels, miou_from_counts(I.astype(float), T.astype(float), P.astype(float), present), \
        changed_total, sweep + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/macro_iou_gap.json")
    a = p.parse_args()
    enable_determinism()
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
    for scene in a.scenes.split(","):
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"artifacts/scannetpp/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.exists(sp) and os.path.isdir(ck)):
            continue
        centers, radii = load_points_radii(ck)
        vmn = torch.load(sp, map_location="cpu", weights_only=True)["valid_mask"].numpy()
        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        row = {}
        for K in sizes:
            pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
            if not pres: continue
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            C = len(pres)
            H, T = cell_histograms(assigned, gt_t, len(centers), C)
            present = np.flatnonzero(T > 0)
            maj = H.argmax(1)
            maj[H.sum(1) == 0] = 0
            N = H.sum(1)
            I = np.zeros(C, np.int64); P = np.zeros(C, np.int64)
            np.add.at(I, maj, H[np.arange(len(maj)), maj]); np.add.at(P, maj, N)
            m_maj = miou_from_counts(I.astype(float), T.astype(float), P.astype(float), present)
            _, m_opt, nch, sw = coordinate_ascent(H, T, present, maj.copy())
            row[f"top{K}"] = {"majority": m_maj, "ascent": m_opt, "gap": m_opt - m_maj,
                              "cells_moved": int(nch), "sweeps": int(sw)}
            log(f"  {scene} top{K}: majority {m_maj:.2f} -> ascent {m_opt:.2f} "
                f"(gap {m_opt-m_maj:+.2f}, {nch:,} cells moved in {sw} sweeps)")
        res[scene] = row
        json.dump(res, open(a.out, "w"), indent=1)

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        print(f"  majority ceiling : {np.mean([x['majority'] for x in ks]):7.2f}")
        print(f"  ascent  ceiling  : {np.mean([x['ascent'] for x in ks]):7.2f}")
        print(f"  GAP              : {np.mean([x['gap'] for x in ks]):+7.2f}")


if __name__ == "__main__":
    main()
