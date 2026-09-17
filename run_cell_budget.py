"""Idea 10d: the foam's cell budget is allocated for photometry, not semantics.

THE CLAIM. Optimal cell density tracks the local variation of the signal being represented. A foam
trained photometrically puts cells where COLOUR varies. Semantics vary elsewhere -- a
uniformly-coloured object needs few cells to render but its BOUNDARY is exactly where segmentation
needs resolution. So expect under-resolved semantic boundaries in flat-coloured regions and
over-resolved cells inside textured ones.

THE FALSIFIABLE FORM, stated before running. Let spacing_i be the mean centre distance from cell i
to its power-adjacency neighbours -- the local cell scale, small where the foam spent budget.
Then:
  * if the budget follows photometry, spacing correlates NEGATIVELY with the local colour gradient;
  * if the budget also happened to follow semantics, spacing would correlate negatively with local
    LABEL disagreement too.
The claim is that the first correlation is real and the second is near zero. If both are strongly
negative the claim is wrong: the photometric budget would already be buying semantic resolution.
If neither is, the spacing statistic is not measuring what it is supposed to and nothing here is
interpretable -- that is the instrument check.

WHY IT STILL MATTERS THAT PURITY IS HIGH. artifacts/oracle_ceiling.json already showed 98.5% of GT
points sit in a cell whose majority label is their own, so the allocation is ADEQUATE even if it is
mismatched. This measurement quantifies the margin: how much bigger cells are where semantics turn
than where colour turns, and therefore how much a semantic-driven splitting criterion (10c) would
have to change before it could move a number.

Boundary cells are defined from GT, not from features: a labelled cell with at least one labelled
adjacency neighbour carrying a different majority label. That keeps the diagnostic independent of
the lifted features, whose over-smoothness is the very thing under suspicion (facet feature-cosine
was already measured as a weak boundary detector, AUC 0.65).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from determinism import enable_determinism
from evaluate_point_cloud_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES

POINTCEPT = r"D:\Downloads\scannet_pointcept"


def per_cell_edge_stats(values, adjacency, offsets):
    """Mean |value_i - value_j| over each cell's adjacency neighbours. `values` is (N,) or (N, D)."""
    v = values if values.ndim == 2 else values[:, None]
    deg = np.diff(offsets)
    src = np.repeat(np.arange(len(deg)), deg)
    diff = np.linalg.norm(v[src] - v[adjacency], axis=-1)
    out = np.zeros(len(deg))
    np.add.at(out, src, diff)
    return out / np.maximum(deg, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tmpl", default="output/scannet_{scene}_nonfrozen/model.pt")
    ap.add_argument("--solved", default="solved_geometric_median_nonfrozen_ogl3.pt")
    ap.add_argument("--out", default="artifacts/cell_budget.json")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = ap.parse_args()
    enable_determinism()
    res = {}

    for scene in args.scenes:
        mp = args.model_tmpl.format(scene=scene)
        sp = f"artifacts/scannet/{scene}/{args.solved}"
        if not (os.path.exists(mp) and os.path.exists(sp)):
            print(f"[skip] {scene}")
            continue
        m = torch.load(mp, map_location="cpu", weights_only=False)
        centers = m["points"].float().numpy().astype(np.float64)
        radii = torch.nn.functional.softplus(m["radii"].float().squeeze(), beta=100)
        radii = radii.numpy().astype(np.float64)
        adjacency = m["adjacency"].long().numpy()
        offsets = m["adjacency_offsets"].long().numpy()
        colour = 0.5 + m["texel_sv_rgb"].float().reshape(centers.shape[0], -1, 3).mean(1).numpy()
        vm = torch.load(sp, map_location="cpu", weights_only=True)["valid_mask"].numpy()

        gt, rawl, _ = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SCENES[scene], scene), "segment20")
        cache = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if os.path.exists(cache):
            assign = np.load(cache)
        else:
            assign = np.asarray(assign_points_to_power_cells(gt, centers, radii, valid=vm, k=64))
            np.save(cache, assign)

        n = centers.shape[0]
        keep = (assign >= 0) & (rawl >= 0)
        n_lab = int(rawl.max()) + 1
        votes = np.zeros((n, n_lab), dtype=np.int32)
        np.add.at(votes, (assign[keep], rawl[keep]), 1)
        has = votes.sum(1) > 0
        cell_lab = np.where(has, votes.argmax(1), -1)

        spacing = per_cell_edge_stats(centers, adjacency, offsets)
        col_grad = per_cell_edge_stats(colour, adjacency, offsets)

        deg = np.diff(offsets)
        src = np.repeat(np.arange(n), deg)
        both = has[src] & has[adjacency]
        diff_lab = both & (cell_lab[src] != cell_lab[adjacency])
        n_both = np.zeros(n); np.add.at(n_both, src, both.astype(np.float64))
        n_diff = np.zeros(n); np.add.at(n_diff, src, diff_lab.astype(np.float64))
        disagree = n_diff / np.maximum(n_both, 1)

        # THIRD CORRELATION, added after the first 10-scene run came back null on BOTH of the two
        # above (rho_colour flipped sign across scenes, mean -0.026). If cell spacing tracks
        # neither colour nor semantics, the obvious remaining explanation is that it tracks the
        # INPUT point cloud: these foams are initialised from the scene's SFM/GT cloud and
        # densified to a target count, so spacing would inherit that cloud's density and never
        # have been free to follow either signal. Local GT spacing = distance to the 8th nearest
        # GT point, averaged over the points a cell owns.
        from scipy.spatial import cKDTree
        dk, _ = cKDTree(np.asarray(gt)).query(np.asarray(gt), k=9, workers=-1)
        gt_local = dk[:, -1]
        s_sum = np.zeros(n); np.add.at(s_sum, assign[keep], gt_local[keep])
        s_cnt = np.zeros(n); np.add.at(s_cnt, assign[keep], 1.0)
        gt_spacing = s_sum / np.maximum(s_cnt, 1)

        sel = has & (n_both > 0)
        bnd = sel & (n_diff > 0)
        inr = sel & (n_diff == 0)
        r_col = spearmanr(spacing[sel], col_grad[sel]).statistic
        r_lab = spearmanr(spacing[sel], disagree[sel]).statistic
        row = {
            "cells": n, "labelled": int(sel.sum()),
            "frac_boundary": float(bnd.sum() / max(sel.sum(), 1)),
            "spacing_boundary": float(np.median(spacing[bnd])),
            "spacing_interior": float(np.median(spacing[inr])),
            "spacing_ratio": float(np.median(spacing[bnd]) / np.median(spacing[inr])),
            "colgrad_boundary": float(np.median(col_grad[bnd])),
            "colgrad_interior": float(np.median(col_grad[inr])),
            "spearman_spacing_colourgrad": float(r_col),
            "spearman_spacing_labeldisagree": float(r_lab),
            "spearman_spacing_gtspacing": float(spearmanr(spacing[sel], gt_spacing[sel]).statistic),
        }
        res[scene] = row
        print(f"{scene}: boundary cells {row['frac_boundary'] * 100:5.1f}%  "
              f"spacing bnd/int {row['spacing_ratio']:.3f}  "
              f"rho(spacing,colourgrad) {r_col:+.3f}  rho(spacing,labeldis) {r_lab:+.3f}  "
              f"rho(spacing,gtspacing) {row['spearman_spacing_gtspacing']:+.3f}")
        json.dump(res, open(args.out, "w"), indent=1)

    if res:
        k = list(res.values())
        print("\n=== mean over scenes ===")
        print(f"frac_boundary {np.mean([v['frac_boundary'] for v in k]) * 100:.1f}%  "
              f"spacing_ratio {np.mean([v['spacing_ratio'] for v in k]):.3f}  "
              f"rho_colour {np.mean([v['spearman_spacing_colourgrad'] for v in k]):+.3f}  "
              f"rho_label {np.mean([v['spearman_spacing_labeldisagree'] for v in k]):+.3f}  "
              f"rho_gtspacing {np.mean([v['spearman_spacing_gtspacing'] for v in k]):+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
