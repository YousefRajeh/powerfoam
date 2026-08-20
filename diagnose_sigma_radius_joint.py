"""Are interior non-owner cells SEPARABLE from surface cells in (sigma, radius) space?

This is the cheapest question that governs the whole surface-concentration direction, and it
needs no training -- only checkpoints we already have.

alpha_i = 1 - exp(-sigma_i * delta_i(r)) is degenerate: infinitely many (sigma, r) pairs give
identical opacity, hence identical rendering, but different geometry. Nothing in the current
loss breaks that tie, so the optimizer settles wherever it drifts. Every proposal on the
table (exp reparameterization, distortion-to-weight gradient, thin-over-thick
canonicalization) is an attempt to break it in a chosen direction.

But all of them are pointless if the two populations are not separable to begin with. So:

  OWNER cells      = cells containing at least one GT point  -> proxy for "on a surface"
  NON-OWNER cells  = cells containing none                   -> proxy for "interior/empty"
                     (~90% of cells under softplus; the number that closed the
                      foam-exclusive clustering direction)

If non-owners are systematically low-sigma / large-r, they are identifiable from cell
parameters alone and a canonicalization or a weight gradient can act on them. If the two
populations overlap heavily, no reparameterization will separate what the loss cannot see,
and the direction should be reconsidered rather than pursued.

Reports per population: sigma and r distributions, the opacity proxy
alpha = 1 - exp(-sigma * 2r) (2r = characteristic chord), and -- the actual answer -- the
AUC of each single feature as a classifier of owner vs non-owner. AUC 0.5 = useless,
>0.75 = a usable signal.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np

from point_cloud_query import assign_points_to_power_cells
from evaluate_point_cloud_miou import load_scannet_pointcept_gt
from run_cluster_classify_eval import SCENES


def auc(scores, labels):
    """Rank-based AUC: P(score[positive] > score[negative]). No sklearn dependency."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so constant features score exactly 0.5
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def describe(name, x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"    {name:<16} n={len(x):>8}  median={q[2]:<10.4g} "
            f"IQR=[{q[1]:.4g}, {q[3]:.4g}]  p5/p95=[{q[0]:.4g}, {q[4]:.4g}]")


def run(run_dir, scene, gt_root, device="cuda"):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{run_dir}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{run_dir}/model.pt")

    centers = model.points.detach().cpu().numpy().astype(np.float64)
    radii = model.get_radii().detach().cpu().numpy().reshape(-1).astype(np.float64)
    sigma = model.get_density().detach().cpu().numpy().reshape(-1).astype(np.float64)
    act = getattr(args, "density_activation", "softplus")

    split = SCENES[scene]
    gt_points, _, _ = load_scannet_pointcept_gt(f"{gt_root}/{split}/{scene}", "segment20")
    assigned = assign_points_to_power_cells(gt_points, centers.astype(np.float32),
                                            radii.astype(np.float32),
                                            valid=np.ones(len(centers), bool), k=64)
    owner = np.zeros(len(centers), dtype=bool)
    owner[np.unique(assigned[assigned >= 0])] = True

    alpha = 1.0 - np.exp(-np.clip(sigma * 2.0 * radii, 0, 60))
    finite = np.isfinite(sigma) & np.isfinite(radii)
    owner, sigma, radii, alpha = owner[finite], sigma[finite], radii[finite], alpha[finite]

    print(f"\n=== {run_dir}  (activation={act}) ===")
    print(f"  cells={len(owner)}  owners={int(owner.sum())} "
          f"({owner.mean()*100:.1f}%)  NON-owners={(~owner).mean()*100:.1f}%")
    for nm, v in (("sigma", sigma), ("radius", radii), ("alpha", alpha)):
        print(f"  {nm}:")
        print(describe("owner", v[owner]))
        print(describe("non-owner", v[~owner]))
    res = {"run": run_dir, "activation": act,
           "non_owner_frac": float((~owner).mean()), "n_cells": int(len(owner))}
    print("  separability (AUC of the feature as an owner-vs-non-owner classifier):")
    for nm, v in (("sigma", sigma), ("radius", radii), ("alpha", alpha),
                  ("sigma*r", sigma * radii)):
        a = auc(v, owner)
        res[f"auc_{nm}"] = a
        verdict = "USABLE" if (a > 0.75 or a < 0.25) else ("weak" if (a > 0.6 or a < 0.4) else "none")
        print(f"    {nm:<10} AUC={a:.3f}   {verdict}")
    for nm, v in (("sigma", sigma), ("radius", radii), ("alpha", alpha)):
        res[f"median_owner_{nm}"] = float(np.median(v[owner]))
        res[f"median_nonowner_{nm}"] = float(np.median(v[~owner]))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--scene", default="scene0062_00")
    p.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = []
    for r in a.runs:
        try:
            rows.append(run(r, a.scene, a.gt_root))
        except Exception as e:
            print(f"[fail] {r}: {type(e).__name__}: {e}")
    json.dump(rows, open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")
    print("READ THIS: if AUC ~ 0.5 for every feature, interior and surface cells are NOT "
          "separable from cell parameters, and no reparameterization can fix what the loss "
          "cannot distinguish -- the direction needs rethinking, cheaply, before more GPU time.")


if __name__ == "__main__":
    main()
