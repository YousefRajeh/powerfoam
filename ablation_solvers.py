"""Ablation: compare Feature-Foam accumulation/solver methods on the same
streamed stats, evaluated against the same held-out operator+observations.

Solvers compared: weighted average, squared-weight average, ridge
(closed-form diagonal), and the streaming cosine geometric median (VALA).
For each, reports held-out cos_sim/angular-error/valid_fraction (same metrics
`feature-foam-evaluate` reports) plus a clustering-cohesion proxy for
segmentation quality: mean cosine similarity between each valid primitive and
its assigned spherical-k-means centroid -- tighter cohesion means cluster
boundaries are less noisy, which is what a downstream discrete segmentation
(feature_foam_lifting.segment) actually depends on.
"""
import argparse
import json
from pathlib import Path

import torch

from feature_foam_lifting.operator import (
    AccumulatedFeatureStats,
    SparseFeatureOperator,
    feature_metrics,
    normalize_features,
    solve_geometric_median_from_stats,
    solve_ridge_closed_form_from_stats,
    solve_weighted_from_stats,
)
from feature_foam_lifting.segment import spherical_kmeans

SOLVERS = ("weighted", "squared-weighted", "ridge-closed-form", "geometric-median")


def solve(stats, name, ridge_mode="default"):
    if name == "weighted":
        x, valid = solve_weighted_from_stats(stats)
        report = {"solver": name}
    elif name == "squared-weighted":
        x, valid = solve_weighted_from_stats(stats, squared=True)
        report = {"solver": name}
    elif name == "ridge-closed-form":
        x, valid, report = solve_ridge_closed_form_from_stats(stats, ridge_mode)
    elif name == "geometric-median":
        x, valid, report = solve_geometric_median_from_stats(stats)
    else:
        raise ValueError(f"unknown solver {name!r}")
    return x, valid, report


def clustering_cohesion(x, valid, num_clusters, num_iters, seed):
    """Mean cosine similarity between each valid primitive and its own
    cluster's centroid, after fitting spherical k-means -- a proxy for how
    "clean"/separable the feature field is, independent of any held-out
    rendering. Not the same thing as segmentation *accuracy* (there's no
    ground-truth label here), but a solver that produces tighter, more
    separable clusters is doing the downstream segmentation step a favor.
    """
    unit = normalize_features(x)
    assignment, centroids = spherical_kmeans(unit, valid, num_clusters, num_iters=num_iters, seed=seed)
    assigned = assignment >= 0
    if not assigned.any():
        return {"mean_cohesion": 0.0, "num_assigned": 0}
    cohesion = (unit[assigned] * centroids[assignment[assigned]]).sum(-1)
    sizes = torch.bincount(assignment[assigned], minlength=num_clusters)
    return {
        "mean_cohesion": float(cohesion.mean()),
        "median_cohesion": float(cohesion.median()),
        "num_assigned": int(assigned.sum()),
        "cluster_sizes": sizes.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description="Ablate Feature-Foam accumulation methods on one stats file")
    p.add_argument("--stats", required=True)
    p.add_argument("--operator", required=True, help="Held-out SparseFeatureOperator .pt (e.g. test_operator.pt)")
    p.add_argument("--observations", required=True, help="Held-out observations tensor .pt matching --operator's rows")
    p.add_argument("--output", required=True)
    p.add_argument("--ridge-mode", default="default", choices=("none", "small", "default", "strong"))
    p.add_argument("--num-clusters", type=int, default=8)
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    stats = AccumulatedFeatureStats.load(args.stats, args.device)
    a = SparseFeatureOperator.load(args.operator, args.device)
    b = torch.load(args.observations, map_location=args.device, weights_only=True).float()
    if b.shape[0] != a.num_rows:
        raise SystemExit("observation count does not match operator rows")

    results = {}
    for name in SOLVERS:
        x, valid, report = solve(stats, name, args.ridge_mode)
        x_unit = x.clone()
        x_unit[valid] = normalize_features(x_unit[valid])
        del x

        # `predicted` alone is (num_rows, F) fp32 -- ~13GB on the garden
        # held-out operator (6.5M rows x 512). Without an explicit free
        # between solvers, four iterations' worth of it plus the (P, F)
        # feature fields accumulate and OOM even a 48GB GPU. Free it (and
        # empty_cache, since PyTorch's allocator otherwise holds the freed
        # block reserved rather than returning it to the driver) right after
        # the metrics that need it are computed.
        predicted = a.matmul(x_unit.float())
        metrics = feature_metrics(predicted, b, a.row_view_ids)
        del predicted
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        per_pixel_keys = ("cosine_per_pixel", "angle_rad_per_pixel", "angle_deg_per_pixel")
        metrics = {k: v for k, v in metrics.items() if k not in per_pixel_keys}

        cohesion = clustering_cohesion(x_unit, valid, args.num_clusters, args.kmeans_iters, args.seed)
        del x_unit
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        results[name] = {
            "solver_report": report,
            "valid_fraction": float(valid.float().mean()),
            "num_valid": int(valid.sum()),
            "heldout": metrics,
            "clustering_cohesion": cohesion,
        }
        print(f"{name:20s} valid_fraction={results[name]['valid_fraction']:.4f}  "
              f"cos_sim={metrics['cos_sim']:.4f}  ang_err_deg={metrics['angular_error_deg_mean']:.2f}  "
              f"cohesion={cohesion['mean_cohesion']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
