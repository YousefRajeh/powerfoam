"""Solve accumulated per-primitive feature stats via the geometric-median
solver (this project's established default -- see ablation_solvers.py and
Experiment-F-scannet.md's solver-comparison note) and save the result in the
{"primitive_features": ..., "valid_mask": ...} format evaluate_point_cloud_miou.py
expects for --powerfoam-features / --gaussian-features.

Thin wrapper -- all solving logic lives in feature_foam_lifting.operator,
this just does load -> solve_geometric_median_from_stats -> save.
"""
import argparse

import torch

from feature_foam_lifting.operator import AccumulatedFeatureStats, solve_geometric_median_from_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    stats = AccumulatedFeatureStats.load(args.stats)
    x, valid, report = solve_geometric_median_from_stats(stats)
    torch.save({"primitive_features": x.cpu(), "valid_mask": valid.cpu()}, args.output)
    print(f"[solve_geometric_median] {int(valid.sum())}/{valid.numel()} valid primitives -> {args.output}")
    print(report)


if __name__ == "__main__":
    main()
