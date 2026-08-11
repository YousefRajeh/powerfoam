"""Experiment A/B overlap statistics (Kish's n_eff, normalized weight
entropy) for any exported SparseFeatureOperator -- generic so it can be
pointed at either PowerFoam's or the Gaussian-splat baseline's operator for
a direct, apples-to-apples comparison.
"""
import argparse
import json
from pathlib import Path

import torch

from feature_foam_lifting.operator import SparseFeatureOperator


def main():
    p = argparse.ArgumentParser(description="Report per-pixel overlap statistics for a SparseFeatureOperator")
    p.add_argument("--operator", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    a = SparseFeatureOperator.load(args.operator, args.device)
    n_eff = a.row_effective_sample_size()
    entropy = a.row_weight_entropy(normalized=True)
    hit = a.row_sums() > 0
    n_eff_hit, entropy_hit = n_eff[hit], entropy[hit]
    row_nnz = torch.bincount(a.row_indices, minlength=a.num_rows).float()

    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    report = {
        "operator": args.operator,
        "num_rows": a.num_rows, "num_primitives": a.num_primitives, "nnz": int(a.values.numel()),
        "hit_rows_fraction": float(hit.float().mean()),
        "n_eff_mean": float(n_eff_hit.mean()), "n_eff_median": float(n_eff_hit.median()),
        "n_eff_quantiles": {str(q): float(torch.quantile(n_eff_hit, q)) for q in qs},
        "entropy_normalized_mean": float(entropy_hit.mean()), "entropy_normalized_median": float(entropy_hit.median()),
        "entropy_normalized_quantiles": {str(q): float(torch.quantile(entropy_hit, q)) for q in qs},
        "near_one_hot_fraction_neff_lt_1_5": float((n_eff_hit < 1.5).float().mean()),
        "fraction_neff_lt_3": float((n_eff_hit < 3).float().mean()),
        "median_candidate_hits_per_pixel": float(row_nnz[hit].median()),
        "mean_candidate_hits_per_pixel": float(row_nnz[hit].mean()),
    }
    if a.row_view_ids is not None:
        per_view = {}
        for v in torch.unique(a.row_view_ids).tolist():
            m = (a.row_view_ids == v) & hit
            if m.any():
                per_view[int(v)] = {"n_eff_mean": float(n_eff[m].mean()), "n_eff_median": float(n_eff[m].median())}
        report["per_view"] = per_view

    print(json.dumps({k: v for k, v in report.items() if k != "per_view"}, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
