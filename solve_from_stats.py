"""Solve per-primitive features from accumulated stats under a CHOSEN solver.

WHY THIS EXISTS. solve_geometric_median.py hardcodes one solver, so every arm of the ablation
inherited the geometric median and `solver` was not actually an ablation axis. Worse, the
Gaussian lifting path (splat-distiller's distill.py:186) divides by accumulated weight -- a
WEIGHTED MEAN -- so putting its output beside geometric-median foam rows would confound
representation with solver: a gaussian-vs-foam delta could be either one.

This solves the SAME stats under any available solver, so every representation can appear
under every solver and the two effects separate.

STATS ARE HUGE (1-3 GB per scene-arm) and are the reason this is a separate step: solve every
solver you want from one stats file, then delete it, rather than re-lifting per solver.
`--delete-stats` does exactly that, which also matters because the remote disk sat at 97%
with a dozen lifts still queued.
"""
import argparse
import os

import torch

from feature_foam_lifting.operator import (AccumulatedFeatureStats,
                                           solve_geometric_median_from_stats,
                                           solve_inverse_variance_from_stats,
                                           solve_ridge_closed_form_from_stats,
                                           solve_weighted_from_stats)

SOLVERS = ("geometric_median", "weighted", "weighted_sq", "ridge", "inverse_variance")


def solve(stats, name):
    """-> (features, valid_mask, report). Normalisation matches each solver's own contract."""
    if name == "geometric_median":
        # already unit-norm by construction: every update step renormalises gm_z
        x, valid, rep = solve_geometric_median_from_stats(stats)
        return x, valid, rep
    if name in ("weighted", "weighted_sq"):
        x, valid = solve_weighted_from_stats(stats, squared=(name == "weighted_sq"))
        return x, valid, {"solver": name}
    if name == "ridge":
        out = solve_ridge_closed_form_from_stats(stats)
        x, valid = out[0], out[1]
        return x, valid, {"solver": "ridge"}
    if name == "inverse_variance":
        out = solve_inverse_variance_from_stats(stats)
        x, valid = out[0], out[1]
        return x, valid, {"solver": "inverse_variance"}
    raise ValueError(f"unknown solver {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True)
    p.add_argument("--out-template", required=True,
                   help="e.g. .../solved_{solver}_ogl3.pt ; {solver} is substituted")
    p.add_argument("--solvers", default="geometric_median,weighted")
    p.add_argument("--delete-stats", action="store_true",
                   help="Remove the stats file once every requested solver has been written. "
                        "Only fires if ALL succeeded, so a failure never destroys the input.")
    a = p.parse_args()

    stats = AccumulatedFeatureStats.load(a.stats)
    wanted = [s for s in a.solvers.split(",") if s]
    written, failed = [], []
    for name in wanted:
        out = a.out_template.format(solver=name)
        if os.path.exists(out):
            print(f"[skip] {name}: {out} exists")
            written.append(out)
            continue
        try:
            x, valid, rep = solve(stats, name)
        except Exception as e:                      # a solver may be unavailable for these stats
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"primitive_features": x.cpu(), "valid_mask": valid.cpu()}, out)
        print(f"[ok] {name}: {int(valid.sum())}/{valid.numel()} valid -> {out}  {rep}")
        written.append(out)

    if a.delete_stats and not failed:
        sz = os.path.getsize(a.stats) / 2**30
        os.remove(a.stats)
        print(f"[cleanup] removed {a.stats} ({sz:.2f} GB)")
    elif a.delete_stats:
        print(f"[cleanup] KEPT {a.stats}: {failed} failed, input preserved for retry")


if __name__ == "__main__":
    main()
