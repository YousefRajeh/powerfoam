"""Overlap mass vs primitive budget on scene0347_00 -- does P alone explain the frozen arm's edge?

THE QUESTION. foamyfoam/03_measurements.md measures a ~10x conditioning gap between the frozen
(matched-budget) and nonfrozen arms. Two candidate causes, not yet separated:

  (1) the frozen arm is GT-DERIVED -- one primitive per ground-truth vertex, so it contains no
      floaters, and floaters carry ~89% of the nonfrozen arm's cross-surface mass; or
  (2) the frozen arm simply has 3x FEWER primitives, and fewer cells per ray is better conditioned
      regardless of where they sit.

If (2) alone explained it, `o` would be a function of P and the frozen point would lie ON the curve
traced by the non-GT-derived budget variants. If the frozen point lies well BELOW that curve, its
placement is buying something P does not.

WHAT THIS CAN AND CANNOT DO. The only budget variants on disk are 500k and 1200k, both LARGER than
the nonfrozen arm's 204k, and neither has a checkpoint -- only reduced accumulators. So:
  * `o = 1 - sum(support2)/sum(support)` is computable for every point (Lemma 2ii). Cheap.
  * cross-surface fraction is NOT -- it needs A and an adjacency graph, i.e. a checkpoint.
  * the curve is anchored at 204k / 500k / 1200k and the frozen point sits at 68k, BELOW the
    fitted range. Reaching it is an extrapolation, and it is reported as such.

So this is a directional test, not a decisive one. It can falsify (2) -- if `o` rises with P and
the frozen point is far below any sane extrapolation, count alone cannot be the story. It cannot
prove (1); that needs a non-GT-derived arm trained at ~68k, which does not exist.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re

import math
import torch

ART = "artifacts/scannet/scene0347_00"

# (label, filename, is_gt_derived)
POINTS = [
    ("frozen (GT-derived)", "stats_truefrozen_ogl3.pt", True),
    ("nonfrozen", "stats_nonfrozen_ogl3.pt", False),
    ("budget 500k", "stats_pf_bud500k_scene0347_00.pt", False),
    ("budget 1200k", "stats_pf_bud1200k_scene0347_00.pt", False),
]


def overlap(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "support" not in d or "support2" not in d:
        print(f"  [skip] {os.path.basename(path)}: no support/support2 (keys: {sorted(d)[:8]})")
        return None
    s = d["support"].double().reshape(-1)
    s2 = d["support2"].double().reshape(-1)
    v = s > 0
    sum_s, sum_s2 = float(s.sum()), float(s2.sum())
    rho = (1.0 - s2[v] / s[v]).numpy()
    out = {
        "P": int(s.numel()),
        "observed": int(v.sum()),
        "overlap_mass": 1.0 - sum_s2 / max(sum_s, 1e-30),
        "rho_mean_unweighted": float(rho.mean()),
        "out_of_range": int(((rho < 0) | (rho >= 1)).sum()),
    }
    for k in list(d):
        d[k] = None
    del d, s, s2, rho
    gc.collect()
    return out


def main():
    rows = []
    for label, fname, is_gt in POINTS:
        r = overlap(f"{ART}/{fname}")
        if r is None:
            print(f"[skip] {label}: {fname} absent")
            continue
        r.update(label=label, file=fname, gt_derived=is_gt)
        rows.append(r)
        flag = "  OUT-OF-RANGE" if r["out_of_range"] else ""
        print(f"{label:<22} P={r['P']:>9,}  observed={r['observed']:>9,}  "
              f"o={r['overlap_mass']:.4f}  rho_mean={r['rho_mean_unweighted']:.4f}{flag}",
              flush=True)

    free = [r for r in rows if not r["gt_derived"]]
    gt = [r for r in rows if r["gt_derived"]]
    json.dump({"points": rows}, open("artifacts/budget_sweep_scene0347.json", "w"), indent=1)

    fits = {}
    if len(free) >= 2 and gt:
        # Pure-python OLS. numpy's LAPACK path aborts with an OpenMP duplicate-runtime error when
        # first touched after torch has loaded these multi-GB files; a two-parameter fit does not
        # need LAPACK. Two functional forms, because the verdict should not hinge on one.
        def ols(xs, ys):
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            m = sxy / sxx
            return m, my - m * mx

        forms = [("o ~ log10 P", lambda o: o, lambda z: z),
                 ("logit(o) ~ log10 P", lambda o: math.log(o / (1 - o)),
                  lambda z: 1 / (1 + math.exp(-z)))]
        for name, fwd, inv in forms:
            xs = [math.log10(r["P"]) for r in free]
            ys = [fwd(r["overlap_mass"]) for r in free]
            m, c = ols(xs, ys)
            resid = max(abs(inv(m * x + c) - inv(y)) for x, y in zip(xs, ys))
            print(f"\n{name}: slope {m:+.4f}  intercept {c:+.4f}  "
                  f"max|resid| {resid:.4f} (o units)")
            for r in gt:
                pred = inv(m * math.log10(r["P"]) + c)
                short = pred - r["overlap_mass"]
                print(f"   {r['label']}: P={r['P']:,}  predicted o={pred:.4f}  "
                      f"measured o={r['overlap_mass']:.4f}  shortfall={short:+.4f} "
                      f"({short / max(resid, 1e-9):.0f}x the fit residual)")
                fits[name] = dict(slope=m, intercept=c, max_resid=resid, predicted=pred,
                                  measured=r["overlap_mass"], shortfall=short)
        print("\n  NOTE: the frozen P is below the fitted range, so this is an extrapolation.")
        print("  A shortfall far larger than the fit residual means primitive count alone does")
        print("  not account for the frozen arm's conditioning.")

    json.dump({"points": rows, "fits": fits},
              open("artifacts/budget_sweep_scene0347.json", "w"), indent=1)
    print("\nwrote artifacts/budget_sweep_scene0347.json")


if __name__ == "__main__":
    main()
