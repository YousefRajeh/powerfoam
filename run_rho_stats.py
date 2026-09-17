"""I3 from BETA_BOUND.md: the per-primitive shrinkage coefficient rho, over all 10 ScanNet scenes.

    rho_j = 1 - (sum_i A_ij^2) / (sum_i A_ij) = 1 - E_A[A_ij]

WHY IT MATTERS. Theorem 2(i) of `splat-distiller/funny probability/BETA_BOUND.md` says the one-shot
row-sum lift does not return the least-squares solution x_hat; it returns x_hat pulled toward the
co-visibility average of its neighbours by EXACTLY the fraction rho_j:

    x_hat_j - x'_j = rho_j (x_hat_j - xbar_j)

so rho_j is the measured approximation error of the lift, per primitive, with no bound and no
unmeasurable constant. rho_j = 0 for every j iff every ray deposits its whole weight on one
primitive iff G = A^T A is diagonal, and then (Theorem 2(iv)) the closed form IS the least-squares
solution. So rho measures ray-support disjointness, which is the property a power diagram has by
construction and a Gaussian cloud does not.

NOTHING NEW IS COMPUTED HERE. `support` and `support2` in our AccumulatedFeatureStats are already
the column sums of A and of A^2 (operator.py:344; support2 is also used as diag(A^T A) by the ridge
solvers, operator.py:174), so rho falls straight out of artifacts that exist for every scene. That
is the point of I3: it is a read-only measurement, not an experiment.

SCALE-FREENESS, which is what makes the two arms comparable. rho_j = 1 - E_A[A_ij] is an
A-weighted MEAN of the weights, so it does not grow with the number of rays that see a primitive --
a cell seen by 10 rays and one seen by 10,000 are on the same scale. What it does depend on is how
many primitives share each ray, which is exactly the quantity under test.

INSTRUMENT CHECK, printed alongside. Property 2 of the SFS paper (rows of A sum to 1) is what makes
rho a fraction at all; for Gaussians it holds only approximately (sum_j A_ij = 1 - T_final). We
cannot read row sums off the reduced accumulators, but we CAN read the implied mean weight per
primitive and the support distribution, and a rho outside [0,1) would be the tell that something is
wrong. Any such primitive is reported rather than clipped.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cluster_classify_eval import SCENES

ARMS = {
    "pf_tfroz": "stats_truefrozen_ogl3.pt",
    "pf_nonfroz": "stats_nonfrozen_ogl3.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/rho_stats.json")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = ap.parse_args()
    res = {}

    for arm, fname in ARMS.items():
        res[arm] = {}
        for scene in args.scenes:
            p = f"artifacts/scannet/{scene}/{fname}"
            if not os.path.exists(p):
                print(f"[skip] {arm}/{scene}")
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            s = d["support"].double().reshape(-1)
            s2 = d["support2"].double().reshape(-1)
            v = s > 0
            rho = (1.0 - s2[v] / s[v]).numpy()
            bad = int(((rho < 0.0) | (rho >= 1.0)).sum())
            q = np.percentile(rho, [5, 25, 50, 75, 95])
            row = {
                "primitives_observed": int(v.sum()),
                "primitives_total": int(s.numel()),
                "rho_mean": float(rho.mean()),
                "rho_p5": float(q[0]), "rho_p25": float(q[1]), "rho_p50": float(q[2]),
                "rho_p75": float(q[3]), "rho_p95": float(q[4]),
                "frac_rho_lt_0.1": float((rho < 0.1).mean()),
                "frac_rho_lt_0.5": float((rho < 0.5).mean()),
                "mean_support": float(s[v].mean()),
                "out_of_range": bad,
            }
            res[arm][scene] = row
            print(f"{arm:<11}{scene}: P={row['primitives_observed']:>8,}  "
                  f"rho {row['rho_mean']:.4f} (p50 {row['rho_p50']:.3f})  "
                  f"rho<0.1 {row['frac_rho_lt_0.1']:.3f}  "
                  f"support {row['mean_support']:.1f}"
                  + (f"  OUT-OF-RANGE {bad}" if bad else ""), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    print("\n=== 10-scene means ===")
    for arm, d in res.items():
        if not d:
            continue
        print(f"{arm:<11} rho_mean {np.mean([v['rho_mean'] for v in d.values()]):.4f}  "
              f"rho_p50 {np.mean([v['rho_p50'] for v in d.values()]):.4f}  "
              f"frac(rho<0.1) {np.mean([v['frac_rho_lt_0.1'] for v in d.values()]):.4f}  "
              f"cells {np.mean([v['primitives_observed'] for v in d.values()]):,.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
