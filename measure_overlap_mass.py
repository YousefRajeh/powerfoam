"""Mean per-ray overlap mass for both PowerFoam arms, from the reduced accumulators alone.

BETA_BOUND.md Lemma 2(ii): the D-weighted mean shrinkage equals the mean per-ray overlap mass,

    sum_j D_jj rho_j / sum_j D_jj  =  sum_i o_i / sum_i s_i ,      o_i = s_i^2 - sum_j A_ij^2

and since `support` = sum_i A_ij s_i = (S 1)_j and `support2` = sum_i A_ij^2 = diag(S)_j, this is

    overlap  =  1 - sum_j support2_j / sum_j support_j

with no gram cache, no x_hat and no CG -- the same two accumulators run_rho_stats.py uses. It is
the middle (sharper) term of Theorem 2's bound, so it says how much of the lift's error budget
exists at all, while measure_cross_surface_gram.py says how much of it is cross-surface.

The frozen arm is the one the paper's matched-budget table describes (P matched to GT vertices);
the gram caches exist only for the nonfrozen arm, so this script is how the two are compared.

CROSS-CHECK, printed when a gram cache result is available: 2 * mass_off must equal
sum_j support_j - sum_j support2_j on the same arm and scene.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ART = "artifacts/scannet"
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
ARMS = {"frozen (matched budget)": "stats_truefrozen_ogl3.pt",
        "nonfrozen": "stats_nonfrozen_ogl3.pt"}


def arm_row(scene: str, fname: str) -> dict | None:
    p = f"{ART}/{scene}/{fname}"
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    s = d["support"].double().reshape(-1)
    s2 = d["support2"].double().reshape(-1)
    v = s > 0
    sum_s, sum_s2 = float(s.sum()), float(s2.sum())
    rho = (1.0 - s2[v] / s[v]).numpy()
    return {
        "P": int(s.numel()),
        "observed": int(v.sum()),
        "overlap_mass": 1.0 - sum_s2 / max(sum_s, 1e-30),   # D-weighted mean rho
        "rho_mean_unweighted": float(rho.mean()),
        "rho_p50": float(np.median(rho)),
        "frac_rho_lt_0.1": float((rho < 0.1).mean()),
        "sum_support": sum_s,
        "sum_support2": sum_s2,
        "out_of_range": int(((rho < 0.0) | (rho >= 1.0)).sum()),
    }


def main():
    res = {a: {} for a in ARMS}
    for arm, fname in ARMS.items():
        for sc in SCENES:
            r = arm_row(sc, fname)
            if r:
                res[arm][sc] = r

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        print(f"{'scene':<15}{'P':>10}{'observed':>10}{'overlap':>10}"
              f"{'rho_mean':>10}{'rho_p50':>10}{'rho<0.1':>9}")
        for sc, r in res[arm].items():
            flag = "  OUT-OF-RANGE" if r["out_of_range"] else ""
            print(f"{sc:<15}{r['P']:>10,}{r['observed']:>10,}{r['overlap_mass']:>10.4f}"
                  f"{r['rho_mean_unweighted']:>10.4f}{r['rho_p50']:>10.4f}"
                  f"{r['frac_rho_lt_0.1']:>9.3f}{flag}")
        if res[arm]:
            om = np.array([r["overlap_mass"] for r in res[arm].values()])
            print(f"{'MEAN':<15}{'':>10}{'':>10}{om.mean():>10.4f}")

    # ---- cross-check against any gram-cache result already written
    for jf in ("artifacts/cross_surface_gram.json", "artifacts/cross_surface_gram_pilot.json"):
        if not os.path.exists(jf):
            continue
        print(f"\n=== cross-check vs {os.path.basename(jf)} (nonfrozen arm) ===")
        for g in json.load(open(jf)):
            sc = g["scene"]
            r = res["nonfrozen"].get(sc)
            if not r:
                continue
            lhs = 2.0 * g["mass_off"]
            rhs = r["sum_support"] - r["sum_support2"]
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
            ok = "ok" if rel < 5e-3 else "MISMATCH"
            print(f"  {sc:<15} 2*mass_off {lhs:.6e}   sumS-sumS2 {rhs:.6e}   "
                  f"rel {rel:.2e}  {ok}")
        break

    json.dump(res, open("artifacts/overlap_mass.json", "w"), indent=1)
    print("\nwrote artifacts/overlap_mass.json")


if __name__ == "__main__":
    main()
