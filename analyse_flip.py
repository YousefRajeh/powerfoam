"""Three questions about whether E_H governs the argmax flip.

Q1  Does the per-primitive AUC hold across 10 scenes, or was scene0062 lucky?
    AUC is a WITHIN-scene ranking statistic: "is a flipped primitive's share of the feature-space
    excess larger than a non-flipped one's?" 0.5 = no information.

Q2  Does it hold for 3DGS as well as foam? Confound to keep in view: the 3DGS conjugate-gradient
    hits its 300-iteration cap (residual ~3e-3) where foam converges to ~1e-6, so `x_hat` is a
    worse reference there and the flip LABELS inherit that error. A lower AUC for 3DGS is therefore
    ambiguous between "the bound is weaker" and "the reference is noisier" -- `cg_residual` is
    printed alongside so the two can be told apart.

Q3  Does E_H predict `flip_frac` ACROSS scenes? This is the harder and more useful claim: AUC says
    the bound ranks primitives within a scene, but a BOUND needs the aggregate -- if E_H is large
    does the scene actually suffer more flips? A statistic can rank well within groups and carry no
    information between them (exactly what happened to us with mIoU, r ~ +0.27).
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

MAP = {"pf_truefrozen": "foam frozen", "gs_froz": "3DGS frozen"}


def load():
    rows = []
    for f in sorted(glob.glob("artifacts/scannet/beta/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "flip_auc" in r and r.get("views") == 12 and r.get("arm") in MAP:
            rows.append(r)
    return rows


def cc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), len(x)
    sp = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
    return np.corrcoef(x, y)[0, 1], sp, len(x)


def main():
    rows = load()
    if not rows:
        print("no rows with flip stats yet"); return
    print(f"{'arm':<13}{'scene':<15}{'flip%':>8}{'AUC':>8}{'gap ratio':>11}"
          f"{'gamma':>9}{'cg resid':>11}{'cg its':>8}")
    for r in sorted(rows, key=lambda r: (r["arm"], r["scene"])):
        ci = r.get("cg_iterations"); cr = r.get("cg_residual")
        ci = ci[-1] if isinstance(ci, list) and ci else ci
        cr = cr[-1] if isinstance(cr, list) and cr else cr
        print(f"{MAP[r['arm']]:<13}{r['scene']:<15}{r['flip_frac']*100:>7.2f}%{r['flip_auc']:>8.3f}"
              f"{r.get('flip_gap_ratio', float('nan')):>11.2f}{r.get('gamma', float('nan')):>9.4f}"
              f"{(f'{cr:.1e}' if isinstance(cr, float) else '-'):>11}"
              f"{(str(ci) if ci is not None else '-'):>8}")

    print("\nQ1/Q2  per-primitive AUC by arm")
    for arm, lbl in MAP.items():
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            print(f"  {lbl:<13} no rows"); continue
        a = np.array([r["flip_auc"] for r in s])
        f = np.array([r["flip_frac"] for r in s])
        print(f"  {lbl:<13} n={len(s):<3} AUC {a.mean():.3f} +- {a.std(ddof=1) if len(a)>1 else 0:.3f}"
              f"  (min {a.min():.3f}, max {a.max():.3f})   flip {f.mean()*100:.2f}%")

    print("\nQ3  does E_H predict flip_frac ACROSS scenes?")
    for arm, lbl in MAP.items():
        s = [r for r in rows if r["arm"] == arm]
        if len(s) < 4:
            print(f"  {lbl:<13} n={len(s)} too few"); continue
        for key, nm in (("gamma", "gamma (relative excess)"),
                        ("gap_A_dx_sq", "E_H (absolute excess)"),
                        ("mean_o", "mean_o"),
                        ("beta_p50", "beta_p50")):
            if key not in s[0]:
                continue
            p, sp, n = cc([r[key] for r in s], [r["flip_frac"] for r in s])
            print(f"  {lbl:<13} {nm:<24} pearson {p:+.3f}  spearman {sp:+.3f}  (n={n})")

    print("\nreminder: AUC ranks WITHIN a scene; Q3 is the aggregate claim a bound needs.")


if __name__ == "__main__":
    main()
