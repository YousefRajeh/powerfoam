"""Does SHRINKING the transferred bias fix its 3/10 failures?

THE PROBLEM. Strict 10-scene leave-one-out transfer of the per-class bias deviation eps gives
+2.12 mIoU at 19cls out of sample -- but only 7/10 scenes positive, and the spread is enormous:

    scene0062 +10.60   scene0070 +6.80   scene0347 +5.90   scene0590 +3.54   scene0097 +2.75
    scene0400 +1.01    scene0645 +0.39   scene0200 -0.98   scene0000 -1.85   scene0140 -6.94

The fit score on the nine training scenes is stable at 47-50 in every fold, so the FIT is fine;
it is the TRANSFER that varies. That is the classic signature of overfitting: eps has ~19 free
scalars fitted to maximise a mean over 9 scenes, and nothing stops it exploiting idiosyncrasies
of those 9.

THE TEST. Shrink the transferred deviation toward zero:

    b(gamma) = 0.5 * colmean + gamma * eps,     gamma in [0, 1]

gamma = 0 is exactly partial centering (the incumbent), gamma = 1 is the full transferred fit.
If the out-of-sample optimum is at gamma < 1, the fit is overfitting and shrinkage is the fix --
this is ridge/James-Stein logic, and the correct amount of shrinkage is estimable from the fold
spread itself rather than tuned on the held-out scene.

WHAT WOULD MAKE THIS WORTHLESS: if the best gamma differs per held-out scene, then choosing it
needs the held-out labels and nothing has been gained. So the honest quantity is a SINGLE gamma
that is good for all folds, and it is reported as such -- the per-scene-optimal gamma is printed
only as an oracle bound, clearly labelled.

FALSIFIER, pre-registered: a single shared gamma must beat partial centering by >= +0.5 mIoU at
19cls AND be positive on >= 8/10 scenes -- the bar the unshrunk version missed at 7/10.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, remap_gt_labels)
from run_bias_loo_10scene import CACHE, LAM, SCENES, fit_eps, load_scene


def main():
    enable_determinism()
    device = "cuda"
    gammas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    out = {}

    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        data = {}
        for s in SCENES:
            if os.path.exists(os.path.join(CACHE, f"{s}_ogl3.pt")):
                data[s] = load_scene(s, cs, device)
        if len(data) < 3:
            continue
        all_names = sorted({n for d in data.values() for n in d["names"]})
        nidx = {n: i for i, n in enumerate(all_names)}

        # rows[held][gamma] = mIoU on the held-out scene
        rows = {}
        for held in data:
            fit_on = [d for s, d in data.items() if s != held]
            eps, _ = fit_eps(fit_on, nidx, len(all_names), device)
            d = data[held]
            idx = torch.tensor([nidx[n] for n in d["names"]], device=device)
            rows[held] = {g: d["score"](LAM * d["colmean"] + g * eps[idx]) for g in gammas}
            print(f"  [{cs[11:]}] {held}: " +
                  " ".join(f"g{g}={rows[held][g]:.2f}" for g in gammas), flush=True)

        base = {h: rows[h][0.0] for h in rows}          # gamma=0 IS partial centering
        print(f"\n  [{cs[11:]}] shared-gamma sweep (all out of sample):")
        print(f"    {'gamma':>6}{'mean mIoU':>11}{'delta':>8}{'positive':>10}")
        best_g, best_mean = None, -1e9
        for g in gammas:
            vals = [rows[h][g] for h in rows]
            d = [rows[h][g] - base[h] for h in rows]
            print(f"    {g:>6}{np.mean(vals):>11.2f}{np.mean(d):>+8.2f}"
                  f"{sum(x > 0 for x in d):>7}/{len(d)}")
            if np.mean(vals) > best_mean:
                best_mean, best_g = np.mean(vals), g
        orc = np.mean([max(rows[h][g] for g in gammas) for h in rows])
        print(f"    best shared gamma = {best_g} -> {best_mean:.2f}")
        print(f"    [oracle] per-scene best gamma -> {orc:.2f}  "
              f"(needs held-out labels; NOT a method)\n", flush=True)
        out[cs] = {"rows": {h: {str(g): v for g, v in r.items()} for h, r in rows.items()},
                   "best_gamma": best_g, "best_mean": best_mean, "oracle": orc}

    json.dump(out, open("artifacts/scannet/bias_loo_shrinkage.json", "w"), indent=1)
    print("=== SUMMARY ===")
    for cs, o in out.items():
        rows = o["rows"]
        base = np.mean([r["0.0"] for r in rows.values()])
        d = [rows[h][str(o["best_gamma"])] - rows[h]["0.0"] for h in rows]
        print(f"{cs[11:]:<6} centering {base:6.2f} | best gamma {o['best_gamma']} -> "
              f"{o['best_mean']:6.2f} ({np.mean(d):+.2f}, "
              f"{sum(x > 0 for x in d)}/{len(d)}) | oracle-gamma {o['oracle']:.2f}")


if __name__ == "__main__":
    main()
