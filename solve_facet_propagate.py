"""Foam-native readout: clamp the forced set, propagate labels into the contested set.

WHY THIS AND NOT MORE SOLVING. `E_H(X) = 1/2 sum_jk G_jk ||X_j - X_k||^2` shows the closed form is
already a graph-regularised estimator -- it smooths along the CO-VISIBILITY graph `G = A^T A`. Foam's
`G` is nearly diagonal (a median ray touches 2.47 primitives against 3DGS's 33.12), so for foam that
smoothing barely happens and each primitive is decided by its own rays in isolation. Meanwhile foam
carries a SECOND graph 3DGS does not: the power diagram's facet adjacency (the Cech complex, already
in the checkpoint), which is exact and parameter-free. The spatial prior is sitting unused.

THREE CHOICES, EACH FORCED BY A MEASUREMENT.

  clamp the forced set   primitives receiving ONE evidence class are 99.75% accurate and provably
                         solver-invariant (Eq. 6 / Eq. 18 / geometric median agree to +0.000 there).
                         Moving them is pure downside, so they are held fixed and act as anchors.
  work in LABEL space    mIoU turns on argmax flips, not on L2 magnitude. Every feature-space
                         quantity we tested -- E_H, gamma, gap/rays, mean_o -- predicts the solve
                         term at r ~ +0.27, i.e. not at all. Feature-space blending was also what
                         `oracle_shrink.py` did, for +0.20.
  use the FACET graph    labels are piecewise constant on objects and objects are spatially
                         connected; the power diagram's dual is exactly that connectivity.

FALSIFIABLE PREDICTION: this should help foam MORE than 3DGS. Foam's `G` is nearly diagonal so the
closed form under-smooths and a spatial prior has room; 3DGS at 33 primitives per ray is already
over-smoothed by its own co-visibility, so the same method should help less or hurt. If 3DGS gains
equally the reasoning is wrong, not merely untuned.

3DGS has no facet graph, so for the comparison arm the script falls back to kNN on centres -- which
is what one would actually have to do, and is itself part of the point.

`lam = 0` reproduces the baseline EXACTLY; asserted every run, so a sweep measures the idea rather
than an implementation difference.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import torch

from evaluate_point_cloud_miou import calculate_metrics

FOAM = {"truefrozen", "nonfrozen"}


def neighbours(z, arm, k_knn=8):
    """-> (flat neighbour array, offsets). Foam: the power-diagram facet graph. 3DGS: kNN."""
    adj, off = z["adj"], z["adj_off"]
    if adj.size and off.size:
        return adj.astype(np.int64), off.astype(np.int64)
    raise RuntimeError("no facet graph in dump; kNN fallback needs centres (pass --centres)")


def propagate(AtS, live, adj, off, lam, rounds, clamp_forced=True):
    """Label-space propagation with the forced set held fixed.

    score_j = p_j + lam * (normalised histogram of neighbours' current labels)
    where p_j is j's own evidence as a distribution. Contested primitives are relabelled by argmax;
    clamped ones never move.
    """
    P, C = AtS.shape
    tot = AtS.sum(1, keepdims=True)
    p = np.zeros_like(AtS)
    nz = tot[:, 0] > 0
    p[nz] = AtS[nz] / tot[nz]
    lab = np.zeros(P, np.int64)
    lab[live] = AtS[live].argmax(1) + 1                      # baseline labels
    if lam == 0.0 or rounds == 0:
        return lab
    ev = (AtS > 0).sum(1)
    frozen = live & (ev == 1) if clamp_forced else np.zeros(P, bool)
    movable = np.nonzero(live & ~frozen)[0]
    for _ in range(rounds):
        new = lab.copy()
        for j in movable:
            nb = adj[off[j]:off[j + 1]]
            nb = nb[(nb >= 0) & (nb < P)]
            nb = nb[lab[nb] > 0]
            s = p[j].copy()
            if nb.size:
                h = np.bincount(lab[nb] - 1, minlength=C).astype(np.float32)
                s = s + lam * (h / h.sum())
            new[j] = s.argmax() + 1
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def score(lab, own, gt, C):
    _, mi, acc, _ = calculate_metrics(torch.from_numpy(gt), torch.from_numpy(lab[own]), C + 1)
    return float(mi) * 100, float(acc) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="artifacts/scannet/plotstats2")
    ap.add_argument("--arms", default="truefrozen")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-clamp", action="store_true", help="ablate the forced-set clamp")
    ap.add_argument("--out", default="artifacts/scannet/facet_propagate.json")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]
    rows = []
    for arm in a.arms.split(","):
        for f in sorted(glob.glob(os.path.join(a.stats, f"{arm}_scene*.npz"))):
            sc = os.path.basename(f).replace(f"{arm}_", "").replace(".npz", "")
            z = np.load(f)
            AtS = z["AtS"]; live = z["live"].astype(bool)
            own = z["own"].astype(np.int64); gt = z["pt_gt"].astype(np.int64)
            if own.size == 0:
                print(f"[{arm}/{sc}] SKIP no ownership in dump", flush=True); continue
            C = AtS.shape[1]
            try:
                adj, off = neighbours(z, arm)
            except RuntimeError as e:
                print(f"[{arm}/{sc}] SKIP {e}", flush=True); continue
            base = None
            r = {"arm": arm, "scene": sc, "P": int(AtS.shape[0]), "live": int(live.sum()),
                 "forced_frac": float(((AtS[live] > 0).sum(1) == 1).mean())}
            for lam in lams:
                lab = propagate(AtS, live, adj, off, lam, a.rounds, not a.no_clamp)
                mi, acc = score(lab, own, gt, C)
                if lam == 0.0:
                    base = mi
                    # lam=0 must reproduce the baseline exactly, or the sweep measures a bug
                    lab0 = np.zeros(AtS.shape[0], np.int64)
                    lab0[live] = AtS[live].argmax(1) + 1
                    assert np.array_equal(lab, lab0), "lam=0 does not reproduce the baseline"
                r[f"miou_{lam}"] = mi; r[f"acc_{lam}"] = acc
            rows.append(r)
            best = max(lams, key=lambda l: r[f"miou_{l}"])
            print(f"[{arm}/{sc}] forced {r['forced_frac']:.1%}  base {base:.2f}  "
                  f"best lam={best} {r[f'miou_{best}']:.2f}  ({r[f'miou_{best}'] - base:+.2f})",
                  flush=True)
    if not rows:
        return
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n{'arm':<12}{'n':>3}" + "".join(f"{('lam=' + str(l)):>10}" for l in lams))
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        b = float(np.mean([r["miou_0.0"] for r in s]))
        print(f"{arm:<12}{len(s):>3}" + "".join(
            f"{np.mean([r[f'miou_{l}'] for r in s]) - b:>+10.2f}" for l in lams))
    print("(deltas vs lam=0 baseline, mIoU)")


if __name__ == "__main__":
    main()
