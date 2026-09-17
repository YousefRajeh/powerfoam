"""FINDINGS4 Eq. (14): a geometry -> mIoU bound with no feature embedding at all.

A53 established that any bound routed through feature-space L2 loses 2-3 orders of magnitude, and
A60 showed why it can never be recovered (the CLIP modality gap caps cos at ~0.31 where the bound
needs > 0.96). Eq. (14) avoids features entirely: it bounds the confusion entries directly from the
geometry of the partition.

    E = U  u  union_j ( C_j  n  Z^{h_j} ),      e = mu(E),  e_c = mu(E n Y_c)

    mIoU(h)  >=  (1/K) sum_c ( pi_c - e_c ) / ( pi_c + e - e_c )                     (14)

`U` is the uncovered set, `C_j` the covered owner region of primitive j, `Z` the semantic boundary,
`Z^h` its h-neighbourhood, and `h_j` a diameter bound for `C_j`. Every mistake of the majority-label
competitor on a covered region must have a differently-labelled point in the same region, hence lies
within `h_j` of `Z`; so all errors are inside `E`, giving `FN_c <= e_c` and `FP_c <= e - e_c`.

INSTANTIATION HERE. For foam, the rendered solid is `Ball(c_j, r_j) n <half-spaces>`, so
`C_j` is contained in that ball and `diam(C_j) <= 2 r_j` -- that is the `h_j` used. A scored point is
COVERED if its owner exists and it lies inside the owner's ball; everything else is in `U`. Distance
to `Z` is the distance to the nearest differently-labelled GT point, computed exactly per class.

THREE LIMITATIONS, carried verbatim from FINDINGS4 because they decide how the result may be quoted:

1. Boundary neighbourhoods and `pi_c` contain label information. This is therefore a DIAGNOSTIC of
   geometric representability, not a label-free certificate. Making it predictive needs an
   independently justified tube bound `mu(Z^h) <= L h` plus a coverage bound.
2. It bounds the **covered-region competitor**, not automatically our reported whole-cell majority
   labelling. An outside-support majority can reverse an inside-support choice, and bounded balls do
   not bound bare power cells. Both labellings are scored below so the difference is visible.
3. It bounds geometric representability, not the learned pipeline's error. Uncovered mass is
   pessimistically counted wrong; a real labelling can do better there.

--selftest verifies:
  1. Eq (14) against brute-force IoU whenever the error set is known exactly;
  2. that containing all errors in E really does give FN_c <= e_c and FP_c <= e - e_c;
  3. monotonicity: growing E (larger h) can only lower the bound;
  4. the weaker envelope-only form max(0, pi_c - u)/(pi_c + u) is implied by it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def miou_lower(pi, e_c, e):
    """Eq (14). pi, e_c are per-class masses; e is the total error mass."""
    num = np.maximum(0.0, pi - e_c)
    den = np.maximum(pi + (e - e_c), 1e-300)
    return float(np.mean(num / den))


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(400):
        K, N = rng.integers(3, 7), 500
        y = rng.integers(0, K, N)
        w = rng.uniform(0.5, 2.0, N); w /= w.sum()
        inE = rng.random(N) < rng.uniform(0.05, 0.7)
        # a competitor that is correct OUTSIDE E and arbitrary inside it -- the situation (14) models
        pred = y.copy()
        pred[inE] = rng.integers(0, K, int(inE.sum()))
        pi = np.array([w[y == c].sum() for c in range(K)])
        e = float(w[inE].sum())
        e_c = np.array([w[inE & (y == c)].sum() for c in range(K)])
        FN = np.array([w[(y == c) & (pred != c)].sum() for c in range(K)])
        FP = np.array([w[(y != c) & (pred == c)].sum() for c in range(K)])
        # 2: containment really gives the claimed bounds
        assert (FN <= e_c + 1e-12).all(), "FN_c <= e_c violated"
        assert (FP <= (e - e_c) + 1e-12).all(), "FP_c <= e - e_c violated"
        # 1: the bound holds against the realised mIoU
        pres = pi > 0
        iou = np.where(pres, (pi - FN) / np.maximum(pi + FP, 1e-300), 0.0)
        real = float(iou[pres].mean())
        lb = miou_lower(pi[pres], e_c[pres], e)
        assert lb <= real + 1e-9, (lb, real)
        # 3: growing E can only lower the bound
        bigger = inE | (rng.random(N) < 0.1)
        e2 = float(w[bigger].sum())
        e_c2 = np.array([w[bigger & (y == c)].sum() for c in range(K)])
        assert miou_lower(pi[pres], e_c2[pres], e2) <= lb + 1e-9
        # 4: the envelope-only form is implied
        u = e
        weak = float(np.mean(np.maximum(0.0, pi[pres] - u) / (pi[pres] + u)))
        assert weak <= lb + 1e-9, (weak, lb)
    print("  selftest OK: Eq(14) lower-bounds realised mIoU whenever errors are contained in E; "
          "FN_c <= e_c and FP_c <= e - e_c hold; the bound is monotone decreasing in E; the "
          "envelope-only form is weaker as expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--h-scale", type=float, default=2.0,
                    help="h_j = h_scale * r_j. Default 2.0 = the ball diameter, the valid bound.")
    ap.add_argument("--out", default="artifacts/scannet/geometry_iou.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    from scipy.spatial import cKDTree
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
    from point_cloud_query import assign_points_to_power_cells

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        recon = arm.replace("pf_", "")
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            K = len(kept)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            P3 = pts[m].astype(np.float64); y = gl[m] - 1          # 0..K-1

            cen, rad, _ = geometry(sc, recon)
            own = assign_points_to_power_cells(P3, cen, rad, valid=None, k=64)

            # covered: owner exists AND the point lies inside the owner's ball
            has_owner = own >= 0
            dist_own = np.full(P3.shape[0], np.inf)
            dist_own[has_owner] = np.linalg.norm(P3[has_owner] - cen[own[has_owner]], axis=1)
            r_own = np.zeros(P3.shape[0]); r_own[has_owner] = rad[own[has_owner]]
            covered = has_owner & (dist_own <= r_own)

            # distance to the semantic boundary Z: nearest differently-labelled GT point, exact
            dZ = np.full(P3.shape[0], np.inf)
            for c in range(K):
                inc = y == c
                if not inc.any():
                    continue
                other = ~inc
                if not other.any():
                    dZ[inc] = np.inf; continue
                dZ[inc] = cKDTree(P3[other]).query(P3[inc], k=1)[0]

            h = a.h_scale * r_own
            inE = (~covered) | (dZ <= h)

            w = np.ones(P3.shape[0]) / P3.shape[0]
            pi = np.array([w[y == c].sum() for c in range(K)])
            e = float(w[inE].sum())
            e_c = np.array([w[inE & (y == c)].sum() for c in range(K)])
            present = pi > 0
            lb = miou_lower(pi[present], e_c[present], e)

            # for reference: the ASA-optimal (majority) labelling's realised mIoU on the same set
            lab = np.zeros(P3.shape[0], np.int64) - 1
            for j in np.unique(own[has_owner]):
                sel = own == j
                if sel.any():
                    lab[sel] = np.bincount(y[sel], minlength=K).argmax()
            ok = lab >= 0
            FN = np.array([w[(y == c) & ~((lab == c) & ok)].sum() for c in range(K)])
            FP = np.array([w[(y != c) & (lab == c) & ok].sum() for c in range(K)])
            maj_miou = float(np.mean(np.maximum(0.0, pi[present] - FN[present]) /
                                     np.maximum(pi[present] + FP[present], 1e-300)))

            rec = {"arm": arm, "scene": sc, "K": K, "n_pts": int(P3.shape[0]),
                   "h_scale": a.h_scale,
                   "frac_uncovered": float(w[~covered].sum()),
                   "frac_in_tube": float(w[covered & (dZ <= h)].sum()),
                   "e_total": e, "median_r": float(np.median(rad)),
                   "median_dZ": float(np.median(dZ[np.isfinite(dZ)])),
                   "miou_lower_bound": lb, "miou_majority_labelling": maj_miou,
                   "vacuous": bool(lb <= 0.0), "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] K={K} | uncovered {100*rec['frac_uncovered']:.1f}% + tube "
                  f"{100*rec['frac_in_tube']:.1f}% = e {100*e:.1f}% | med r {rec['median_r']:.4f} "
                  f"med dZ {rec['median_dZ']:.4f} | mIoU >= {100*lb:.2f} vs majority labelling "
                  f"{100*maj_miou:.2f} {'VACUOUS' if rec['vacuous'] else 'NON-VACUOUS'} "
                  f"{rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
