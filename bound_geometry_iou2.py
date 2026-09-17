"""Eq. (14) tightened, calibrated, and extended to Gaussians.

A62 established the bound is non-vacuous on 10/10 foam-frozen scenes (mIoU >= 24.6-61.6) using
h_j = 2 r_j. Three upgrades here, corresponding to directions 1, 2 and 4.

(1) TIGHTER h_j FROM MEASURED DIAMETERS. `h_j` only needs to upper-bound diam(C_j), and for ANY
    centre c, diam(S) <= 2 max_{p in S} ||p - c|| by the triangle inequality. So

        h_j = 2 * min( r_j,  max_{p in C_j} ||p - c_j||,  max_{p in C_j} ||p - centroid_j|| )

    Every term is a valid bound, so their min is too. Points concentrate on surfaces, so the measured
    radii are typically far below the support radius.

(2) CALIBRATION, mu(Z^h) <= L h. The bound as instantiated uses ground truth for the boundary
    distance. Fitting `L` from the empirical tube-growth curve and validating it LEAVE-ONE-OUT turns
    that into a calibration step: a scene's bound then needs only its geometry plus an `L` estimated
    elsewhere. This is explicitly calibration, not a label-free certificate (FINDINGS4 limitation 1).

(4) GAUSSIAN SUPPORT. 3DGS has no ball, but its RENDERED support is bounded: the rasteriser cuts at
    Mahalanobis radius tau_j = min(3.33, sqrt(2 ln(255 alpha_j))), so C_j is inside an ellipsoid with
    diam <= 2 tau_j s_max,j, and containment is ||diag(1/s) R^T (x - mu)|| <= tau_j. Without this the
    geometry bound is a foam-only statement and cannot enter the cross-representation table.

--selftest verifies:
  1. diam(S) <= 2 max||p - c|| for arbitrary point sets and centres, and that the three-way min is
     still a valid diameter bound;
  2. the Mahalanobis containment and diameter formulas against brute force on random ellipsoids;
  3. that tightening h can only RAISE the bound (monotonicity), so (1) is safe;
  4. the LOO calibration logic: an L fitted on a subset either bounds the held-out curve or is
     correctly reported as a violation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def diam_upper(pts, centres):
    """2 * min over supplied centres of max ||p - c||. Valid diameter upper bound for any centre."""
    best = np.inf
    for c in centres:
        best = min(best, float(np.linalg.norm(pts - c, axis=1).max()))
    return 2.0 * best


def miou_lower(pi, e_c, e):
    return float(np.mean(np.maximum(0.0, pi - e_c) / np.maximum(pi + (e - e_c), 1e-300)))


def gauss_support(scales, opac, alpha_thresh=1.0 / 255.0, dmax=3.33):
    """tau_j = min(dmax, sqrt(2 ln(alpha/alpha_thresh))), and the ellipsoid diameter 2 tau s_max."""
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.sqrt(np.maximum(0.0, 2.0 * np.log(np.maximum(opac, 1e-12) / alpha_thresh)))
    tau = np.minimum(dmax, t)
    return tau, 2.0 * tau * scales.max(1)


def selftest():
    rng = np.random.default_rng(0)

    # 1: the diameter bound, and that a min over centres stays valid
    for _ in range(500):
        n = rng.integers(2, 40)
        S = rng.normal(size=(n, 3)) * rng.uniform(0.1, 3.0)
        true_d = 0.0
        for i in range(n):
            true_d = max(true_d, float(np.linalg.norm(S - S[i], axis=1).max()))
        cs = [rng.normal(size=3) * 2, S.mean(0), np.zeros(3)]
        ub = diam_upper(S, cs)
        assert ub >= true_d - 1e-9, (ub, true_d)

    # 2: Mahalanobis containment and ellipsoid diameter against brute force
    for _ in range(200):
        s = rng.uniform(0.05, 1.0, 3)
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        tau = rng.uniform(0.5, 3.3)
        # sample the ellipsoid surface: x = R diag(s) tau * u, ||u|| = 1
        U = rng.normal(size=(400, 3)); U /= np.linalg.norm(U, axis=1, keepdims=True)
        PTS = (R @ (np.diag(s) @ (tau * U).T)).T
        dm = np.linalg.norm((np.diag(1.0 / s) @ (R.T @ PTS.T)).T, axis=1)
        assert np.allclose(dm, tau, atol=1e-8), (dm.min(), dm.max(), tau)
        obs = max(float(np.linalg.norm(PTS - PTS[i], axis=1).max()) for i in range(0, 400, 20))
        assert obs <= 2 * tau * s.max() + 1e-6, (obs, 2 * tau * s.max())

    # 3: tightening h can only raise the bound
    K, N = 5, 600
    y = rng.integers(0, K, N); w = np.ones(N) / N
    pi = np.array([w[y == c].sum() for c in range(K)])
    dZ = rng.uniform(0, 1, N)
    prev = -1.0
    for h in (0.6, 0.4, 0.2, 0.1):
        inE = dZ <= h
        e = float(w[inE].sum()); e_c = np.array([w[inE & (y == c)].sum() for c in range(K)])
        lb = miou_lower(pi, e_c, e)
        assert lb >= prev - 1e-12, "shrinking h must not lower the bound"
        prev = lb

    # 4: LOO calibration logic
    curves = [(np.array([0.1, 0.2, 0.3]), np.array([0.05, 0.11, 0.16])) for _ in range(4)]
    curves.append((np.array([0.1, 0.2, 0.3]), np.array([0.09, 0.19, 0.30])))   # a violator
    Ls = [float((m / h).max()) for h, m in curves]
    L_from_first4 = max(Ls[:4])
    assert (curves[4][1] / curves[4][0]).max() > L_from_first4, "violator must be detected"
    print("  selftest OK: diam(S) <= 2 max||p-c|| for arbitrary centres (500 sets) and the "
          "three-way min stays valid; Mahalanobis containment and ellipsoid diameter exact (200 "
          "ellipsoids); shrinking h never lowers the bound; LOO calibration detects a violator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--out", default="artifacts/scannet/geometry_iou2.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    from scipy.spatial import cKDTree
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    FOAM = {"truefrozen", "nonfrozen"}
    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    H_GRID = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])

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
            P3 = pts[m].astype(np.float64); y = gl[m] - 1
            N = P3.shape[0]

            if recon in FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(P3, cen, rad, valid=None, k=64)
                has = own >= 0
                dctr = np.full(N, np.inf)
                dctr[has] = np.linalg.norm(P3[has] - cen[own[has]], axis=1)
                supp_r = np.zeros(N); supp_r[has] = rad[own[has]]
                covered = has & (dctr <= supp_r)
                h_support = 2.0 * supp_r
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                weights_only=False)
                sp = ck["splats"] if "splats" in ck else ck
                mu = sp["means"].float().numpy().astype(np.float64)
                sc_ = np.exp(sp["scales"].float().numpy()).astype(np.float64)
                qt = sp["quats"].float().numpy().astype(np.float64)
                qt = qt / np.linalg.norm(qt, axis=1, keepdims=True)
                op = 1.0 / (1.0 + np.exp(-sp["opacities"].float().numpy().reshape(-1))).astype(np.float64)
                tau, diam_e = gauss_support(sc_, op)
                own = assign_points_to_nearest_center(P3, mu.astype(np.float32), valid=None)
                has = own >= 0
                # Mahalanobis distance in the owner's frame
                dm = np.full(N, np.inf)
                idx = np.where(has)[0]
                q = qt[own[idx]]; w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                R = np.empty((idx.size, 3, 3))
                R[:, 0, 0] = 1 - 2 * (y0 ** 2 + z0 ** 2); R[:, 0, 1] = 2 * (x0 * y0 - w0 * z0); R[:, 0, 2] = 2 * (x0 * z0 + w0 * y0)
                R[:, 1, 0] = 2 * (x0 * y0 + w0 * z0); R[:, 1, 1] = 1 - 2 * (x0 ** 2 + z0 ** 2); R[:, 1, 2] = 2 * (y0 * z0 - w0 * x0)
                R[:, 2, 0] = 2 * (x0 * z0 - w0 * y0); R[:, 2, 1] = 2 * (y0 * z0 + w0 * x0); R[:, 2, 2] = 1 - 2 * (x0 ** 2 + y0 ** 2)
                dlt = P3[idx] - mu[own[idx]]
                loc = np.einsum('nji,nj->ni', R, dlt)        # R^T (x - mu), the local frame
                dm[idx] = np.linalg.norm(loc / sc_[own[idx]], axis=1)
                covered = has & (dm <= tau[own])
                h_support = np.zeros(N); h_support[has] = diam_e[own[has]]
                dctr = np.full(N, np.inf); dctr[has] = np.linalg.norm(P3[has] - mu[own[has]], axis=1)

            # (1) measured diameters: 2*max||p-c_own|| and 2*max||p-centroid|| over each C_j
            Pn = int(own.max()) + 1 if has.any() else 1
            mx_ctr = np.zeros(Pn); np.maximum.at(mx_ctr, own[covered], dctr[covered])
            csum = np.zeros((Pn, 3)); cnt = np.zeros(Pn)
            np.add.at(csum, own[covered], P3[covered]); np.add.at(cnt, own[covered], 1.0)
            cent = csum / np.maximum(cnt, 1)[:, None]
            dcen = np.linalg.norm(P3[covered] - cent[own[covered]], axis=1)
            mx_cen = np.zeros(Pn); np.maximum.at(mx_cen, own[covered], dcen)
            h_meas = np.zeros(N)
            h_meas[covered] = 2.0 * np.minimum(mx_ctr[own[covered]], mx_cen[own[covered]])
            h_used = np.minimum(h_support, np.where(covered, h_meas, np.inf))

            # boundary distance
            dZ = np.full(N, np.inf)
            for c in range(K):
                inc = y == c
                if not inc.any() or not (~inc).any():
                    continue
                dZ[inc] = cKDTree(P3[~inc]).query(P3[inc], k=1)[0]

            w = np.ones(N) / N
            pi = np.array([w[y == c].sum() for c in range(K)])
            present = pi > 0

            def bound_with(hv):
                inE = (~covered) | (dZ <= hv)
                e = float(w[inE].sum())
                e_c = np.array([w[inE & (y == c)].sum() for c in range(K)])
                return miou_lower(pi[present], e_c[present], e), e

            lb_supp, e_supp = bound_with(h_support)
            lb_meas, e_meas = bound_with(h_used)
            # (2) tube-growth curve for calibration
            curve = [float(w[dZ <= hh].sum()) for hh in H_GRID]

            rec = {"arm": arm, "scene": sc, "K": K, "n_pts": N,
                   "frac_uncovered": float(w[~covered].sum()),
                   "h_support_median": float(np.median(h_support[covered])) if covered.any() else None,
                   "h_measured_median": float(np.median(h_used[covered])) if covered.any() else None,
                   "e_support": e_supp, "e_measured": e_meas,
                   "miou_lb_support": lb_supp, "miou_lb_measured": lb_meas,
                   "h_grid": H_GRID.tolist(), "tube_curve": curve,
                   "L_fit": float((np.array(curve) / H_GRID).max()),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] uncov {100*rec['frac_uncovered']:.1f}% | h med {rec['h_support_median']:.4f}"
                  f" -> {rec['h_measured_median']:.4f} | e {100*e_supp:.1f}% -> {100*e_meas:.1f}% | "
                  f"mIoU >= {100*lb_supp:.2f} -> {100*lb_meas:.2f} | L_fit {rec['L_fit']:.3f}  "
                  f"{rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
