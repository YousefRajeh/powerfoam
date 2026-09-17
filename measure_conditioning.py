"""The conditioning certificate FINDINGS4 proposes, measured per arm.

FINDINGS4 removed both representation framings we had: spatial disjointness does NOT bound the number
of contributors per ray (along n disjoint cells, alpha_j = 1/(n-j+1) gives every weight 1/n and
dispersion 1 - 1/n), and Gaussians have no dispersion floor (a single-contributor ray has o = 0; the
real rasteriser culls and terminates blending). Low mean dispersion does not even imply good
conditioning: M pure rays plus one ray [0, 1/2, 1/2] has mean dispersion -> 0 while two columns are
identical and C is singular.

What replaces them is measurable. With d_j = sum_i A_ij, G_jj = sum_i A_ij^2, r_i = sum_j A_ij:

    q_j = G_jj / d_j            exposure-normalised SELF-contribution
    s_j = (sum_{k != j} G_jk) / d_j = ([A^T r]_j - G_jj) / d_j        coupling
    gamma = min_j (q_j - s_j) = min_j (2 G_jj - [A^T r]_j) / d_j

Gershgorin applied to D^-1 G, which is similar to the symmetric C = D^-1/2 G D^-1/2, gives

    gamma > 0   ==>   gamma I <= C <= r_max I,    kappa_2(C) <= r_max / gamma

and since q_j + s_j <= r_max <= 1, the condition min_j q_j > 1/2 is sufficient. This is a classical
diagonal-dominance argument specialised here, not a new matrix theorem.

THE QUESTION, stated so it can fail: **does frozen foam produce more columns with large q_j and a
larger certified lower spectral edge, at comparable exposure and reconstruction quality?**

Reported per arm: the full q - s distribution (not just the min, which one bad column controls),
min_j q_j, the fraction with q_j > 1/2, gamma, the certified kappa bound when gamma > 0, exposure
tails, and the count of zero / poorly observed columns. FINDINGS4 is explicit that poorly observed
primitives must NOT be silently dropped to manufacture a favourable minimum, so exclusions are
counted and reported rather than applied quietly. A negative gamma means the CERTIFICATE fails, not
that the matrix is ill-conditioned.

COST: two passes over nnz. Pass 1 computes d_j, G_jj and r_i (three independent reductions over the
same non-zeros). Pass 2 computes [A^T r]_j, which cannot start until r is complete. Nothing else
touches nnz; the rest is O(P).

--selftest verifies, before any GPU time:
  1. q, s, gamma against dense reference matrices;
  2. the Gershgorin implication really holds -- gamma > 0 => lambda_min(C) >= gamma, checked against
     a dense eigendecomposition, and lambda_max(C) <= r_max;
  3. the certificate is CONSERVATIVE, i.e. gamma <= lambda_min(C), so failing it does not imply bad
     conditioning (the point FINDINGS4 insists on);
  3b. the POSITIVE branch actually fires: perfectly one-hot rays certify, and gamma degrades
     monotonically as rays are made more dispersed. Random sparse operators never certify, so
     without this case the gamma > 0 path would be dead code in the test;
  4. the counterexample from FINDINGS4: low mean dispersion with a singular C, where gamma correctly
     reports failure while dispersion looks excellent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def certificate(A):
    """(q, s, gamma, r_max) from a dense A. Mirrors exactly what the GPU path computes."""
    d = A.sum(0)
    r = A.sum(1)
    Gjj = (A ** 2).sum(0)
    Atr = A.T @ r
    live = d > 0
    q = np.zeros_like(d); s = np.zeros_like(d)
    q[live] = Gjj[live] / d[live]
    s[live] = (Atr[live] - Gjj[live]) / d[live]
    gamma = float((q[live] - s[live]).min()) if live.any() else float("nan")
    return q, s, gamma, float(r.max()), live


def selftest():
    rng = np.random.default_rng(0)
    n_cert_ok = n_cert_fail = 0
    for _ in range(300):
        R, P = 40, 8
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = 1.0
        A[A.sum(1) == 0, :] = 1.0
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        q, s, gamma, rmax, live = certificate(A)

        d = A.sum(0); G = A.T @ A
        Dh = np.diag(1.0 / np.sqrt(d))
        C = Dh @ G @ Dh
        ev = np.linalg.eigvalsh(C)

        # 1: q and s against their definitions
        assert np.allclose(q, np.diag(G) / d)
        assert np.allclose(s, (G.sum(1) - np.diag(G)) / d)

        # 2 & 3: the implication holds, and is conservative
        assert ev.max() <= rmax + 1e-9, (ev.max(), rmax)
        if gamma > 0:
            n_cert_ok += 1
            assert ev.min() >= gamma - 1e-9, (ev.min(), gamma)
            assert (ev.max() / max(ev.min(), 1e-300)) <= rmax / gamma + 1e-6
        else:
            n_cert_fail += 1
        assert gamma <= ev.min() + 1e-9, "certificate must be conservative"

    # 3b: EXERCISE THE POSITIVE BRANCH. Random sparse operators never certify (0/300 above), so
    #     without this the gamma > 0 path is untested. Near-one-hot rays are the regime foam is
    #     claimed to occupy: a perfectly one-hot ray gives s_j = 0 and q_j = sum r^2 / sum r, so
    #     gamma > 0 should hold and tighten as rays become purer.
    prev_g = None
    for purity in (1.0, 0.98, 0.9, 0.75):
        R, P = 60, 6
        A = np.zeros((R, P))
        main = rng.integers(0, P, R)
        A[np.arange(R), main] = purity
        if purity < 1.0:
            for i in range(R):
                others = [k for k in range(P) if k != main[i]]
                A[i, rng.choice(others)] = 1.0 - purity
        A *= rng.uniform(0.6, 1.0, (R, 1))
        q, s_, g, rmax, live = certificate(A)
        d = A.sum(0); Gm = A.T @ A
        C = np.diag(1/np.sqrt(d)) @ Gm @ np.diag(1/np.sqrt(d))
        ev = np.linalg.eigvalsh(C)
        if purity == 1.0:
            assert g > 0, f"perfectly one-hot rays must certify, got gamma={g}"
            assert ev.min() >= g - 1e-9
            assert (rmax / g) >= ev.max() / max(ev.min(), 1e-300) - 1e-6
        if prev_g is not None:
            assert g <= prev_g + 1e-9, "gamma should not improve as rays get more dispersed"
        prev_g = g

    # 4: FINDINGS4's counterexample -- tiny mean dispersion, singular C
    M = 200
    A = np.zeros((M + 1, 3)); A[:M, 0] = 1.0; A[M, 1] = 0.5; A[M, 2] = 0.5
    r = A.sum(1); o = (r ** 2 - (A ** 2).sum(1)).sum() / max(r.sum(), 1e-30)
    q, s, gamma, rmax, live = certificate(A)
    d = A.sum(0); G = A.T @ A
    C = np.diag(1/np.sqrt(d)) @ G @ np.diag(1/np.sqrt(d))
    ev = np.linalg.eigvalsh(C)
    assert o < 0.01, f"mean dispersion should look excellent, got {o}"
    assert ev.min() < 1e-9, "C should be singular here"
    assert gamma <= 1e-9, f"certificate must FAIL here, got gamma={gamma}"
    print(f"  selftest OK: q/s match definitions; gamma>0 => lambda_min>=gamma and "
          f"kappa<=r_max/gamma ({n_cert_ok}/300 certified, {n_cert_fail} failed); certificate is "
          f"conservative; FINDINGS4 counterexample has mean dispersion {o:.4f} (excellent) yet "
          f"singular C and gamma={gamma:.2e} (correctly fails)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen,pf_nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--tau-sweep", action="store_true",
                    help="sweep an exposure threshold tau and report gamma(tau) against the SCORED-"
                         "POINT mass retained. Restricting to S makes the Gershgorin row sum "
                         "s_j^S = sum_{k in S, k != j} G_jk/d_j <= s_j, so reusing the full s_j is "
                         "CONSERVATIVE and the bound on the principal submatrix C_SS is valid. The "
                         "resulting claim is about the sub-operator on well-exposed primitives, so "
                         "the retained evaluation mass must be quoted with it.")
    ap.add_argument("--out", default="artifacts/scannet/conditioning.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    from determinism import enable_determinism
    enable_determinism()
    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            CH = 100_000_000

            def scatter(nout, idx, src):
                acc = torch.zeros(nout, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), CH):
                    e0 = min(s0 + CH, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc.float()

            # PASS 1: d_j, G_jj, r_i -- three reductions over the same non-zeros
            d = scatter(P, col, val)
            Gjj = scatter(P, col, val * val)
            r = scatter(R, row, val)
            # PASS 2: [A^T r]_j -- needs r complete
            Atr = scatter(P, col, val * r[row])

            live = d > 0
            hit = torch.zeros(R, dtype=torch.bool, device=dev); hit[row] = True
            rmax = float(r[hit].max())
            q = torch.zeros(P, device=dev); s_ = torch.zeros(P, device=dev)
            q[live] = Gjj[live] / d[live]
            s_[live] = (Atr[live] - Gjj[live]) / d[live]
            gap = (q - s_)[live]
            gamma = float(gap.min())


            # ---- tau sweep: gamma on the well-exposed sub-operator, against evaluation mass kept
            sweep = None
            if a.tau_sweep:
                import glob as _g, os as _o
                from diagnose_holes import GT_ROOT, geometry
                from diagnose_scannet_miou import load_scannet_pointcept_gt
                from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
                from point_cloud_query import (assign_points_to_power_cells,
                                               assign_points_to_nearest_center)
                dd = [x for x in _g.glob(_o.path.join(GT_ROOT, "*", sc)) if _o.path.isdir(x)][0]
                pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
                n2i = {n: i for i, n in enumerate(names)}
                pr_ = set(np.unique(raw).tolist())
                kept_ = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pr_]
                gl = remap_gt_labels(raw, [n2i[n] for n in kept_]).astype(np.int64)
                vis = np.load(_o.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
                mm = (gl > 0) & vis
                rec_ = arm.replace("pf_", "")
                if rec_ in XB.FOAM:
                    cen, rad, _u = geometry(sc, rec_)
                    own = assign_points_to_power_cells(pts[mm], cen, rad, valid=None, k=64)
                else:
                    ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                    weights_only=False)
                    spp = ck["splats"] if "splats" in ck else ck
                    own = assign_points_to_nearest_center(pts[mm], spp["means"].float().numpy(),
                                                          valid=None)
                ok_ = own >= 0
                wv = torch.zeros(P, device=dev)
                wv.index_add_(0, torch.from_numpy(own[ok_]).to(dev),
                              torch.ones(int(ok_.sum()), device=dev))
                tot_pts = float(wv.sum())
                gapf = (q - s_)
                sweep = []
                for frac in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5):
                    tau = frac * float(d[live].median())
                    S = live & (d >= tau)
                    if not bool(S.any()):
                        continue
                    gS = float(gapf[S].min())
                    sweep.append({"tau_rel_median_d": frac, "tau": tau,
                                  "n_kept": int(S.sum()),
                                  "frac_prims_kept": float(S.sum() / max(int(live.sum()), 1)),
                                  "frac_raymass_kept": float(d[S].sum() / d[live].sum()),
                                  "frac_points_kept": float(wv[S].sum() / max(tot_pts, 1e-30)),
                                  "gamma": gS,
                                  "kappa_bound": (rmax / gS) if gS > 0 else None})

            ql = q[live]
            pct = [float(torch.quantile(gap.float(), t)) for t in (0.01, 0.05, 0.25, 0.5)]
            rec = {"arm": arm, "scene": sc, "views": int(a.views), "P": int(P), "nnz": int(nnz),
                   "n_live": int(live.sum()), "n_dead": int((~live).sum()),
                   "r_max": rmax, "gamma": gamma,
                   "kappa_bound": (rmax / gamma) if gamma > 0 else None,
                   "q_min": float(ql.min()), "q_mean": float(ql.mean()),
                   "q_median": float(ql.median()),
                   "frac_q_gt_half": float((ql > 0.5).float().mean()),
                   "frac_gap_pos": float((gap > 0).float().mean()),
                   "gap_p01": pct[0], "gap_p05": pct[1], "gap_p25": pct[2], "gap_p50": pct[3],
                   "d_p01": float(torch.quantile(d[live].float(), 0.01)),
                   "d_median": float(d[live].median()),
                   "tau_sweep": sweep,
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] live {rec['n_live']:,} dead {rec['n_dead']:,} | q med "
                  f"{rec['q_median']:.4f} min {rec['q_min']:.4f} | q>1/2 on "
                  f"{100*rec['frac_q_gt_half']:.1f}% | gap>0 on {100*rec['frac_gap_pos']:.1f}% "
                  f"(p05 {rec['gap_p05']:+.4f}) | gamma {gamma:+.4e} "
                  f"{'CERTIFIED kappa<=' + format(rec['kappa_bound'], '.1f') if gamma > 0 else 'certificate FAILS'}"
                  f"  {rec['wall_s']}s", flush=True)
            del row, col, val, gid, Treg
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
