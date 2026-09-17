"""Off-diagonal Gram mass rho, and the column-wise diagonal-dominance fraction.

The Gram-diagonality conjecture (boundResearch5) needs one number per arm: how far `G = A^T A` is
from diagonal. Forming G is impossible at P ~ 2.5M, but for a NONNEGATIVE operator it is not needed.

    sum_{j,k} G_jk = 1^T A^T A 1 = ||A 1||^2 = sum_i r_i^2        (r_i = row mass)
    sum_j   G_jj  = ||A||_F^2   = sum_i sum_j A_ij^2

    => rho := (sum_{j != k} G_jk) / (sum_j G_jj) = (sum_i r_i^2) / ||A||_F^2  -  1

One pass over nnz, exact, no Gram. And `rho + 1` is precisely the ray-mass-weighted PARTICIPATION
RATIO -- the effective number of primitives a ray depends on. A one-hot ray contributes 0 to rho; a
ray split n ways contributes n - 1. So the "1.01 vs 7.8 primitives per ray" statistic and the
off-diagonal Gram mass are the SAME quantity, which is what makes the conjecture's hypothesis
directly measurable.

Also reported, per column, using d_j = sum_i A_ij and G_jj = sum_i A_ij^2:

    q_j = G_jj / d_j,   s_j = ([A^T r]_j - G_jj) / d_j,   q_j + s_j = [A^T r]_j / d_j

Since r_i ~ 1 (measured mean 0.9965-0.9994), q_j + s_j ~ 1, so `q_j > 1/2` is exactly column-wise
diagonal dominance of G. This script checks that arithmetic rather than assuming it.

--selftest verifies:
  1. the rho identity against a densely formed A^T A on random nonnegative operators;
  2. rho = 0 exactly for one-hot rays, and rho = n-1 for rays split n ways;
  3. rho + 1 equals the participation-ratio interpretation term by term;
  4. q_j + s_j = [A^T r]_j / d_j exactly, and reduces to 1 when every r_i = 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def rho_dense(A):
    G = A.T @ A
    off = G.sum() - np.trace(G)
    return off / np.trace(G)


def rho_streamed(A):
    r = A.sum(1)
    return float((r ** 2).sum() / (A ** 2).sum() - 1.0)


def selftest():
    rng = np.random.default_rng(0)
    # 1: identity against a dense Gram
    for _ in range(300):
        R, P = int(rng.integers(5, 60)), int(rng.integers(3, 40))
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.3)
        if (A ** 2).sum() == 0:
            continue
        assert abs(rho_dense(A) - rho_streamed(A)) < 1e-9 * max(1.0, abs(rho_dense(A)))
    # 2: one-hot gives exactly 0; an n-way equal split gives exactly n-1
    for _ in range(200):
        R, P = 40, 12
        A = np.zeros((R, P))
        for i in range(R):
            A[i, rng.integers(0, P)] = rng.uniform(0.1, 1.0)
        assert abs(rho_streamed(A)) < 1e-12, rho_streamed(A)
        assert abs(rho_dense(A)) < 1e-12
    for n in (1, 2, 3, 5, 8):
        A = np.zeros((30, 10))
        for i in range(30):
            js = rng.choice(10, size=n, replace=False)
            A[i, js] = rng.uniform(0.2, 1.0)          # equal within a row
            A[i, js] = A[i, js][0]
        assert abs(rho_streamed(A) - (n - 1)) < 1e-9, (n, rho_streamed(A))
    # 3: participation-ratio reading, term by term
    for _ in range(100):
        A = rng.random((20, 9)) * (rng.random((20, 9)) < 0.5)
        r = A.sum(1); sq = (A ** 2).sum(1)
        keep = sq > 0
        pr = (r[keep] ** 2 / sq[keep])                 # per-ray effective count
        w = sq[keep] / sq[keep].sum()
        assert abs((w * pr).sum() - (rho_streamed(A) + 1)) < 1e-9
    # 4: q + s identity
    for _ in range(200):
        A = rng.random((25, 11)) * (rng.random((25, 11)) < 0.4)
        d = A.sum(0); live = d > 0
        r = A.sum(1); G = A.T @ A
        q = np.diag(G)[live] / d[live]
        s = ((A.T @ r)[live] - np.diag(G)[live]) / d[live]
        assert np.allclose(q + s, (A.T @ r)[live] / d[live], atol=1e-12)
        A1 = A / np.maximum(A.sum(1, keepdims=True), 1e-30)     # rows exactly 1
        d1 = A1.sum(0); l1 = d1 > 0; r1 = A1.sum(1); G1 = A1.T @ A1
        q1 = np.diag(G1)[l1] / d1[l1]
        s1 = ((A1.T @ r1)[l1] - np.diag(G1)[l1]) / d1[l1]
        assert np.allclose(q1 + s1, 1.0, atol=1e-9)
    print("  selftest OK: rho identity matches a dense A^T A (300 operators); rho = 0 for one-hot "
          "rows and exactly n-1 for n-way equal splits; rho+1 equals the mass-weighted participation "
          "ratio; q_j + s_j = [A^T r]_j/d_j exactly and equals 1 when rows sum to 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen,gs_froz,pf_nonfrozen,gs_unfroz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/gram_diagonality.json")
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
    CH = 100_000_000

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()

            def scat(n, idx, src):
                acc = torch.zeros(n, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), CH):
                    e0 = min(s0 + CH, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc

            r = scat(R, row, val)                       # row mass r_i
            sq_row = scat(R, row, val * val)            # sum_j A_ij^2 per ray
            d = scat(P, col, val)                       # column exposure d_j
            Gjj = scat(P, col, val * val)               # G_jj
            # [A^T r]_j
            Atr = torch.zeros(P, device=dev, dtype=torch.float64)
            for s0 in range(0, nnz, CH):
                e0 = min(s0 + CH, nnz)
                Atr.index_add_(0, col[s0:e0], (val[s0:e0].double() * r[row[s0:e0]]))

            fro2 = float(sq_row.sum())
            rho = float((r ** 2).sum()) / fro2 - 1.0
            live = d > 0
            q = (Gjj[live] / d[live])
            qs = (Atr[live] / d[live])
            s = qs - q
            # per-ray participation ratio, mass weighted
            rk = sq_row > 0
            pr = (r[rk] ** 2 / sq_row[rk])

            rec = {"arm": arm, "scene": sc, "P": int(P), "R": int(R), "nnz": int(nnz),
                   "live": int(live.sum().item()),
                   "rho": rho,
                   "participation_mean": float((sq_row[rk] / sq_row[rk].sum() * pr).sum()),
                   "participation_median": float(pr.median()),
                   "r_mean": float(r[r > 0].mean()), "r_max": float(r.max()),
                   "q_median": float(q.median()),
                   "frac_q_gt_half": float((q > 0.5).double().mean()),
                   "q_plus_s_mean": float(qs.mean()),
                   "q_plus_s_min": float(qs.min()), "q_plus_s_max": float(qs.max()),
                   "frac_col_diag_dominant": float((q > s).double().mean()),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] rho {rho:8.4f} | participation mean "
                  f"{rec['participation_mean']:6.3f} med {rec['participation_median']:5.3f} | "
                  f"q med {rec['q_median']:.3f} q>1/2 {100*rec['frac_q_gt_half']:5.1f}% "
                  f"diag-dom {100*rec['frac_col_diag_dominant']:5.1f}% | q+s "
                  f"[{rec['q_plus_s_min']:.4f},{rec['q_plus_s_max']:.4f}] mean "
                  f"{rec['q_plus_s_mean']:.4f}  {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
