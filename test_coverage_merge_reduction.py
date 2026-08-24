"""
test_coverage_merge_reduction.py

Second experiment, motivated by Bodin & Sambridge (2009) but stripped of MCMC.

The only part of that paper that touches our problem is the IDEA that the model
parameterisation should follow data coverage (their Sec 2.1 p.1413, Sec 5
p.1432).  Their machinery for realising it (rj-MCMC) is intractable at our
scale.  This script tests the cheap DETERMINISTIC version of the same idea:

  coverage-driven column merging -- every cell with zero (or near-zero) ray
  sensitivity is absorbed into its nearest sufficiently-sampled cell, turning a
  RANK-DEFICIENT system into a smaller FULL-RANK one that needs no damping.

Measured: rank, condition number, and RMSE against a lambda-swept damped solve.
Reuses the ray geometry of test_bodin_rjmcmc_coverage.py (same RNG seed).

Run:  D:\conda\envs\powerfoam\python.exe D:\Downloads\powerfoam\test_coverage_merge_reduction.py
"""
import numpy as np
from scipy.spatial import cKDTree
import test_bodin_rjmcmc_coverage as B


def build_A(NB):
    bi = np.clip((B.PTS[:, 0] * NB).astype(int), 0, NB - 1)
    bj = np.clip((B.PTS[:, 1] * NB).astype(int), 0, NB - 1)
    A = np.zeros((B.NRAY, NB * NB))
    np.add.at(A, (B.RID, bi * NB + bj), B.DLS)
    cx = (np.arange(NB) + 0.5) / NB
    CX, CY = np.meshgrid(cx, cx, indexing='ij')
    C = np.stack([CX.ravel(), CY.ravel()], 1)
    return A, C


def stats(A, tag):
    s = np.linalg.svd(A, compute_uv=False)
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    r = int((s > tol).sum())
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    condr = s[0] / s[r - 1]
    print(f"  {tag:34s} shape={A.shape}  rank={r}  "
          f"cond={cond:.3e}  cond(rank-truncated)={condr:.3e}")
    return r, cond


def solve_ls(A, d, lam=0.0):
    H = A.T @ A + lam * np.eye(A.shape[1])
    return np.linalg.lstsq(H, A.T @ d, rcond=None)[0]


def to_grid(s_cells, assign, NB):
    """map per-(merged)-cell slowness back to the fine evaluation grid, clipped
    to the physically admissible velocity range (same range as the rj-MCMC prior)"""
    gi = np.clip((B.GRID[:, 0] * NB).astype(int), 0, NB - 1)
    gj = np.clip((B.GRID[:, 1] * NB).astype(int), 0, NB - 1)
    s_full = s_cells[assign]
    v = 1.0 / np.clip(s_full[gi * NB + gj], 1e-6, None)
    return np.clip(v, B.VMIN, B.VMAX)


def rmse(v, m=None):
    e = v - B.VT
    return float(np.sqrt(np.mean(e[m] ** 2))) if m is not None else float(np.sqrt(np.mean(e ** 2)))


def main(NB=10):
    A, C = build_A(NB)
    d = B.d_obs
    n = NB * NB
    colcov = A.sum(0)                      # total ray length in each cell
    dead = colcov <= 0.0
    print("=" * 78)
    print("Coverage-driven merging vs global damping (no MCMC)")
    print("=" * 78)
    print(f"  grid {NB}x{NB} = {n} unknowns, {B.NRAY} rays")
    print(f"  zero-coverage cells: {dead.sum()}/{n} = {100*dead.mean():.2f}%   "
          f"(our ScanNet scene: 9.44% cells untouched)")

    stats(A, "FULL system A")

    # --- variant 1: drop the dead columns ---
    keep = ~dead
    A1 = A[:, keep]
    stats(A1, "dead columns DROPPED")

    # --- variant 2: also merge weakly-sampled columns into nearest strong one ---
    for q in (0.10, 0.25):
        thr = np.quantile(colcov[keep], q)
        strong = colcov > thr
        tree = cKDTree(C[strong])
        idx_strong = np.where(strong)[0]
        assign = np.empty(n, dtype=int)
        assign[strong] = np.searchsorted(idx_strong, np.where(strong)[0])
        assign[~strong] = tree.query(C[~strong], k=1)[1]
        M = np.zeros((n, strong.sum()))
        M[np.arange(n), assign] = 1.0
        A2 = A @ M                          # merged forward operator
        r2, c2 = stats(A2, f"merged (coverage<q{q:.2f} absorbed)")
        s2 = solve_ls(A2, d, 0.0)           # NO damping at all
        v2 = to_grid(s2, assign, NB)
        chi2 = float(np.sum(((A2 @ s2 - d) / B.SIG_D) ** 2)) / B.NRAY
        print(f"      undamped solve: chi2/N={chi2:.3f}  RMSE all={rmse(v2):.4f} "
              f"covered={rmse(v2, B.M_COV):.4f}  dead={rmse(v2, B.M_DEAD):.4f}")

    # --- baseline: full system, damping sweep (needs a tuned lambda) ---
    print("\n  full system with Tikhonov damping (needs a tuned lambda):")
    ident = np.arange(n)
    best = None
    for lam in np.logspace(-6, 2, 9):
        s = solve_ls(A / B.SIG_D, d / B.SIG_D, lam)
        v = to_grid(s, ident, NB)
        chi2 = float(np.sum(((A @ s - d) / B.SIG_D) ** 2)) / B.NRAY
        rr = rmse(v)
        print(f"      lam={lam:8.2g} chi2/N={chi2:9.3f} RMSE all={rr:.4f} "
              f"covered={rmse(v, B.M_COV):.4f} dead={rmse(v, B.M_DEAD):.4f}")
        if best is None or rr < best[0]:
            best = (rr, lam)
    print(f"      best lambda={best[1]:.2g} (selected using the TRUE model)")

    # --- undamped solve on the FULL rank-deficient system, for reference ---
    s0 = np.linalg.lstsq(A / B.SIG_D, d / B.SIG_D, rcond=None)[0]
    v0 = to_grid(s0, ident, NB)
    print(f"\n  undamped pinv on FULL rank-deficient system: RMSE all={rmse(v0):.4f} "
          f"covered={rmse(v0, B.M_COV):.4f} dead={rmse(v0, B.M_DEAD):.4f}")


if __name__ == '__main__':
    for nb in (10, 14, 20):
        main(nb)
        print()
