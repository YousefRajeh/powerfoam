"""Validate the beta computation against brute force, and test whether beta is even well-posed.

Three things are checked, in increasing order of consequence:

1. IMPLEMENTATION. The vectorised two-pass beta in compute_beta.py must equal a literal
   per-ray transcription of the published definition.

2. THE k=1 IDENTITY. A ray touching one primitive has sigma_i = 0, so beta_i = 0 exactly.

3. WELL-POSEDNESS -- the one that matters. When G = A^T A is singular the least-squares
   minimiser is NOT unique. The excess L(X') - L(Xhat) is invariant to which minimiser is
   chosen (it depends on Xhat only through A Xhat, which is common to all of them). beta is
   NOT: it is built from Delta_ij = ||xhat_j - b_i||, which reads Xhat ITSELF. So two exact
   minimisers with identical loss can give wildly different beta, and "the" beta of a scene
   is undefined. This test constructs that situation explicitly.

If (3) fires, a measured beta_max is a property of whichever solver happened to run, not of the
data -- and on real scenes, where Xhat blows up to 1e7 in a near-null direction, that is exactly
what a huge beta_max is reporting.
"""
from __future__ import annotations
import numpy as np
import torch


def beta_reference(A, X, B):
    """Literal transcription of the definition, one ray at a time. Slow and obviously correct."""
    R = A.shape[0]
    out = np.full(R, np.nan)
    for i in range(R):
        nz = np.nonzero(A[i])[0]
        if nz.size == 0:
            continue
        w = A[i, nz]
        s = w.sum()
        if s <= 0:
            continue
        w = w / s                                  # beta is defined under a weight DISTRIBUTION
        delta = np.linalg.norm(X[nz] - B[i][None, :], axis=-1)
        mu = float((w * delta).sum())
        sig2 = float((w * (delta - mu) ** 2).sum())
        out[i] = sig2 / mu ** 2 if mu > 1e-6 else np.nan
    return out


def beta_vectorised(A, X, B):
    """The two-pass form compute_beta.py uses, on dense inputs."""
    dev = "cpu"
    At = torch.tensor(A, dtype=torch.float64)
    Xt = torch.tensor(X, dtype=torch.float64)
    Bt = torch.tensor(B, dtype=torch.float64)
    row, col = torch.nonzero(At, as_tuple=True)
    val = At[row, col]
    R = At.shape[0]
    rowsum = torch.zeros(R, dtype=torch.float64).index_add_(0, row, val)
    rs = rowsum.clamp_min(torch.finfo(torch.float64).eps)
    d = (Xt[col] - Bt[row]).norm(dim=-1)
    mu = torch.zeros(R, dtype=torch.float64).index_add_(0, row, val * d) / rs
    w = val / rs[row]
    m2 = torch.zeros(R, dtype=torch.float64).index_add_(0, row, w * (d - mu[row]) ** 2)
    beta = torch.full((R,), float("nan"), dtype=torch.float64)
    ok = (rowsum > 0) & (mu > 1e-6)
    beta[ok] = m2[ok].clamp_min(0) / mu[ok] ** 2
    return beta.numpy()


def loss(A, X, B):
    return float(((A @ X - B) ** 2).sum())


def make_rowstochastic(R, P, kmax, rng):
    A = np.zeros((R, P))
    for i in range(R):
        k = rng.integers(1, kmax + 1)
        idx = rng.choice(P, k, replace=False)
        w = rng.random(k) + 1e-3
        A[i, idx] = w / w.sum()
    return A


def main():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("1. IMPLEMENTATION: vectorised two-pass vs literal per-ray reference")
    worst = 0.0
    for s in range(5):
        r = np.random.default_rng(s)
        A = make_rowstochastic(200, 30, 4, r)
        X = r.normal(size=(30, 8))
        B = r.normal(size=(200, 8))
        a, b = beta_reference(A, X, B), beta_vectorised(A, X, B)
        m = np.isfinite(a) & np.isfinite(b)
        err = np.abs(a[m] - b[m]).max()
        worst = max(worst, err)
        print(f"   seed{s}: max |ref - vec| = {err:.3e}   ({m.sum()} rays compared)")
    print(f"   => {'PASS' if worst < 1e-9 else 'FAIL'} (worst {worst:.3e})")

    print("\n" + "=" * 78)
    print("2. k=1 IDENTITY: a ray touching one primitive must give beta_i = 0 exactly")
    r = np.random.default_rng(7)
    A = np.zeros((100, 20)); A[np.arange(100), r.integers(0, 20, 100)] = 1.0
    X = r.normal(size=(20, 8)); B = r.normal(size=(100, 8))
    b = beta_vectorised(A, X, B)
    fin = b[np.isfinite(b)]
    print(f"   max beta over {fin.size} single-hit rays = {np.abs(fin).max():.3e}")
    print(f"   => {'PASS' if np.abs(fin).max() < 1e-12 else 'FAIL'}")

    print("\n" + "=" * 78)
    print("3. WELL-POSEDNESS under a SINGULAR G: is beta a property of the data at all?")
    # Two primitives that are never separated by any ray: every ray weights them equally, so
    # only (x_1 + x_2) is determined and (x_1 - x_2) is a null direction of G.
    R, P, F = 60, 4, 3
    A = np.zeros((R, P))
    rr = np.random.default_rng(3)
    for i in range(R):
        A[i, 0] = A[i, 1] = 0.5 * rr.random() + 0.05
        A[i, 2] = rr.random() + 0.05
        A[i] /= A[i].sum()
    B = rr.normal(size=(R, F))
    G = A.T @ A
    ev = np.linalg.eigvalsh(G)
    print(f"   eigenvalues of G: {np.array2string(ev, precision=4)}")
    print(f"   => G is {'SINGULAR' if ev.min() < 1e-12 else 'nonsingular'} "
          f"(null direction = x_0 - x_1)")

    Xls = np.linalg.lstsq(A, B, rcond=None)[0]          # minimum-norm minimiser
    null = np.zeros((P, F)); null[0] = 1.0; null[1] = -1.0
    print(f"\n   {'t':>10}{'loss':>14}{'||X||':>12}{'beta_max':>14}{'beta_p50':>12}")
    for t in [0.0, 1.0, 10.0, 1000.0, 1e5]:
        Xt = Xls + t * null
        bt = beta_vectorised(A, Xt, B)
        f = bt[np.isfinite(bt)]
        print(f"   {t:>10.0f}{loss(A, Xt, B):>14.6f}{np.linalg.norm(Xt):>12.3e}"
              f"{f.max():>14.4e}{np.median(f):>12.4e}")
    print("\n   Every row above is an EXACT minimiser: identical loss, identical A X.")
    print("   The excess L(X') - L(Xhat) is therefore identical for all of them, but beta is")
    print("   not -- it grows without bound along the null direction. beta is a property of")
    print("   the SOLVER's output, not of the data, whenever G is rank-deficient.")

    # and the excess really is invariant
    D = A.sum(0)
    Xp = np.zeros_like(Xls)
    lv = D > 0                      # primitive 3 is touched by no ray; dividing by 0 gives NaN
    Xp[lv] = (A.T @ B)[lv] / D[lv, None]
    exc = [loss(A, Xp, B) - loss(A, Xls + t * null, B) for t in [0.0, 1e5]]
    print(f"\n   excess at t=0: {exc[0]:.10f}   at t=1e5: {exc[1]:.10f}   "
          f"(identical: {abs(exc[0] - exc[1]) < 1e-9})")


if __name__ == "__main__":
    main()
