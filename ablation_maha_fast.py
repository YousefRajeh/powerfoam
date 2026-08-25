"""GEMM formulation of the exact Mahalanobis argmin, with a precision guard.

The straightforward implementation materialises a (points x gaussians x 3) intermediate before
reducing, which is memory-bound: measured 0.2e9 pairs/s, i.e. 770 s for one scene and an
extrapolated 6-8 h across the ablation.

The quadratic form expands into an inner product of two 10-vectors:

    m2 = (x-mu)^T A (x-mu),   A = Sigma^-1
       = x^T A x  -  2 mu^T A x  +  mu^T A mu

    u(x)   = [x1^2, x2^2, x3^2, x1x2, x1x3, x2x3, x1, x2, x3, 1]
    v(i)   = [A11, A22, A33, 2A12, 2A13, 2A23,
              -2(A mu)_1, -2(A mu)_2, -2(A mu)_3, mu^T A mu]
    m2     = u(x) . v(i)

so the whole point-by-gaussian score matrix is a single (P,10) @ (10,N) GEMM -- tensor-core
eligible, and the output is (P,N) rather than (P,N,3).

PRECISION IS THE CATCH, and it is why this file verifies rather than assumes. The expansion
computes a small result as a difference of large terms: with scales floored at 1e-6 x extent,
A entries reach ~1e10 and x^T A x ~ 1e12, while the winning m2 may be O(1). In float32 that
is catastrophic cancellation. Two mitigations, both applied:

  * TF32 is disabled for these matmuls. TF32 keeps only 10 mantissa bits and would make the
    cancellation far worse.
  * Coordinates are shifted so the point cloud is centred at the origin, which shrinks the
    magnitude of every term in the expansion.

`assign_fast` therefore cross-checks its own argmin against the numerically stable factored
implementation on a random subset, and REFUSES to return if they disagree.
"""
import time

import torch

from ablation_maha import assign_exact


def _pack(means, scales, R):
    """-> V (10,N) with the per-Gaussian coefficients of the expanded quadratic form."""
    # A = R diag(s^-2) R^T
    inv2 = (1.0 / scales.clamp_min(1e-12)) ** 2                    # (N,3)
    A = torch.einsum("nij,nj,nkj->nik", R, inv2, R)                # (N,3,3), symmetric
    Amu = torch.einsum("nij,nj->ni", A, means)                     # (N,3)
    muAmu = (means * Amu).sum(-1)                                  # (N,)
    V = torch.stack([
        A[:, 0, 0], A[:, 1, 1], A[:, 2, 2],
        2 * A[:, 0, 1], 2 * A[:, 0, 2], 2 * A[:, 1, 2],
        -2 * Amu[:, 0], -2 * Amu[:, 1], -2 * Amu[:, 2],
        muAmu,
    ], dim=0)                                                      # (10,N)
    return V


def _u(x):
    """-> (P,10) point features of the expanded quadratic form."""
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    one = torch.ones_like(x1)
    return torch.stack([x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3,
                        x1, x2, x3, one], dim=1)


def assign_fast(points, means, scales, R, pt_chunk=16384, g_tile=262144, device="cuda",
                verify_n=256, progress=None):
    """Exact-in-intent Mahalanobis argmin via GEMM, verified against the stable path.

    Returns (idx, m2, info). Raises RuntimeError if the verification subset disagrees --
    a silent fallback would put a numerically corrupted assignment into the ablation.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False       # 10 mantissa bits would wreck this
    try:
        P, N = points.shape[0], means.shape[0]
        # Shift to the point cloud's centroid: every term in the expansion shrinks, which is
        # what makes the cancellation survivable in float32.
        origin = points.mean(0).to(device)
        pts = (points.to(device) - origin)
        mus = (means - origin)

        V = _pack(mus, scales, R)                                   # (10,N)
        idx = torch.full((P,), -1, dtype=torch.long, device=device)
        best = torch.full((P,), float("inf"), device=device)

        t0 = time.time()
        for p0 in range(0, P, pt_chunk):
            u = _u(pts[p0:p0 + pt_chunk])                            # (C,10)
            cb = torch.full((u.shape[0],), float("inf"), device=device)
            ci = torch.full((u.shape[0],), -1, dtype=torch.long, device=device)
            for a in range(0, N, g_tile):
                b = min(a + g_tile, N)
                m2 = u @ V[:, a:b]                                   # (C,T)
                v, j = m2.min(1)
                upd = v < cb
                cb[upd] = v[upd]
                ci[upd] = j[upd] + a
                del m2
            idx[p0:p0 + pt_chunk] = ci
            best[p0:p0 + pt_chunk] = cb
            if progress and (p0 // pt_chunk) % 4 == 0:
                done = min(p0 + pt_chunk, P)
                el = time.time() - t0
                progress(f"    maha-gemm {done}/{P}  {el:.0f}s  eta {el/done*(P-done):.0f}s")
        dt = time.time() - t0

        # verification against the stable factored implementation
        n = min(verify_n, P)
        sel = torch.randperm(P, device=device)[:n]
        ref_idx, _ = assign_exact(pts[sel], mus, scales, R, pt_chunk=n,
                                  g_tile=min(g_tile, 131072), device=device)
        agree = (ref_idx == idx[sel]).float().mean().item()
        if agree < 1.0:
            raise RuntimeError(
                f"GEMM Mahalanobis disagrees with the stable path on "
                f"{(1-agree)*100:.2f}% of {n} verification points -- float32 cancellation in "
                f"the expanded quadratic form. Use ablation_maha.assign_exact instead.")
        return idx, best, {"seconds": dt, "pairs_per_s": P * N / max(dt, 1e-9),
                           "verified_on": n}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
