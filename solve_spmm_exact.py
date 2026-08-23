"""The EXACT coupled solve, made fast by forming S -- no cone, no basis, no approximation.

WHY THIS EXISTS SEPARATELY FROM solve_cone_fast.py

The coefficient-space reformulation gets ~270x by never leaving the 7-dimensional cone basis, but
it pays for that with an approximation: the solution is restricted to the span of each cell's
top-K observed embeddings. A research agent pointed out we had attributed the whole speedup to
that restriction when most of it comes from forming S at all:

    once S = A^T A exists, the FULL 512-channel operator is just  S @ F,
    a sparse-times-dense product, with NO basis and NO restriction.

At 16.2M edges and F of shape (204k, 512) that is ~65 GB of memory traffic, i.e. ~0.1-0.4 s,
against the 33 s / 3.02 TiB of the matrix-free ray-streaming version. So the exact operator is
50-150x faster than what we were doing, and the cone basis buys a further ~10x on top of THAT,
not 70x on top of the matrix-free path.

WHAT THIS UNLOCKS. Every earlier unconstrained result was measured with the slow operator, which
is why the ridge sweep took hours and why only scene0347_00 was ever swept. With this path the
whole lambda sweep is minutes, so the claim "the exact solve is 9 mIoU worse and lambda is a
'how far off the CLIP cone may I wander' dial" can be re-established on several scenes rather
than one -- which matters, because the cone result is currently trending negative on the 10-scene
run and the unconstrained comparison it is measured against deserves the same scrutiny.

The one thing this path CANNOT do is impose the non-negativity constraint: projecting onto the
cone requires the per-cell basis, which is exactly what this drops. So the two solvers are
complementary -- this is the honest exact reference, solve_cone_fast is the constrained method.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from solve_cone_fast import build, cache_path, D
from gram_blocks import prune_edges


class SparseGram:
    """S as a symmetric CSR-like operator applied to a dense (P, D) block of channels.

    Only the upper triangle is stored, so each application scatters the block AND its transpose,
    skipping the diagonal on the second pass so it is not counted twice -- the same convention as
    the block-sparse version, and the same place a bug hid there (off-diagonals were counted four
    times, caught at 4.5e-02 relative error).
    """

    def __init__(self, keys, vals, P):
        self.j = (keys // P).long()
        self.l = (keys % P).long()
        self.v = vals.float()
        self.diag = self.j == self.l
        self.P = P

    def matvec(self, X, chunk=2_000_000):
        """X: (P, D) -> S @ X, chunked over EDGES so the (chunk, D) product is bounded."""
        out = torch.zeros_like(X)
        E = self.j.numel()
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            j, l, v = self.j[s:e], self.l[s:e], self.v[s:e]
            out.index_add_(0, j, v[:, None] * X[l])
            off = ~self.diag[s:e]
            if bool(off.any()):
                out.index_add_(0, l[off], v[off][:, None] * X[j[off]])
        return out

    def diagonal(self):
        d = torch.zeros(self.P, device=self.v.device)
        d[self.j[self.diag]] = self.v[self.diag]
        return d

    def row_sums(self):
        """d = S 1, the SQS majorizer: S <= diag(d) in the Loewner order (task 27)."""
        d = torch.zeros(self.P, device=self.v.device)
        d.index_add_(0, self.j, self.v)
        off = ~self.diag
        d.index_add_(0, self.l[off], self.v[off])
        return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--kmax", type=int, default=6)
    p.add_argument("--topk", type=int, default=6)      # unused here; keeps build() signature happy
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--max-edges", type=int, default=60_000_000)
    p.add_argument("--merge-limit", type=int, default=20_000_000)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="lambda as a fraction of mean(diag(S))")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--save-diagonal", default=None)
    a = p.parse_args()
    device = "cuda"

    cache = build(a, device)
    P = cache["P"]
    Atb = cache["Atb"].float()
    support = cache["support"].float()
    kk, vv, mass = prune_edges(cache["S_keys"], cache["S_vals"].float(), P, a.max_edges)
    S = SparseGram(kk, vv, P)
    del cache
    torch.cuda.empty_cache()

    valid = support > 0
    x_diag = torch.zeros(P, D, device=device)
    x_diag[valid] = Atb[valid] / support[valid][:, None]

    diag = S.diagonal()
    lam = a.ridge * float(diag[valid].mean())
    print(f"[ridge] lambda = {lam:.6g}", flush=True)

    # SQS majorizer, task 27: S <= diag(d), so Jacobi on (d + lambda) is a PROVABLE preconditioner
    d = S.row_sums()
    M = (d + lam).clamp_min(1e-20)
    print(f"[sqs] d = S1: median {float(d[valid].median()):.4e}  max {float(d[valid].max()):.4e}"
          f"   (predicted lambda_max of S)", flush=True)

    def apply(X):
        return S.matvec(X) + lam * X

    # preconditioned CG on (S + lambda I) x = A^T b, started at the diagonal incumbent
    x = x_diag.clone()
    t0 = time.time()
    r = Atb - apply(x)
    z = r / M[:, None]
    pdir = z.clone()
    rz = (r * z).sum()
    bn = Atb.norm()
    print(f"[cg] start relative residual {float(r.norm()/bn):.6f}", flush=True)
    for it in range(a.iters):
        Ap = apply(pdir)
        den = (pdir * Ap).sum()
        if float(den) <= 0:
            print(f"[cg] non-positive curvature at {it}", flush=True)
            break
        al = rz / den
        x = x + al * pdir
        r = r - al * Ap
        rel = float(r.norm() / bn)
        if it % 20 == 0 or rel < a.rtol:
            print(f"[cg] iter {it:4d}  relative residual {rel:.6e}  ({time.time()-t0:.1f}s)",
                  flush=True)
        if rel < a.rtol:
            break
        z = r / M[:, None]
        rzn = (r * z).sum()
        pdir = z + (rzn / rz) * pdir
        rz = rzn
    print(f"[cg] {time.time()-t0:.1f}s total", flush=True)

    # how far off the CLIP cone did the UNCONSTRAINED solve go? (chunked -- the (N,512) gather
    # trap has bitten four times in this project)
    idx = torch.where(valid)[0]
    n_over, tot, mx = 0, 0, 0.0
    for s in range(0, idx.numel(), 200_000):
        ii = idx[s:s + 200_000]
        nrm = x[ii].norm(dim=-1)
        n_over += int((nrm > 1).sum()); tot += ii.numel(); mx = max(mx, float(nrm.max()))
    print(f"[cone] ||f|| > 1 on {n_over/max(tot,1)*100:.2f}% of valid cells, max {mx:.3f}   "
          f"(impossible for a convex combination of unit vectors)", flush=True)

    torch.save({"primitive_features": x.cpu(), "valid_mask": valid.cpu()}, a.output)
    print(f"[solve_spmm_exact] {int(valid.sum())}/{P} valid -> {a.output}", flush=True)
    if a.save_diagonal:
        torch.save({"primitive_features": x_diag.cpu(), "valid_mask": valid.cpu()},
                   a.save_diagonal)
        print(f"[solve_spmm_exact] diagonal baseline -> {a.save_diagonal}", flush=True)


if __name__ == "__main__":
    main()
