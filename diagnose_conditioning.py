"""Conditioning diagnostics that cost two matvecs, plus the provable SQS step size.

TASK 28 -- what actually determines our conditioning?

Two quantities, both free once S is cached:

  c_j = sum_r A[r,j]           column sums: total alpha mass a cell ever accumulates
  G_jj = sum_r A[r,j]^2        the diagonal of A^T A

and one rigorous bound that needs nothing else:

  cond(G) >= max_j G_jj / min_j G_jj

because lambda_max >= max_j G_jj and lambda_min <= min_j G_jj for any SPD matrix.

The hypothesis being tested is that our conditioning is NOT set by angular coverage but by
OCCLUSION. A cell hidden behind an opaque surface has T ~ 0 on every ray, hence c_j ~ 0 -- a
near-zero COLUMN of A. That is a rank deficiency, not ill-conditioning: the information was never
collected, and no preconditioner recovers it. If the c_j histogram has a heavy spike at zero, the
fix is to prune/pin/prior those cells, not to solve harder.

TASK 27 -- the provable step size.

Disjoint cells plus telescoping transmittance make A row-substochastic: a ray visits each cell at
most once and sum_k alpha_k T_k = 1 - T_final <= 1. Jensen on the non-negative rows then gives a
LOEWNER-order bound (stronger than a spectral one):

    A^T A  <=  diag(d),    d_j = sum_r A[r,j] s_r = (A^T A 1)_j = row sums of S

In coefficient space with f_j = U_j^T a_j this becomes

    a^T H a = (Ua)^T S (Ua) <= sum_j d_j ||U_j^T a_j||^2 <= sum_j d_j lambda_max(U_j U_j^T) ||a_j||^2

so the per-cell Lipschitz constant is L_j = d_j * lambda_max(U_j U_j^T) -- a K x K eigenvalue per
cell, batched. Compare against what we currently use, L_j = sum_l ||B_{jl}||_F, which is a
Gershgorin-style bound using the Frobenius norm as a stand-in for the spectral norm. Since
||G_{jl}||_F <= K for unit rows, the current bound is roughly K * d_j, while the SQS bound is
d_j * lambda_max(Gram_j) <= K * d_j. So SQS should be tighter by lambda_max(Gram_j)/K, and a
LARGER admissible step means faster convergence. This script measures that ratio rather than
assuming it.
"""
import argparse

import torch

from solve_cone_fast import cache_path, D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--kmax", type=int, default=12)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--views", type=int, default=54)
    p.add_argument("--topk", type=int, default=6)
    a = p.parse_args()
    device = "cuda"

    cp = cache_path(a.scene, a.kmax, a.sam_level, a.views)
    c = torch.load(cp, map_location=device, weights_only=True)
    P = c["P"]
    keys, svals = c["S_keys"], c["S_vals"].float()
    support = c["support"].float()          # = column sums c_j of A
    j = keys // P
    l = keys % P
    is_diag = j == l

    # ---- Task 28a: the diagonal of G, and the rigorous cond lower bound ----------------
    Gjj = torch.zeros(P, device=device)
    Gjj[j[is_diag]] = svals[is_diag]
    alive = Gjj > 0
    q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=device)
    gq = torch.quantile(Gjj[alive].float(), q)
    print(f"=== {a.scene}: {P:,} cells, {keys.numel():,} edges ===")
    print(f"\nG_jj (diagonal of A^T A) over {int(alive.sum()):,} observed cells")
    print(f"  min {gq[0]:.4e}   p1 {gq[1]:.4e}   median {gq[2]:.4e}   p99 {gq[3]:.4e}   "
          f"max {gq[4]:.4e}")
    print(f"  RIGOROUS BOUND: cond(G) >= max/min = {float(gq[4]/gq[0].clamp_min(1e-30)):.3e}")

    # ---- Task 28b: column sums, i.e. is the problem occlusion-limited? ------------------
    print(f"\ncolumn sums c_j = sum_r A[r,j]  (total alpha mass a cell ever receives)")
    zero = int((support <= 0).sum())
    print(f"  EXACTLY zero (never touched by any ray): {zero:,} / {P:,} "
          f"({zero/P*100:.2f}%)  <- pure rank deficiency")
    sq = torch.quantile(support[support > 0].float(), q)
    print(f"  over observed cells: min {sq[0]:.4e}  p1 {sq[1]:.4e}  median {sq[2]:.4e}  "
          f"p99 {sq[3]:.4e}  max {sq[4]:.4e}")
    for t in (1e-4, 1e-3, 1e-2, 1e-1):
        n = int(((support > 0) & (support < t * sq[2])).sum())
        print(f"  below {t:g} x median: {n:,} ({n/P*100:.2f}%) -- effectively unobserved")

    # ---- Task 27: SQS step size vs the Frobenius bound we currently use -----------------
    d = torch.zeros(P, device=device)
    d.index_add_(0, j, svals)
    off = ~is_diag
    d.index_add_(0, l[off], svals[off])      # row sums of the symmetric S
    print(f"\nd_j = (A^T A 1)_j  [SQS majorizer: A^T A <= diag(d), so lambda_max <= 1 after scaling]")
    dq = torch.quantile(d[alive].float(), q)
    print(f"  min {dq[0]:.4e}   median {dq[2]:.4e}   max {dq[4]:.4e}")
    print(f"  predicted lambda_max(A^T A) ~ max_j d_j = {float(dq[4]):.4e}")

    K = a.topk + 1
    U = c["U"][:, :a.topk].float()
    Ufull = torch.zeros(P, K, U.shape[-1], device=device)
    Ufull[:, :a.topk] = U
    # the augmented direction is a unit vector too; its exact value does not change lambda_max
    # materially, and this diagnostic is about the RATIO of the two bounds
    lam = torch.zeros(P, device=device)
    CH = 100_000
    for s in range(0, P, CH):
        e = min(s + CH, P)
        Gm = torch.bmm(Ufull[s:e], Ufull[s:e].transpose(1, 2))
        lam[s:e] = torch.linalg.eigvalsh(Gm)[:, -1]
    L_sqs = d * lam
    print(f"\nper-cell Lipschitz bounds")
    print(f"  lambda_max(U_j U_j^T): median {float(lam[alive].median()):.3f} of K={K} "
          f"(K would mean perfectly correlated basis vectors, 1 means orthonormal)")
    lq = torch.quantile(L_sqs[alive].float(), q)
    print(f"  L_j^SQS = d_j * lambda_max(Gram_j): median {float(lq[2]):.4e}")
    print(f"\n  A SMALLER L_j means a LARGER admissible step. Compare against the Frobenius")
    print(f"  row-block bound currently in use by running solve_cone_fast with --precond 1")
    print(f"  and reading its [precond] line.")


if __name__ == "__main__":
    main()
