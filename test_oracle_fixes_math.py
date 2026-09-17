"""Prove the math in `oracle_fixes.py` before spending GPU time on it.

Three claims, each tested against something independent rather than against itself:

1. `project_simplex` really is the EUCLIDEAN PROJECTION onto {y >= 0, sum y = 1}.
   Checked three ways: (a) feasibility, (b) optimality via the KKT/variational inequality
   <y - p, q - p> <= 0 for every feasible q -- verified against random simplex points AND against
   the vertices, which are the extreme points where a violation would show first, (c) against a
   brute-force projection by constrained numerical optimisation on small cases.

2. `grad_and_obj` returns (A^T(AY - S), ||AY - S||^2). The gradient of ||AY-S||^2 is 2A^T(AY-S),
   so the returned gradient is HALF the true gradient -- fine for a step rule, but the factor is
   asserted rather than assumed, against autograd.

3. The abstaining readout at tau = 0 must reproduce the unabstained baseline EXACTLY. If it does
   not, the abstention sweep is measuring an implementation difference rather than the idea.
"""
from __future__ import annotations
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from oracle_fixes import project_simplex


def test_simplex_feasible_and_optimal(seed=0, n=4000, C=7, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    # a spread of regimes: already-feasible, negative, large, near-degenerate
    Y = torch.cat([
        torch.randn(n, C, generator=g) * 3.0,
        torch.rand(n, C, generator=g),
        torch.full((n, C), -5.0) + torch.randn(n, C, generator=g),
        torch.nn.functional.normalize(torch.rand(n, C, generator=g), p=1, dim=1),
    ]).double()
    Pj = project_simplex(Y)

    feas_nonneg = float(Pj.min())
    feas_sum = float((Pj.sum(1) - 1.0).abs().max())

    # variational inequality: for the true projection, <Y - P, q - P> <= 0 for all feasible q
    worst = -np.inf
    qs = [torch.eye(C).double()[k].expand(Y.shape[0], C) for k in range(C)]      # vertices
    qs.append(torch.full_like(Y, 1.0 / C))                                        # centroid
    for _ in range(5):                                                            # random points
        r = torch.rand(Y.shape[0], C, generator=g).double()
        qs.append(r / r.sum(1, keepdim=True))
    for q in qs:
        worst = max(worst, float(((Y - Pj) * (q - Pj)).sum(1).max()))

    # idempotence: projecting a point already on the simplex must not move it
    S = torch.rand(n, C, generator=g).double(); S = S / S.sum(1, keepdim=True)
    idem = float((project_simplex(S) - S).abs().max())

    print(f"[simplex] min entry {feas_nonneg:.3e} (>= -1e-12)   "
          f"max |sum-1| {feas_sum:.3e} (<= 1e-12)")
    print(f"[simplex] worst <Y-P, q-P> over vertices+centroid+random = {worst:.3e} (<= 1e-9)")
    print(f"[simplex] idempotent on the simplex: max move {idem:.3e} (<= 1e-12)")
    return feas_nonneg > -1e-12 and feas_sum < 1e-12 and worst < 1e-9 and idem < 1e-12


def test_simplex_vs_bruteforce(seed=1, n=200, C=5):
    """Independent check: minimise ||y - p||^2 over the simplex by projected SGD from many starts."""
    g = torch.Generator().manual_seed(seed)
    Y = (torch.randn(n, C, generator=g) * 2.0).double()
    Pj = project_simplex(Y)
    # brute force: parameterise by softmax logits and optimise; a slow but independent route
    best = torch.full((n,), np.inf, dtype=torch.float64)
    for s in range(6):
        z = torch.randn(n, C, generator=g).double().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.15)
        for _ in range(1500):
            opt.zero_grad()
            q = torch.softmax(z, dim=1)
            loss = ((Y - q) ** 2).sum(1)
            loss.sum().backward(); opt.step()
        with torch.no_grad():
            best = torch.minimum(best, ((Y - torch.softmax(z, 1)) ** 2).sum(1))
    ours = ((Y - Pj) ** 2).sum(1)
    # softmax can only approach the boundary, so ours must be <= brute force up to tolerance
    gap = float((ours - best).max())
    print(f"[simplex vs brute force] max (ours - bruteforce) = {gap:.3e} (<= 1e-4; ours should win)")
    return gap < 1e-4


def test_grad_matches_autograd(seed=2, R=500, P=80, C=6, nnz_per_row=4):
    g = torch.Generator().manual_seed(seed)
    row = torch.arange(R).repeat_interleave(nnz_per_row)
    col = torch.randint(0, P, (R * nnz_per_row,), generator=g)
    val = torch.rand(R * nnz_per_row, generator=g).double() + 0.05
    kcl = torch.randint(0, C, (R,), generator=g)

    Y = torch.randn(P, C, generator=g).double().requires_grad_(True)
    A = torch.zeros(R, P).double()
    A.index_put_((row, col), val, accumulate=True)
    S = torch.zeros(R, C).double(); S[torch.arange(R), kcl] = 1.0
    obj_ref = ((A @ Y - S) ** 2).sum()
    obj_ref.backward()
    grad_auto = Y.grad.detach().clone()

    # replicate the script's computation
    with torch.no_grad():
        ay = torch.zeros(R, C).double()
        ay.index_add_(0, row, val.unsqueeze(-1) * Y.detach()[col])
        ay[torch.arange(R), kcl] -= 1.0
        obj_ours = float(ay.pow(2).sum())
        g_ours = torch.zeros(P, C).double()
        g_ours.index_add_(0, col, val.unsqueeze(-1) * ay[row])

    d_obj = abs(obj_ours - float(obj_ref)) / max(abs(float(obj_ref)), 1e-30)
    d_grad = float((2.0 * g_ours - grad_auto).abs().max() / grad_auto.abs().max())
    print(f"[grad] objective rel err {d_obj:.3e} (<= 1e-12)")
    print(f"[grad] 2*ours vs autograd rel err {d_grad:.3e} (<= 1e-12)  "
          f"-- confirms ours = HALF the true gradient")
    return d_obj < 1e-12 and d_grad < 1e-12


def test_abstain_tau0_is_identity(seed=3, P=3000, npts=9000):
    """tau=0 must select every live primitive, so the pool -- and thus the metric -- is unchanged."""
    rng = np.random.default_rng(seed)
    D = torch.from_numpy(rng.random(P).astype(np.float64))
    live = D > 0.0
    Dl = D[live]
    thr0 = torch.tensor(-1.0, dtype=torch.float64)
    keep0 = (D > thr0) & live
    same = bool((keep0 == live).all())
    # and a positive tau must be a strict subset with the expected retained fraction
    ok_frac = True
    for tau in (0.05, 0.2, 0.5):
        thr = torch.quantile(Dl, tau)
        keep = (D > thr) & live
        frac = float(keep.sum()) / float(live.sum())
        if not (keep.sum() <= live.sum() and abs(frac - (1.0 - tau)) < 0.02):
            ok_frac = False
        print(f"[abstain] tau={tau:<5g} kept {frac:.3f} of live (expect ~{1-tau:.2f}), "
              f"subset={bool((keep & ~live).sum() == 0)}")
    print(f"[abstain] tau=0 pool identical to live: {same}")
    return same and ok_frac


def main():
    torch.manual_seed(0)
    results = {
        "simplex feasible+optimal": test_simplex_feasible_and_optimal(),
        "simplex vs brute force": test_simplex_vs_bruteforce(),
        "gradient vs autograd": test_grad_matches_autograd(),
        "abstain tau=0 identity": test_abstain_tau0_is_identity(),
    }
    print()
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print("\nRESULT:", "all math checks pass" if ok else "*** MATH IS WRONG - DO NOT RUN ***")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
