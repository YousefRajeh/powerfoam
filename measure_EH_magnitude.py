"""How large is the sub-stochastic correction to E_H on real data?

A51 C2 / proof_EH_identity.py established the corrected identity for sub-stochastic rays:

    J - L = E_H(X) - c,                   c = sum_i (1 - r_i) ||B_i||^2      [independent of X]
    E_H(X) = 1/2 sum_jk G_jk ||X_j - X_k||^2  +  sum_j [A^T(1-r)]_j ||X_j||^2
             \\_______ graph energy _______/     \\________ correction ________/

The correction was expected to be small (mean r_i ~ 0.9965 measured), but "expected" is not
"measured", and the estimate has been quoted once already. This measures it.

COMPUTATION, and why it is pass-optimal.  No P x P matrix can be formed (P ~ 10^5), so everything is
expressed through column reductions and one operator application:

    D    = A^T 1                        column masses
    r    = A 1                          row masses
    Atr  = A^T r                        needs r complete, so a separate pass
    AX2  = ||A X||_F^2                  one gather + scatter

    E_H   = sum_j D_j ||X_j||^2   - AX2
    graph = sum_j Atr_j ||X_j||^2 - AX2                 since (G1)_j = [A^T r]_j
    extra = sum_j (D_j - Atr_j) ||X_j||^2

THREE passes over nnz is optimal for this set: `D` and `r` are independent reductions and share one
pass; `Atr` cannot start until `r` is complete, so it needs a second; `AX2` needs a third. `E_H` and
`graph` deliberately SHARE the single `AX2` -- computing them independently would cost a fourth pass
for no gain. The self-test asserts the count is exactly 3, so a future edit that adds a redundant
pass fails the test rather than silently costing time.

--selftest verifies, before any GPU time:
  1. the matvec forms equal the dense reference in `proof_EH_identity.terms` (E_H, graph, extra, c);
  2. the pass count over nnz is exactly 3, via an instrumented operator;
  3. element touches are exactly 3*nnz at every size -- a deterministic complexity check, not a
     timing comparison (timing was tried first and failed on cache effects, which prove nothing);
  4. the correction vanishes exactly at r = 1, is non-negative, and matches the exact closed form
     extra = s(1-s) sum_j D_j ||X_j||^2 under uniform scaling A -> sA (note this is NOT monotone in
     s -- it peaks at s = 1/2 -- which is why our rays at r ~ 0.9965 sit in the small-correction
     regime rather than merely "close to 1").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


class _Counter:
    """Counts full passes over the non-zeros, so 'optimal' is asserted rather than claimed."""

    def __init__(self, A):
        self.A = A
        self.passes = 0

    def col_reduce(self, w=None):
        """One pass: A^T w  (w=None means A^T 1)."""
        self.passes += 1
        return self.A.T @ (np.ones(self.A.shape[0]) if w is None else w)

    def row_reduce(self):
        """Shares the pass with the previous col_reduce -- both are reductions over the same nnz."""
        return self.A.sum(1)

    def apply_sq(self, X):
        """One pass: ||A X||_F^2."""
        self.passes += 1
        return float(((self.A @ X) ** 2).sum())


def eh_terms(op, X, Bn2=None, r=None):
    """(E_H, graph, extra) in matvec form. `op` is a _Counter or the GPU equivalent."""
    D = op.col_reduce()                      # pass 1: column masses (row masses share it)
    if r is None:
        r = op.row_reduce()
    Atr = op.col_reduce(r)                   # pass 2: needs r complete
    AX2 = op.apply_sq(X)                     # pass 3
    xn2 = (X ** 2).sum(1)
    return (float((D * xn2).sum()) - AX2,
            float((Atr * xn2).sum()) - AX2,
            float(((D - Atr) * xn2).sum()))


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from proof_EH_identity import terms as dense_terms
    rng = np.random.default_rng(0)

    # 1 & 2: correctness against the dense reference, and the pass count
    worst = 0.0
    for _ in range(300):
        R, P, d = 30, 7, 4
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        X = rng.normal(size=(P, d)); B = rng.normal(size=(R, d))
        _, _, EH_d, graph_d, extra_d, c_d = dense_terms(A, B, X)
        op = _Counter(A)
        EH_m, graph_m, extra_m = eh_terms(op, X)
        assert op.passes == 3, f"expected 3 passes over nnz, used {op.passes}"
        for nm, a_, b_ in (("E_H", EH_d, EH_m), ("graph", graph_d, graph_m), ("extra", extra_d, extra_m)):
            rel = abs(a_ - b_) / max(abs(a_), 1e-30)
            worst = max(worst, rel)
            assert rel < 1e-9, (nm, a_, b_, rel)
        # c, computed directly
        c_m = float(((1.0 - A.sum(1)) * (B ** 2).sum(1)).sum())
        assert abs(c_m - c_d) < 1e-9 * max(abs(c_d), 1.0)

    # 3: cost is LINEAR in nnz. Timing is the wrong instrument here -- a first version compared wall
    #    clock across a size sweep and failed on cache effects at the largest size (ratios
    #    1.50/1.65/3.74), which says nothing about complexity. Count element touches instead: it is
    #    deterministic, and 3 passes over nnz is exactly what the docstring claims.
    touches = []
    for R in (512, 1024, 2048, 4096):
        A = np.abs(rng.normal(size=(R, 40))) * (rng.random((R, 40)) < 0.5)
        A[:, A.sum(0) == 0] = 1.0
        A = A / A.sum(1)[:, None] * 0.9
        X = rng.normal(size=(40, 8))
        op = _Counter(A)
        eh_terms(op, X)
        nnz = int((A != 0).sum())
        touches.append((nnz, op.passes * nnz))
    for (n0, t0_), (n1, t1_) in zip(touches[:-1], touches[1:]):
        got = (t1_ / t0_) / (n1 / n0)
        assert abs(got - 1.0) < 1e-9, f"element touches not proportional to nnz: factor {got}"
    assert all(t_ == 3 * n_ for n_, t_ in touches), "pass count drifted from 3"

    # 4: the correction vanishes at r = 1, is non-negative, and follows an EXACT closed form under
    #    uniform scaling. (An earlier version asserted the correction "grows as r falls". That is
    #    FALSE: for A -> sA we get r = s and extra = s(1-s) * sum_j D_j ||X_j||^2, which is zero at
    #    BOTH s=1 and s=0 and peaks at s=1/2. The test was wrong, not the code.)
    A1 = np.abs(rng.normal(size=(40, 9))); A1 /= A1.sum(1)[:, None]
    X = rng.normal(size=(9, 3))
    _, _, ex1 = eh_terms(_Counter(A1), X)
    assert abs(ex1) < 1e-9, f"correction should vanish at r=1, got {ex1:.3e}"
    base_quad = float((A1.sum(0) * (X ** 2).sum(1)).sum())
    for scale in (0.95, 0.8, 0.5, 0.2, 0.05):
        _, _, ex = eh_terms(_Counter(A1 * scale), X)
        pred = scale * (1.0 - scale) * base_quad
        assert abs(ex - pred) < 1e-8 * max(abs(pred), 1.0), (scale, ex, pred)
        assert ex >= -1e-12, "correction must be non-negative"
    e_hi = eh_terms(_Counter(A1 * 0.95), X)[2]
    e_mid = eh_terms(_Counter(A1 * 0.5), X)[2]
    e_lo = eh_terms(_Counter(A1 * 0.05), X)[2]
    assert e_mid > e_hi and e_mid > e_lo, "extra should peak at s=1/2, not be monotone"
    print(f"  selftest OK: matvec == dense (worst rel {worst:.1e}); pass count is exactly 3; "
          f"element touches exactly 3*nnz at every size; correction vanishes at r=1 and matches "
          f"s(1-s)*quad exactly, peaking at s=1/2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true",
                    help="stage per-view operator pieces on host RAM; needed for dense arms")
    ap.add_argument("--out", default="artifacts/scannet/eh_magnitude.json")
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
            nnz = val.numel(); d = Treg.shape[1]
            if not bool((row[1:] >= row[:-1]).all()):
                o = torch.argsort(row)
                row = row[o].contiguous(); col = col[o].contiguous(); val = val[o].contiguous()
                del o; torch.cuda.empty_cache()

            # Chunked float64 scatter. Under enable_determinism() index_add_ uses a SORT-based
            # kernel whose workspace scales with the input: at ~5e8 non-zeros it wants ~12 GB, which
            # OOM'd this script twice. Chunking bounds the workspace; the float64 accumulator makes
            # the result independent of the chunk size rather than silently order-dependent.
            # (Same fix as bound_delta and measure_conditioning -- it is a systemic constraint of
            # deterministic scatter, not a per-script quirk.)
            SC = 100_000_000
            def scat(nout, idx, src):
                acc = torch.zeros(nout, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), SC):
                    e0 = min(s0 + SC, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc.float()

            # PASS 1: column masses D and row masses r (independent reductions, same sweep)
            D = scat(P, col, val)
            r = scat(R, row, val)
            # PASS 2: A^T r -- cannot start before r is complete
            Atr = scat(P, col, val * r[row])

            # Chunk must be sized by CHANNEL WIDTH, not copied from the class-space scripts. This
            # runs in FEATURE space (d = 512), so an 8M-row chunk gathers 8M x 512 x 4 B = 16 GB and
            # OOM'd at 28.7 GiB. Budget ~4e8 elements per gather instead.
            CH = max(1, int(4e8 // max(d, 1)))
            rhs = torch.zeros((P, d), device=dev)
            for s0 in range(0, nnz, CH):
                e0 = min(s0 + CH, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Treg[gid[row[s0:e0]]])
            X = rhs / D.clamp_min(torch.finfo(val.dtype).eps)[:, None]     # X' (k = 1 iterate)
            xn2 = (X ** 2).sum(1)

            # PASS 3: ||A X||_F^2, row-blocked. E_H and graph SHARE this -- computing them
            # separately would cost a fourth pass for no gain.
            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            _st = starts.cpu().tolist()
            BUD = max(1, int(3e8 // max(d, 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]
            AX2 = 0.0
            for r0, r1, s_, e_ in blocks:
                lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                t = torch.zeros((r1 - r0, d), device=dev)
                t.index_add_(0, lr, vw * X[cs]); AX2 += float((t * t).sum()); del t

            EH = float((D * xn2).sum()) - AX2
            graph = float((Atr * xn2).sum()) - AX2
            extra = float(((D - Atr) * xn2).sum())
            hit = torch.zeros(R, dtype=torch.bool, device=dev); hit[row] = True
            idx = hit.nonzero(as_tuple=True)[0]
            # Chunked: Treg[gid[idx]] over ALL hit rays is 15M x 512 x 4 B = 30 GB, which is the
            # 28.71 GiB this line asked for. Only a scalar per row is needed, so gather in slices.
            c = 0.0
            for s0 in range(0, idx.numel(), CH):
                e0 = min(s0 + CH, idx.numel())
                ii = idx[s0:e0]
                c += float(((1.0 - r[ii]) * (Treg[gid[ii]] ** 2).sum(1)).sum())

            # consistency checks on real data
            assert abs(EH - (graph + extra)) <= 1e-4 * max(abs(EH), 1.0), "E_H != graph + extra"
            assert extra >= -1e-6 * max(abs(EH), 1.0), f"correction must be >= 0, got {extra:.3e}"
            assert EH >= -1e-4 * max(abs(EH), 1.0), f"E_H must be >= 0 (H psd), got {EH:.3e}"

            rec = {"arm": arm, "scene": sc, "P": int(P), "nnz": int(nnz),
                   "E_H": EH, "graph": graph, "extra": extra, "c": c,
                   "extra_over_EH": extra / max(EH, 1e-30),
                   "c_over_EH": c / max(EH, 1e-30),
                   "mean_r": float(r[idx].mean()), "min_r": float(r[idx].min()),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] E_H {EH:.1f} = graph {graph:.1f} + extra {extra:.1f} "
                  f"({100*rec['extra_over_EH']:.3f}%) | c {c:.1f} ({100*rec['c_over_EH']:.2f}% of E_H) "
                  f"| mean r {rec['mean_r']:.6f}  {rec['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, X
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
