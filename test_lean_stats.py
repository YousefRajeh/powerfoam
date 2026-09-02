"""Verify `lean=True` changes ONLY the accumulators we deliberately dropped.

The claim being tested is narrow and total: for the two solvers we actually run
(geometric-median, weighted), a lean accumulator must produce BITWISE-identical output to a full
one. Not "close" -- identical. The lean path skips `support2`/`sq_numerator`/`support_iv`/
`numerator_iv` updates but must not perturb the order or content of any surviving index_add_, and
bitwise equality is the only check that proves no reassociation crept in.

Also asserted:
  - reliability() is identical (it feeds feature consensus, and reads support/intra_sum/
    sum_view_weight_sq/numerator -- all retained)
  - the three solvers that NEED the dropped fields raise a clear error rather than silently
    returning zeros, which is the actual danger of removing a buffer
  - save/load round-trips the lean flag, and a lean file does not get silently re-inflated to (P, F)
    by the backward-compatibility zero-fill in load()

Run:  python test_lean_stats.py
"""
import sys

sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")

import torch

from feature_foam_lifting.operator import (AccumulatedFeatureStats,
                                           solve_geometric_median_from_stats,
                                           solve_inverse_variance_from_stats,
                                           solve_ridge_closed_form_from_stats,
                                           solve_weighted_from_stats)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, F, V = 5000, 512, 12


def build(lean, seed=0):
    """Identical view sequence into both accumulators -- same order, same values."""
    st = AccumulatedFeatureStats.zeros(P, F, device=DEV, lean=lean)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for v in range(V):
        nnz = int(torch.randint(2000, 6000, (1,), generator=g, device=DEV).item())
        cols = torch.randint(0, P, (nnz,), generator=g, device=DEV)
        vals = torch.rand(nnz, generator=g, device=DEV) + 1e-3
        b = torch.nn.functional.normalize(torch.randn(nnz, F, generator=g, device=DEV), dim=-1)
        st.accumulate_view(cols, vals, b)
    return st


def main():
    # CUDA index_add_ reduces with atomics, so its summation order varies run to run and a plain
    # bitwise comparison would fail for reasons that have nothing to do with `lean`. Force
    # deterministic kernels so equality is actually achievable and the test means what it claims;
    # the full-vs-full control below proves the floor really is zero under this setting.
    torch.use_deterministic_algorithms(True)

    full, lean = build(False), build(True)
    control = build(False)          # same construction as `full` -- the noise floor
    ok = True

    print(f"P={P} F={F} views={V} device={DEV}  (deterministic algorithms ON)\n")
    for name, solve in (("geometric_median", solve_geometric_median_from_stats),
                        ("weighted", solve_weighted_from_stats)):
        n_ctl = int((solve(full)[0] != solve(control)[0]).sum())
        print(f"{name:18s} CONTROL full-vs-full differing: {n_ctl:,}"
              f"   {'(floor is zero -- bitwise test is meaningful)' if n_ctl == 0 else '(NONZERO FLOOR)'}")
        ok &= (n_ctl == 0)
    print()
    for name, solve in (("geometric_median", solve_geometric_median_from_stats),
                        ("weighted", solve_weighted_from_stats)):
        a = solve(full)
        b = solve(lean)
        xa, va = a[0], a[1]
        xb, vb = b[0], b[1]
        n_diff = int((xa != xb).sum())
        same_valid = bool((va == vb).all())
        print(f"{name:18s} features differing: {n_diff:,}/{xa.numel():,}   "
              f"valid masks equal: {same_valid}   -> "
              f"{'BITWISE IDENTICAL' if n_diff == 0 and same_valid else 'DIFFERS'}")
        ok &= (n_diff == 0 and same_valid)

    ra, rb = full.reliability(), lean.reliability()
    for k in ra:
        if torch.is_tensor(ra[k]):
            d = int((ra[k] != rb[k]).sum())
            print(f"reliability[{k:12s}] differing: {d:,}   "
                  f"-> {'identical' if d == 0 else 'DIFFERS'}")
            ok &= (d == 0)

    print()
    for name, fn in (("weighted(squared=True)", lambda s: solve_weighted_from_stats(s, squared=True)),
                     ("ridge", solve_ridge_closed_form_from_stats),
                     ("inverse_variance", solve_inverse_variance_from_stats)):
        try:
            fn(lean)
            print(f"{name:24s} -> NO ERROR (BAD: would return zeros silently)")
            ok = False
        except ValueError as e:
            print(f"{name:24s} -> raises ValueError as intended ({str(e)[:48]}...)")

    import os
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "lean_roundtrip.pt")
    lean.save(p)
    rt = AccumulatedFeatureStats.load(p, device=DEV)
    inflated = any(t.numel() for t in (rt.support2, rt.sq_numerator, rt.support_iv, rt.numerator_iv))
    x0 = solve_geometric_median_from_stats(lean)[0]
    x1 = solve_geometric_median_from_stats(rt)[0]
    n_rt = int((x0 != x1).sum())
    print(f"\nsave/load: lean flag preserved={rt.lean}  re-inflated={inflated}  "
          f"gm differing after round-trip: {n_rt:,}")
    ok &= rt.lean and not inflated and n_rt == 0
    os.remove(p)

    mem_full = sum(t.numel() * t.element_size() for t in
                   (full.support2, full.sq_numerator, full.support_iv, full.numerator_iv))
    print(f"\nbuffers dropped by lean at P={P}: {mem_full/1e6:.1f} MB "
          f"-> at P=2,252,236: {mem_full/P*2_252_236/1e9:.2f} GB")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
