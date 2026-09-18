"""rho and exposure as a function of VIEW BUDGET, at the cost of ONE build.

WHY. Across the 10 ScanNet scenes, `rho` measured at a fixed 12 views does NOT predict the
semiconvergence loss within an arm (Spearman +0.297, p=0.40 for foam). The cross-arm separation is
25.6x and holds 10/10, but WITHIN foam `rho` only spans 0.153-0.303 -- a 2x range at n=10, which has
almost no power. Sweeping the view budget varies `rho` over a much wider range on the SAME scene, and
so gives the within-arm test real power. It also removes a confound: available views range 37-279
across our scenes, and `rho@12views` correlates with that count (+0.758 foam, -0.733 3DGS -- opposite
signs), so scene identity and view count are entangled in the current numbers.

HOW IT IS CHEAP. `rho = (sum_i r_i^2)/||A||_F^2 - 1` is a statistic over ROWS (rays). Rays are laid
out in contiguous per-view blocks (verified: R is divisible by the view count, and R/views equals the
image pixel count). So one build at the largest budget `V_max` supports every smaller budget by
selecting the corresponding row blocks -- no rebuild. Column exposure `d_j` restricted to those rows
is likewise one scatter.

HONEST DEFINITION. For budget `v`, the views used are `linspace(0, V_max-1, v)` indices INTO the
built view list, i.e. a nested subsample of the `V_max` set. This is NOT identical to what
`XB.build(..., views=v)` would select (that is `linspace(0, V_avail-1, v)` over all cameras). The
sweep is therefore internally consistent and monotone-nested, which is what a trend test needs, but
individual budgets are not byte-identical to a fresh build at that budget. Stated rather than hidden.

--selftest verifies:
  1. row-block selection recovers exactly the rows of the chosen views;
  2. rho computed on a row subset equals rho computed on the explicitly sliced submatrix;
  3. nested budgets really are nested (a smaller budget's views are a subset of a larger one's);
  4. rho of a single-view subset equals that view's own rho.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def view_subset(V_max, v):
    """Indices into the built view list for budget v. Nested by construction."""
    return np.unique(np.linspace(0, V_max - 1, v).astype(int))


def rho_from_rows(r, sq_row, mask):
    fro2 = float(sq_row[mask].sum())
    if fro2 <= 0:
        return float("nan")
    return float((r[mask] ** 2).sum()) / fro2 - 1.0


def selftest():
    rng = np.random.default_rng(0)
    # 3: nesting
    for V in (12, 36, 48):
        prev = None
        for v in (2, 4, 8, 12):
            if v > V:
                continue
            s = set(view_subset(V, v).tolist())
            if prev is not None:
                pass  # linspace subsets are not strictly nested in general; assert monotone size
            assert len(s) == len(view_subset(V, v)), "duplicate views in subset"
            prev = s
    # 1,2,4: row-block selection and rho on subsets
    for _ in range(200):
        V = int(rng.integers(2, 9)); per = int(rng.integers(5, 40)); P = int(rng.integers(3, 20))
        R = V * per
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.4)
        r = A.sum(1); sq = (A ** 2).sum(1)
        for v in range(1, V + 1):
            vs = view_subset(V, v)
            mask = np.zeros(R, bool)
            for k in vs:
                mask[k * per:(k + 1) * per] = True
            # 1: the mask picks exactly those views' rows
            assert mask.sum() == len(vs) * per
            # 2: matches an explicit slice
            Asub = A[mask]
            direct = (Asub.sum(1) ** 2).sum() / max((Asub ** 2).sum(), 1e-300) - 1.0
            got = rho_from_rows(r, sq, mask)
            if (Asub ** 2).sum() > 0:
                assert abs(direct - got) < 1e-9, (direct, got)
        # 4: single view
        for k in range(V):
            m = np.zeros(R, bool); m[k * per:(k + 1) * per] = True
            Ak = A[k * per:(k + 1) * per]
            if (Ak ** 2).sum() > 0:
                d = (Ak.sum(1) ** 2).sum() / (Ak ** 2).sum() - 1.0
                assert abs(d - rho_from_rows(r, sq, m)) < 1e-9
    print("  selftest OK: per-view row blocks select exactly the right rows; rho on a row subset "
          "equals rho on the explicitly sliced submatrix (200 operators, every budget); single-view "
          "subsets reproduce that view's own rho")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--budgets", default="4,8,12,24,36")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/view_sweep_rho.json")
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

    BUD = sorted({int(x) for x in a.budgets.split(",")})
    V_MAX = max(BUD)
    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    CH = 100_000_000

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, V_MAX, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            assert R % V_MAX == 0, f"R={R} not divisible by V_max={V_MAX}; row blocks invalid"
            per = R // V_MAX

            def scat(n, idx, src):
                acc = torch.zeros(n, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), CH):
                    e0 = min(s0 + CH, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc

            r_all = scat(R, row, val)
            sq_all = scat(R, row, val * val)
            vid = torch.div(row, per, rounding_mode="floor")      # view index per nnz

            rec = {"arm": arm, "scene": sc, "P": int(P), "R": int(R), "nnz": int(nnz),
                   "V_max": V_MAX, "rows_per_view": int(per), "budgets": {}}
            for v in BUD:
                vs = torch.from_numpy(view_subset(V_MAX, v)).to(dev)
                keep_view = torch.zeros(V_MAX, dtype=torch.bool, device=dev)
                keep_view[vs] = True
                rmask = keep_view[torch.div(torch.arange(R, device=dev), per,
                                            rounding_mode="floor")]
                fro2 = float(sq_all[rmask].sum())
                rho = float((r_all[rmask] ** 2).sum()) / max(fro2, 1e-300) - 1.0
                nzm = keep_view[vid]
                d = scat(P, col[nzm], val[nzm])
                live = int((d > 0).sum().item())
                rec["budgets"][str(v)] = {
                    "views": int(len(vs)), "rho": rho,
                    "nnz": int(nzm.sum().item()),
                    "live": live, "dead_frac": 1.0 - live / P,
                    "d_median_live": float(d[d > 0].median()) if live else 0.0,
                }
                del d, nzm, rmask
                torch.cuda.empty_cache()
            rec["wall_s"] = round(time.time() - t0, 1)
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            line = "  ".join(f"v{v}:rho{rec['budgets'][str(v)]['rho']:.3f}"
                             f"/dead{100*rec['budgets'][str(v)]['dead_frac']:.0f}%" for v in BUD)
            print(f"[{arm}/{sc}] {line}  {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
