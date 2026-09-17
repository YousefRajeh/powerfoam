"""How much of the co-visibility mass in S = A^T A couples cells that are NOT power-diagram
neighbours? That fraction is what a surface-block solver can remove and the row-sum lift cannot.

WHY. BETA_BOUND.md Theorem 2 gives the exact excess of the closed-form lift,

    L(x') = L(x_hat) + || A D^-1 L_G x_hat ||^2 ,      L_G = D - S,  S = A^T A,  D = diag(S 1)

so the error lives entirely on the OFF-DIAGONAL of S: the pairs of cells that share ray weight.
Generalising to any SPD preconditioner M, x_hat - x'_M = M^-1 (M - S) x_hat, so if M is chosen to
contain a block of S exactly, that block's coupling costs nothing and only the rest survives.

In a foam the off-diagonal edges split into two physically different kinds:

  * ADJACENT (j, l share a power-diagram face). The ray crossed one surface spanning a few cells.
    The features should agree; this coupling is benign and a block solver absorbs it exactly.
  * NON-ADJACENT. Residual transmittance carried past the front cell onto something behind it --
    a depth jump. The features must NOT agree, and this is where || x_j - x_l || in the Dirichlet
    bound is large. This is the coupling that produces feature bleed across silhouettes.

The number this script reports,

    cross-surface fraction  =  (off-diagonal S mass on non-adjacent pairs) / (all off-diagonal mass)

is therefore the fraction of the lift's error budget that a surface-block/adjacency-restricted
solver could target, and the fraction that is irreducible for a per-primitive preconditioner.
If it is small, the row-sum lift is already near-exact and the fix is not worth building.

NOTHING IS TRAINED OR SOLVED HERE. Both inputs already exist for all ten scenes:
`gram_cache_*.pt` (S, built unpruned by solve_cone_fast.py) and `adjacency_nonfrozen.pt`
(CSR power-diagram adjacency). No x_hat, no CG, no feature maps. Read-only.

INSTRUMENT CHECKS, printed per scene and not assumed:
  * S_keys must decode to lo <= hi (upper triangle, each unordered pair once).
  * sum_j D_jj from `support` must equal diag(S) + 2 * off-diagonal mass, since D = diag(S 1).
    A mismatch means the cache and the support vector are not the same operator.
  * the implied mean per-ray overlap mass  sum_i o_i / sum_j D_jj  must lie in [0, 1).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ART = "artifacts/scannet"
# hardest first: highest mean k_i / highest measured beta in the paper's Table 1
SCENES = ["scene0070_00", "scene0645_00", "scene0140_00", "scene0590_00", "scene0000_00",
          "scene0347_00", "scene0062_00", "scene0200_00", "scene0400_00", "scene0097_00"]

CHUNK = 1 << 24


def find_cache(scene: str) -> str | None:
    d = f"{ART}/{scene}"
    if not os.path.isdir(d):
        return None
    cands = sorted(f for f in os.listdir(d)
                   if f.startswith("gram_cache_K6_") and f.endswith(".pt") and "bgdrop" not in f)
    if not cands:
        cands = sorted(f for f in os.listdir(d)
                       if f.startswith("gram_cache_") and f.endswith(".pt") and "bgdrop" not in f)
    return f"{d}/{cands[0]}" if cands else None


def adjacency_keys(scene: str, P: int) -> np.ndarray:
    """Sorted int64 array of lo*P + hi for every undirected power-diagram edge."""
    d = torch.load(f"{ART}/{scene}/adjacency_nonfrozen.pt", map_location="cpu", weights_only=False)
    adj = d["adjacent"].numpy().astype(np.int64)
    off = d["offsets"].numpy().astype(np.int64)
    Pa = int(d["num_primitives"])
    assert Pa == P, f"adjacency P={Pa} != gram P={P}"
    deg = np.diff(off)
    src = np.repeat(np.arange(off.size - 1, dtype=np.int64), deg)
    assert src.size == adj.size, (src.size, adj.size)
    lo = np.minimum(src, adj)
    hi = np.maximum(src, adj)
    keys = np.unique(lo * P + hi)          # unique() also sorts
    del adj, off, src, lo, hi, deg
    gc.collect()
    return keys


def measure(scene: str, verbose: bool = True) -> dict | None:
    cpath = find_cache(scene)
    if cpath is None:
        print(f"[skip] {scene}: no gram cache")
        return None

    c = torch.load(cpath, map_location="cpu", weights_only=False)
    P = int(c["P"])
    keys = c["S_keys"].numpy().astype(np.int64)
    vals = c["S_vals"].double().numpy()
    support = c["support"].double().numpy().reshape(-1)     # D_jj = sum_i A_ij
    n_views = int(c.get("n_views", -1))
    for k in list(c):                                        # U/Atb/top_w dominate the file
        if k not in ("P",):
            c[k] = None
    del c
    gc.collect()

    j = keys // P
    l = keys % P
    assert bool((j <= l).all()), "S_keys are not upper-triangular"

    diag_m = j == l
    mass_diag = float(vals[diag_m].sum())
    off = ~diag_m
    n_off = int(off.sum())
    mass_off = float(vals[off].sum())

    # ---- instrument check: D = diag(S 1)
    sum_D = float(support.sum())
    implied = mass_diag + 2.0 * mass_off
    rel = abs(sum_D - implied) / max(sum_D, 1e-30)

    # mean per-ray overlap mass = sum_i o_i / sum_i s_i  (Lemma 2(ii))
    overlap_frac = 2.0 * mass_off / max(sum_D, 1e-30)

    # ---- classify off-diagonal edges against power-diagram adjacency
    akeys = adjacency_keys(scene, P)
    ok = keys[off]
    ov = vals[off]
    del keys, vals, j, l, diag_m, off
    gc.collect()

    adj_mass = 0.0
    adj_cnt = 0
    for s in range(0, ok.size, CHUNK):
        blk = ok[s:s + CHUNK]
        pos = np.searchsorted(akeys, blk)
        pos_c = np.minimum(pos, akeys.size - 1)
        hit = akeys[pos_c] == blk
        adj_mass += float(ov[s:s + CHUNK][hit].sum())
        adj_cnt += int(hit.sum())
    cross_mass = mass_off - adj_mass
    cross_cnt = n_off - adj_cnt

    row = {
        "scene": scene,
        "cache": os.path.basename(cpath),
        "n_views": n_views,
        "P": P,
        "cells_observed": int((support > 0).sum()),
        "off_diag_edges": n_off,
        "adjacent_edges": adj_cnt,
        "nonadjacent_edges": cross_cnt,
        "mass_diag": mass_diag,
        "mass_off": mass_off,
        "mass_adjacent": adj_mass,
        "mass_nonadjacent": cross_mass,
        "cross_surface_fraction": cross_mass / max(mass_off, 1e-30),
        "cross_surface_edge_fraction": cross_cnt / max(n_off, 1),
        "mean_overlap_mass": overlap_frac,
        "sum_D": sum_D,
        "D_consistency_rel_err": rel,
    }
    if verbose:
        print(f"{scene}  P={P:>8,}  views={n_views:>4}  edges={n_off:>12,}", flush=True)
        print(f"    off-diagonal mass split : adjacent {adj_mass/mass_off:6.2%}   "
              f"NON-adjacent {row['cross_surface_fraction']:6.2%}")
        print(f"    by edge count           : adjacent {adj_cnt/max(n_off,1):6.2%}   "
              f"NON-adjacent {row['cross_surface_edge_fraction']:6.2%}")
        print(f"    mean per-ray overlap o  : {overlap_frac:.4f}"
              f"   [D check rel.err {rel:.2e}{'  <-- MISMATCH' if rel > 1e-3 else ''}]", flush=True)
    del ok, ov, akeys, support
    gc.collect()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--out", default="artifacts/cross_surface_gram.json")
    args = ap.parse_args()

    rows = []
    for s in args.scenes:
        try:
            r = measure(s)
        except Exception as e:                                   # keep going; report at the end
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
            continue
        if r:
            rows.append(r)
            json.dump(rows, open(args.out, "w"), indent=1)

    if not rows:
        print("nothing measured")
        return
    cf = np.array([r["cross_surface_fraction"] for r in rows])
    om = np.array([r["mean_overlap_mass"] for r in rows])
    print(f"\n=== {len(rows)} scenes ===")
    print(f"cross-surface fraction of off-diagonal S mass: "
          f"mean {cf.mean():.2%}  min {cf.min():.2%}  max {cf.max():.2%}")
    print(f"mean per-ray overlap mass o_i               : "
          f"mean {om.mean():.4f}  min {om.min():.4f}  max {om.max():.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
