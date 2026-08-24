"""Measure d_ij on a real scene: is the difference across each facet determined by the data?

    d_ij = sum_r (A_ri - A_rj)^2 / (S_ii + S_jj) = (S_ii + S_jj - 2 S_ij) / (S_ii + S_jj)

See `facet_identifiability.py` for the derivation and for the validation against Curtis & Snieder
(Geophysics 62(4):1524-1532, 1997) Figure 1, which this measure reproduces exactly -- including
their two zero eigenvalues, localised to the two facets where the difference is unresolvable.

WHAT WOULD MAKE THIS MEASURE UNINTERESTING, stated before running it so the result cannot be
rationalised afterwards:
  - If d is ~1 on essentially every facet, there are no null directions in the partition, the
    "cells are a solver design variable" angle has nothing to bite on, and we drop it.
  - If d is bimodal or has a substantial low-d population, then a real fraction of facets are
    invisible to the data, and re-parameterising to make them visible is a lever.
The prediction to test afterwards is that refinement gain concentrates on LOW-d facets, since that
is where the data genuinely cannot separate neighbours and a graph prior supplies the information.

S IS SYMMETRIC AND ONLY THE UPPER TRIANGLE IS STORED. The cache emits each unordered pair once
(this was a real bug: off-diagonals were counted 4x and the Hessian was wrong at 4.5e-02 until a
`right >= left` filter fixed it to 6.2e-07). So a facet (i,j) must be looked up as the ordered key
min*P + max, not both ways.
"""
import argparse
import os

import numpy as np
import torch

from facet_identifiability import facet_identifiability


def load_true_facets(path):
    z = np.load(path)
    for a, b in (("edges", None), ("i", "j"), ("src", "dst")):
        if a in z:
            e = z[a] if b is None else np.stack([z[a], z[b]], axis=1)
            return np.ascontiguousarray(e.astype(np.int64))
    raise SystemExit(f"no edge array in {path}: keys = {list(z.keys())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--facet-graph", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--kmax", type=int, default=6)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--merge-limit", type=int, default=20_000_000)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--keep-background", action="store_true")
    a = p.parse_args()
    device = "cuda"

    from solve_cone_fast import build
    cache = build(a, device)
    P = cache["P"]
    keys = cache["S_keys"].to(device)
    vals = cache["S_vals"].float().to(device)
    print(f"[gram] P={P}  stored entries={keys.numel()}", flush=True)

    # diagonal S_ii
    kj, kl = (keys // P).long(), (keys % P).long()
    diag = torch.zeros(P, device=device, dtype=torch.float64)
    dm = kj == kl
    diag[kj[dm]] = vals[dm].double()

    edges = torch.from_numpy(load_true_facets(a.facet_graph)).to(device)
    lo = torch.minimum(edges[:, 0], edges[:, 1]).long()
    hi = torch.maximum(edges[:, 0], edges[:, 1]).long()
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    want = lo * P + hi
    print(f"[facets] {want.numel()} unordered facets", flush=True)

    # look up S_ij by sorted-key search; facets with no stored entry have S_ij = 0 exactly
    order = torch.argsort(keys)
    ks, vs = keys[order], vals[order]
    pos = torch.searchsorted(ks, want)
    pos_c = pos.clamp(max=ks.numel() - 1)
    hit = (pos < ks.numel()) & (ks[pos_c] == want)
    S_ij = torch.where(hit, vs[pos_c], torch.zeros_like(vs[pos_c])).double()
    print(f"[facets] S_ij present for {int(hit.sum())}/{hit.numel()} "
          f"({float(hit.float().mean())*100:.2f}%) -- absent means the two cells share NO ray",
          flush=True)

    S_ii, S_jj = diag[lo], diag[hi]
    live = (S_ii + S_jj) > 0
    d = torch.zeros_like(S_ii)
    d[live] = torch.from_numpy(
        facet_identifiability(S_ii[live].cpu().numpy(), S_jj[live].cpu().numpy(),
                              S_ij[live].cpu().numpy())).to(device)

    dl = d[live].cpu().numpy()
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"\n[d_ij] over {live.sum()} facets with data "
          f"({int((~live).sum())} dead facets excluded)")
    print("  quantiles: " + "  ".join(f"q{int(q*100)}={np.quantile(dl,q):.4f}" for q in qs))
    print(f"  mean {dl.mean():.4f}  min {dl.min():.4e}  max {dl.max():.4f}")
    for t in (1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5):
        print(f"  frac d < {t:<7g}: {float((dl < t).mean())*100:6.2f}%")

    np.savez(a.output, d=d.cpu().numpy().astype(np.float32),
             lo=lo.cpu().numpy().astype(np.int32), hi=hi.cpu().numpy().astype(np.int32),
             live=live.cpu().numpy(), S_ii=S_ii.cpu().numpy().astype(np.float32),
             S_jj=S_jj.cpu().numpy().astype(np.float32),
             S_ij=S_ij.cpu().numpy().astype(np.float32))
    print(f"\n[d_ij] wrote {a.output}")


if __name__ == "__main__":
    main()
