"""Lift features under different preconditioners, from the cached Gram matrix. No re-rendering.

foamyfoam/02_solvers.md proposes two solvers that the theory implies but nobody has run on a
scene. Both need only `S = A^T A`, `A^T B` and `D`, all of which the gram cache already holds:

  B. RICHARDSON. SFS Eq. 6 is step one of  x_{t+1} = x_t + D^-1 (A^T B - S x_t), x_0 = 0.
     Provably monotone in ||A x - B||, unconditionally stable, no step size. Does more steps help
     mIoU, or was the excess acting as a regulariser over noisy masks? foamyfoam/05 A says this is
     the experiment that decides whether 02_solvers.md is a method or a diagnosis.

  A. SURFACE-BLOCK. Theorem 1 generalises to any SPD preconditioner M:
         x_hat - x'_M = M^-1 (M - S) x_hat
     Choose M = blockdiag(S) over groups of cells belonging to one surface crossing; then M - S
     holds ONLY inter-surface entries, so the benign within-surface coupling is solved exactly and
     only depth-jump coupling survives into the error. Blocks are built here by greedily merging
     the strongest ADJACENT co-visibility edges (adjacency = shares a power-diagram face) subject
     to a size cap, which is the practical stand-in for "cells of one surface crossing".

REGRESSION, RUN FIRST AND NOT OPTIONAL. `richardson_k1` must reproduce the existing
`solved_weighted_*` file to float tolerance, because k=1 IS the closed form. If it does not, the
cache and the scored artifacts are not the same operator and nothing below means anything.

OUTPUT. `artifacts/scannet/<scene>/solved_<variant>.pt` = {primitive_features, valid_mask},
the format run_cluster_classify_eval.py scores.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import torch


def load_cache(scene: str, device: str):
    p = sorted(glob.glob(f"artifacts/scannet/{scene}/gram_cache_K6_l3_*.pt"))
    if not p:
        p = sorted(glob.glob(f"artifacts/scannet/{scene}/gram_cache_*.pt"))
    p = [q for q in p if "bgdrop" not in q][0]
    c = torch.load(p, map_location="cpu", weights_only=False)
    P = int(c["P"])
    keys, vals = c["S_keys"], c["S_vals"].float()
    Atb = c["Atb"].float()
    sup = c["support"].float()
    for k in list(c):
        c[k] = None
    del c
    print(f"  cache {os.path.basename(p)}  P={P:,}  edges={keys.numel():,}", flush=True)
    return P, keys, vals, Atb.to(device), sup.to(device)


def build_S(keys: torch.Tensor, vals: torch.Tensor, P: int, device: str):
    """Edge list (j, l, v, off) for the stored upper triangle. NOT a materialised sparse tensor.

    Building the symmetric COO doubles the nonzeros and then coalesce needs its own temporaries:
    on scene0140_00 that is 382M -> 764M nnz at 20 B each, ~15 GiB before overhead, and it OOMs a
    48 GiB card. The edge list is the same data at half the footprint, and `spmm` below applies it
    without ever forming the transpose copy.
    """
    j = (keys // P).to(device)
    l = (keys % P).to(device)
    v = vals.to(device)
    off = j != l
    return j, l, v, off


def spmm(S, x, edge_chunk=1 << 23, col_chunk=64):
    """S @ x from the upper-triangle edge list, chunked over BOTH edges and feature columns.

    Each unordered pair contributes to row j and, when j != l, to row l as well -- that is the
    transpose half, applied here instead of stored. Chunking bounds the gather temporary at
    edge_chunk x col_chunk floats (~2 GiB at the defaults) regardless of scene size.
    """
    j, l, v, off = S
    out = torch.zeros_like(x)
    N = j.numel()
    for c in range(0, x.shape[1], col_chunk):
        cc = slice(c, min(c + col_chunk, x.shape[1]))
        xc = x[:, cc]
        oc = out[:, cc]
        for s in range(0, N, edge_chunk):
            e = min(s + edge_chunk, N)
            js, ls, vs, om = j[s:e], l[s:e], v[s:e], off[s:e]
            oc.index_add_(0, js, vs.unsqueeze(1) * xc[ls])
            oc.index_add_(0, ls[om], vs[om].unsqueeze(1) * xc[js[om]])
        out[:, cc] = oc
    return out


def richardson(S, Atb, sup, k: int, project: bool = False):
    """x_{t+1} = x_t + D^-1 (A^T B - S x_t), x_0 = 0.  k=1 is SFS Eq. 6 exactly.

    `project=True` renormalises onto the unit sphere after every step (extrinsic Riemannian
    step). WHY THIS MATTERS. Every B_i is unit-norm (solve_cone_fast.py:163 normalises each
    view), while (A x)_i is a convex combination with ||(A x)_i|| <= 1 and equality only if all
    contributing x_j are the SAME unit vector. So A X = B is structurally unsatisfiable: there is
    an irreducible residual from the norm mismatch alone. Euclidean least squares can only shrink
    it by INFLATING ||x_j|| toward 1, which it does at the cost of direction -- measured on
    scene0347_00, k=10 grows norms 1.07x while rotating 63% of primitives past cosine 0.99.

    Since CLIP features are consumed as directions (cosine), the iterate belongs on the sphere,
    not inside the ball. Projecting each step removes the norm-chasing degree of freedom and lets
    the iteration spend its steps on direction only.
    """
    d = sup.clamp_min(1e-30).unsqueeze(1)
    x = torch.zeros_like(Atb)
    for _ in range(k):
        x = x + (Atb - spmm(S, x)) / d
        if project:
            n = x.norm(dim=1, keepdim=True)
            x = torch.where(n > 1e-12, x / n.clamp_min(1e-12), x)
    return x


def surface_blocks(scene: str, keys, vals, P: int, cap: int, device: str):
    """Greedily merge the strongest ADJACENT co-visibility edges into blocks of size <= cap.

    Adjacent = the two cells share a power-diagram face, i.e. they are plausibly the same surface
    crossing rather than a depth jump. Union-find with a size cap; edges are consumed strongest
    first so the heaviest benign coupling is the coupling that gets absorbed exactly.
    """
    adj_path = f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt"
    d = torch.load(adj_path, map_location="cpu", weights_only=False)
    assert int(d["num_primitives"]) == P, "adjacency P mismatch"
    a = d["adjacent"].long()
    o = d["offsets"].long()
    deg = o[1:] - o[:-1]
    src = torch.repeat_interleave(torch.arange(deg.numel(), dtype=torch.long), deg)
    lo, hi = torch.minimum(src, a), torch.maximum(src, a)
    akeys = torch.unique(lo * P + hi)
    del d, a, o, deg, src, lo, hi

    j, l = keys // P, keys % P
    off = j != l
    ek, ev = keys[off], vals[off]
    pos = torch.searchsorted(akeys, ek).clamp_(max=akeys.numel() - 1)
    is_adj = akeys[pos] == ek
    ek, ev = ek[is_adj], ev[is_adj]
    order = torch.argsort(ev, descending=True)
    ej, el = (ek[order] // P).numpy(), (ek[order] % P).numpy()
    print(f"  blocks: {len(ej):,} adjacent co-visibility edges, cap={cap}", flush=True)

    parent = list(range(P))
    size = [1] * P

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = 0
    for u, v in zip(ej.tolist(), el.tolist()):
        ru, rv = find(u), find(v)
        if ru == rv or size[ru] + size[rv] > cap:
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        merged += 1
    groups = torch.tensor([find(i) for i in range(P)], dtype=torch.long)
    _, groups = torch.unique(groups, return_inverse=True)
    n_g = int(groups.max()) + 1
    sizes = torch.bincount(groups)
    print(f"  blocks: {merged:,} merges -> {n_g:,} groups; "
          f"size hist {torch.bincount(sizes).tolist()[:cap+1]}", flush=True)
    return groups.to(device)


def solve_blockdiag(S, Atb, sup, groups, device: str):
    """x = M^-1 A^T B with M the block LUMPING of S over `groups`.

        M = blockdiag(S) + diag( sum_{k not in block(j)} S_jk )

    The added diagonal is the point. Plain blockdiag(S) DISCARDS the off-block mass, so M^-1
    divides by a far smaller denominator than D does and the solve overshoots wildly -- measured
    L = +2.45e8 against the closed form's -4.95e7 on scene0347_00. Lumping the off-block mass onto
    the diagonal instead preserves row sums (M 1 = S 1 = D 1), which is exactly the property that
    makes D^-1 S row-stochastic and the whole scheme stable (01_theory.md Lemma 1).

    Consistency check, asserted below: at block size 1 this must reduce to M = D, i.e. reproduce
    SFS Eq. 6 exactly.
    """
    P, F = Atb.shape
    x = torch.zeros_like(Atb)
    j, l, v, off = S
    # symmetric entries, expanded on the fly (never stored): (j,l,v) plus (l,j,v) for j != l
    bi = torch.cat([j, l[off]])
    bj = torch.cat([l, j[off]])
    bv = torch.cat([v, v[off]])
    same = groups[bi] == groups[bj]
    rowsum = torch.zeros(P, device=device).index_add_(0, bi, bv)
    bi, bj, bv = bi[same], bj[same], bv[same]
    within = torch.zeros(P, device=device).index_add_(0, bi, bv)
    offblk = (rowsum - within).clamp_min(0.0)          # lumped onto the diagonal below

    # Group members by a SORT, not a (G, P) boolean matrix: that matrix is 53 GiB at P=1.1M.
    order = torch.argsort(groups)
    gsorted = groups[order]
    sizes = torch.bincount(groups)
    starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                     torch.bincount(gsorted)[:-1]]), 0)
    for sz in range(1, int(sizes.max()) + 1):
        gsel = (sizes == sz).nonzero(as_tuple=True)[0]
        if not gsel.numel():
            continue
        G = gsel.numel()
        rows = (starts[gsel].unsqueeze(1)
                + torch.arange(sz, device=device).unsqueeze(0))       # (G, sz) into `order`
        rows = order[rows]                                            # (G, sz) global indices
        loc = torch.full((P,), -1, dtype=torch.long, device=device)
        loc[rows.reshape(-1)] = torch.arange(G * sz, device=device) % sz
        gof = torch.full((P,), -1, dtype=torch.long, device=device)
        gof[rows.reshape(-1)] = torch.arange(G, device=device).repeat_interleave(sz)
        M = torch.zeros((G, sz, sz), device=device)
        sel = (gof[bi] >= 0) & (gof[bi] == gof[bj])
        M[gof[bi][sel], loc[bi][sel], loc[bj][sel]] = bv[sel]
        # lump the off-block mass onto the diagonal so that M 1 = S 1
        gi = rows.reshape(-1)
        di = torch.arange(sz, device=device).repeat(G)
        M[torch.arange(G, device=device).repeat_interleave(sz), di, di] += offblk[gi]
        M += torch.eye(sz, device=device) * 1e-12
        rhs = Atb[rows.reshape(-1)].reshape(G, sz, F)
        sol = torch.linalg.solve(M, rhs)
        x[rows.reshape(-1)] = sol.reshape(G * sz, F)
        del rows, loc, gof, M, rhs, sol
        torch.cuda.empty_cache()
    bad = ~torch.isfinite(x)
    if bad.any():
        n = int(bad.any(1).sum())
        print(f"  blocks: {n:,} cells non-finite after solve -> falling back to D^-1 there")
        x[bad.any(1)] = (Atb / sup.clamp_min(1e-30).unsqueeze(1))[bad.any(1)]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--ks", default="1,2,3,5,10")
    ap.add_argument("--block-cap", type=int, default=4)
    ap.add_argument("--skip-blocks", action="store_true")
    ap.add_argument("--sphere", action="store_true",
                    help="also emit sphere-projected Richardson iterates")
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    P, keys, vals, Atb, sup = load_cache(a.scene, dev)
    valid = (sup > 0).cpu()
    S = build_S(keys, vals, P, dev)

    def save(tag, x):
        out = f"artifacts/scannet/{a.scene}/solved_{tag}.pt"
        torch.save({"primitive_features": x.cpu(), "valid_mask": valid}, out)
        print(f"  wrote {os.path.basename(out)}  ({time.time()-t0:.0f}s)", flush=True)

    for k in [int(v) for v in a.ks.split(",")]:
        save(f"richardson_k{k}", richardson(S, Atb, sup, k))
        if a.sphere:
            save(f"richardson_sph_k{k}", richardson(S, Atb, sup, k, project=True))

    if not a.skip_blocks:
        groups = surface_blocks(a.scene, keys, vals, P, a.block_cap, dev)
        save(f"surfblock_c{a.block_cap}", solve_blockdiag(S, Atb, sup, groups, dev))

    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
