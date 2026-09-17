"""The co-visibility graph G = A^T A as a neighbourhood graph, sparsified to top-K per node.

WHY THIS GRAPH. Every other neighbourhood in this project is CHOSEN from geometry -- position
k-NN, feature k-NN, Mahalanobis, radius, Delaunay, k-means star, codebook. `paper-A`'s
`tab:adjacency` shows the choice is worth 5.65 mIoU, and that the best Gaussian choice is the one
with no semantic content at all. G is not chosen: it is the coupling the renderer already defines,
G_jl = sum_i A_ij A_il = the ray weight primitives j and l share. `foamyfoam/01_theory.md` Cor. 2
says it is exactly the graph the lifting error lives on.

FOAM-ENABLED, not foam-exclusive. G exists for any representation, but it is sparse only because a
foam ray crosses ~12 cells; with 3DGS's ~33 overlapping primitives per ray the same object is far
denser (see `gram_blocks.py`). And its off-diagonal is geometrically LOCAL in a foam (adjacent
cells) versus NONLOCAL in 3DGS (projection overlap across depth), so it is well-founded as a
diffusion graph for one and not the other.

SPARSIFICATION. G has 16M-382M off-diagonal entries per scene, far denser than a K=30 k-NN graph
(~3M). Top-K by edge weight per node matches the K used in `tab:adjacency` and keeps the strongest
coupling, which is the coupling Cor. 2 weights the error by.
"""
from __future__ import annotations

import glob

import torch


def load_gram_edges(scene: str, device: str = "cpu"):
    """Off-diagonal (j, l, w) of G = A^T A for a scene, each unordered pair once."""
    p = [q for q in sorted(glob.glob(f"artifacts/scannet/{scene}/gram_cache_K6_l3_*.pt"))
         if "bgdrop" not in q]
    if not p:
        p = [q for q in sorted(glob.glob(f"artifacts/scannet/{scene}/gram_cache_*.pt"))
             if "bgdrop" not in q]
    c = torch.load(p[0], map_location="cpu", weights_only=False)
    P = int(c["P"])
    keys, vals = c["S_keys"], c["S_vals"].float()
    for k in list(c):
        c[k] = None
    del c
    j, l = keys // P, keys % P
    off = j != l
    return P, j[off].to(device), l[off].to(device), vals[off].to(device)


def top_k_edges(j: torch.Tensor, l: torch.Tensor, w: torch.Tensor, P: int, K: int,
                blocks: int = 32):
    """Symmetric edge list keeping each node's K strongest co-visibility partners.

    Processed in blocks of source nodes so peak memory is bounded by one block's edges, not by the
    whole mirrored list. scene0140_00 has 382M unordered pairs -> 764M directed; materialising and
    sorting that at once OOMs a 48 GiB card.

    TWO BUGS THIS REPLACES, both in one line (`argsort(src * (val.max()+1) - val)`):
      * MEMORY -- it built a 764M-element float key and sorted it globally.
      * CORRECTNESS -- that key is float32, whose mantissa is exact only to 2^24 = 16.7M. With
        P = 203,952 the products stayed exact by luck; at P = 1,118,823 they do not, so the
        ordering would have silently corrupted on the large scenes rather than failing.
    Ordering is now done per block with an int64 composite key, which is exact for any P * K that
    fits in 63 bits.
    """
    dev = j.device
    out_s, out_d, out_v = [], [], []
    step = max(1, (P + blocks - 1) // blocks)
    # `j` is sorted ascending (the gram cache stores sorted keys), so the forward direction of each
    # block is a contiguous slice found by searchsorted; the reverse direction needs a mask.
    for a in range(0, P, step):
        b = min(a + step, P)
        lo = int(torch.searchsorted(j, torch.tensor(a, device=dev)))
        hi = int(torch.searchsorted(j, torch.tensor(b, device=dev)))
        fs, fd, fv = j[lo:hi], l[lo:hi], w[lo:hi]
        m = (l >= a) & (l < b)
        rs, rd, rv = l[m], j[m], w[m]
        src = torch.cat([fs, rs])
        dst = torch.cat([fd, rd])
        val = torch.cat([fv, rv])
        if src.numel() == 0:
            continue
        # exact int64 ordering: source ascending, then weight descending
        rank_v = torch.argsort(torch.argsort(val, descending=True))
        order = torch.argsort(src * src.new_tensor(src.numel() + 1) + rank_v)
        src, dst, val = src[order], dst[order], val[order]
        starts = torch.ones_like(src, dtype=torch.bool)
        starts[1:] = src[1:] != src[:-1]
        run_id = torch.cumsum(starts.long(), 0) - 1
        run_start = torch.zeros(int(run_id[-1]) + 1, dtype=torch.long, device=dev)
        run_start.scatter_(0, run_id[starts], torch.arange(src.numel(), device=dev)[starts])
        keep = (torch.arange(src.numel(), device=dev) - run_start[run_id]) < K
        out_s.append(src[keep]); out_d.append(dst[keep]); out_v.append(val[keep])
        del src, dst, val, order, rank_v, run_id, run_start, starts, keep, m, fs, fd, fv, rs, rd, rv
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(out_s), torch.cat(out_d), torch.cat(out_v)


def covis_graph(scene: str, P: int, K: int = 30, device: str = "cuda"):
    """(src, dst, weight) top-K co-visibility graph for `scene`."""
    Pc, j, l, w = load_gram_edges(scene, device)
    assert Pc == P, f"gram cache P={Pc} != model P={P}"
    src, dst, val = top_k_edges(j, l, w, P, K)
    deg = torch.zeros(P, device=device).index_add_(
        0, src, torch.ones(src.numel(), device=device))
    print(f"  covis graph: {src.numel():,} directed edges, mean degree "
          f"{float(deg[deg > 0].mean()):.2f}, isolated {int((deg == 0).sum()):,}/{P:,}",
          flush=True)
    return src, dst, val


def ppr_diffuse(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, P: int,
                alpha: float = 0.5, iters: int = 20, weights: torch.Tensor | None = None,
                sphere: bool = False, edge_chunk: int = 1 << 22):
    """Personalised PageRank smoothing: x <- (1-a) x0 + a * rownorm(W) x, iterated.

    Same operator as `run_adjacency_eval.diffuse`. Its fixed point is a SMOOTHING of x0, not the
    least-squares solution -- unlike Richardson, whose fixed point is x_hat and which
    `foamyfoam/05_open_questions.md` A measured as harmful. Isolated nodes keep x0 exactly.

    `sphere=True` renormalises after every step, so the iterate stays on the unit sphere and each
    update is an extrinsic spherical (Karcher-style) average of the neighbours rather than a
    Euclidean convex combination that dives through the interior of the ball. CLIP features are
    trained and consumed on the sphere, so smoothing them chordally shrinks norms toward the mean
    direction and mixes a magnitude change into what should be a pure rotation.
    """
    if weights is None:
        weights = torch.ones(src.numel(), device=x.device)
    deg = torch.zeros(P, device=x.device).index_add_(0, src, weights)
    wn = weights / deg.clamp_min(1e-30)[src]
    a = torch.where(deg > 0, torch.full((P,), alpha, device=x.device),
                    torch.zeros(P, device=x.device))[:, None]
    p = x.clone()
    E = src.numel()
    for _ in range(iters):
        # The gather p[dst] is (E, F): at E = 33.3M and F = 512 that is 68 GiB, which OOMs a
        # 48 GiB card on scene0140_00. Chunking over edges bounds it at edge_chunk x F.
        acc = torch.zeros_like(p)
        for s0 in range(0, E, edge_chunk):
            s1 = min(s0 + edge_chunk, E)
            acc.index_add_(0, src[s0:s1], p[dst[s0:s1]] * wn[s0:s1, None])
        p = (1 - a) * x + a * acc
        del acc
        if sphere:
            n = p.norm(dim=1, keepdim=True)
            p = torch.where(n > 1e-12, p / n.clamp_min(1e-12), p)
    return p
