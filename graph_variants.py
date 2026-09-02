"""Every grouping / neighbourhood construction, as one interchangeable axis of the ablation.

WHY ALL OF THEM. "Neighbourhood" is an input to BOTH feature consensus and diffusion, so the graph is
a lattice axis, not an implementation detail. Foam gets its graph exactly (the shared-facet dual of
the power diagram); 3DGS must construct one, and which construction is fair is precisely the open
question. Several candidates exist in the literature and in the baseline itself, so we test all of
them rather than defending one.

THE ISOTROPIC ROUTE IS NOT PRIVILEGED -- measured, not assumed. The hard-assignment limit of a
Gaussian mixture is argmin_i[||x-mu_i||^2 - 2 sigma^2 log w_i], EXACTLY a power diagram with
r_i^2 = 2 sigma^2 log w_i, whose dual is the regular (weighted Delaunay) triangulation; verified at
100.0000% agreement for equal-isotropic Gaussians. But OUR Gaussians are overwhelmingly anisotropic:

    axis ratio (max/min): median 14.8 | only 4.5% below 2 | 59.1% above 10 | p99 = 63,728

so ~95% of primitives violate the assumption that identity requires, and the best power-diagram fit
to an anisotropic hard assignment agrees on only 66.1% of points. `regular` is therefore ONE
approximation among several, and the anisotropy-aware `knn_maha` is at least as principled here.
Neither is the default.

Every builder returns `(src, dst, weight_or_None)` over VALID primitives only, with no self-loops,
symmetrised -- `diffuse` row-normalises, so an asymmetric edge set would behave differently
depending on scan direction.
"""
import numpy as np
import torch


def _sym_dedup(src, dst, P):
    """Symmetrise, drop self-loops, and remove duplicate edges."""
    keep = src != dst
    src, dst = src[keep], dst[keep]
    s2 = torch.cat([src, dst])
    d2 = torch.cat([dst, src])
    key = s2.to(torch.int64) * int(P) + d2.to(torch.int64)
    _, first_idx = np.unique(key.detach().cpu().numpy(), return_index=True)
    sel = torch.from_numpy(first_idx).to(src.device)
    return s2[sel], d2[sel]


def _knn_generic(X, idx, K, chunk):
    """kNN inside an arbitrary embedding, chunked over rows."""
    s, d = [], []
    n = X.shape[0]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dist = torch.cdist(X[a:b], X)
        dist[torch.arange(b - a, device=X.device), torch.arange(a, b, device=X.device)] = float("inf")
        nb = dist.topk(min(K, n - 1), largest=False).indices
        s.append(idx[a:b].repeat_interleave(nb.shape[1]))
        d.append(idx[nb.reshape(-1)])
        del dist, nb
    return torch.cat(s), torch.cat(d)


def knn_pos(pos, vm, K=30, chunk=2048, **kw):
    idx = torch.nonzero(vm).squeeze(1)
    s, d = _knn_generic(pos[idx], idx, K, chunk)
    return _sym_dedup(s, d, pos.shape[0]) + (None,)


def knn_feat(feat, vm, K=30, chunk=2048, **kw):
    idx = torch.nonzero(vm).squeeze(1)
    s, d = _knn_generic(feat[idx], idx, K, chunk)
    return _sym_dedup(s, d, feat.shape[0]) + (None,)


def knn_maha(pos, vm, scales, quats, K=30, chunk=512, **kw):
    """kNN under each Gaussian's OWN metric: d_ij = (mu_j-mu_i)^T Sigma_i^-1 (mu_j-mu_i).

    Directional by construction (d_ij != d_ji), which is the point -- an elongated Gaussian reaches
    far along its long axis and hardly at all across it, and at a median axis ratio of 14.8 that
    difference dominates. Symmetrising afterwards makes the neighbourhood the union of both reaches.
    """
    idx = torch.nonzero(vm).squeeze(1)
    X, S, Q = pos[idx], scales[idx].clamp_min(1e-9), quats[idx]
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    s, d = [], []
    n = X.shape[0]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        diff = X[None, :, :] - X[a:b, None, :]                       # (b, n, 3)
        loc = torch.einsum('bij,bnj->bni', R[a:b].transpose(1, 2), diff) / S[a:b, None, :]
        dist = (loc ** 2).sum(-1)
        dist[torch.arange(b - a, device=X.device), torch.arange(a, b, device=X.device)] = float("inf")
        nb = dist.topk(min(K, n - 1), largest=False).indices
        s.append(idx[a:b].repeat_interleave(nb.shape[1]))
        d.append(idx[nb.reshape(-1)])
        del diff, loc, dist, nb
    return _sym_dedup(torch.cat(s), torch.cat(d), pos.shape[0]) + (None,)


def radius_graph(pos, vm, K=30, decay=1.0, chunk=2048, **kw):
    """Same neighbourhood as knn_pos but with distance-decayed weights: near neighbours count more.
    Isolates 'position-aware weighting' from 'which primitives are neighbours at all'."""
    src, dst, _ = knn_pos(pos, vm, K=K, chunk=chunk)
    dd = (pos[src] - pos[dst]).norm(dim=-1)
    return src, dst, torch.exp(-dd / (decay * dd.median().clamp_min(1e-9)))


def _radfoam_delaunay_edges(pos_valid, device):
    """Delaunay edges from radfoam's CUDA backend -- exact, on GPU, at full scale.

    WHY THIS EXISTS. The scipy path below TIMED OUT at 900s on our SMALLEST scene (881k
    primitives) even with its 150k subsample, so it is not merely slow -- it cannot produce the
    arm at all. It also made `delaunay`/`regular` approximate while every other builder is exact,
    which is a fairness problem in any table that compares them. This backend removes both issues
    and keeps all groupings on the GPU, so reported build times are measured on equal terms.

    THE PERMUTATION IS NOT OPTIONAL. radfoam's backend spatially RE-SORTS the points (radfoam's
    own scene.py keeps `orig_indices = perm` for exactly this reason), so both the CSR row order
    and the neighbour values are in triangulation-internal positions. Mapping them back through
    `permutation()` is required; skipping it silently yields a graph over scrambled identities
    that still looks structurally valid (right degree, symmetric, no self-loops) and would
    quietly corrupt every downstream result. Verified against scipy in test_gpu_delaunay.py.
    """
    import sys
    if "D:/Downloads/foamvol" not in sys.path:
        sys.path.insert(0, "D:/Downloads/foamvol")
    import radfoam

    pts = pos_valid.to(device=device, dtype=torch.float32).contiguous()
    tri = radfoam.Triangulation(pts)
    perm = tri.permutation().to(torch.long)              # triangulation slot -> row of `pts`
    adj = tri.point_adjacency().to(torch.long)           # neighbour slots, CSR values
    off = tri.point_adjacency_offsets().to(torch.long)   # CSR offsets over slots
    counts = off[1:] - off[:-1]
    slot_src = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts)
    return perm[slot_src], perm[adj]


def _delaunay_edges(points, heights, max_n=150_000, seed=0):
    """Edges of a (weighted) Delaunay triangulation.

    A regular triangulation of {(mu_i, r_i^2)} is the lower convex hull of the lifted points
    (mu_i, ||mu_i||^2 - r_i^2); with r_i = 0 this is the ordinary Delaunay triangulation. scipy on
    >1e6 points in 3D is not tractable here, so above `max_n` we SUBSAMPLE and RETURN THAT FACT --
    an approximate graph honestly labelled, never a silent truncation.
    """
    from scipy.spatial import Delaunay
    n = points.shape[0]
    sub = None
    if n > max_n:
        sub = np.random.default_rng(seed).choice(n, max_n, replace=False)
        points, heights = points[sub], heights[sub]
    lifted = np.concatenate([points, (np.square(points).sum(1) - heights)[:, None]], 1)
    tri = Delaunay(lifted)
    simp = tri.simplices
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    pairs = [p for p in pairs if max(p) < simp.shape[1]]
    e = np.concatenate([simp[:, list(p)] for p in pairs], 0)
    if sub is not None:
        e = sub[e]
    return e, (sub is not None)


def delaunay_graph(pos, vm, device="cuda", backend="gpu", **kw):
    """Ordinary Delaunay. `backend="gpu"` uses radfoam's CUDA triangulation (exact, full scale);
    "cpu" is the scipy path kept only as a cross-check, since it subsamples above 150k and timed
    out at 900s on our smallest scene."""
    idx = torch.nonzero(vm).squeeze(1)
    if backend == "gpu":
        s, d = _radfoam_delaunay_edges(pos[idx], device)
        out = _sym_dedup(idx[s], idx[d], pos.shape[0])
        return out[0], out[1], None
    pts = pos[idx].detach().cpu().numpy().astype(np.float64)
    e, approx = _delaunay_edges(pts, np.zeros(len(pts)))
    loc = idx.cpu().numpy()
    src = torch.from_numpy(loc[e[:, 0]]).to(device)
    dst = torch.from_numpy(loc[e[:, 1]]).to(device)
    out = _sym_dedup(src, dst, pos.shape[0])
    return out[0], out[1], None


def regular_graph(pos, vm, scales, opacity, device="cuda", **kw):
    """Weighted Delaunay with r_i^2 = 2 sigma_i^2 log w_i -- the dual of the power diagram the
    mixture's hard-assignment limit produces. sigma_i is the geometric mean of the three scales,
    which is an APPROXIMATION our data mostly violates (see the module docstring)."""
    idx = torch.nonzero(vm).squeeze(1)
    pts = pos[idx].detach().cpu().numpy().astype(np.float64)
    sig = scales[idx].clamp_min(1e-9).log().mean(-1).exp().detach().cpu().numpy().astype(np.float64)
    w = opacity[idx].detach().cpu().numpy().astype(np.float64).clip(1e-6, 1 - 1e-6)
    r2 = 2.0 * (sig ** 2) * np.log(w)
    e, approx = _delaunay_edges(pts, r2)
    loc = idx.cpu().numpy()
    src = torch.from_numpy(loc[e[:, 0]]).to(device)
    dst = torch.from_numpy(loc[e[:, 1]]).to(device)
    out = _sym_dedup(src, dst, pos.shape[0])
    return out[0], out[1], None


def _star_edges(lab, idx, P, device):
    """Hard clustering -> star graph through each cluster's first member.

    NOT cliques: a cluster of size m has m(m-1) clique edges, which for 1.2M primitives is
    catastrophic. A star preserves within-cluster connectivity for message passing at O(m).
    """
    order = torch.argsort(lab)
    lab_s, idx_s = lab[order], idx[order]
    if lab_s.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=device)
        return z, z, None
    bnd = torch.nonzero(torch.diff(lab_s)).squeeze(1) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), bnd])
    ends = torch.cat([bnd, torch.tensor([lab_s.numel()], device=device)])
    s, d = [], []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a < 2:
            continue
        s.append(idx_s[a].repeat(b - a - 1))
        d.append(idx_s[a + 1:b])
    if not s:
        z = torch.empty(0, dtype=torch.long, device=device)
        return z, z, None
    out = _sym_dedup(torch.cat(s), torch.cat(d), P)
    return out[0], out[1], None


def kmeans_graph(pos, vm, n_clusters=2048, iters=12, seed=0, device="cuda", **kw):
    idx = torch.nonzero(vm).squeeze(1)
    X = pos[idx]
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:n_clusters]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        C.index_reduce_(0, lab, X, "mean", include_self=False)
    return _star_edges(torch.cdist(X, C).argmin(1), idx, pos.shape[0], device)


def codebook_graph(feat, vm, root=64, leaf=5, iters=10, seed=0, device="cuda", **kw):
    """OpenGaussian's two-level codebook (root x leaf = 320) over FEATURE space -- the BASELINE's own
    grouping, and therefore the fairest external comparator: if our pipeline only wins because it
    uses a different neighbourhood than OpenGaussian, this arm exposes it."""
    idx = torch.nonzero(vm).squeeze(1)
    X = torch.nn.functional.normalize(feat[idx], dim=-1)
    g = torch.Generator(device=X.device).manual_seed(seed)

    def km(Z, k, it):
        C = Z[torch.randperm(Z.shape[0], generator=g, device=Z.device)[:k]].clone()
        for _ in range(it):
            l = (Z @ C.T).argmax(1)
            C = torch.zeros_like(C).index_reduce_(0, l, Z, "mean", include_self=False)
            C = torch.nn.functional.normalize(C, dim=-1)
        return (Z @ C.T).argmax(1)

    root_lab = km(X, root, iters)
    lab = torch.zeros_like(root_lab)
    nxt = 0
    for r in range(root):
        m = root_lab == r
        cnt = int(m.sum())
        if cnt == 0:
            continue
        k = min(leaf, cnt)
        lab[m] = nxt + (km(X[m], k, iters) if k > 1 else
                        torch.zeros(cnt, dtype=torch.long, device=X.device))
        nxt += k
    return _star_edges(lab, idx, feat.shape[0], device)


# `regular` is DELIBERATELY ABSENT from the 3DGS grouping set (the function is kept: it remains
# meaningful on foam, where the power diagram IS the representation). Three reasons, all measured:
#   1. No GPU backend. radfoam's Triangulation takes points only; the regular/weighted triangulation
#      needs the 4D lifted hull with w_i = ||x_i||^2 - r_i^2, which exists only on the scipy path.
#   2. That path is not viable at scale -- TIMEOUT at 900s on our SMALLEST scene (881k), even with
#      its 150k subsample. Keeping it would make one arm approximate while the other seven are exact.
#   3. It is weakly motivated for Gaussians anyway: the power-diagram identity requires equal
#      ISOTROPIC covariance, and our Gaussians have median axis ratio 14.8 with only 4.5% below 2;
#      the best power-diagram fit to an anisotropic hard assignment agrees on just 66.1% of points.
BUILDERS = {
    "knn_pos": knn_pos, "knn_maha": knn_maha, "knn_feat": knn_feat, "radius": radius_graph,
    "delaunay": delaunay_graph,
    "kmeans": kmeans_graph, "codebook": codebook_graph,
}
