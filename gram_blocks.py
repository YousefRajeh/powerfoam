"""Solve in COEFFICIENT space: precompute the block-sparse Hessian once, then iterate cheaply.

THE PROBLEM WITH THE MATRIX-FREE SOLVE

The cone-constrained unknown is `a` of shape (P, K) with K ~ 7 -- seven non-negative coefficients
per cell over its own observed CLIP embeddings. But every FISTA iteration expands it to
f = U a in 512 dimensions, streams ~810M ray nonzeros through that, and immediately contracts
back to 7. Measured on an A6000: the 512-dim scatter/gather costs 308 ms for ONE view, so a full
A^T A application over 54 views x 2 passes is ~33 s and touches 3.02 TiB. The same operation in
7 dimensions costs 7.7 ms per view, 40x cheaper. The 512-dim vectors exist only to be contracted
away again.

THE REFORMULATION

Write S = A^T A (P x P, sparse: cells j and l couple only if some ray crosses BOTH). Then for
the reduced variable,

    (H a)_j = sum_l  S_{jl} * (U_j U_l^T) a_l  =  sum_l  B_{jl} a_l ,     B_{jl} = S_{jl} G_{jl}

with G_{jl} = U_j U_l^T a K x K matrix. Precompute B once per edge; afterwards every iteration is
a block-sparse matvec with K x K blocks and the 512-dim features never appear in the inner loop.

WHY THIS IS A POWER-DIAGRAM MOVE. S is sparse ONLY because the cells are disjoint and a ray
crosses ~12 of them, so coupling means adjacency or occluder/occludee. A 3DGS method has 50+
overlapping primitives per ray, so its S is far denser and this reformulation would not be
tractable. Whether it is tractable HERE is an empirical question about nnz(S), which is exactly
what --measure-only answers before any of the expensive machinery is built.

BUILDING S. Rows (pixels) are disjoint, so S = sum over rows of the outer product of that row's
nonzeros. For a row with n_r entries that is n_r^2 pairs, and sum_r n_r^2 is ~1.9e8 per view at
a median of 12 hits/pixel -- large but chunkable. Pairs are generated with the standard
"expand ranges" trick on the row-sorted triples, coalesced per chunk, then merged.

Only the LOWER-OR-EQUAL triangle is stored (j <= l) since S is symmetric; the matvec applies
both B_{jl} and its transpose, with the diagonal handled once.
"""
import torch

_PAIR_BUDGET = 20_000_000     # pair-instances materialized at once; ~0.5 GiB of int64 keys
_MERGE_LIMIT = 60_000_000     # coalesce once this many uncoalesced pairs have piled up


def _coalesce(keys, vals):
    """Sum duplicate keys. Returns sorted unique keys and summed values.

    `torch.unique` sorts internally and needs roughly 2-3x the input in workspace, so this is the
    memory high-water mark of the whole build. On scene0140_00 (372k cells, 215 views) letting
    24 chunks of 40M pile up before coalescing meant ~960M keys at 8 bytes plus their values plus
    the sort workspace, and it OOM'd at 38.9 GiB allocated. Hence the smaller chunk budget above
    and the size-triggered merge in `maybe_merge` rather than a fixed chunk count -- the right
    trigger is total pairs pending, which scales with the scene, not the number of chunks, which
    does not.
    """
    uk, inv = torch.unique(keys, return_inverse=True)
    out = torch.zeros(uk.numel(), device=vals.device, dtype=vals.dtype)
    out.index_add_(0, inv, vals)
    return uk, out


def maybe_merge(key_store, val_store, limit=_MERGE_LIMIT, force=False):
    """Coalesce when enough pairs have accumulated. Returns True if a merge happened."""
    total = sum(k.numel() for k in key_store)
    if not force and total < limit:
        return False
    torch.cuda.empty_cache()          # hand back the per-view scratch before the sort
    k, v = merge(key_store, val_store)
    key_store.clear(); val_store.clear()
    key_store.append(k); val_store.append(v)
    torch.cuda.empty_cache()
    return True


def accumulate_view_pairs(cols, vals, slots, P, key_store, val_store):
    """Fold one view's contribution to S into the running (key, value) lists.

    `cols`/`vals` are the view's compacted nonzeros in row-major (pixel) order and `slots` is the
    per-pixel count, so the triples are already grouped by row -- which is what makes the range
    expansion below valid without an explicit sort.
    """
    device = cols.device
    slots = slots.long()
    npix = slots.numel()
    row_start = torch.cumsum(slots, 0) - slots            # first nonzero index of each pixel
    n_at = torch.repeat_interleave(slots, slots)          # per-nonzero: how many entries its row has
    start_at = torch.repeat_interleave(row_start, slots)  # per-nonzero: where its row begins

    nnz = cols.numel()
    # process nonzeros in chunks whose expanded pair count stays under the budget
    pair_counts = n_at
    cum = torch.cumsum(pair_counts, 0)
    total_pairs = int(cum[-1]) if nnz else 0

    s = 0
    while s < nnz:
        # largest e such that pairs in [s, e) <= budget
        base = cum[s] - pair_counts[s]
        e = int(torch.searchsorted(cum, base + _PAIR_BUDGET).item())
        e = max(e, s + 1)
        e = min(e, nnz)
        n_chunk = pair_counts[s:e]
        tot = int(n_chunk.sum())
        # left index of each pair: the nonzero itself, repeated n_r times
        left = torch.repeat_interleave(torch.arange(s, e, device=device), n_chunk)
        # right index: walk the row's range [start, start+n)
        off = torch.arange(tot, device=device) - torch.repeat_interleave(
            torch.cumsum(n_chunk, 0) - n_chunk, n_chunk)
        right = start_at[left] + off
        # Emit each UNORDERED pair once. The expansion above walks the full row for every
        # nonzero, so an off-diagonal pair appears twice -- once as (j,l) and once as (l,j) --
        # and both collapse onto the same upper-triangle key, storing 2*S_{jl}. Since matvec
        # then applies the block AND its transpose, the off-diagonal would be counted four
        # times instead of twice while the diagonal stayed correct. Measured symptom before
        # this filter: relative error 4.5e-02 against the matrix-free operator.
        sel = right >= left
        left, right = left[sel], right[sel]
        cj, cl = cols[left], cols[right]
        lo = torch.minimum(cj, cl)
        hi = torch.maximum(cj, cl)
        keys = lo * P + hi
        v = vals[left] * vals[right]
        k2, v2 = _coalesce(keys, v)
        key_store.append(k2)
        val_store.append(v2)
        del left, right, off, cj, cl, lo, hi, keys, v, k2, v2
        s = e
    return total_pairs


def merge(key_store, val_store):
    k = torch.cat(key_store)
    v = torch.cat(val_store)
    return _coalesce(k, v)


class BlockSparseHessian:
    """H in coefficient space: K x K blocks on the upper triangle of S, applied symmetrically."""

    def __init__(self, j_idx, l_idx, blocks, P, K):
        self.j, self.l, self.B, self.P, self.K = j_idx, l_idx, blocks, P, K
        self.diag_mask = j_idx == l_idx

    def bytes(self):
        return self.B.numel() * 4 + self.j.numel() * 8 + self.l.numel() * 8

    def row_block_norms(self):
        """Per-cell L_j = sum_l ||B_{jl}||_2, a Gershgorin-style bound on that cell's share of
        the Lipschitz constant.

        A single global step 1/L uses the LARGEST eigenvalue over the whole problem, so every
        cell moves at the pace of the worst-conditioned one. Here S_jj = sum_r A[r,j]^2 varies by
        orders of magnitude between a cell seen head-on in many views and one glimpsed edge-on in
        two, so a global step is drastically too small for most cells. Note the true diagonal of H
        is S_jj * ||u_jk||^2 = S_jj for every k (the basis vectors are unit), so the natural
        preconditioner is a per-CELL scalar and the non-negativity projection stays a plain clamp
        -- no per-cell NNLS needed, unlike a full 7x7 block metric.
        """
        dev = self.B.device
        rs = torch.zeros(self.P, device=dev)
        # spectral norm <= Frobenius norm; use Frobenius as the cheap valid upper bound
        fro = self.B.flatten(1).norm(dim=1)
        rs.index_add_(0, self.j, fro)
        off = ~self.diag_mask
        rs.index_add_(0, self.l[off], fro[off])
        return rs

    def matvec(self, a, chunk=8_000_000):
        """(H a)_j = sum_l B_{jl} a_l, applying each stored upper-triangle block both ways."""
        out = torch.zeros_like(a)
        E = self.j.numel()
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            j, l, B = self.j[s:e], self.l[s:e], self.B[s:e]
            # j <- B @ a_l
            out.index_add_(0, j, torch.bmm(B, a[l].unsqueeze(-1)).squeeze(-1))
            # l <- B^T @ a_j, skipping the diagonal so it is not counted twice
            off = ~self.diag_mask[s:e]
            if bool(off.any()):
                out.index_add_(0, l[off],
                               torch.bmm(B[off].transpose(1, 2),
                                         a[j[off]].unsqueeze(-1)).squeeze(-1))
        return out


def prune_edges(keys, svals, P, max_edges, verbose=True):
    """Keep the largest-magnitude off-diagonal couplings, always keeping the diagonal.

    Edge count does not scale with cells alone -- it scales with cells TIMES view count, because
    every additional view discovers new occluder/occludee pairs. Measured: scene0347_00 (204k
    cells, 54 views) has 16.2M edges, but scene0140_00 (373k cells, 215 views) has 330M+, which
    at K=7 is 64.7 GB of blocks on a 48 GB card. So the coefficient-space reformulation needs a
    budget to be usable on the large scenes, not just the small ones.

    Pruning by |S_jl| is the right truncation because S_jl = sum_r A[r,j]A[r,l] is a sum of
    non-negative products -- there is no cancellation, so a small entry really does mean a weak
    coupling rather than a large one that happened to cancel. That is a property of THIS matrix
    (A >= 0) and would not hold for a general signed operator.

    Diagonal entries are exempt: S_jj is the self-term the whole diagonal estimator is built on,
    and dropping any of it would silently change the incumbent the solve starts from.
    """
    E = keys.numel()
    if E <= max_edges:
        return keys, svals, 1.0
    j = keys // P
    l = keys % P
    is_diag = j == l
    n_diag = int(is_diag.sum())
    budget = max(max_edges - n_diag, 0)
    off = ~is_diag
    off_vals = svals[off]
    if budget == 0 or off_vals.numel() <= budget:
        return keys, svals, 1.0
    thresh = torch.topk(off_vals, budget, largest=True, sorted=False).values.min()
    keep = is_diag | (svals >= thresh)
    kept_mass = float(svals[keep].sum() / svals.sum())
    if verbose:
        print(f"[prune] {E:,} -> {int(keep.sum()):,} edges (budget {max_edges:,}), "
              f"retaining {kept_mass*100:.3f}% of total coupling mass, "
              f"all {n_diag:,} diagonal entries kept", flush=True)
    return keys[keep], svals[keep], kept_mass


def build_blocks(keys, svals, U, P, K, edge_chunk=200_000):
    """B_e = S_e * (U_j U_l^T). Chunked over edges: gathering U for all edges at once would be
    (E, K, 512) which is hundreds of GB at realistic E."""
    j = (keys // P).to(torch.int64)
    l = (keys % P).to(torch.int64)
    E = j.numel()
    B = torch.empty(E, K, K, device=U.device, dtype=torch.float32)
    for s in range(0, E, edge_chunk):
        e = min(s + edge_chunk, E)
        Uj = U[j[s:e]].float()               # (chunk, K, D); U may be fp16 to fit
        Ul = U[l[s:e]].float()
        B[s:e] = torch.einsum("ekd,eld->ekl", Uj, Ul) * svals[s:e, None, None]
        del Uj, Ul
    return BlockSparseHessian(j, l, B, P, K)
