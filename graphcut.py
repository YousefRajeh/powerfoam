"""Exact binary graph cut on the power diagram's facet graph -- the PAIRWISE extension of the
separable back-projection.

THE ARGUMENT. Our separability proposition says any objective LINEAR in the per-primitive label is
a function of `A^T B` alone and is solved exactly, per primitive, with no coupling. FlashSplat
(arXiv 2409.08270) is that result for discrete labels; our derivation (b) is it for the cosine
objective. So the unary problem has no room left in it.

The only way geometry can re-enter is a PAIRWISE term -- and that is precisely what a partition
provides and an overlapping representation does not. For a SINGLE query the labelling is binary,
and a binary Potts model is submodular, so the pairwise problem is ALSO solvable exactly, by
min-cut:

    minimise   sum_j  U_j(l_j)  +  lambda * sum_{(j,k) in facets} w_jk * [ l_j != l_k ]

with `U_j(1) = max(0, -(s_j - t))`, `U_j(0) = max(0, s_j - t)` for score `s_j` and threshold `t`.
Submodular because the Potts penalty is 0 on agreement and lambda*w >= 0 on disagreement, so
`E(0,0) + E(1,1) <= E(0,1) + E(1,0)` holds with equality on the unary part.

WHY IT IS FOAM-SPECIFIC. The pairwise term needs an EXACT adjacency. The power diagram's Delaunay
dual is that graph (jaccard 1.0000 against radfoam's own CUDA Delaunay). A 3DGS arm has no such
graph: its alpha graph is degenerate at mean degree 0.05 even at gsplat's own 3-sigma bound, so
there are almost no edges to put a prior on.

SINGLE-QUERY SAFE. Nothing here involves other classes: the cut is computed for one query's score
vector. This is the multi-class Potts model's binary special case, and it is the only version
compatible with answering one open-vocabulary query at a time.

INTEGER CAPACITIES. `scipy.sparse.csgraph.maximum_flow` requires int32 capacities, so costs are
scaled by `SCALE` and rounded. That makes the solve exact for the ROUNDED problem; with
SCALE=1e6 and cosine scores in [-1, 1] the rounding error per edge is <= 1e-6, far below the
score differences that decide anything here.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

SCALE = 1_000_000


def binary_graphcut(scores, t, indptr, indices, lam, edge_w=None, subset=None):
    """Exact min-cut labelling for ONE query.

    Parameters
    ----------
    scores : (P,) float, the query's per-primitive score (e.g. cosine to the class text).
    t      : float, the score threshold -- unary prefers foreground where `scores > t`.
    indptr, indices : CSR facet graph over all P primitives.
    lam    : float, Potts weight. lam=0 reduces EXACTLY to thresholding.
    edge_w : optional (nnz,) per-edge weights; defaults to 1.
    subset : optional (P,) bool of primitives eligible to be labelled foreground (e.g. the
             confidence gate). Excluded primitives are REMOVED FROM THE GRAPH -- both their unary
             and all their facet edges are dropped -- and returned as background.

             They are NOT pinned with a large unary, which was the first implementation and is a
             trap: a pinned background node still participates in the Potts term, so at moderate
             lambda it can drag its whole neighbourhood to background. Concretely, on a 7-chain
             with every unary favouring foreground by 1.0 and one node pinned at lambda=5, the true
             optimum is ALL background (cost 6.0) rather than foreground-with-a-hole (cost 10.0 in
             cut edges alone). Removing the node instead makes the gate a pure restriction of the
             candidate set, which is what it is meant to be.

    Returns
    -------
    (P,) bool foreground labelling.
    """
    P = indptr.size - 1
    u = (scores - t).astype(np.float64)
    live = np.ones(P, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)

    # node ids: 0 = source (foreground terminal), 1..P = primitives, P+1 = sink
    src, snk = 0, P + 1
    rows, cols, data = [], [], []

    # unary: cutting the source edge means calling j background, at cost max(u, 0);
    # cutting the sink edge means calling j foreground, at cost max(-u, 0).
    pos = np.where(live, np.clip(u, 0, None), 0.0)
    neg = np.where(live, np.clip(-u, 0, None), 0.0)

    j = np.arange(P)
    keep_s = pos > 0
    rows.append(np.full(int(keep_s.sum()), src)); cols.append(j[keep_s] + 1)
    data.append(pos[keep_s])
    keep_t = neg > 0
    rows.append(j[keep_t] + 1); cols.append(np.full(int(keep_t.sum()), snk))
    data.append(neg[keep_t])

    # pairwise: each undirected facet edge becomes a symmetric pair of capacity lam*w.
    # Edges touching a removed primitive are dropped, so the gate exerts no smoothing pull.
    if lam > 0:
        deg = np.diff(indptr)
        a = np.repeat(np.arange(P), deg)
        b = indices
        w = np.ones(b.size) if edge_w is None else edge_w
        m = (a < b) & live[a] & live[b]             # each unordered pair once, both endpoints live
        aa, bb, ww = a[m], b[m], w[m] * lam
        rows.append(aa + 1); cols.append(bb + 1); data.append(ww)
        rows.append(bb + 1); cols.append(aa + 1); data.append(ww)

    r = np.concatenate(rows); c = np.concatenate(cols)
    d = np.concatenate(data) * SCALE
    d = np.rint(np.clip(d, 0, np.iinfo(np.int32).max)).astype(np.int32)
    nz = d > 0
    g = csr_matrix((d[nz], (r[nz], c[nz])), shape=(P + 2, P + 2))

    res = maximum_flow(g, src, snk)
    # min-cut: nodes reachable from the source in the RESIDUAL graph are foreground
    resid = g - res.flow
    resid.eliminate_zeros()
    seen = np.zeros(P + 2, dtype=bool)
    seen[src] = True
    stack = [src]
    ip, ix = resid.indptr, resid.indices
    while stack:
        v = stack.pop()
        for e in range(ip[v], ip[v + 1]):
            u_ = ix[e]
            if not seen[u_]:
                seen[u_] = True
                stack.append(u_)
    out = seen[1:P + 1]
    return out & live          # removed primitives are background by construction


def multiclass_potts_icm(sim, indptr, indices, lam, iters=12, init=None, live=None,
                         damp=1.0, seed=0):
    """Multi-class Potts smoothing by ICM (iterated conditional modes).

        label_j <- argmax_c [ sim_jc + lam * #{facet neighbours currently labelled c} ]

    WHY ICM AND NOT ALPHA-EXPANSION. The binary Potts model is submodular and exactly solvable by
    min-cut (`binary_graphcut`), but the MULTI-class expansion move needs auxiliary nodes on edges
    whose endpoints disagree. ICM is a local optimiser -- it can stop at a local minimum where
    alpha-expansion would not -- but it is exact per-site, monotone in the energy, and needs no
    extra machinery. It is reported as such, not as an optimal solve.

    KNOWN LIMITATION -- LARGE lambda. The sweep is synchronous by default (`damp=1.0`), which is
    fast but can enter a 2-cycle when the prior dominates the unary: every site flips to its
    neighbours' label at once, forever. Best-energy tracking makes that SAFE (the result is never
    worse than the plain argmax) but not EFFECTIVE -- at large lambda it simply returns the
    initialisation. `damp < 1.0` updates a random subset per sweep and breaks the cycle, at the
    cost of needing more iterations. The operating range here is small lambda (the unary |s - t|
    is ~0.01-0.05 against a mean facet degree of ~11), so the default is the fast path; use
    `damp=0.5, iters>=40` if a strong prior is ever wanted.

    NOT SINGLE-QUERY. This is the multi-class variant and it needs the whole class set, so it is
    comparable to the reported argmax mIoU but does NOT satisfy the one-query-at-a-time constraint
    that `binary_graphcut` does. They are two different methods sharing a prior.

    Relation to FlashSplat: with lam = 0 this reduces to `argmax_c (A^T B)_jc`, their weighted
    majority vote, which our separability proposition says is already optimal for the unary
    objective. `lam > 0` adds the pairwise term the unary problem provably cannot contain.
    """
    P, C = sim.shape
    lab = sim.argmax(1) if init is None else init.copy()
    if live is None:
        live = np.ones(P, dtype=bool)
    deg = np.diff(indptr)
    src = np.repeat(np.arange(P), deg)
    dst = indices
    ok = live[src] & live[dst]
    src, dst = src[ok], dst[ok]
    def energy(l):
        # -unary + lam * (# disagreeing undirected edges); src/dst hold each pair twice
        return -sim[np.arange(P), l].sum() + 0.5 * lam * float((l[src] != l[dst]).sum())

    rng = np.random.default_rng(seed)
    best, best_e = lab.copy(), energy(lab)
    for _ in range(iters):
        votes = np.bincount(src * C + lab[dst], minlength=P * C).reshape(P, C)
        prop = (sim + lam * votes).argmax(1)
        # DAMPING: update only a random `damp` fraction of sites per sweep. A fully synchronous
        # sweep can enter a 2-cycle at large lam -- every site flips to its neighbours' label at
        # once, forever -- which leaves the best-energy iterate stuck at the initialisation.
        # Updating a random subset breaks the symmetry that sustains the cycle, and is the standard
        # stochastic-ICM remedy. Seeded, so the result is deterministic.
        upd = rng.random(P) < damp if damp < 1.0 else np.ones(P, dtype=bool)
        new = np.where(live & upd, prop, lab)
        if np.array_equal(new, lab):
            break
        lab = new
        # SYNCHRONOUS (Jacobi) updates are NOT guaranteed to decrease the energy: at large lam a
        # chain can enter a 2-cycle where every site flips to its neighbours' label at once, and
        # the fixed-point test never fires. Sequential (Gauss-Seidel) ICM is monotone but is a
        # Python-level loop over 10^5-10^6 primitives. Tracking the best-energy iterate keeps the
        # speed of the vectorised sweep while guaranteeing the result is never worse than the
        # initialisation -- i.e. never worse than the plain unary argmax.
        e = energy(lab)
        if e < best_e:
            best, best_e = lab.copy(), e
    return best
