"""Audit of the remaining load-bearing pieces of the frozen stack: mode_vote_refine and
coverage_filter.

Both sit upstream of every ScanNet++ number. `mode_vote_refine` rewrites features before the
classifier ever runs, and its CSR gather (`cand[:, 1:][mask] = adjacent[flat]`) is exactly the kind
of hand-rolled ragged indexing that fails silently -- a wrong `flat` would mix up whose neighbours
are whose and still produce plausible output. `coverage_filter` decides which GT points are scored
at all, so a defect there moves every mIoU in the table.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from run_normlift_refine_eval import mode_vote_refine
from run_spp_eval import coverage_filter


def _chain(P):
    """Path graph 0-1-2-...-(P-1) in CSR, as (adjacent, offsets)."""
    nbrs = [[j for j in (i - 1, i + 1) if 0 <= j < P] for i in range(P)]
    deg = [len(n) for n in nbrs]
    offsets = torch.tensor([0] + list(np.cumsum(deg)), dtype=torch.long)
    adjacent = torch.tensor([j for n in nbrs for j in n], dtype=torch.long)
    return adjacent, offsets


# ---------------------------------------------------------------------------- mode_vote_refine

def test_constant_field_is_a_fixed_point():
    """If every cell already agrees, no vote can improve anything and the field must be untouched.
    Catches an off-by-one in the CSR gather that would import a neighbour's row wholesale."""
    P, C = 8, 5
    u = torch.zeros(P, C); u[:, 0] = 1.0
    adj, off = _chain(P)
    pos = torch.arange(P).float()[:, None].repeat(1, 3)
    out = mode_vote_refine(u, torch.ones(P), pos, adj, off)
    assert torch.allclose(out, u, atol=1e-6)


def test_isolated_dissenter_is_flipped_towards_its_neighbourhood():
    """The operator's entire purpose: one cell disagreeing with a confident, coherent neighbourhood
    should adopt the neighbourhood's feature. If this fails the pass is a no-op in the real stack."""
    P, C = 9, 4
    u = torch.zeros(P, C); u[:, 0] = 1.0
    u[4] = 0.0; u[4, 1] = 1.0                      # the dissenter
    adj, off = _chain(P)
    pos = torch.arange(P).float()[:, None].repeat(1, 3) * 0.01   # tight, so distance kernel ~1
    out = mode_vote_refine(u, torch.ones(P), pos, adj, off)
    assert out[4].argmax().item() == 0, out[4]


def test_refinement_only_ever_copies_an_existing_feature():
    """`refined[rows] = U[take, best_j]` is a COPY, never a blend. This matters: NormLift's own
    ablation shows linear blending drifts off the CLIP manifold (33.6% semantic drift), and the
    whole justification for mode-voting over smoothing is that it stays on-manifold."""
    torch.manual_seed(0)
    P, C = 12, 6
    u = torch.nn.functional.normalize(torch.randn(P, C), dim=-1)
    adj, off = _chain(P)
    pos = torch.randn(P, 3) * 0.01
    out = mode_vote_refine(u, torch.rand(P), pos, adj, off)
    for i in range(P):
        d = (u - out[i][None, :]).norm(dim=-1)
        assert float(d.min()) < 1e-5, f"row {i} is not equal to any input row -- blending occurred"


def test_zero_reliability_neighbours_cannot_outvote_self():
    """Voter weight is R_j * distance. With every neighbour at R=0 the neighbourhood score is 0 and
    nothing should exceed self + delta."""
    P, C = 7, 3
    u = torch.zeros(P, C); u[:, 0] = 1.0
    u[3] = 0.0; u[3, 1] = 1.0
    R = torch.zeros(P); R[3] = 1.0
    adj, off = _chain(P)
    pos = torch.arange(P).float()[:, None].repeat(1, 3) * 0.01
    out = mode_vote_refine(u, R, pos, adj, off)
    assert out[3].argmax().item() == 1, "a zero-reliability neighbourhood overrode the cell"


def test_chunking_does_not_change_the_result():
    """The CSR gather is rebuilt per chunk from `offsets[s:e]`; if the within-chunk offset algebra
    were wrong, results would depend on chunk size. This is the sharpest test of that indexing."""
    torch.manual_seed(1)
    P, C = 40, 5
    u = torch.nn.functional.normalize(torch.randn(P, C), dim=-1)
    adj, off = _chain(P)
    pos = torch.randn(P, 3) * 0.05
    R = torch.rand(P)
    a = mode_vote_refine(u, R, pos, adj, off, chunk=P)          # single chunk
    b = mode_vote_refine(u, R, pos, adj, off, chunk=7)          # ragged chunks
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_degree_varies_across_nodes_without_leaking_neighbours():
    """A star graph: the hub sees every leaf, each leaf sees only the hub. If the padded candidate
    table leaked, a leaf would be able to vote using another leaf's feature."""
    P, C = 6, 3
    adjacent, offsets = [], [0]
    nbrs = {0: [1, 2, 3, 4, 5], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0]}
    for i in range(P):
        adjacent += nbrs[i]; offsets.append(len(adjacent))
    adj = torch.tensor(adjacent); off = torch.tensor(offsets)
    u = torch.zeros(P, C); u[:, 0] = 1.0
    u[0] = 0.0; u[0, 1] = 1.0                       # hub disagrees with all leaves
    u[5] = 0.0; u[5, 2] = 1.0                       # one leaf disagrees with everything
    pos = torch.zeros(P, 3)
    out = mode_vote_refine(u, torch.ones(P), pos, adj, off)
    # leaf 5's ONLY neighbour is the hub, so it may only ever adopt the hub's feature (class 1),
    # never class 0 which it could reach solely by leaking another leaf.
    assert out[5].argmax().item() in (1, 2), out[5]


# ---------------------------------------------------------------------------- coverage_filter

def test_far_points_are_dropped_and_near_points_kept():
    centers = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    valid = np.ones(4, bool)
    pts = np.array([[0., 0, 0.1], [1, 0, 0.05], [0, 0, 50.0]])
    assigned = np.array([0, 1, 0])
    keep, spacing, med = coverage_filter(pts, assigned, centers, valid, k_spacing=2.0)
    assert abs(spacing - 1.0) < 1e-6, spacing        # unit-spaced centres
    assert keep.tolist() == [True, True, False]


def test_k_spacing_scales_the_threshold():
    centers = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    valid = np.ones(3, bool)
    pts = np.array([[0., 0, 1.5]])
    assigned = np.array([0])
    assert coverage_filter(pts, assigned, centers, valid, 1.0)[0].tolist() == [False]
    assert coverage_filter(pts, assigned, centers, valid, 2.0)[0].tolist() == [True]


def test_spacing_uses_only_VALID_centres():
    """Invalid primitives must not set the scene scale. Two valid centres 10 apart plus a cloud of
    invalid ones packed at 0.01 -- if the mask were ignored, spacing would collapse to 0.01 and the
    filter would discard essentially everything."""
    tight = np.stack([np.linspace(0, 0.1, 12), np.zeros(12), np.zeros(12)], 1)
    centers = np.concatenate([np.array([[0., 0, 0], [10, 0, 0]]), tight])
    valid = np.zeros(len(centers), bool); valid[:2] = True
    pts = np.array([[0., 0, 3.0]])
    keep, spacing, _ = coverage_filter(pts, np.array([0]), centers, valid, k_spacing=1.0)
    assert abs(spacing - 10.0) < 1e-6, spacing
    assert keep.tolist() == [True]


def test_unassigned_points_do_not_silently_index_the_last_centre():
    """AUDIT TARGET. `centers[assigned]` with assigned == -1 wraps to the LAST centre in numpy, so
    an unowned point is measured against an arbitrary primitive. Whether that point is then kept is
    luck of the draw. Documented here as the real behaviour so the scoring path can be reasoned
    about: an unowned point kept by coverage is scored pred=0 vs gt=c, i.e. penalised."""
    centers = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [99, 0, 0]])
    valid = np.ones(4, bool)
    pts = np.array([[99., 0, 0]])                 # sits exactly on the LAST centre
    keep_unassigned = coverage_filter(pts, np.array([-1]), centers, valid, 2.0)[0]
    # distance measured to centers[-1] == the point itself -> 0 -> kept, despite owning nothing
    assert keep_unassigned.tolist() == [True], \
        "behaviour changed: -1 no longer wraps -- check callers that pass unowned points"
