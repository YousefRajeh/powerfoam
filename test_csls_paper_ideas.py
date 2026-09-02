"""Verify the math of the three CSLS ideas BEFORE trusting any sweep that uses them.

Every case below is hand-computable. The point is not coverage, it is catching the failure mode that
has cost this project real GPU time twice: an arm that is algebraically degenerate but still emits a
plausible mIoU (`top5_pairwise` was a monotone transform of argmax and tested nothing).

Run: python -m pytest test_csls_paper_ideas.py -q
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import torch

from run_csls_paper_ideas_eval import r_class, r_cell, mutual_rank, norm01
from run_derived_stack_eval import rank_encode
from run_simplex_diffusion_eval import diffuse

DEV = "cpu"

# 4 cells x 3 classes. Columns chosen so every top-k is unambiguous.
CV = torch.tensor([
    [0.10, 0.90, 0.50],
    [0.20, 0.70, 0.60],
    [0.80, 0.10, 0.55],
    [0.40, 0.30, 0.05],
])


def test_r_class_matches_hand_computation():
    """r_S(t_c): mean of the top-K cosines DOWN each class column."""
    got = r_class(CV, 2)
    # class0 top2 = .80,.40 -> .60 | class1 = .90,.70 -> .80 | class2 = .60,.55 -> .575
    assert torch.allclose(got, torch.tensor([0.60, 0.80, 0.575]), atol=1e-6), got


def test_r_cell_matches_hand_computation():
    """r_T(f_i): mean of the top-K cosines ACROSS each cell row -- the discarded source-side term."""
    got = r_cell(CV, 2)
    # row0 top2 = .90,.50 -> .70 | row1 = .70,.60 -> .65 | row2 = .80,.55 -> .675 | row3 = .40,.30 -> .35
    assert torch.allclose(got, torch.tensor([0.70, 0.65, 0.675, 0.35]), atol=1e-6), got


def test_r_class_and_r_cell_are_different_axes():
    """Guards the classic transposition bug: these must not be computable from one another."""
    assert r_class(CV, 2).shape[0] == CV.shape[1]
    assert r_cell(CV, 2).shape[0] == CV.shape[0]


def test_mutual_rank_places_each_class_top_cell_at_one():
    """mutual_rank[i,c] = 1 - rank_of_i_in_class_c's_list / K, and 0 outside the list."""
    mr = mutual_rank(CV, 2)
    # class0 ordering by cosine: cell2(.80) then cell3(.40) -> values 1.0, 0.5; cells 0,1 -> 0
    assert mr[2, 0] == 1.0 and mr[3, 0] == 0.5
    assert mr[0, 0] == 0.0 and mr[1, 0] == 0.0
    # class1: cell0(.90) then cell1(.70)
    assert mr[0, 1] == 1.0 and mr[1, 1] == 0.5
    # class2: cell1(.60) then cell2(.55)
    assert mr[1, 2] == 1.0 and mr[2, 2] == 0.5
    # exactly K nonzeros per class, no more
    assert (mr > 0).sum(0).tolist() == [2, 2, 2]


def _realistic():
    """CLIP text-vs-cell cosines are tightly clustered: mean ~0.25, sd ~0.02, ~100 classes. The 4x3
    toy CANNOT exercise reordering -- with 3 classes the r_K spread flips nothing -- so the
    non-degeneracy checks need a matrix at the real scale. Hand-computed value checks stay on CV."""
    torch.manual_seed(0)
    return 0.25 + 0.02 * torch.randn(700, 100)


def test_mutual_rank_is_not_a_monotone_transform_of_cosine():
    """The lesson from top5_pairwise: an 'idea' that preserves each cell's class ORDER changes
    nothing downstream. Mutual-NN must actually reorder cells, or the arm is vacuous."""
    cv = _realistic()
    base = cv - 0.5 * r_class(cv, 30)[None, :]
    with_mutual = base + 1.0 * cv.std(dim=1).mean() * mutual_rank(cv, 30)
    moved = (base.argmax(1) != with_mutual.argmax(1)).float().mean()
    assert moved > 0.005, f"mutual-NN moved only {moved:.4%} of argmaxes -- effectively a no-op"


def test_mutual_rank_handles_k_larger_than_n():
    mr = mutual_rank(CV, 99)
    assert mr.shape == CV.shape and torch.isfinite(mr).all()
    assert (mr > 0).sum(0).tolist() == [4, 4, 4]      # all cells enter when K >= N


def test_norm01_is_monotone_and_bounded():
    x = torch.tensor([5.0, -1.0, 3.0, 100.0])
    n = norm01(x)
    assert n.min() == 0.0 and n.max() == 1.0
    assert torch.equal(x.argsort(), n.argsort())      # order preserved
    # rank-based, so an outlier must NOT compress the rest (min-max would give ~0.06 here)
    assert abs(float(n[x.argsort()[-2]]) - 2 / 3) < 1e-6


# --------------------------------------------------------------------------------------------
# The central claim of the analysis: the stack is invariant to per-CELL monotone transforms.
# If this is false, the explanation for seven negative results is wrong.
# --------------------------------------------------------------------------------------------

def test_rank_encode_is_invariant_to_per_cell_affine_transforms():
    torch.manual_seed(0)
    s = torch.randn(50, 7)
    a = torch.rand(50, 1) * 3 + 0.1          # positive per-cell scale
    b = torch.randn(50, 1) * 5               # per-cell shift
    assert torch.equal(rank_encode(s, 8.0, DEV), rank_encode(a * s + b, 8.0, DEV))


def test_rank_encode_is_NOT_invariant_to_per_class_shifts():
    """The complement: per-class terms (CSLS, lambda-centering) DO survive. Without this the
    invariance claim would prove too much -- it would say nothing can ever work."""
    torch.manual_seed(0)
    s = torch.randn(50, 7)
    w = torch.randn(1, 7)
    assert not torch.equal(rank_encode(s, 8.0, DEV), rank_encode(s - w, 8.0, DEV))


def test_nicdm_mutual_collapses_onto_divisive_after_rank_encode():
    """Direct check of the falsification test built into the sweep: dividing additionally by a
    per-cell sqrt(r_T) must be invisible."""
    rK = r_class(CV, 2).clamp_min(1e-6)
    rT = r_cell(CV, 2).clamp_min(1e-6)
    div = CV / rK[None, :] ** 0.5
    nicdm = CV / (rK[None, :] ** 0.5 * rT[:, None] ** 0.5)
    assert torch.equal(rank_encode(div, 8.0, DEV), rank_encode(nicdm, 8.0, DEV))


def test_divisive_scaling_actually_reorders():
    """cos/r_K must differ from cos - r_K/2, else arm C silently duplicates the baseline."""
    cv = _realistic()
    rK = r_class(cv, 30).clamp_min(1e-6)
    moved = ((cv - 0.5 * rK[None, :]).argmax(1) != (cv / rK[None, :]).argmax(1)).float().mean()
    assert moved > 0.01, f"divisive scaling moved only {moved:.4%} of argmaxes"


# --------------------------------------------------------------------------------------------
# diffuse() semantics, since arm B relies entirely on anchor/edge_w behaving as documented.
# --------------------------------------------------------------------------------------------

def _two_node_graph():
    src = torch.tensor([0, 1]); dst = torch.tensor([1, 0])
    deg = torch.tensor([1, 1])
    return src, dst, deg


def test_anchor_one_means_node_keeps_its_own_evidence():
    src, dst, deg = _two_node_graph()
    p0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = diffuse(p0, src, dst, deg, alpha=0.9, iters=5, anchor=torch.tensor([1.0, 1.0]))
    assert torch.allclose(out, p0, atol=1e-6), out


def test_anchor_zero_reproduces_unanchored_diffusion():
    src, dst, deg = _two_node_graph()
    p0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    a = diffuse(p0, src, dst, deg, alpha=0.5, iters=3, anchor=torch.zeros(2))
    b = diffuse(p0, src, dst, deg, alpha=0.5, iters=3)
    assert torch.allclose(a, b, atol=1e-6)


def test_one_diffusion_step_matches_hand_computation():
    src, dst, deg = _two_node_graph()
    p0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = diffuse(p0, src, dst, deg, alpha=0.5, iters=1)
    # each node has one neighbour, row-stochastic weight 1: p = .5*p0 + .5*p0[neighbour]
    assert torch.allclose(out, torch.tensor([[0.5, 0.5], [0.5, 0.5]]), atol=1e-6), out


def test_edge_weights_are_row_normalised_so_a_global_scale_is_a_no_op():
    """edge_w is normalised per row inside diffuse, so only RELATIVE neighbour weights matter.
    Confirms B_edgew varies neighbour influence rather than smuggling in a global strength knob."""
    src = torch.tensor([0, 0, 1, 2]); dst = torch.tensor([1, 2, 0, 0])
    deg = torch.tensor([2, 1, 1])
    p0 = torch.eye(3)
    w = torch.tensor([1.0, 3.0, 1.0, 1.0])
    a = diffuse(p0, src, dst, deg, alpha=0.7, iters=2, edge_w=w)
    b = diffuse(p0, src, dst, deg, alpha=0.7, iters=2, edge_w=w * 17.0)
    assert torch.allclose(a, b, atol=1e-6)
    c = diffuse(p0, src, dst, deg, alpha=0.7, iters=2, edge_w=torch.ones(4))
    assert not torch.allclose(a, c, atol=1e-6), "unequal edge weights had no effect"


def test_diffuse_preserves_unit_sum_rows():
    """rank_encode emits a simplex point per cell; diffusion is a convex combination, so it must
    stay on the simplex or argmax comparisons across arms are not commensurate."""
    src = torch.tensor([0, 0, 1, 2]); dst = torch.tensor([1, 2, 0, 0])
    deg = torch.tensor([2, 1, 1])
    p0 = rank_encode(torch.randn(3, 5), 8.0, DEV)
    out = diffuse(p0, src, dst, deg, alpha=0.6, iters=4)
    assert torch.allclose(out.sum(1), torch.ones(3), atol=1e-5)
