"""Audit of the two load-bearing quantities: the mIoU metric and the oracle-ceiling protocol.

WHY THIS FIRST. Every conclusion in the ScanNet++ work is a ratio against 91.92 -- "we capture 29% of
the ceiling, so the loss is in the features, not the partition". That claim routes through
`calculate_metrics` (for both the achieved score AND the ceiling) and through `score_pred`'s
ownership convention. A defect in either does not invalidate one arm, it redirects the research
programme. It has never been tested.

These are characterisation tests as much as correctness tests: several behaviours below are
deliberate inherited quirks of OpenGaussian's protocol, and pinning them stops a future "cleanup"
from silently changing what our numbers mean.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from evaluate_point_cloud_miou import calculate_metrics
from run_overnight import score_pred


def _m(gt, pred, n):
    return calculate_metrics(torch.tensor(gt), torch.tensor(pred), n)


def test_perfect_prediction_scores_one():
    ious, miou, acc, macc = _m([1, 1, 2, 2, 3], [1, 1, 2, 2, 3], 4)
    assert abs(miou - 1.0) < 1e-9 and abs(acc - 1.0) < 1e-9 and abs(macc - 1.0) < 1e-9


def test_all_wrong_scores_zero():
    _, miou, acc, _ = _m([1, 1, 2, 2], [2, 2, 1, 1], 3)
    assert miou == 0.0 and acc == 0.0


def test_two_class_case_matches_hand_computation():
    # gt   : 1 1 1 2
    # pred : 1 1 2 2
    # class1: I=2, U=|{0,1,2} u {0,1}|=3 -> 2/3 ; class2: I=1, U=|{3} u {2,3}|=2 -> 1/2
    # torch accumulates in float32, so 1e-9 is below representable precision here (the true
    # discrepancy is 4e-8). Tolerance is 1e-6, still far tighter than any effect we report.
    _, miou, acc, macc = _m([1, 1, 1, 2], [1, 1, 2, 2], 3)
    assert abs(miou - (2 / 3 + 1 / 2) / 2) < 1e-6, miou
    assert abs(acc - 3 / 4) < 1e-6
    # per-class accuracy: class1 2/3, class2 1/1
    assert abs(macc - (2 / 3 + 1.0) / 2) < 1e-6


def test_label_zero_is_ignored_entirely():
    """gt == 0 marks unannotated GT. Predictions there are zeroed and the points leave the metric,
    so they can neither help nor hurt."""
    _, miou_a, acc_a, _ = _m([0, 1, 1, 2], [9, 1, 1, 2], 3)
    _, miou_b, acc_b, _ = _m([0, 1, 1, 2], [0, 1, 1, 2], 3)
    assert miou_a == miou_b == 1.0 and acc_a == acc_b == 1.0


def test_miou_averages_only_over_classes_PRESENT_in_gt():
    """A class in the label set but absent from this scene must not drag the mean to 0. This is what
    makes the score per-scene and is why our tables average over present classes."""
    _, miou, _, _ = _m([1, 1], [1, 1], 5)          # classes 2,3,4 absent from gt
    assert miou == 1.0


def test_predicting_an_absent_class_is_not_directly_penalised():
    """INHERITED QUIRK, pinned deliberately. Class 3 is not in gt, so its IoU is excluded from the
    mean; the damage shows up only as a false negative for the true class. Hallucinating classes the
    scene does not contain is therefore cheaper than confusing two present classes -- relevant to how
    hub classes (`shelf`, `doorframe`) score."""
    _, miou_absent, _, _ = _m([1, 1, 1, 1], [1, 1, 1, 3], 4)
    _, miou_present, _, _ = _m([1, 1, 1, 2], [1, 1, 1, 1], 4)
    # both lose one point of class 1, but the second also zeroes a present class
    assert miou_absent == 3 / 4                      # only class1 counted: I=3,U=4
    assert miou_present < miou_absent


def test_gt_class_never_predicted_counts_as_zero_iou():
    _, miou, _, _ = _m([1, 1, 2, 2], [1, 1, 1, 1], 3)
    # class1: I=2 U=4 -> .5 ; class2: I=0 U=2 -> 0
    assert abs(miou - 0.25) < 1e-9


# --------------------------------------------------------------------------------------------
# score_pred: ownership. Everything about coverage/culling depends on this convention.
# --------------------------------------------------------------------------------------------

def test_unowned_gt_points_are_counted_as_errors():
    """A GT point that no cell owns gets pred = 0 while gt != 0, so it is a false negative for its
    true class. Coverage therefore costs mIoU directly -- which is the whole basis of the coverage
    filter and the culling work. If unowned points were silently dropped instead, every coverage
    conclusion would be wrong."""
    gt = torch.tensor([1, 1, 1, 1])
    assigned = np.array([0, 0, -1, -1])
    owned = assigned >= 0
    cls = np.array([0])                       # cell 0 predicts class 0 -> label 1
    miou_partial, _ = score_pred(cls, assigned, owned, gt, 1, 4)
    # 2 of 4 points owned and correct; 2 unowned -> pred 0 -> I=2, U=4 -> 0.5
    assert abs(miou_partial - 50.0) < 1e-6, miou_partial
    full = score_pred(cls, np.array([0, 0, 0, 0]), np.ones(4, bool), gt, 1, 4)[0]
    assert abs(full - 100.0) < 1e-6


def test_score_pred_offsets_class_ids_by_one():
    """cls index c maps to label c+1, because 0 is reserved for 'ignore'. An off-by-one here would
    shift every class and still produce a plausible-looking mIoU."""
    gt = torch.tensor([1, 2])
    assigned = np.array([0, 1])
    miou = score_pred(np.array([0, 1]), assigned, np.ones(2, bool), gt, 2, 2)[0]
    assert abs(miou - 100.0) < 1e-6


# --------------------------------------------------------------------------------------------
# The oracle ceiling: "label every cell with the majority GT label it owns".
# --------------------------------------------------------------------------------------------

def _oracle(assigned, gt, n_cells, n_cls):
    """Reference implementation of the ceiling, written independently of the diagnostic script."""
    best = np.zeros(n_cells, dtype=np.int64)
    for c in range(n_cells):
        m = assigned == c
        if m.sum() == 0:
            continue
        lab = gt[m]
        lab = lab[lab > 0]
        if lab.numel() == 0:
            continue
        best[c] = int(torch.bincount(lab).argmax()) - 1
    return best


def test_majority_labelling_is_NOT_the_mIoU_ceiling():
    """AUDIT FINDING (2026-08-31). The vault called 91.92 "the best ANY per-cell method could
    score". It is not. Majority-per-cell maximises ACCURACY; under macro-IoU a rare class can be
    worth more in a cell where it is a MINORITY, because every class contributes equally to the mean
    regardless of size. Deterministic counterexample:

        cell A: 10 pts class1        cell B: 3 pts class1, 2 pts class2
        majority (A->1, B->1): class1 13/15=.867, class2 0      -> mIoU .433
        better   (A->1, B->2): class1 10/13=.769, class2 2/5=.4 -> mIoU .585

    So 91.92 is a LOWER bound on the true per-cell ceiling. The published conclusion survives and
    strengthens -- we capture at most 29% of it -- but the wording must be corrected, and the gap
    between the two ceilings is itself a measurable quantity: it bounds what a macro-IoU-aware
    decision rule (Nowozin CVPR 2014) could buy on top of the features we already have."""
    assigned = np.array([0] * 10 + [1] * 5)
    gt = torch.tensor([1] * 10 + [1, 1, 1, 2, 2])
    owned = np.ones(15, bool)
    maj = score_pred(np.array([0, 0]), assigned, owned, gt, 2, 15)[0]
    alt = score_pred(np.array([0, 1]), assigned, owned, gt, 2, 15)[0]
    assert alt > maj, f"expected the non-majority labelling to win: {alt} vs {maj}"
    assert abs(maj - 43.33) < 0.1 and abs(alt - 58.46) < 0.1, (maj, alt)


def _unused_oracle_upper_bound_probe():
    """Kept for provenance: this random search is what surfaced the finding above."""
    rng = np.random.default_rng(0)
    for trial in range(25):
        n_pts, n_cells, n_cls = 60, 7, 4
        assigned = rng.integers(0, n_cells, n_pts)
        gt = torch.from_numpy(rng.integers(1, n_cls + 1, n_pts))
        owned = np.ones(n_pts, bool)
        ceil = score_pred(_oracle(assigned, gt, n_cells, n_cls), assigned, owned, gt, n_cls, n_pts)[0]
        for _ in range(20):
            other = rng.integers(0, n_cls, n_cells)
            got = score_pred(other, assigned, owned, gt, n_cls, n_pts)[0]
            assert got <= ceil + 1e-6, f"labelling beat the oracle: {got} > {ceil}"


def test_oracle_is_one_hundred_when_cells_are_pure():
    """A partition whose cells each contain a single GT class must have ceiling 100. This is the
    sanity check that ties the ceiling to cell PURITY, the quantity we report beside it (0.976)."""
    assigned = np.array([0, 0, 1, 1, 2, 2])
    gt = torch.tensor([1, 1, 2, 2, 3, 3])
    ceil = score_pred(_oracle(assigned, gt, 3, 3), assigned, np.ones(6, bool), gt, 3, 6)[0]
    assert abs(ceil - 100.0) < 1e-6


def test_oracle_falls_when_a_cell_straddles_two_classes():
    """Impure cell -> the minority points are unrecoverable by ANY per-cell method."""
    assigned = np.array([0, 0, 0, 0])
    gt = torch.tensor([1, 1, 1, 2])
    ceil = score_pred(_oracle(assigned, gt, 1, 2), assigned, np.ones(4, bool), gt, 2, 4)[0]
    # The cell takes class 1, so pred == 1 for ALL FOUR points, including the class-2 one.
    # class1: I=3, U=|{0,1,2} u {0,1,2,3}|=4 -> 0.75  (NOT 1.0 -- the union includes the point the
    # cell wrongly claims).  class2: I=0, U=1 -> 0.  mean = 0.375.
    assert abs(ceil - 37.5) < 1e-6, ceil
