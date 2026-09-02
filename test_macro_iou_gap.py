"""Verify the macro-IoU gap machinery against the REAL metric before trusting any gap number.

The whole measurement is a fast reimplementation of mIoU in terms of per-class counts. If that
reimplementation disagrees with `calculate_metrics` even slightly, the "gap" is an artefact of two
different metrics rather than a property of the labelling -- exactly the failure this project has
been bitten by before. So the central test drives random labellings through BOTH paths and demands
agreement.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from run_macro_iou_gap import cell_histograms, miou_from_counts, coordinate_ascent
from run_overnight import score_pred


def _reference(labels, assigned, gt, n_pts, n_cls):
    """Ground truth: the metric actually used in every reported number."""
    owned = assigned >= 0
    return score_pred(labels, assigned, owned, gt, n_cls, n_pts)[0]


def _counts(H, T, labels):
    n_cells, n_cls = H.shape
    N = H.sum(1)
    I = np.zeros(n_cls, np.int64); P = np.zeros(n_cls, np.int64)
    np.add.at(I, labels, H[np.arange(n_cells), labels])
    np.add.at(P, labels, N)
    return I, P


def test_count_based_miou_matches_calculate_metrics():
    """The load-bearing test: random scenes, random labellings, both paths must agree."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        n_pts, n_cells, n_cls = 80, 6, 4
        assigned = rng.integers(0, n_cells, n_pts)
        raw = rng.integers(1, n_cls + 1, n_pts)
        gt = torch.from_numpy(raw).long()
        labels = rng.integers(0, n_cls, n_cells)
        H, T = cell_histograms(assigned, gt, n_cells, n_cls)
        I, P = _counts(H, T, labels)
        present = np.flatnonzero(T > 0)
        fast = miou_from_counts(I.astype(float), T.astype(float), P.astype(float), present)
        ref = _reference(labels, assigned, gt, n_pts, n_cls)
        assert abs(fast - ref) < 1e-4, f"{fast} vs {ref}"


def test_matches_reference_when_some_points_are_unowned():
    """Unowned points are scored pred=0 (a false negative), so they belong in T but in no cell.
    If cell_histograms put them in H the gap would be measured against an inflated ceiling."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n_pts, n_cells, n_cls = 60, 5, 3
        assigned = rng.integers(-1, n_cells, n_pts)      # -1 => unowned
        gt = torch.from_numpy(rng.integers(1, n_cls + 1, n_pts)).long()
        labels = rng.integers(0, n_cls, n_cells)
        H, T = cell_histograms(assigned, gt, n_cells, n_cls)
        I, P = _counts(H, T, labels)
        fast = miou_from_counts(I.astype(float), T.astype(float), P.astype(float),
                                np.flatnonzero(T > 0))
        assert abs(fast - _reference(labels, assigned, gt, n_pts, n_cls)) < 1e-4


def test_matches_reference_when_gt_contains_ignore_label():
    """gt == 0 points are dropped by the metric entirely; they must not enter T, H or N."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        n_pts, n_cells, n_cls = 70, 5, 3
        assigned = rng.integers(0, n_cells, n_pts)
        gt = torch.from_numpy(rng.integers(0, n_cls + 1, n_pts)).long()   # includes 0
        labels = rng.integers(0, n_cls, n_cells)
        H, T = cell_histograms(assigned, gt, n_cells, n_cls)
        I, P = _counts(H, T, labels)
        fast = miou_from_counts(I.astype(float), T.astype(float), P.astype(float),
                                np.flatnonzero(T > 0))
        assert abs(fast - _reference(labels, assigned, gt, n_pts, n_cls)) < 1e-4


def test_histogram_rows_sum_to_owned_scored_points():
    assigned = np.array([0, 0, 1, -1, 2])
    gt = torch.tensor([1, 2, 2, 3, 0])
    H, T = cell_histograms(assigned, gt, 3, 3)
    assert H.sum() == 3                       # point 3 unowned, point 4 is ignore-label
    assert H[0].tolist() == [1, 1, 0] and H[1].tolist() == [0, 1, 0]
    assert T.tolist() == [1, 2, 1]            # the unowned class-3 point still counts in T


def test_ascent_never_decreases_miou():
    """Coordinate ascent must be monotone: it only accepts a move with positive total gain."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        n_pts, n_cells, n_cls = 120, 8, 5
        assigned = rng.integers(0, n_cells, n_pts)
        gt = torch.from_numpy(rng.integers(1, n_cls + 1, n_pts)).long()
        H, T = cell_histograms(assigned, gt, n_cells, n_cls)
        present = np.flatnonzero(T > 0)
        maj = H.argmax(1); maj[H.sum(1) == 0] = 0
        I, P = _counts(H, T, maj)
        before = miou_from_counts(I.astype(float), T.astype(float), P.astype(float), present)
        _, after, _, _ = coordinate_ascent(H, T, present, maj.copy())
        assert after >= before - 1e-9, f"ascent went DOWN: {before} -> {after}"


def test_ascent_result_is_reproducible_through_the_real_metric():
    """The improved labelling must also score higher under calculate_metrics, not just under the
    fast count path -- otherwise the gain lives in the reimplementation."""
    rng = np.random.default_rng(4)
    n_pts, n_cells, n_cls = 200, 10, 5
    assigned = rng.integers(0, n_cells, n_pts)
    gt = torch.from_numpy(rng.integers(1, n_cls + 1, n_pts)).long()
    H, T = cell_histograms(assigned, gt, n_cells, n_cls)
    present = np.flatnonzero(T > 0)
    maj = H.argmax(1); maj[H.sum(1) == 0] = 0
    lab_opt, fast_opt, _, _ = coordinate_ascent(H, T, present, maj.copy())
    ref_maj = _reference(maj, assigned, gt, n_pts, n_cls)
    ref_opt = _reference(lab_opt, assigned, gt, n_pts, n_cls)
    assert abs(fast_opt - ref_opt) < 1e-4
    assert ref_opt >= ref_maj - 1e-9


def test_pure_cells_are_never_moved():
    """The shortcut that makes this tractable: a class absent from a cell can never be optimal for
    it, so pure cells have exactly one candidate. If this were false the search would be wrong."""
    assigned = np.array([0, 0, 0, 1, 1, 1])
    gt = torch.tensor([1, 1, 1, 2, 2, 2])                  # both cells perfectly pure
    H, T = cell_histograms(assigned, gt, 2, 2)
    maj = H.argmax(1)
    lab, _, moved, _ = coordinate_ascent(H, T, np.flatnonzero(T > 0), maj.copy())
    assert moved == 0 and lab.tolist() == maj.tolist()


def test_ascent_finds_the_known_counterexample():
    """The deterministic case from test_metric_audit: majority is beatable, and the search must
    actually find it. Without this, a zero gap could mean 'no headroom' OR 'broken search'."""
    assigned = np.array([0] * 10 + [1] * 5)
    gt = torch.tensor([1] * 10 + [1, 1, 1, 2, 2])
    H, T = cell_histograms(assigned, gt, 2, 2)
    present = np.flatnonzero(T > 0)
    maj = H.argmax(1)
    assert maj.tolist() == [0, 0]                          # majority labels cell 1 as class 1
    lab, opt, moved, _ = coordinate_ascent(H, T, present, maj.copy())
    assert lab.tolist() == [0, 1] and moved == 1
    ref_maj = _reference(maj, assigned, gt, 15, 2)
    ref_opt = _reference(lab, assigned, gt, 15, 2)
    assert abs(ref_maj - 43.33) < 0.1 and abs(ref_opt - 58.46) < 0.1, (ref_maj, ref_opt)
