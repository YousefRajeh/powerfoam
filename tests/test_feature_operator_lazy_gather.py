"""End-to-end check that accumulate_feature_stats_for_views's lazy-gather closure
(see feature_operator.py's docstring) produces identical AccumulatedFeatureStats
to eagerly gathering each view's full (nnz, F) feature tensor up front -- the
caller-side half of the exact OOM fix (the library-side half is proven in
feature-foam-lifting/tests/test_operator.py::
test_accumulate_view_lazy_gather_matches_materialized_b)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from feature_foam_lifting.operator import AccumulatedFeatureStats, _nnz_chunks
import feature_foam_lifting.operator as op_module
from powerfoam.feature_operator import accumulate_feature_stats_for_views


class FakeCamera:
    def __init__(self, height, width):
        self.height = height
        self.width = width


class FakeModel:
    """Mimics PowerfoamScene enough for accumulate_feature_stats_for_views:
    a fixed random hit pattern per view, `max_hits_per_pixel` slots per pixel."""

    def __init__(self, num_primitives, max_hits_per_pixel, device):
        self.points = torch.zeros(num_primitives, 3, device=device)
        self._num_primitives = num_primitives
        self._max_hits_per_pixel = max_hits_per_pixel
        self._device = device

    def export_feature_operator(self, camera, transmittance_threshold, max_intersections, max_hits_per_pixel):
        assert max_hits_per_pixel == self._max_hits_per_pixel
        num_pixels = camera.height * camera.width
        device = self._device
        # Ragged hit counts per pixel, 1..max_hits_per_pixel.
        slots_used = torch.randint(1, max_hits_per_pixel + 1, (num_pixels,), device=device)
        out_col = torch.randint(0, self._num_primitives, (num_pixels * max_hits_per_pixel,), device=device)
        out_val = torch.rand(num_pixels * max_hits_per_pixel, device=device)
        overflow_counter = torch.zeros(1, device=device)
        return out_col, out_val, slots_used, overflow_counter, None


def _run(num_primitives, height, width, feature_dim, max_hits_per_pixel, batch_size, force_small_chunks):
    torch.manual_seed(0)
    device = "cpu"
    model = FakeModel(num_primitives, max_hits_per_pixel, device)
    camera = FakeCamera(height, width)
    fmap = torch.randn(height, width, feature_dim, device=device)

    old_max = op_module._MAX_GATHER_ELEMENTS
    if force_small_chunks:
        op_module._MAX_GATHER_ELEMENTS = feature_dim * 10
    try:
        stats = accumulate_feature_stats_for_views(
            model, [camera], [0], feature_map_loader=lambda view_id: fmap,
            max_hits_per_pixel=max_hits_per_pixel, batch_size=batch_size,
        )
    finally:
        op_module._MAX_GATHER_ELEMENTS = old_max
    return stats


def test_lazy_gather_matches_eager_materialization_with_forced_chunking():
    # Reference: batch_size=1, chunking NOT forced (full-size internal chunk budget)
    # so the internal _nnz_chunks loop in accumulate_view takes exactly one iteration
    # -- functionally identical to eagerly materializing `b` in one shot.
    ref = _run(num_primitives=15, height=4, width=5, feature_dim=6, max_hits_per_pixel=3,
               batch_size=1, force_small_chunks=False)
    # Same random seed/inputs, but force the lazy gather to be pulled in several small
    # internal chunks (and batch multiple... well only one view here, batch_size=4 just
    # exercises accumulate_views' tuple-unpacking path instead of accumulate_view directly).
    chunked = _run(num_primitives=15, height=4, width=5, feature_dim=6, max_hits_per_pixel=3,
                    batch_size=4, force_small_chunks=True)

    for field in ("support", "support2", "numerator", "sq_numerator", "sum_view_weight_sq",
                  "intra_sum", "support_iv", "numerator_iv", "gm_z", "gm_weight"):
        a, b = getattr(ref, field), getattr(chunked, field)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    assert ref.num_views == chunked.num_views == 1
