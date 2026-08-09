"""Host-side orchestration for exporting Feature Foam sparse rendering-weight
operators from a trained, frozen PowerFoam checkpoint.

The actual per-(pixel, primitive) weight computation is a forward-only twin of
the real rendering kernel (``Rasterizer.export_operator_kernel`` in
``powerfoam/rasterize.py``, invoked via ``PowerfoamScene.export_feature_operator``)
so the exported weight is exactly what the real renderer used for RGB at that
pixel/primitive pair. This module only loops over views, compacts each view's
fixed-capacity per-pixel output into COO triplets, and concatenates across views
into a single ``SparseFeatureOperator`` (from the separately-installed,
renderer-independent ``feature_foam_lifting`` solver package).
"""

import torch

from feature_foam_lifting.operator import AccumulatedFeatureStats, SparseFeatureOperator


def export_operator_for_views(
    model,
    cameras,
    view_ids,
    transmittance_threshold=1e-3,
    max_intersections=1024,
    max_hits_per_pixel=64,
):
    """Export and concatenate a sparse operator for the given cameras.

    Parameters
    ----------
    model : PowerfoamScene
        Loaded checkpoint. Must NOT have had ``sort_points()``/``resample()``
        called on it after ``load_pt`` -- see docs/feature-foam-phase1... for why
        primitive-index stability across separate train/test export runs depends
        on this.
    cameras : sequence of TorchCamera
    view_ids : sequence of int
        Stable identifiers for each camera (e.g. dataset index), stored per-row
        so downstream feature-map alignment can match rows back to images.
    """
    device = model.points.device
    all_rows, all_cols, all_vals = [], [], []
    all_view_ids, all_pixels = [], []  # one entry per ROW (per pixel), NOT per nonzero
    total_rows_so_far = 0
    total_dropped = 0

    for view_id, camera in zip(view_ids, cameras):
        H, W = camera.height, camera.width
        num_pixels = H * W

        out_col, out_val, slot_counter, overflow_counter, _ = model.export_feature_operator(
            camera,
            transmittance_threshold=transmittance_threshold,
            max_intersections=max_intersections,
            max_hits_per_pixel=max_hits_per_pixel,
        )

        dropped = int(overflow_counter.item())
        if dropped > 0:
            print(
                f"[export_feature_operator] view {view_id}: dropped {dropped} hits "
                f"beyond max_hits_per_pixel={max_hits_per_pixel} -- consider raising the cap"
            )
        total_dropped += dropped

        slots_used = slot_counter.clamp(max=max_hits_per_pixel)
        slot_arange = torch.arange(max_hits_per_pixel, device=device)
        keep_mask = (slot_arange[None, :] < slots_used[:, None]).reshape(-1)

        cols = out_col[keep_mask].long()
        vals = out_val[keep_mask]

        row_local = torch.arange(num_pixels, device=device).repeat_interleave(
            max_hits_per_pixel
        )[keep_mask]
        rows = row_local + total_rows_so_far

        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(vals)

        # SparseFeatureOperator.row_view_ids/row_pixels are indexed in [0, num_rows)
        # -- one entry per pixel/row, independent of how many nonzeros (primitive
        # hits) that row has -- NOT one entry per nonzero like rows/cols/vals above.
        pixel_arange = torch.arange(num_pixels, device=device)
        all_view_ids.append(torch.full((num_pixels,), int(view_id), dtype=torch.long, device=device))
        all_pixels.append(torch.stack([pixel_arange // W, pixel_arange % W], dim=-1))

        total_rows_so_far += num_pixels

    if total_dropped > 0:
        print(f"[export_feature_operator] total dropped hits across all views: {total_dropped}")

    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    return SparseFeatureOperator(
        row_indices=torch.cat(all_rows) if all_rows else empty_long,
        col_indices=torch.cat(all_cols) if all_cols else empty_long,
        values=torch.cat(all_vals) if all_vals else torch.zeros(0, dtype=torch.float32, device=device),
        num_rows=total_rows_so_far,
        num_primitives=model.points.shape[0],
        row_view_ids=torch.cat(all_view_ids) if all_view_ids else empty_long,
        row_pixels=torch.cat(all_pixels) if all_pixels else torch.zeros((0, 2), dtype=torch.long, device=device),
    )


def accumulate_feature_stats_for_views(
    model,
    cameras,
    view_ids,
    feature_map_loader,
    transmittance_threshold=1e-3,
    max_intersections=1024,
    max_hits_per_pixel=64,
    batch_size=4,
):
    """Stream-accumulate per-primitive sufficient statistics (support, support2,
    numerator, sq_numerator -- see AccumulatedFeatureStats) across ANY number of
    views, at memory bounded by ONE view's nnz and ONE view's feature map at a
    time -- NOT the sum across all views.

    This is why export_operator_for_views (above) was capped at ~10 train views:
    it concatenates every view's triples into one batch SparseFeatureOperator, so
    its memory cost scales with total nnz across all views combined. Solvers 0
    (weighted average), 0b (squared-weight average), and 1b (closed-form ridge)
    only ever need the four per-primitive accumulators below -- they don't need
    the raw triples once reduced -- so folding each view in immediately and
    discarding its triples gives IDENTICAL results (see
    tests/test_operator.py::test_streaming_accumulation_matches_batch_operator in
    the feature_foam_lifting repo) at memory cost independent of view count. This
    lets every train view contribute, closing the "unseen by the sparse subset"
    coverage gaps a 10-view batch export leaves behind.

    `feature_map_loader` MUST be lazy (called once per view, right before that
    view is needed) -- pre-loading every view's (H, W, F) feature map into a
    list before this function runs defeats the entire point (161 views x
    648x420x512 float32 is ~90GB; one at a time is ~550MB).

    `batch_size` views' exported triples are queued up (each view's own,
    never-concatenated (nnz, channels) gather) and folded in via one
    `accumulate_views` call per batch. CORRECTED after benchmarking on the real
    garden checkpoint (161 views, 1.2M primitives, 512-d): batching this way is
    NOT a free throughput win. Every queued-but-not-yet-reduced view's `b`
    gather (several GB each at F=512) stays alive in the `batch` list until the
    whole batch flushes, so peak memory genuinely scales with `batch_size` --
    batch_size=16 OOMs a 48GB GPU where batch_size=8 does not. And the measured
    wall-clock was flat across batch_size in {1, 4, 8} (~0.25-0.48 s/view all
    three, with the batch_size=1 number likely inflated by one-time Warp kernel
    compilation rather than genuine per-call overhead) -- per-view rendering and
    disk I/O dominate, not Python dispatch, so there is little to gain from
    batching here regardless. Keep `batch_size` small (1-4); it exists mainly so
    a caller CAN trade a bounded amount of extra memory for fewer Python calls
    if their profiling shows dispatch overhead actually matters for their
    workload, not because it does here.

    (Full ridge_pcg is NOT reproducible from these stats -- it needs the coupled
    A^T A, not just its diagonal support2 -- so it stays batch/view-limited via
    export_operator_for_views + ridge_pcg.)

    Parameters
    ----------
    feature_map_loader : callable(view_id) -> (H, W, F) tensor
        Loads a single view's feature map on demand.
    """
    device = model.points.device
    stats = None
    total_dropped = 0
    batch = []  # list of (cols, vals, b) tuples, one per view -- never concatenated

    def flush_batch():
        nonlocal batch
        if not batch:
            return
        stats.accumulate_views(batch)  # folds each view in via its own internally-chunked accumulate_view call
        batch = []

    for view_id, camera in zip(view_ids, cameras):
        H, W = camera.height, camera.width
        num_pixels = H * W
        fmap = feature_map_loader(view_id)
        if stats is None:
            stats = AccumulatedFeatureStats.zeros(model.points.shape[0], fmap.shape[-1], device=device)

        out_col, out_val, slot_counter, overflow_counter, _ = model.export_feature_operator(
            camera,
            transmittance_threshold=transmittance_threshold,
            max_intersections=max_intersections,
            max_hits_per_pixel=max_hits_per_pixel,
        )

        dropped = int(overflow_counter.item())
        if dropped > 0:
            print(f"[accumulate_feature_stats] view {view_id}: dropped {dropped} hits beyond max_hits_per_pixel={max_hits_per_pixel}")
        total_dropped += dropped

        slots_used = slot_counter.clamp(max=max_hits_per_pixel)
        slot_arange = torch.arange(max_hits_per_pixel, device=device)
        keep_mask = (slot_arange[None, :] < slots_used[:, None]).reshape(-1)

        cols = out_col[keep_mask].long()
        vals = out_val[keep_mask]
        row_local = torch.arange(num_pixels, device=device).repeat_interleave(max_hits_per_pixel)[keep_mask]
        pix_y, pix_x = row_local // W, row_local % W

        batch.append((cols, vals, fmap.to(device)[pix_y, pix_x]))

        del out_col, out_val, fmap  # this view's triples/feature map are captured above; free the rest now
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()

    if total_dropped > 0:
        print(f"[accumulate_feature_stats] total dropped hits across all views: {total_dropped}")
    print(f"[accumulate_feature_stats] folded in {stats.num_views} views, "
          f"valid_fraction={stats.diagnostics()['valid_fraction']:.4f}")
    return stats
