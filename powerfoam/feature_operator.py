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
    weight_transform=None,
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

    Each view's feature gather is passed to `accumulate_view` as a LAZY callable
    `(start, end) -> (end-start, F)` closed over that view's own feature map,
    rather than the materialized `fmap[pix_y, pix_x]` tensor this function used
    to build eagerly. Diagnosed root cause of a real OOM at high-resolution
    views with many primitives (e.g. ScanNet's 1296x968 images,
    >~500k-670k primitives): building that full `(nnz, F)` gather up front
    happened BEFORE `accumulate_view`'s own internal `_nnz_chunks` chunking ever
    got a chance to help (that chunking only applies to a `b` that's already
    fully materialized -- it can't un-materialize an already-OOMing gather).
    Passing a lazy callable instead lets `accumulate_view` gather each
    bounded-size chunk on demand, inside its own loop, so peak memory per view
    is independent of that view's resolution/nnz. This is an exact fix, not a
    heuristic: it's the same streaming reduction as before, just with the
    gather deferred to the point it's actually consumed (see
    AccumulatedFeatureStats.accumulate_view's docstring and
    feature-foam-lifting/tests/test_operator.py::
    test_accumulate_view_lazy_gather_matches_materialized_b for the proof that
    this produces bit-identical accumulator state, including the
    view-order-sensitive geometric-median/reliability fields, to gathering `b`
    fully up front).

    `batch_size` views' exported triples are queued up (each view's own lazy
    gather closure, never a materialized (nnz, channels) tensor) and folded in
    via one `accumulate_views` call per batch. CORRECTED after benchmarking on
    the real garden checkpoint (161 views, 1.2M primitives, 512-d): batching
    this way is NOT a free throughput win, and the measured wall-clock was flat
    across batch_size in {1, 4, 8} (~0.25-0.48 s/view) -- per-view rendering and
    disk I/O dominate, not Python dispatch. Now that queued entries are lazy
    closures rather than materialized gathers, batch_size no longer trades
    memory for fewer Python calls the way it used to (each view's gather is
    still bounded-size regardless of when it runs) -- it's kept only because
    there's no reason to remove it, not because it matters for memory anymore.

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
    batch = []  # list of (cols, vals, lazy_b, row_local, feature_dim) tuples, one per view

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

        # weight_transform: reshape lifting weights to test the sharpness hypothesis vs
        # splat-style alpha-blending (which concentrates on dominant surface splats,
        # unlike the volumetric operator's up-to-64 cells per ray):
        #   'top1' -- each pixel's feature goes ONLY to its max-weight cell (hard
        #             surface assignment, the sharpest splat analogue)
        #   'sq'   -- w^2 (soft sharpening; renormalization is irrelevant because the
        #             weighted solve normalizes per-primitive anyway)
        if weight_transform == "top1":
            vmat = out_val.reshape(num_pixels, max_hits_per_pixel)
            top = vmat.argmax(dim=1)
            hard = torch.zeros_like(vmat)
            ar = torch.arange(num_pixels, device=device)
            hard[ar, top] = vmat[ar, top]
            out_val = hard.reshape(-1)
        elif weight_transform == "sq":
            out_val = out_val * out_val

        cols = out_col[keep_mask].long()
        vals = out_val.reshape(-1)[keep_mask]
        row_local = torch.arange(num_pixels, device=device).repeat_interleave(max_hits_per_pixel)[keep_mask]

        feature_dim = fmap.shape[-1]
        fmap_dev = fmap.to(device)

        def lazy_b(start, end, fmap_dev=fmap_dev, row_local=row_local, W=W):
            chunk_row_local = row_local[start:end]
            return fmap_dev[chunk_row_local // W, chunk_row_local % W]

        batch.append((cols, vals, lazy_b, row_local, feature_dim))

        del out_col, out_val, fmap  # fmap_dev is captured by lazy_b's closure; freed once this view's batch entry flushes
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()

    if total_dropped > 0:
        print(f"[accumulate_feature_stats] total dropped hits across all views: {total_dropped}")
    print(f"[accumulate_feature_stats] folded in {stats.num_views} views, "
          f"valid_fraction={stats.diagnostics()['valid_fraction']:.4f}")
    return stats
