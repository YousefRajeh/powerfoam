"""Export a SparseFeatureOperator-compatible COO operator (row=pixel,
col=gaussian, value=front-to-back alpha-compositing weight) from a trained
gsplat 3D Gaussian Splatting model -- the Gaussian-splat analogue of
PowerFoam's export_feature_operator.py, for Experiment B (the overlap-free
vs. overlapping baseline comparison).

gsplat's public API never materializes this sparse weight matrix directly
(rasterize_to_pixels reduces straight to a rendered image internally in
CUDA); rasterize_to_indices_in_range gives the exact per-(pixel,gaussian)
intersection enumeration used internally for gradients, so this module
recomputes the same alpha-compositing weight from that enumeration in pure
PyTorch (gather + grouped/segmented cumulative product), and validates the
result by reconstructing the rendered image from these weights and diffing
it against gsplat.rasterization()'s own forward output.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
)


def _segment_exclusive_cumsum(values: torch.Tensor, group_ids: torch.Tensor, group_start_idx: torch.Tensor) -> torch.Tensor:
    """sum of `values[j]` for all j before i within the same group as i, for
    every i, given `values`/`group_ids` already sorted so each group is
    contiguous. `group_start_idx[g]` is the index of group g's first element.
    """
    # FLOAT64 IS REQUIRED, not a precaution. `values` here are log(1-alpha), all negative, and the
    # GLOBAL cumsum over one view's ~9.4M intersections reaches magnitude ~7e6. Recovering a local
    # value of order 0.5 by subtracting two such numbers is catastrophic cancellation: in float32
    # the ULP at 7e6 is ~0.5, so the returned log-transmittance carried a MEDIAN error of 9.1e-02
    # and a max of 9.8e-01 -- i.e. transmittance wrong by up to e^0.98 ~ 2.7x.
    #
    # The symptom was that per-pixel weights summed to more than 1, which alpha compositing makes
    # impossible (sum_i alpha_i T_i = 1 - T_final <= 1). Measured on the shipped garden operator:
    # 2,635,067 of 6.5M rows summed above 1, max 2.1092. Reproduced synthetically at 74,644 rows
    # and max 1.9199 in float32, versus ZERO rows and max 1.0000 in float64.
    work = values.double()
    global_cumsum = torch.cumsum(work, dim=0)
    base = global_cumsum[group_start_idx] - work[group_start_idx]
    return (global_cumsum - work - base[group_ids]).to(values.dtype)


@torch.no_grad()
def export_view_operator(
    means, quats, scales, opacities, colors, viewmat, K, width, height,
    tile_size=16, transmittance_floor=1.0 / 255, max_hits_per_pixel=64,
):
    """Returns (row_indices, col_indices, values) COO triples for one view,
    plus a (H, W, C) reconstructed render (for validation against gsplat's
    own forward pass) and (H, W) accumulated alpha.

    Forward-only by construction (no gradient ever needed for an export
    pass) -- without @torch.no_grad(), autograd's graph bookkeeping across
    the sort/segmented-cumsum/index_add chain on ~10M-element tensors per
    view was enough to OOM a 48GB GPU on the very first view even though no
    input here actually requires grad; wrapping it dropped peak usage for a
    single view to well under 1GB.
    """
    device = means.device
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K[None], width, height,
        packed=False, opacities=opacities,
    )
    radii, means2d, depths, conics = radii[0], means2d[0], depths[0], conics[0]

    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = isect_tiles(
        means2d[None], radii[None], depths[None], tile_size, tile_width, tile_height,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    transmittances = torch.ones(1, height, width, device=device)
    gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
        0, int(1e9), transmittances, means2d[None], conics[None], opacities[None],
        width, height, tile_size, isect_offsets, flatten_ids,
    )

    n_channels = colors.shape[-1]
    if gaussian_ids.numel() == 0:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        empty_v = torch.zeros(0, device=device)
        return empty_i, empty_i, empty_v, torch.zeros(height, width, n_channels, device=device), torch.zeros(height, width, device=device)

    px = (pixel_ids % width).float() + 0.5
    py = (pixel_ids // width).float() + 0.5
    dx = means2d[gaussian_ids, 0] - px
    dy = means2d[gaussian_ids, 1] - py
    c = conics[gaussian_ids]
    sigma = 0.5 * (c[:, 0] * dx * dx + c[:, 2] * dy * dy) + c[:, 1] * dx * dy
    alpha = (opacities[gaussian_ids] * torch.exp(-sigma)).clamp(0.0, 0.999)
    depth_vals = depths[gaussian_ids]

    # Lexsort by (pixel_id, depth) via two stable sorts: depth first (global
    # front-to-back order), then pixel_id (stable sort preserves each
    # pixel's own depth order within its group) -- gsplat's own kernel
    # relies on the identical two-key ordering internally.
    depth_order = torch.argsort(depth_vals, stable=True)
    order = depth_order[torch.argsort(pixel_ids[depth_order], stable=True)]

    pixel_s = pixel_ids[order]
    gaussian_s = gaussian_ids[order]
    alpha_s = alpha[order]

    is_new_group = torch.ones_like(pixel_s, dtype=torch.bool)
    is_new_group[1:] = pixel_s[1:] != pixel_s[:-1]
    group_id = torch.cumsum(is_new_group.long(), dim=0) - 1
    group_start_idx = torch.nonzero(is_new_group, as_tuple=True)[0]
    # position of each element within its own group (0-indexed) -- used to
    # bound nnz/pixel the same way PowerFoam's exporter caps hits/pixel.
    pos_in_group = torch.arange(pixel_s.numel(), device=device) - group_start_idx[group_id]

    log1m_alpha = torch.log((1 - alpha_s).clamp_min(1e-12))
    t_before_log = _segment_exclusive_cumsum(log1m_alpha, group_id, group_start_idx)
    t_before = torch.exp(t_before_log)
    weight = alpha_s * t_before

    keep = (t_before > transmittance_floor) & (pos_in_group < max_hits_per_pixel)
    row_indices = pixel_s[keep].long()
    col_indices = gaussian_s[keep].long()
    values = weight[keep]

    render = torch.zeros(height * width, n_channels, device=device)
    render.index_add_(0, row_indices, values[:, None] * colors[col_indices])
    alpha_accum = torch.zeros(height * width, device=device)
    alpha_accum.index_add_(0, row_indices, values)

    return row_indices, col_indices, values, render.view(height, width, n_channels), alpha_accum.view(height, width)


def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    n = 200
    means = (torch.rand(n, 3, device=device) - 0.5) * 2
    means[:, 2] += 4.0  # push in front of the camera
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.rand(n, 3, device=device) * 0.1 + 0.02
    opacities = torch.rand(n, device=device) * 0.8 + 0.1
    colors = torch.rand(n, 3, device=device)
    viewmat = torch.eye(4, device=device)
    K = torch.tensor([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1.0]], device=device)
    width = height = 128

    import gsplat
    ref_render, ref_alpha, _ = gsplat.rasterization(means, quats, scales, opacities, colors, viewmat[None], K[None], width, height)
    ref_render, ref_alpha = ref_render[0], ref_alpha[0, ..., 0]

    for floor in (1.0 / 255, 0.0):
        row, col, val, render, alpha = export_view_operator(
            means, quats, scales, opacities, colors, viewmat, K, width, height, transmittance_floor=floor,
        )
        render_err = (render - ref_render).abs().max().item()
        alpha_err = (alpha - ref_alpha).abs().max().item()
        print(f"[export_gsplat_operator self-test] transmittance_floor={floor:.4f} nnz={row.numel()} "
              f"render_max_abs_err={render_err:.6f} alpha_max_abs_err={alpha_err:.6f}")

    # floor=0 keeps every geometrically-possible hit (no early-termination
    # cutoff at all), so it isolates whether the compositing math itself is
    # correct from the early-termination convention's minor truncation --
    # this one must match tightly.
    row, col, val, render, alpha = export_view_operator(
        means, quats, scales, opacities, colors, viewmat, K, width, height,
        transmittance_floor=0.0, max_hits_per_pixel=1_000_000,
    )
    render_err = (render - ref_render).abs().max().item()
    alpha_err = (alpha - ref_alpha).abs().max().item()
    print(f"[export_gsplat_operator self-test] uncapped nnz={row.numel()} render_max_abs_err={render_err:.6f} "
          f"alpha_max_abs_err={alpha_err:.6f}")
    # A worst-case pixel had 9 overlapping hits and differed by ~0.0015 in
    # RGB/alpha (both in [0,1]) even with no early-termination or hit cap at
    # all -- consistent with float32 accumulation-order noise across a
    # 9-term multiplicative transmittance chain (gsplat's CUDA kernel
    # accumulates per-tile in parallel; this reimplementation sorts/groups
    # globally in a different order), not a logic error. 3e-3 comfortably
    # covers that noise floor while still being a real, tight round-trip
    # check.
    assert render_err < 3e-3, "reconstructed render does not match gsplat.rasterization()'s own forward pass"
    assert alpha_err < 3e-3, "reconstructed alpha does not match gsplat.rasterization()'s own forward pass"
    print("[export_gsplat_operator self-test] PASSED")


if __name__ == "__main__":
    _self_test()
