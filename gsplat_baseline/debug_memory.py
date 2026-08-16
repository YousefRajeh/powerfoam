import gsplat_env_gsview  # noqa: F401

import sys
from pathlib import Path

import torch
from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles, rasterize_to_indices_in_range

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda"


def mb():
    return torch.cuda.memory_allocated() / 1e6


def main():
    torch.cuda.reset_peak_memory_stats()
    print(f"start: {mb():.0f} MB")
    ckpt = torch.load(r"D:\Downloads\powerfoam\artifacts\garden_gsplat\ckpt.pt", map_location=DEVICE, weights_only=False)
    means = ckpt["means"].to(DEVICE)
    quats = ckpt["quats"].to(DEVICE)
    scales = torch.exp(ckpt["scales"]).to(DEVICE)
    opacities = torch.sigmoid(ckpt["opacities"]).to(DEVICE)
    colors = torch.sigmoid(ckpt["colors"]).to(DEVICE)
    K = ckpt["K"].to(DEVICE)
    width, height = ckpt["width"], ckpt["height"]
    camtoworlds = ckpt["camtoworlds"]
    viewmat = torch.linalg.inv(camtoworlds[0]).to(DEVICE)
    print(f"after ckpt load: {mb():.0f} MB, n_gaussians={means.shape[0]}")

    with torch.no_grad():
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means, None, quats, scales, viewmat[None], K[None], width, height, packed=False, opacities=opacities,
        )
        print(f"after fully_fused_projection: {mb():.0f} MB")
        radii, means2d, depths, conics = radii[0], means2d[0], depths[0], conics[0]
        print("radii stats: max", radii.float().max().item(), "mean", radii.float().mean().item())
        n_valid = (radii[:, 0] > 0).sum().item()
        print(f"n_valid (radii>0): {n_valid} / {means.shape[0]}")

        tile_size = 16
        tile_width = (width + tile_size - 1) // tile_size
        tile_height = (height + tile_size - 1) // tile_size
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d[None], radii[None], depths[None], tile_size, tile_width, tile_height,
        )
        print(f"after isect_tiles: {mb():.0f} MB, n_isects={isect_ids.numel()}, "
              f"max_tiles_per_gauss={tiles_per_gauss.max().item()}, sum_tiles_per_gauss={tiles_per_gauss.sum().item()}")

        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
        print(f"after isect_offset_encode: {mb():.0f} MB")

        transmittances = torch.ones(1, height, width, device=DEVICE)
        gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
            0, int(1e9), transmittances, means2d[None], conics[None], opacities[None],
            width, height, tile_size, isect_offsets, flatten_ids,
        )
        print(f"after rasterize_to_indices_in_range: {mb():.0f} MB, n_hits={gaussian_ids.numel()}")

    print(f"peak: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")


if __name__ == "__main__":
    main()
