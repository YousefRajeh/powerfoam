"""Render a PCA-pseudocolor preview of a solved Feature Foam feature field,
composited through the sparse operator for a couple of held-out test views.
Quick, non-viewer way to eyeball "how the features look" spatially before
building the interactive viewer (Phase 2)."""

import argparse

import numpy as np
import torch
from PIL import Image

from feature_foam_lifting.operator import SparseFeatureOperator


def pca_to_rgb(x, valid_mask, low_pct=1.0, high_pct=99.0):
    # Normalize to unit length before PCA: direction is what carries the
    # cosine-similarity-meaningful signal (this is the whole basis of the
    # lifting problem); magnitude is largely a solver artifact -- e.g. ridge_pcg
    # at this scale produces primitive norms ranging from ~0 to ~32 (vs
    # weighted_average's tight [0.8, 1.0]) for weakly-identified primitives, and
    # letting that dominate a global percentile normalization clips whole
    # spatial regions to solid black/white regardless of their actual direction.
    x_valid = x[valid_mask]
    x_valid = x_valid / x_valid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mean = x_valid.mean(0, keepdim=True)
    xc = x_valid - mean
    _, _, v = torch.pca_lowrank(xc, q=3)
    proj_valid = xc @ v[:, :3]

    lo = torch.quantile(proj_valid, low_pct / 100.0, dim=0)
    hi = torch.quantile(proj_valid, high_pct / 100.0, dim=0)
    proj_valid = ((proj_valid - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    rgb = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)
    rgb[valid_mask] = proj_valid
    return rgb


def render_view(operator, rgb, view_id, height, width):
    rendered = operator.matmul(rgb)
    mask = operator.row_view_ids == view_id
    coords = operator.row_pixels[mask]
    img = torch.zeros(height, width, 3, device=rgb.device, dtype=rgb.dtype)
    img[coords[:, 0], coords[:, 1]] = rendered[mask]
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operator", default="artifacts/garden/test_operator.pt")
    p.add_argument("--primitive-features", default="artifacts/garden/roundtrip/x_ridge.pt")
    p.add_argument("--views", default="0,3,7,12")
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--width", type=int, default=648)
    p.add_argument("--output", default="artifacts/garden/pca_preview.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    a = SparseFeatureOperator.load(args.operator, args.device)
    field = torch.load(args.primitive_features, map_location=args.device, weights_only=True)
    x = field["primitive_features"].float()
    valid_mask = field["valid_mask"]

    pca_rgb = pca_to_rgb(x, valid_mask)

    view_ids = [int(v) for v in args.views.split(",")]
    panels = [render_view(a, pca_rgb, v, args.height, args.width) for v in view_ids]
    grid = torch.cat(panels, dim=1)  # side by side

    img = (grid.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(args.output)
    print(f"wrote {args.output} ({img.shape[1]}x{img.shape[0]}), views={view_ids}")


if __name__ == "__main__":
    main()
