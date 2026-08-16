"""Held-out PSNR/SSIM for the trained gsplat Replica room_0 checkpoint, on
the same held-out test views used everywhere else in this comparison --
mirrors eval_psnr.py's role for garden.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import gsplat

CKPT_PATH = r"D:\Downloads\powerfoam\artifacts\replica_room0_gsplat\ckpt.pt"
DEVICE = "cuda"


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    means, quats, scales, opacities, colors = (
        ckpt["means"].to(DEVICE), ckpt["quats"].to(DEVICE),
        torch.exp(ckpt["scales"]).to(DEVICE), torch.sigmoid(ckpt["opacities"]).to(DEVICE),
        torch.sigmoid(ckpt["colors"]).to(DEVICE),
    )
    K, width, height = ckpt["K"].to(DEVICE), ckpt["width"], ckpt["height"]
    camtoworlds = ckpt["camtoworlds"]
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)
    test_idx = ckpt["test_idx"]
    image_paths = ckpt["image_paths"]

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    psnrs, ssims = [], []
    with torch.no_grad():
        for view_id in test_idx.tolist():
            gt = torch.from_numpy(np.asarray(Image.open(image_paths[view_id]).convert("RGB"))).float().to(DEVICE) / 255.0
            render, _, _ = gsplat.rasterization(
                means, quats, scales, opacities, colors, viewmats_all[view_id][None], K[None], width, height,
            )
            render = render[0].clamp(0, 1)
            p = float(psnr_metric(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            s = float(ssim_metric(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            psnrs.append(p)
            ssims.append(s)

    print(f"[eval_psnr_replica] n_views={len(test_idx)}")
    print(f"[eval_psnr_replica] Average PSNR: {sum(psnrs) / len(psnrs):.4f}")
    print(f"[eval_psnr_replica] Average SSIM: {sum(ssims) / len(ssims):.4f}")


if __name__ == "__main__":
    main()
