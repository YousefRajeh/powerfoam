"""Held-out reconstruction quality (PSNR/SSIM) for the trained gsplat garden
checkpoint, on the same 24 test views used everywhere else in this
comparison -- a control check for Experiment B: if the two representations'
base reconstruction quality differs a lot, that's a confound sitting
underneath the overlap/lifting-quality comparison, not just overlap itself.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import os

import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import gsplat

CKPT_DIR = os.environ.get("GSPLAT_ARTIFACT_DIR", r"D:\Downloads\powerfoam\artifacts\garden_gsplat")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")
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
            gt = torch.from_numpy(np.asarray(imageio.imread(image_paths[view_id])[..., :3])).float().to(DEVICE) / 255.0
            render, _, _ = gsplat.rasterization(
                means, quats, scales, opacities, colors, viewmats_all[view_id][None], K[None], width, height,
            )
            render = render[0].clamp(0, 1)
            p = float(psnr_metric(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            s = float(ssim_metric(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            psnrs.append(p)
            ssims.append(s)
            print(f"[eval_psnr] view {view_id}: PSNR={p:.3f} SSIM={s:.4f}")

    print(f"\n[eval_psnr] Average PSNR: {sum(psnrs) / len(psnrs):.4f}")
    print(f"[eval_psnr] Average SSIM: {sum(ssims) / len(ssims):.4f}")


if __name__ == "__main__":
    main()
