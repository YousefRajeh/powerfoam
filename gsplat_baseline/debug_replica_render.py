import gsplat_env_gsview  # noqa: F401

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio

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
    train_idx, test_idx, image_paths = ckpt["train_idx"], ckpt["test_idx"], ckpt["image_paths"]
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    for label, view_id in [("train[0]", int(train_idx[0])), ("train[last]", int(train_idx[-1])), ("test[0]", int(test_idx[0]))]:
        gt = torch.from_numpy(np.array(Image.open(image_paths[view_id]).convert("RGB"))).float().to(DEVICE) / 255.0
        with torch.no_grad():
            render, alpha, _ = gsplat.rasterization(
                means, quats, scales, opacities, colors, viewmats_all[view_id][None], K[None], width, height,
            )
        render = render[0].clamp(0, 1)
        p = float(psnr_metric(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        Image.fromarray((render.cpu().numpy() * 255).astype(np.uint8)).save(
            rf"D:\Downloads\powerfoam\artifacts\replica_room0_gsplat\debug_{label.replace('[','_').replace(']','')}_render.png")
        Image.fromarray((gt.cpu().numpy() * 255).astype(np.uint8)).save(
            rf"D:\Downloads\powerfoam\artifacts\replica_room0_gsplat\debug_{label.replace('[','_').replace(']','')}_gt.png")
        print(f"{label} (view {view_id}): PSNR={p:.3f} mean_alpha={float(alpha.mean()):.4f} "
              f"render_mean={float(render.mean()):.4f} gt_mean={float(gt.mean()):.4f}")


if __name__ == "__main__":
    main()
