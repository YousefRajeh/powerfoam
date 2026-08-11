"""Minimal 3D Gaussian Splatting training for the garden scene (Experiment B
baseline). Uses plain gsplat's public API (rasterization + DefaultStrategy)
plus splat-distiller's own COLMAP Parser for data loading, per the standard
train/test split convention already used for the PowerFoam side
(sorted-by-filename, index % 8 == 0 held out) so both representations are
evaluated against the identical 24 held-out views.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.colmap import Parser  # noqa: E402

import gsplat  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402

import os

DATA_DIR = r"D:\Downloads\powerfoam\data\mipnerf360\garden"
DOWNSAMPLE = 8
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30000"))
DEVICE = "cuda"


def build_split(num_images):
    """Same convention as PowerFoam's export_feature_operator.py /
    accumulate_feature_stats.py: sorted-by-filename index, test = idx % 8 == 0
    (not gsplat's own off-by-one `(idx+1) % test_every`), so both
    representations share the exact same 24 held-out views."""
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def init_gaussians(points, points_rgb, scene_scale):
    n = points.shape[0]
    means = torch.from_numpy(points).float().to(DEVICE)
    # Nearest-neighbor-distance scale init (standard 3DGS heuristic): each
    # Gaussian starts about as big as its local point spacing.
    dists = torch.cdist(means[: min(n, 20000)], means).topk(4, dim=-1, largest=False).values[:, 1:]
    mean_nn_dist = dists.mean()
    scales = torch.log(mean_nn_dist * torch.ones(n, 3, device=DEVICE)).clone()
    quats = torch.zeros(n, 4, device=DEVICE)
    quats[:, 0] = 1.0
    opacities = torch.logit(0.1 * torch.ones(n, device=DEVICE))
    colors = torch.from_numpy(points_rgb / 255.0).float().to(DEVICE).clamp(1e-4, 1 - 1e-4)
    colors = torch.logit(colors)  # store as pre-sigmoid like scales/opacities

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    return torch.nn.ParameterDict(params).to(DEVICE)


def main():
    parser = Parser(data_dir=DATA_DIR, factor=DOWNSAMPLE, normalize=False, test_every=8)
    train_idx, test_idx = build_split(len(parser.image_names))
    print(f"[train_garden] {len(parser.image_names)} images total: {len(train_idx)} train, {len(test_idx)} test")

    camera_id = parser.camera_ids[0]
    K = torch.from_numpy(parser.Ks_dict[camera_id]).float().to(DEVICE)
    width, height = parser.imsize_dict[camera_id]

    camtoworlds = torch.from_numpy(parser.camtoworlds).float()
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)

    def load_image(i):
        img = torch.from_numpy(np.asarray(__import__("imageio.v2", fromlist=["imread"]).imread(parser.image_paths[i])[..., :3])).float() / 255.0
        return img.to(DEVICE)

    params = init_gaussians(parser.points, parser.points_rgb, 1.0)
    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {
        name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        for name, lr in lrs.items()
    }

    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    t0 = time.time()
    for step in range(MAX_STEPS):
        i = int(train_idx[step % len(train_idx)])
        gt = load_image(i)[None]  # (1, H, W, 3)
        viewmat = viewmats_all[i][None]

        means, quats, scales, opacities, colors = (
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
        )
        renders, alphas, info = gsplat.rasterization(
            means, quats, scales, opacities, colors, viewmat, K[None], width, height,
            packed=True, absgrad=strategy.absgrad,
        )
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = F.l1_loss(renders, gt)
        ssim_val = ssim(renders.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = 0.8 * l1 + 0.2 * (1 - ssim_val)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()

        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

        if step % 500 == 0 or step == MAX_STEPS - 1:
            elapsed = time.time() - t0
            print(f"[train_garden] step {step:6d}/{MAX_STEPS} loss={float(loss):.4f} "
                  f"n_gaussians={means.shape[0]} elapsed={elapsed:.1f}s")

    out_dir = Path(r"D:\Downloads\powerfoam\artifacts\garden_gsplat")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "means": params["means"].detach().cpu(),
        "quats": params["quats"].detach().cpu(),
        "scales": params["scales"].detach().cpu(),
        "opacities": params["opacities"].detach().cpu(),
        "colors": params["colors"].detach().cpu(),
        "K": K.cpu(), "width": width, "height": height,
        "camtoworlds": camtoworlds, "train_idx": train_idx, "test_idx": test_idx,
        "image_paths": parser.image_paths, "image_names": parser.image_names,
        "step": MAX_STEPS,
    }, out_dir / "ckpt.pt")
    print(f"[train_garden] wrote {out_dir / 'ckpt.pt'}, final n_gaussians={params['means'].shape[0]}")


if __name__ == "__main__":
    main()
