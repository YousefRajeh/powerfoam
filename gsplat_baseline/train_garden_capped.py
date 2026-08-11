"""Matched-primitive-count rerun of the Experiment B baseline: same training
recipe as train_garden.py, but capped near PowerFoam's 1.2M primitive count
whenever densification pushes past it, by pruning the lowest-opacity
Gaussians back down to the budget. Isolates overlap from raw model capacity
-- the confound flagged (and left unrun) in the original Experiment B
writeup.
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.colmap import Parser  # noqa: E402

import gsplat  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402
from gsplat.strategy.ops import remove  # noqa: E402

DATA_DIR = r"D:\Downloads\powerfoam\data\mipnerf360\garden"
DOWNSAMPLE = 8
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30000"))
TARGET_MAX_GAUSSIANS = int(os.environ.get("TARGET_MAX_GAUSSIANS", "1200000"))
DEVICE = "cuda"


def build_split(num_images):
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def init_gaussians(points, points_rgb):
    n = points.shape[0]
    means = torch.from_numpy(points).float().to(DEVICE)
    dists = torch.cdist(means[: min(n, 20000)], means).topk(4, dim=-1, largest=False).values[:, 1:]
    mean_nn_dist = dists.mean()
    scales = torch.log(mean_nn_dist * torch.ones(n, 3, device=DEVICE)).clone()
    quats = torch.zeros(n, 4, device=DEVICE)
    quats[:, 0] = 1.0
    opacities = torch.logit(0.1 * torch.ones(n, device=DEVICE))
    colors = torch.from_numpy(points_rgb / 255.0).float().to(DEVICE).clamp(1e-4, 1 - 1e-4)
    colors = torch.logit(colors)

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    return torch.nn.ParameterDict(params).to(DEVICE)


@torch.no_grad()
def cap_to_budget(params, optimizers, state, target_max):
    """Prune the lowest-opacity Gaussians until at or under target_max. Same
    mechanism DefaultStrategy._prune_gs already uses (gsplat.strategy.ops.remove,
    which keeps params and optimizer state consistent), just driven by a
    count budget instead of an opacity/scale threshold."""
    n = params["means"].shape[0]
    if n <= target_max:
        return 0
    n_remove = n - target_max
    opacities = torch.sigmoid(params["opacities"].flatten())
    # Remove EXACTLY the n_remove lowest-opacity Gaussians by rank (argsort),
    # not by a value threshold. A threshold + "<=" comparison breaks under
    # ties: gsplat's own periodic opacity reset (reset_opa) sets a huge
    # cohort of Gaussians to the exact same value, and if the kth-smallest
    # value happens to land on that tied plateau, "<=" captures the WHOLE
    # plateau -- this is what silently emptied the model to 0 Gaussians in
    # an earlier run (removed 1,286,832 of 1,286,832 instead of ~86,832).
    # argsort-by-rank has no such failure mode regardless of ties.
    remove_idx = torch.argsort(opacities)[:n_remove]
    mask = torch.zeros(n, dtype=torch.bool, device=opacities.device)
    mask[remove_idx] = True
    assert int(mask.sum().item()) == n_remove
    remove(params=params, optimizers=optimizers, state=state, mask=mask)
    torch.cuda.empty_cache()
    return n_remove


def save_checkpoint(out_path, params, K, width, height, camtoworlds, train_idx, test_idx, image_paths, image_names, step):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    torch.save({
        "means": params["means"].detach().cpu(),
        "quats": params["quats"].detach().cpu(),
        "scales": params["scales"].detach().cpu(),
        "opacities": params["opacities"].detach().cpu(),
        "colors": params["colors"].detach().cpu(),
        "K": K.cpu(), "width": width, "height": height,
        "camtoworlds": camtoworlds, "train_idx": train_idx, "test_idx": test_idx,
        "image_paths": image_paths, "image_names": image_names,
        "step": step,
    }, tmp_path)
    tmp_path.replace(out_path)


def main():
    parser = Parser(data_dir=DATA_DIR, factor=DOWNSAMPLE, normalize=False, test_every=8)
    train_idx, test_idx = build_split(len(parser.image_names))
    print(f"[train_garden_capped] {len(parser.image_names)} images total: {len(train_idx)} train, "
          f"{len(test_idx)} test, target_max_gaussians={TARGET_MAX_GAUSSIANS}")

    camera_id = parser.camera_ids[0]
    K = torch.from_numpy(parser.Ks_dict[camera_id]).float().to(DEVICE)
    width, height = parser.imsize_dict[camera_id]

    camtoworlds = torch.from_numpy(parser.camtoworlds).float()
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)

    def load_image(i):
        img = torch.from_numpy(np.asarray(__import__("imageio.v2", fromlist=["imread"]).imread(parser.image_paths[i])[..., :3])).float() / 255.0
        return img.to(DEVICE)

    out_dir = Path(r"D:\Downloads\powerfoam\artifacts\garden_gsplat_capped")
    ckpt_path = out_dir / "ckpt.pt"
    resume_path = out_dir / "resume.pt"

    start_step = 0
    if resume_path.exists():
        # Optimizer momentum/variance isn't saved (only params), so resuming
        # re-warms Adam's state over the next few hundred steps rather than
        # continuing bit-exact -- acceptable to salvage the ~11.5k steps
        # already paid for rather than restart from scratch a third time.
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(ckpt[k].to(DEVICE)) for k in ("means", "quats", "scales", "opacities", "colors")
        }).to(DEVICE)
        start_step = int(ckpt["step"])
        print(f"[train_garden_capped] resuming from {resume_path} at step {start_step}, "
              f"n_gaussians={params['means'].shape[0]}", flush=True)
    else:
        params = init_gaussians(parser.points, parser.points_rgb)

    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {
        name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        for name, lr in lrs.items()
    }

    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    t0 = time.time()
    step = start_step
    try:
        for step in range(start_step, MAX_STEPS):
            i = int(train_idx[step % len(train_idx)])
            gt = load_image(i)[None]
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
            n_capped = cap_to_budget(params, optimizers, strategy_state, TARGET_MAX_GAUSSIANS)
            if n_capped:
                print(f"Step {step}: capped {n_capped} lowest-opacity GSs. Now having {params['means'].shape[0]} GSs.", flush=True)

            if step % 500 == 0 or step == MAX_STEPS - 1:
                elapsed = time.time() - t0
                print(f"[train_garden_capped] step {step:6d}/{MAX_STEPS} loss={float(loss):.4f} "
                      f"n_gaussians={params['means'].shape[0]} elapsed={elapsed:.1f}s", flush=True)

            # Resumable checkpoint every 2000 steps: if this run dies
            # silently (no Python traceback, e.g. an OS/driver-level kill --
            # which is what happened twice before this was added), the next
            # attempt can pick up from here instead of restarting from
            # scratch and re-hitting whatever step/condition triggered it.
            if step > 0 and step % 2000 == 0:
                save_checkpoint(resume_path, params, K, width, height, camtoworlds,
                                 train_idx, test_idx, parser.image_paths, parser.image_names, step)
                print(f"[train_garden_capped] checkpointed at step {step} -> {resume_path}", flush=True)
    except Exception:
        print(f"[train_garden_capped] CRASHED at step {step}:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise

    save_checkpoint(ckpt_path, params, K, width, height, camtoworlds,
                     train_idx, test_idx, parser.image_paths, parser.image_names, MAX_STEPS)
    print(f"[train_garden_capped] wrote {ckpt_path}, final n_gaussians={params['means'].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
