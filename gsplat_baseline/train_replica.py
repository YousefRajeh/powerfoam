"""3D Gaussian Splatting training on the Replica room_0 scene (Experiment D:
real open-vocabulary segmentation ground truth), matching the geometry setup
already validated for garden (same DefaultStrategy densification, same L1
plus SSIM loss). Loads poses directly from Replica's traj_w_c.txt (no
COLMAP/SfM available for this data) and, since there is no SfM point cloud
to initialize from, uses the same "scatter around camera positions" init
PowerFoam's ReplicaDataset config falls back to (random_unbounded) so
neither representation gets an unfairly better starting point.
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
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

import gsplat  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402

DATA_DIR = r"D:\Downloads\powerfoam\data\replica\room_0"
NATIVE_WIDTH, NATIVE_HEIGHT = 640, 480
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30000"))
INIT_POINTS = int(os.environ.get("INIT_POINTS", "20000"))
DEPTH_WEIGHT = float(os.environ.get("DEPTH_WEIGHT", "0.1"))
DEPTH_SCALE = 1.0 / 1000.0  # mm -> m, per the dataset's own readme
DEVICE = "cuda"


def build_split(num_images):
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def load_poses_and_images():
    rgb_dir = Path(DATA_DIR) / "rgb"
    traj_path = Path(DATA_DIR) / "traj_w_c.txt"
    num_frames = len(list(rgb_dir.glob("rgb_*.png")))
    poses = np.loadtxt(traj_path).reshape(-1, 4, 4).astype(np.float32)
    assert poses.shape[0] == num_frames, f"{poses.shape[0]} poses vs {num_frames} rgb frames"
    image_paths = [str(rgb_dir / f"rgb_{i}.png") for i in range(num_frames)]
    return torch.from_numpy(poses), image_paths


POINTCLOUD_PATH = r"D:\Downloads\powerfoam\data\replica\room_0\pointcloud.pt"


def load_shared_pointcloud(n):
    """The same depth-derived point cloud (build_replica_pointcloud.py) the
    PowerFoam side initializes from, subsampled to the same count -- an
    earlier version of this function did a blind gaussian scatter around
    camera positions instead (no depth used, since Replica ships no SfM),
    and that collapsed training to 0 Gaussians outright: too few scattered
    points ever landed near a real surface to survive gsplat's periodic
    opacity reset. Using the real, shared point cloud fixes the root cause
    and keeps both representations' starting point identical in kind, not
    just similar in spirit."""
    cloud = torch.load(POINTCLOUD_PATH, map_location=DEVICE, weights_only=True)
    points, colors = cloud["points"].to(DEVICE), cloud["colors"].to(DEVICE)
    if points.shape[0] > n:
        idx = torch.from_numpy(np.random.default_rng(0).choice(points.shape[0], size=n, replace=False))
        points, colors = points[idx], colors[idx]
    return points, colors


def init_gaussians(camtoworlds):
    means, point_colors = load_shared_pointcloud(INIT_POINTS)
    n = means.shape[0]
    dists = torch.cdist(means[: min(n, 20000)], means).topk(4, dim=-1, largest=False).values[:, 1:]
    mean_nn_dist = dists.mean().clamp_min(1e-3)
    scales = torch.log(mean_nn_dist * torch.ones(n, 3, device=DEVICE)).clone()
    quats = torch.zeros(n, 4, device=DEVICE)
    quats[:, 0] = 1.0
    opacities = torch.logit(0.1 * torch.ones(n, device=DEVICE))
    colors = torch.logit(point_colors.clamp(1e-4, 1 - 1e-4))

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    return torch.nn.ParameterDict(params).to(DEVICE)


def save_checkpoint(out_path, params, K, width, height, camtoworlds, train_idx, test_idx, image_paths, step):
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
        "image_paths": image_paths, "step": step,
    }, tmp_path)
    tmp_path.replace(out_path)


def main():
    camtoworlds, image_paths = load_poses_and_images()
    train_idx, test_idx = build_split(len(image_paths))
    print(f"[train_replica] {len(image_paths)} frames total: {len(train_idx)} train, {len(test_idx)} test", flush=True)

    width, height = NATIVE_WIDTH, NATIVE_HEIGHT
    focal = 0.5 * width  # hfov=90 deg, same convention as data_loader/replica.py on the PowerFoam side
    K = torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]], device=DEVICE)
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)

    depth_dir = Path(DATA_DIR) / "depth"

    def load_image(i):
        img = torch.from_numpy(np.asarray(Image.open(image_paths[i]).convert("RGB"))).float() / 255.0
        return img.to(DEVICE)

    def load_depth(i):
        d = np.asarray(Image.open(depth_dir / f"depth_{i}.png")).astype(np.float32) * DEPTH_SCALE
        return torch.from_numpy(d.copy()).to(DEVICE)

    out_dir = Path(r"D:\Downloads\powerfoam\artifacts\replica_room0_gsplat")
    ckpt_path = out_dir / "ckpt.pt"
    resume_path = out_dir / "resume.pt"

    start_step = 0
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(ckpt[k].to(DEVICE)) for k in ("means", "quats", "scales", "opacities", "colors")
        }).to(DEVICE)
        start_step = int(ckpt["step"])
        print(f"[train_replica] resuming from {resume_path} at step {start_step}, n_gaussians={params['means'].shape[0]}", flush=True)
    else:
        params = init_gaussians(camtoworlds)

    # A first attempt copied garden's absolute LR values unchanged and
    # produced a visibly hazy, floater-covered reconstruction (train-view
    # PSNR ~16dB despite a training loss that looked normal) with the loss
    # oscillating rather than settling -- the standard signature of a
    # position/scale learning rate that's too aggressive relative to this
    # scene's actual physical extent. LR_SCALE lets that be dialed down
    # without duplicating this whole script; scene_scale is also now
    # computed from the real camera extent instead of hardcoded to 1.0
    # (DefaultStrategy's grow/prune size thresholds are relative to it).
    lr_scale = float(os.environ.get("LR_SCALE", "1.0"))
    lrs = {"means": 1.6e-4 * lr_scale, "scales": 5e-3 * lr_scale, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15) for name, lr in lrs.items()}

    camera_centers = camtoworlds[:, :3, 3]
    scene_scale = float(torch.linalg.norm(camera_centers.std(dim=0)) * 3.0)
    print(f"[train_replica] lr_scale={lr_scale} scene_scale={scene_scale:.4f}", flush=True)

    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    t0 = time.time()
    step = start_step
    try:
        for step in range(start_step, MAX_STEPS):
            i = int(train_idx[step % len(train_idx)])
            gt = load_image(i)[None]
            gt_depth = load_depth(i)[None]
            viewmat = viewmats_all[i][None]

            means, quats, scales, opacities, colors = (
                params["means"], params["quats"], torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
            )
            # RGB+ED: expected depth (per-pixel weighted mean of contributing
            # Gaussians' depths, sum_i w_i z_i / sum_i w_i) as a 4th output
            # channel. Real depth supervision is standard practice for every
            # 3DGS-style method trained on Replica specifically (SplaTAM,
            # GS-SLAM, MonoGS, etc.) -- Replica's camera trajectory is
            # "purely randomly" sampled per its own readme, not a smooth
            # continuous path, which is a harder viewpoint distribution than
            # vanilla 3DGS is usually validated on; depth loss is the
            # established stabilizer for exactly this setting, not a
            # bespoke workaround.
            renders, alphas, info = gsplat.rasterization(
                means, quats, scales, opacities, colors, viewmat, K[None], width, height,
                packed=True, absgrad=strategy.absgrad, render_mode="RGB+ED",
            )
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

            rgb_render, depth_render = renders[..., :3], renders[..., 3]
            l1 = F.l1_loss(rgb_render, gt)
            ssim_val = ssim(rgb_render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
            valid_depth = gt_depth > 0
            depth_l1 = F.l1_loss(depth_render[valid_depth], gt_depth[valid_depth]) if valid_depth.any() else torch.zeros((), device=DEVICE)
            loss = 0.8 * l1 + 0.2 * (1 - ssim_val) + DEPTH_WEIGHT * depth_l1

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers.values():
                opt.step()

            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

            if step % 500 == 0 or step == MAX_STEPS - 1:
                elapsed = time.time() - t0
                print(f"[train_replica] step {step:6d}/{MAX_STEPS} loss={float(loss):.4f} "
                      f"n_gaussians={params['means'].shape[0]} elapsed={elapsed:.1f}s", flush=True)

            if step > 0 and step % 2000 == 0:
                save_checkpoint(resume_path, params, K, width, height, camtoworlds, train_idx, test_idx, image_paths, step)
                print(f"[train_replica] checkpointed at step {step} -> {resume_path}", flush=True)
    except Exception:
        print(f"[train_replica] CRASHED at step {step}:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise

    save_checkpoint(ckpt_path, params, K, width, height, camtoworlds, train_idx, test_idx, image_paths, MAX_STEPS)
    print(f"[train_replica] wrote {ckpt_path}, final n_gaussians={params['means'].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
