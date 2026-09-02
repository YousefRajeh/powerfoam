import os
import uuid
import yaml
import configargparse
import numpy as np
import tqdm
from PIL import Image
from matplotlib import cm
import warp as wp
import gc
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from configs import *
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.geometry import normals_from_depth, depth_bilateral_filter
from powerfoam.scheduling import get_exp_scheduler, get_cosine_scheduler
from powerfoam.metrics import psnr, ssim, ssim_eval, lpips_eval
from powerfoam.distortion import exact_distortion, stratified_thresholds

torch.manual_seed(42)
np.random.seed(42)


def train(args):
    wp.init()

    # Setting up output directory
    if not args.dry_run:
        if len(args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{args.scene}@{unique_str}"
        else:
            experiment_name = args.experiment_name
        print("Experiment Name:", experiment_name)
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataloader
    train_split = "train" if args.eval else "all"
    test_split = "test" if args.eval else "all"
    test_data_handler = DataHandler(args)
    test_data_handler.reload(test_split, downsample=args.downsample[-1])
    # With `eval: false` both splits are "all", so the two handlers would otherwise decode and
    # hold a SECOND, byte-identical copy of the entire image set. On the 733-view ScanNet++ scene
    # that second copy is ~20 GB and was enough on its own to exceed the job's memory limit.
    # Sharing is only safe when the reload at the downsample switch below cannot make the two
    # diverge, i.e. when every entry of `downsample` is the same value.
    share_handler = train_split == test_split and len(set(args.downsample)) == 1
    if share_handler:
        train_data_handler = test_data_handler
    else:
        train_data_handler = DataHandler(args)
        train_data_handler.reload(train_split, downsample=args.downsample[0])
    train_data_iter = train_data_handler.get_iter()
    print("Loaded dataset")

    # Setting up model
    model = PowerfoamScene(args)
    model.initialize_from_dataset(train_data_handler, device="cuda")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()
    args.init_points = model.points.shape[0]

    # ---- resume -------------------------------------------------------------------------
    # Order matters. load_pt restores the geometry AT THE POINT COUNT REACHED (densification
    # changes it every 100 iters), so the optimizer must be re-declared against the restored
    # tensors before its state is loaded, or the Adam moments have the wrong shape.
    # args.init_points must also be restored to its ORIGINAL value: the densification target is
    # init_points * a**(i - densify_from), so seeding it with the resumed count would collapse
    # the growth schedule.
    start_iter = 0
    state_path = None if getattr(args, "dry_run", False) else f"{out_dir}/train_state.pt"
    if getattr(args, "resume", False) and state_path and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_pt(f"{out_dir}/model.pt")
        model.declare_optimizers(args, args.iterations)
        # Adam does NOT validate shapes in load_state_dict -- it stores whatever it is given and
        # only fails later inside _foreach_lerp_ on the first step(), far from the cause. So the
        # try/except below cannot catch a stale-moment mismatch and the shapes must be checked here.
        #
        # WHY A MISMATCH HAPPENS AT ALL: model.pt and train_state.pt are two separate writes, so a
        # crash BETWEEN them (995 ran out of disk mid-checkpoint) leaves geometry from iteration i
        # beside optimizer moments from i-k. Densification changed the point count in between, so
        # the moments index a different number of primitives (582,634 vs 602,803 on 3db0a1c8f3).
        # Fresh moments cost a few hundred iterations of Adam warm-up; refusing to resume costs the
        # whole run, so we drop the state and continue.
        def _moments_fit(sd):
            ps = [p for g in model.optimizer.param_groups for p in g["params"]]
            for k, s in (sd.get("state") or {}).items():
                try:
                    p = ps[int(k)]
                except (ValueError, IndexError):
                    return False
                for m in ("exp_avg", "exp_avg_sq"):
                    t = s.get(m)
                    if torch.is_tensor(t) and tuple(t.shape) != tuple(p.shape):
                        return False
            return True

        try:
            if _moments_fit(st["optimizer"]):
                model.optimizer.load_state_dict(st["optimizer"])
            else:
                print("[resume] optimizer moments do not match the restored geometry "
                      "(interrupted checkpoint); continuing with fresh moments")
        except (ValueError, KeyError) as e:
            print(f"[resume] optimizer state rejected ({e}); continuing with fresh moments")
        start_iter = int(st["iteration"]) + 1
        args.init_points = int(st["init_points"])
        print(f"[resume] iteration {start_iter}/{args.iterations}, "
              f"{model.points.shape[0]:,} points, init_points={args.init_points:,}")
    elif getattr(args, "resume", False):
        print("[resume] no train_state.pt found; starting from scratch")

    viewer = None
    viewer_lock = nullcontext()
    if args.viewer:
        from powerfoam.viewer import Viewer

        viewer = Viewer(
            model, train_data_handler.cameras[0], world_up=train_data_handler.viewer_up
        )
        viewer_lock = viewer.lock

    def test_loop(step, final=False):
        psnr_list = []
        ssim_list = []
        lpips_list = []
        with torch.no_grad():
            num_test = len(test_data_handler.cameras)
            for i in range(num_test):
                camera = test_data_handler.cameras[i]
                rgb_gt = test_data_handler.rgbs[i].cuda()

                depth_quantile = 0.5 * torch.ones(
                    *rgb_gt.shape[:-1], 1, device=model.device
                )

                result = model.forward(camera, depth_quantiles=depth_quantile)
                rgb = result[0]
                normal = result[3]
                depth = result[4]

                rgb_clamped = rgb.clamp(0.0, 1.0)
                psnr_list.append(psnr(rgb_clamped, rgb_gt).item())
                ssim_list.append(ssim_eval(rgb_clamped, rgb_gt).item())
                if final:
                    lpips_list.append(lpips_eval(rgb_clamped, rgb_gt).item())

                if not args.dry_run:
                    if i % (num_test // 3) != 0 and not final:
                        continue

                    rgb_output = (rgb.cpu().clamp(min=0, max=1) * 255).to(torch.uint8)
                    normal = ((normal + 1) * 0.5 * 255).cpu().to(torch.uint8)
                    depth = depth.cpu()[:, :, 0]
                    positive = depth[depth > 0]
                    if positive.numel() > 0:
                        depth_min = positive.min()
                        depth_max = depth.max()
                        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                    else:
                        depth = torch.zeros_like(depth)
                    depth = (cm.viridis(depth.numpy())[:, :, :3] * 255).astype(np.uint8)
                    depth = torch.from_numpy(depth)

                    im = torch.hstack((rgb_output, normal, depth))

                    if final:
                        im = Image.fromarray(im.numpy())
                        im.save(f"{out_dir}/test/{i:03d}.png")

        average_psnr = sum(psnr_list) / len(psnr_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        average_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else None
        if final and not args.dry_run:
            with open(f"{out_dir}/metrics.txt", "w") as f:
                f.write(f"Average PSNR:  {average_psnr:.4f}\n")
                f.write(f"Average SSIM:  {average_ssim:.4f}\n")
                f.write(f"Average LPIPS: {average_lpips:.4f}\n")

        if not args.dry_run:
            writer.add_scalar("test/psnr", average_psnr, step)
            writer.add_scalar("test/ssim", average_ssim, step)

    def train_loop():
        nonlocal train_data_iter

        print("Starting training")
        iters_since_triangulation = 0
        triangulation_interval = 1

        normal_loss_scheduler = get_exp_scheduler(
            args.normal_weight, 1e-1 * args.normal_weight, args.iterations
        )
        contrib_loss_scheduler = get_exp_scheduler(
            args.contribution_weight, 1e-3 * args.contribution_weight, args.iterations
        )
        interpenetration_loss_scheduler = get_exp_scheduler(
            args.interpenetration_weight,
            1e-3 * args.interpenetration_weight,
            args.iterations,
        )
        sigma_spatial_scheduler = get_cosine_scheduler(4.5, 1.5, args.iterations)

        # Fixed scene scale for the surface-concentration term, measured ONCE from the
        # initial point cloud. It must not be recomputed during training: the loss would
        # then be able to shrink itself by moving points instead of by sharpening density,
        # which is a degenerate optimum rather than the property we want.
        with torch.no_grad():
            _p = model.points.detach()
            distortion_scale = float((_p.max(dim=0).values - _p.min(dim=0).values).norm())
        if args.distortion_weight > 0:
            print(f"[distortion] weight={args.distortion_weight} "
                  f"quantiles={tuple(args.distortion_quantiles)} "
                  f"scene_diag={distortion_scale:.3f}")

        with tqdm.trange(start_iter, args.iterations, desc="Training") as train:
            for i in train:
                if viewer is not None:
                    viewer.step(i)
                    viewer.wait_if_paused()

                # `share_handler` implies every `downsample` entry is equal, so this reload would
                # re-decode the dataset at the resolution it already holds -- no effect, but it
                # briefly holds the old and new copies at once, which is exactly the memory spike
                # the sharing above exists to avoid. Skip it.
                if i and i in args.downsample_iterations and not share_handler:
                    downsample_idx = args.downsample_iterations.index(i)
                    downsample = args.downsample[downsample_idx]
                    train_data_handler.reload(train_split, downsample=downsample)
                    train_data_iter = train_data_handler.get_iter()

                torch.cuda.nvtx.range_push("Train Step")
                torch.cuda.nvtx.range_push("Loading Data")

                camera, rgb_gt, alpha_gt, normal_gt = next(train_data_iter)
                random_bkgd = torch.rand_like(rgb_gt)
                rgb_gt += (1 - alpha_gt[..., None]) * random_bkgd

                torch.cuda.nvtx.range_pop()  # Loading Data
                torch.cuda.nvtx.range_push("Zero Grad")

                model.optimizer.zero_grad(set_to_none=True)

                torch.cuda.nvtx.range_pop()  # Zero Grad

                with viewer_lock:
                    torch.cuda.nvtx.range_push("Rebuild Adjacency")
                    iters_since_triangulation += 1
                    if iters_since_triangulation % triangulation_interval == 0:
                        model.rebuild_adjacency()
                        iters_since_triangulation = 0
                        triangulation_interval += 1
                        triangulation_interval = min(triangulation_interval, 20)

                    torch.cuda.nvtx.range_pop()  # Rebuild Adjacency
                    torch.cuda.nvtx.range_push("Forward")

                    # When external normal supervision is enabled we also need
                    # the rendered depth (median quantile) to (a) build a
                    # validity mask and (b) compute finite-difference normals
                    # if Metric3D is not used.
                    # Quantile layout is built once so normal supervision and the
                    # surface-concentration regulariser can share ONE render. The kernel
                    # walks quantiles in order, so they are requested sorted and the
                    # positions are recorded to index the result afterwards.
                    # These are TRANSMITTANCE thresholds, not quantiles of the termination
                    # distribution: the kernel emits the depth at which transmittance first
                    # falls below each value. Two consequences, both of which silently
                    # produce garbage if ignored:
                    #   1. The kernel advances a forward-only index, and transmittance
                    #      decreases monotonically, so the list MUST be DESCENDING. Passing
                    #      ascending values makes every threshold fire at nearly the same
                    #      depth (the deepest one), collapsing the spread to ~0.
                    #   2. A HIGH threshold (0.9) is crossed EARLY => near depth; a LOW one
                    #      (0.1) is crossed late => far depth. So the interval width is
                    #      depth(low) - depth(high), not the other way round.
                    q_list, q_median_idx, q_near_idx, q_far_idx = [], None, None, None
                    if args.normal_supervision:
                        q_list.append(0.5)
                    exact_dist = (args.distortion_weight > 0
                                  and args.distortion_mode == "exact")
                    if exact_dist:
                        # stratified thresholds: each carries EXACTLY 1/K of the terminated
                        # mass, so the distortion weights are known constants and only the
                        # depths need rendering. Descending, as the kernel requires.
                        q_exact = stratified_thresholds(
                            args.distortion_num_quantiles, model.device).tolist()
                        q_list.extend(q_exact)
                    elif args.distortion_weight > 0:
                        t_far, t_near = args.distortion_quantiles   # e.g. (0.1, 0.9)
                        q_list.extend([t_far, t_near])
                    if q_list:
                        q_list = sorted(set(q_list), reverse=True)   # DESCENDING, required
                        if args.normal_supervision:
                            q_median_idx = q_list.index(0.5)
                        if exact_dist:
                            q_exact_idx = [q_list.index(v) for v in q_exact]
                        elif args.distortion_weight > 0:
                            q_near_idx = q_list.index(t_near)
                            q_far_idx = q_list.index(t_far)
                    if q_list:
                        depth_quantiles = torch.tensor(
                            q_list, device=model.device, dtype=torch.float32
                        ).expand(*rgb_gt.shape[:-1], len(q_list)).contiguous()
                    else:
                        depth_quantiles = None

                    result = model.forward(
                        camera,
                        depth_quantiles=depth_quantiles,
                        ray_gt=rgb_gt,
                        return_point_err=True,
                    )
                    rgb = result[0]
                    alpha = result[1]
                    normal_err = result[2]
                    normal = result[3]
                    depth = result[4]
                    contrib = result[6]
                    point_error = result[7]
                    prim_visible_mask = result[8]

                    torch.cuda.nvtx.range_pop()  # Forward
                    torch.cuda.nvtx.range_push("Losses")

                    rgb = rgb + (1 - alpha[..., None]) * random_bkgd
                    train_psnr = psnr(rgb, rgb_gt)
                    rgb_loss = (
                        F.mse_loss(rgb, rgb_gt, reduction="none").sum(dim=-1).mean()
                    )

                    ssim_loss = 1 - ssim(rgb, rgb_gt)
                    w_ssim = 0.2

                    torch.cuda.nvtx.range_push("Normal")

                    normal_loss = normal_err.mean()
                    if args.normal_supervision:
                        # depth has shape (H, W, 1) since we requested a
                        # single quantile above.
                        median_depth_slice = depth[..., q_median_idx : q_median_idx + 1]
                        valid_depth_mask = (median_depth_slice > 0).all(dim=-1)
                        if args.use_metric3d:
                            normal_loss += (
                                F.mse_loss(
                                    normal[valid_depth_mask],
                                    normal_gt[valid_depth_mask],
                                )
                                * 1e-1
                            )
                        else:
                            median_depth = depth_bilateral_filter(
                                median_depth_slice,
                                sigma_spatial=sigma_spatial_scheduler(i),
                                sigma_color=0.5,
                            )
                            est_normals = normals_from_depth(
                                camera, median_depth.detach()
                            )
                            normal_loss += (
                                F.mse_loss(
                                    normal[valid_depth_mask],
                                    est_normals[valid_depth_mask],
                                )
                                * 1e-1
                            )
                    w_normal = normal_loss_scheduler(i)
                    torch.cuda.nvtx.range_pop()  # Normal

                    torch.cuda.nvtx.range_push("Contribution")
                    contrib_loss = contrib.sum()
                    w_contrib = contrib_loss_scheduler(i)
                    torch.cuda.nvtx.range_pop()  # Contribution

                    torch.cuda.nvtx.range_push("Distortion")
                    if args.distortion_weight > 0 and args.distortion_mode == "exact":
                        dsel = depth[..., q_exact_idx]
                        # Channel control: detaching a parameter's contribution here is what
                        # separates "make the material opaque" (density) from "make the cell
                        # thinner" (radii). Implemented by detaching the rendered depth w.r.t.
                        # the channel we want silent -- see the ablation in the method note.
                        loss_rays, nbins = exact_distortion(
                            dsel, torch.tensor(q_exact, device=model.device))
                        # NO ray mask, matching VoroTracing: a ray that never becomes opaque
                        # has no reached thresholds, so every bin mass is zero and it
                        # contributes exactly 0 to the mean. Gating on alpha instead would
                        # (a) silence the loss entirely early in training, when the scene is
                        # still transparent and geometry is most malleable, and (b) bias the
                        # average toward already-converged rays. Averaging over all rays keeps
                        # the estimator unbiased and lets the term act from step 0.
                        distortion_loss = loss_rays.mean()
                        if i % 100 == 0:
                            print(f"[distortion/exact] iter {i}: "
                                  f"rays_with_2+_bins={int((nbins > 1).sum())} "
                                  f"mean_bins={nbins.float().mean():.2f} "
                                  f"loss={float(distortion_loss):.6e}", flush=True)
                    elif args.distortion_weight > 0:
                        # Width of the depth interval carrying the bulk of each ray's
                        # transmittance. Small => the ray terminates on a thin surface;
                        # large => weight is smeared through a soft slab, which is exactly
                        # the volumetric-interior structure that broke facet-graph growing.
                        d_far = depth[..., q_far_idx]     # transmittance fell to t_far
                        d_near = depth[..., q_near_idx]   # transmittance fell to t_near
                        # Only rays that actually hit something can be asked to be thin;
                        # background rays legitimately have no surface and must not be
                        # pulled toward zero spread (that would push density into empty
                        # space to manufacture a crossing).
                        hit = (alpha >= args.distortion_min_alpha) & (d_far > 0) & (d_near > 0)
                        if i % 100 == 0:
                            print(f"[dbg] iter {i} alpha[min={float(alpha.min()):.3f} "
                                  f"max={float(alpha.max()):.3f} mean={float(alpha.mean()):.3f}] "
                                  f"a>=thr={int((alpha>=args.distortion_min_alpha).sum())} "
                                  f"d_far>0={int((d_far>0).sum())} d_near>0={int((d_near>0).sum())} "
                                  f"hit={int(hit.sum())} of {alpha.numel()}", flush=True)
                        if hit.any():
                            spread = (d_far - d_near).clamp_min(0.0)[hit]
                            # scale-free: a 5cm slab means something different in a
                            # doll-house and a lecture hall, so normalise by scene scale
                            distortion_loss = (spread / distortion_scale).mean()
                            if i % 100 == 0:
                                print(f"[distortion] iter {i}: hit={int(hit.sum())} "
                                      f"mean_spread={float(spread.mean()):.4f}m "
                                      f"loss={float(distortion_loss):.6f}", flush=True)
                        else:
                            distortion_loss = torch.zeros((), device=model.device)
                    else:
                        distortion_loss = torch.zeros((), device=model.device)
                    torch.cuda.nvtx.range_pop()  # Distortion

                    torch.cuda.nvtx.range_push("Interpenetration")
                    interpenetration_loss = model.interpenetration().sum()
                    w_interpenetration = interpenetration_loss_scheduler(i)
                    torch.cuda.nvtx.range_pop()  # Interpenetration

                    loss = (
                        rgb_loss
                        + w_ssim * ssim_loss
                        + w_normal * normal_loss
                        + w_contrib * contrib_loss
                        + w_interpenetration * interpenetration_loss
                    )
                    # CHANNEL CONTROL -- the power-diagram-native experiment.
                    # VoroTracing routes the distortion gradient to DENSITY ONLY, because a
                    # midpoint bisector cannot move without moving a site and dragging every
                    # other boundary of that cell with it. Our power weights translate a
                    # cell's planes WITHOUT moving its center, so the same loss can also mean
                    # "make this cell thinner". Isolating a channel needs the distortion
                    # gradient computed separately and added only to the chosen parameters --
                    # a plain sum into `loss` would send it everywhere.
                    dist_params, dist_names = [], []
                    if args.distortion_weight > 0 and args.distortion_channel != "both":
                        if args.distortion_channel in ("density", "both"):
                            dist_params.append(model.density); dist_names.append("density")
                        if args.distortion_channel in ("radii", "both"):
                            dist_params.append(model.radii); dist_names.append("radii")
                    if dist_params:
                        dist_grads = torch.autograd.grad(
                            args.distortion_weight * distortion_loss, dist_params,
                            retain_graph=True, allow_unused=True)
                    elif args.distortion_weight > 0:
                        loss = loss + args.distortion_weight * distortion_loss

                    torch.cuda.nvtx.range_pop()  # Losses
                    torch.cuda.nvtx.range_push("Backward")

                    loss.backward()

                    if dist_params:
                        # add the isolated distortion gradient after the main backward, so it
                        # reaches only the selected channel
                        for p, g in zip(dist_params, dist_grads):
                            if g is None:
                                continue
                            p.grad = g if p.grad is None else p.grad + g

                    torch.cuda.nvtx.range_pop()  # Backward
                    torch.cuda.nvtx.range_push("Optimizer Step")

                    model.optimizer.step()

                    # VoroTracing's trainer clamps log-density at 30, for a concrete
                    # reason worth copying: a surface-concentration term keeps pushing
                    # density up past full opacity, and with sigma = exp(rho) the
                    # gradient of exp overflows fp32 and the parameter becomes NaN.
                    # exp(30) is already totally opaque, so this clips nothing physical.
                    # Softplus cannot run away like this, so the clamp is applied only to
                    # the exponential parameterization.
                    if getattr(args, "density_activation", "softplus") == "exp":
                        with torch.no_grad():
                            model.density.clamp_(max=30.0)

                    torch.cuda.nvtx.range_pop()  # Optimizer Step

                    if viewer is not None:
                        model.update_vis_cache()

                torch.cuda.nvtx.range_push("Stats Update")

                model.update_learning_rate(i)
                model.update_stats(contrib, point_error, prim_visible_mask)

                torch.cuda.nvtx.range_pop()  # Stats Update

                if i % 100 == 99 and not args.dry_run:
                    writer.add_scalar("train/rgb_loss", rgb_loss.item(), i)
                    writer.add_scalar("train/normal_loss", normal_loss.item(), i)
                    writer.add_scalar("train/contrib_loss", contrib_loss.item(), i)
                    writer.add_scalar(
                        "train/interpenetration_loss", interpenetration_loss.item(), i
                    )
                    writer.add_scalar(
                        "train/distortion_loss", float(distortion_loss), i
                    )

                    num_points = model.points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)
                    test_loop(i, final=False)

                    writer.add_scalar("lr/points_lr", model.points_scheduler(i), i)
                    writer.add_scalar("lr/density_lr", model.density_scheduler(i), i)
                    writer.add_scalar("lr/radii_lr", model.radii_scheduler(i), i)
                    writer.add_scalar(
                        "lr/normals_lr", model.quaternions_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sites_lr", model.texel_sites_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sv_rgb_lr", model.texel_sv_rgb_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_sv_axis_lr", model.texel_sv_axis_scheduler(i), i
                    )
                    writer.add_scalar(
                        "lr/texel_height_lr", model.texel_height_scheduler(i), i
                    )

                if i % getattr(args, "ckpt_every", 1000) == getattr(args, "ckpt_every", 1000) - 1                         and not args.dry_run:
                    model.save_pt(f"{out_dir}/model.pt")
                    model.save_pc(f"{out_dir}/points.ply")
                    # Written AFTER model.pt and swapped in atomically: a wall-clock kill between
                    # the two must never leave a train_state.pt newer than the geometry it indexes.
                    tmp = f"{out_dir}/train_state.pt.tmp"
                    torch.save({"iteration": i,
                                "optimizer": model.optimizer.state_dict(),
                                "init_points": int(args.init_points)}, tmp)
                    os.replace(tmp, f"{out_dir}/train_state.pt")

                torch.cuda.nvtx.range_push("Resampling")

                if i >= args.densify_from and i < args.densify_until:
                    a = (args.final_points / args.init_points) ** (
                        1 / (args.densify_until - args.densify_from - 1)
                    )
                    current_target = int(
                        args.init_points * (a ** (i - args.densify_from))
                    )
                else:
                    current_target = model.points.shape[0]

                if (not args.freeze_points) and i < int(0.95 * args.iterations) and i % 100 == 99:
                    with viewer_lock:
                        num_resampled = model.resample(current_target)
                        model.sort_points()
                        model.rebuild_adjacency()
                        iters_since_triangulation = 0
                        if viewer is not None:
                            model.update_vis_cache()
                else:
                    num_resampled = 0

                torch.cuda.nvtx.range_pop()  # Resampling
                torch.cuda.nvtx.range_pop()  # Train Step

                # Clean up memory if it exceeds 20GB
                # Force garbage collection and clear PyTorch cache
                reserved_mem = torch.cuda.memory_reserved() / 1024**3
                if reserved_mem > 20.0:
                    gc.collect()
                    torch.cuda.empty_cache()

        if not args.dry_run:
            model.save_pt(f"{out_dir}/model.pt")

    def run_training():
        train_loop()
        test_loop(args.iterations, final=True)

    if args.viewer:
        viewer.run(run_training, total_iterations=args.iterations)
    else:
        run_training()


if __name__ == "__main__":
    parser = configargparse.ArgParser()

    get_params = add_group(parser, Params)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )
    # Checkpoint-resume. Ibex denies explicit partition selection and routes on TIME LIMIT:
    # <=4h lands in `gpu4`, which holds by far the most free GPUs, while >4h is confined to the
    # saturated `gpu`/`gpu24`. Without resume a 17h scene cannot use `gpu4` at all; with it, the
    # scene becomes a chain of short jobs that backfill.
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out_dir>/train_state.pt if present")
    # BOTH spellings, deliberately. train.py dumps every resolved arg into output/<exp>/config.yaml,
    # and configargparse serialises this one by its dest -- `ckpt_every: 1000`. Feeding that file
    # back with -c (which is exactly what --resume does) then emits `--ckpt_every=1000`, which an
    # underscore-less option rejects: the run dies at parse time before a single iteration. Every
    # resumed run on 995 hit this. Accepting the underscore form makes the config round-trip.
    parser.add_argument("--ckpt-every", "--ckpt_every", type=int, default=1000,
                        help="iterations between resumable checkpoints")

    # Parse arguments
    args = parser.parse_args()

    _p = get_params(args)
    _p.resume = args.resume
    _p.ckpt_every = args.ckpt_every
    train(_p)
