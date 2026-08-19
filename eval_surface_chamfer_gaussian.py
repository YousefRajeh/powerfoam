"""Gaussian-baseline counterpart of eval_surface_chamfer.py -- same TSDF -> mesh -> Chamfer
pipeline, so foam and Gaussian geometry are measured by one extractor.

Depth: gsplat's `render_mode="RGB+ED"` gives EXPECTED depth (accumulated depth / alpha) in
camera-space z, plus the accumulated alpha used for masking. Note the honest asymmetry with
the foam script, which uses the MEDIAN (0.5-transmittance) depth: vanilla gsplat exposes no
median-depth mode (2DGS ships its own kernel for that). For well-converged opaque surfaces
the two nearly coincide, and our own quantile sweep on foam moved CD-L1 by <0.1cm across
q=0.3..0.5, so the difference is small relative to the effect being measured -- but it is a
protocol difference and should be stated wherever these numbers are reported.

Everything downstream (TSDF voxel/truncation, alpha mask, mesh sampling, Chamfer with
accuracy/completeness/F-score) is shared with the foam script by construction.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rajehyl/splat-distiller")
sys.path.insert(0, "/home/rajehyl/powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import open3d as o3d
from gsplat import rasterization
from gsplat_ext import Parser, Dataset

from eval_surface_chamfer import chamfer
from run_cluster_classify_eval import SCENES


def render_median_depth(means, quats, scales, opacities, colors, sh_degree,
                        viewmat, K, W, H, n_slices=48, near=0.1, far=6.0):
    """Median depth (2DGS convention) for gsplat, which exposes no median mode.

    Exact definition: the depth at which accumulated transmittance first falls below 0.5,
    i.e. the smallest t with A(t) >= 0.5 where A is accumulated alpha. gsplat composites
    Gaussians sorted by camera-space center depth, so restricting the render to Gaussians
    with center depth < t yields precisely A(t). Sweeping t over `n_slices` thresholds
    therefore samples the true A(t) curve, and since A is monotone the per-pixel crossing
    is recovered by linear interpolation between the bracketing slices -- second-order
    accurate, no CUDA kernel required (2DGS ships one; GOF avoids the issue with a level
    set).

    Returns (median_depth HxW, valid HxW in {0,1}). A pixel has a well-defined median depth
    exactly when its accumulated alpha reaches 0.5 somewhere in the sweep, so `valid` is that
    crossing indicator -- NOT the alpha of the last slice evaluated, which would drop
    legitimately-crossed pixels (an earlier version did that and completeness collapsed to
    19cm).
    """
    device = means.device
    zc = (means @ viewmat[0, :3, :3].T + viewmat[0, :3, 3])[:, 2]      # camera-space z
    ts = torch.linspace(near, far, n_slices, device=device)
    ones = torch.ones((means.shape[0], 1), device=device)

    A_prev = torch.zeros((H, W), device=device)
    t_prev = torch.full((H, W), near, device=device)
    med = torch.zeros((H, W), device=device)
    done = torch.zeros((H, W), dtype=torch.bool, device=device)
    A_last = torch.zeros((H, W), device=device)

    for t in ts:
        m = zc < t
        if not bool(m.any()):
            continue
        _, ra, _ = rasterization(
            means=means[m], quats=quats[m], scales=torch.exp(scales[m]),
            opacities=torch.sigmoid(opacities[m]), colors=ones[m],
            viewmats=viewmat, Ks=K[None], width=W, height=H,
            sh_degree=None, render_mode="RGB", packed=False)
        A = ra[0, ..., 0]
        cross = (~done) & (A >= 0.5)
        if bool(cross.any()):
            # linear interpolation of the 0.5 crossing between the bracketing slices
            denom = (A - A_prev).clamp_min(1e-8)
            frac = ((0.5 - A_prev) / denom).clamp(0.0, 1.0)
            med = torch.where(cross, t_prev + frac * (t - t_prev), med)
            done = done | cross
        A_prev, t_prev, A_last = A, torch.full_like(t_prev, float(t)), A
        if bool(done.all()):
            break
    return med, done.float()


def load_splats(ckpt_path, device):
    """splat-distiller .pt ({'splats': {...}}) or a graphdeco-style .ply."""
    if str(ckpt_path).endswith(".ply"):
        from gsplat_ext import GaussianPrimitive
        prim = GaussianPrimitive()
        prim.from_file(str(ckpt_path))
        prim.to(device)
        g, c = prim.geometry, prim.color
        return (g["means"], g["quats"], g["scales"], g["opacities"], c["colors"],
                int(g.get("sh_degree", 3)))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)["splats"]
    colors = torch.cat([ck["sh0"], ck["shN"]], dim=1)
    return (ck["means"], ck["quats"], ck["scales"], ck["opacities"], colors, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0062_00")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--sdf-trunc", type=float, default=0.08)
    p.add_argument("--depth-trunc", type=float, default=6.0)
    p.add_argument("--min-alpha", type=float, default=0.5)
    p.add_argument("--n-sample", type=int, default=1_000_000)
    p.add_argument("--thresh", type=float, default=0.05)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--depth-mode", choices=["expected", "median"], default="median",
                   help="median = 2DGS convention (transmittance crossing 0.5), computed by "
                        "alpha sweep since gsplat has no median mode; expected = gsplat RGB+ED")
    p.add_argument("--n-slices", type=int, default=48)
    p.add_argument("--out-mesh", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scene = args.scene
    ckpt = args.ckpt or f"/home/rajehyl/gaussian_baseline_scannet/{scene}/ckpts/ckpt_29999_rank0.pt"
    data_dir = args.data_dir or f"/home/rajehyl/powerfoam/data/scannet/{scene}_colmap"

    means, quats, scales, opacities, colors, sh_degree = load_splats(ckpt, device)
    print(f"[{scene}] {means.shape[0]} gaussians from {Path(ckpt).name}", flush=True)

    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=100000)
    ds = Dataset(parser, split="train")
    n = len(ds) if args.max_views is None else min(args.max_views, len(ds))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    n_used = 0
    for i in range(n):
        d = ds[i]
        c2w = d["camtoworld"].to(device).float()
        K = d["K"].to(device).float()
        H, W = int(d["image"].shape[0]), int(d["image"].shape[1])
        viewmat = torch.linalg.inv(c2w)[None]           # world -> camera
        if args.depth_mode == "median":
            depth, alpha = render_median_depth(
                means, quats, scales, opacities, colors, sh_degree,
                viewmat, K, W, H, n_slices=args.n_slices, far=args.depth_trunc)
        else:
            rc, ra, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities), colors=colors,
                viewmats=viewmat, Ks=K[None], width=W, height=H,
                sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
            depth = rc[0, ..., -1]                       # expected depth, camera-space z
            alpha = ra[0, ..., 0]
        dep = torch.where(alpha >= args.min_alpha, depth, torch.zeros_like(depth))
        dep_np = dep.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(dep_np).any() or (dep_np > 0).sum() < 100:
            continue

        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        extr = viewmat[0].detach().cpu().numpy().astype(np.float64)
        color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(np.nan_to_num(dep_np)), depth_scale=1.0,
            depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, extr)
        n_used += 1
        if n_used % 25 == 0:
            print(f"  fused {n_used}", flush=True)

    print(f"[{scene}] fused {n_used} {args.depth_mode}-depth views")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    print(f"  mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    if args.out_mesh:
        o3d.io.write_triangle_mesh(args.out_mesh, mesh)
    if len(mesh.triangles) == 0:
        print("  EMPTY MESH")
        return

    rec = np.asarray(mesh.sample_points_uniformly(number_of_points=args.n_sample).points)
    split = SCENES[scene]
    gt = np.load(Path(args.gt_root) / split / scene / "coord.npy").astype(np.float64)
    res = chamfer(rec, gt, args.thresh)
    print(f"[{scene}] GAUSSIAN: acc={res['accuracy']*100:.2f}cm comp={res['completeness']*100:.2f}cm "
          f"CD-L1={res['chamfer_l1']*100:.2f}cm F@{args.thresh*100:.0f}cm={res['fscore']:.3f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"scene": scene, "ckpt": str(ckpt), "views_fused": n_used,
                       "n_gaussians": int(means.shape[0]), "full": res}, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
