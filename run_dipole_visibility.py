"""Visibility pass for the dipole surface: keep only the surface a ray actually terminates on.

THE PROBLEM THIS SOLVES. Extracting a dipole surface from every solid cell over-produces badly:
70.5% of the samples land more than 5 cm from any ScanNet GT vertex, and the extracted area is
185.8 m^2 (448.9 m^2 unfrozen) for a room. The cause is that the foam is opaque almost everywhere
-- alpha at L=2r has median 0.891 and p90 = 1.000 -- so essentially every cell owns a dipole face,
and most of those faces are INTERIOR: inside walls and furniture, behind whatever is visible.

WHY NO PER-CELL SCALAR FIXES IT, measured before writing this. Thresholding on accumulated
lifting weight (`support`, the operator's own alpha*trans sum) or on per-cell alpha both trade
completeness away without buying accuracy -- support>p90 gives 78.2% beyond 5 cm, WORSE than the
70.5% baseline, while GT->surface degrades 1.16 -> 6.86 cm. Even the top 1% of cells by support is
89.5% beyond 5 cm. Cell selection is the wrong frame, because a single cell's face is partly
visible and partly interior: the face spans the whole power cell, and only the patch a ray
terminates on is real surface. Visibility is a property of a SAMPLE and a VIEW, not of a cell.

THE TEST. Render the median-depth map per training view (the same quantity 2DGS fuses, which
PowerFoam computes analytically), then project every surface sample into every view and keep it
where its own ray distance agrees with the rendered depth. A sample kept in at least one view is
front-most somewhere, and is therefore surface rather than interior.

    keep(sample) = OR over views of [ in frustum AND alpha >= min_alpha
                                      AND | ||p - eye|| - depth(pixel) | <= tol ]

TWO CONVENTIONS THAT ARE EASY TO GET WRONG, both inherited from eval_surface_chamfer.py rather
than re-derived here:
  * The renderer marches along NORMALISED ray directions, so its depth is ray distance from the
    eye, not planar z. Samples are therefore compared with ||p - eye||, not with z_cam.
  * `VisOptions` is a warp struct that zero-initialises, and the default built inside
    `Rasterizer.visualize()` never sets `depth_quantile`. Leaving it 0.0 makes the transmittance
    crossing never fire and the depth render silently all-zero. It is set explicitly.

The alpha mask is not cosmetic: without it every pixel gets a depth, including rays through empty
space that merely accumulate enough density to cross the quantile.

TOLERANCE IS A KNOB AND IS SWEPT. `--tols` takes several values; a verdict that only holds at one
tolerance is not a verdict. Reported against the two things that must move together: the excess
(surface->GT, and the fraction beyond 5 cm) must fall WITHOUT completeness (GT->surface) rising,
since throwing surface away trivially improves the first and ruins the second -- which is exactly
how the per-cell filters failed.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import warp as wp
import configargparse

from configs import Params, add_group
from data_loader import DataHandler
from determinism import enable_determinism
from powerfoam.rasterize import VisOptions
from powerfoam.scene import PowerfoamScene
from run_dipole_surface import DENSITY_THRESH, extract_surface
from run_percell_masked import SPLIT

GT_ROOT = r"D:\Downloads\scannet_pointcept"
VARIANT = {"pf_tfroz": "truefrozen", "pf_nonfroz": "nonfrozen"}
FEATURES = {"pf_tfroz": "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt",
            "pf_nonfroz": "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt"}


def visible_mask(model, cameras, pts, tols, min_alpha=0.5, quantile=0.5, max_views=None,
                 dev="cuda", chunk=4_000_000):
    """Boolean (len(tols), S): sample was front-most in at least one view, per tolerance."""
    vis_options = VisOptions()
    vis_options.transmittance_threshold = 1e-3
    vis_options.max_intersections = 1024
    vis_options.depth_quantile = quantile          # 0.0 would silently render all-zero depth
    vis_options.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    P = torch.as_tensor(pts, dtype=torch.float32, device=dev)
    S = P.shape[0]
    keep = torch.zeros(len(tols), S, dtype=torch.bool, device=dev)
    n = len(cameras) if max_views is None else min(max_views, len(cameras))
    for i in range(n):
        cam = cameras[i]
        out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis_options)
        depth, alpha = out[1], out[3]
        if depth.ndim == 3:
            depth = depth[..., 0]
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        H, W = depth.shape
        prm = cam.to_open3d()
        K = torch.as_tensor(np.asarray(prm.intrinsic.intrinsic_matrix), dtype=torch.float32,
                            device=dev)
        E = torch.as_tensor(np.asarray(prm.extrinsic), dtype=torch.float32, device=dev)
        R, t = E[:3, :3], E[:3, 3]
        eye = -R.T @ t
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for s in range(0, S, chunk):
            p = P[s:s + chunk]
            xc = p @ R.T + t                       # world -> camera
            z = xc[:, 2]
            front = z > 1e-6
            u = (fx * xc[:, 0] / z.clamp_min(1e-6) + cx).round().long()
            v = (fy * xc[:, 1] / z.clamp_min(1e-6) + cy).round().long()
            inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inside.any():
                continue
            idx = torch.nonzero(inside).squeeze(1)
            uu, vv = u[idx], v[idx]
            d_ren = depth[vv, uu]
            a_ren = alpha[vv, uu]
            # renderer depth is RAY DISTANCE from the eye, not planar z
            d_smp = (p[idx] - eye).norm(dim=-1)
            ok_alpha = (a_ren >= min_alpha) & (d_ren > 0)
            for k, tol in enumerate(tols):
                hit = ok_alpha & ((d_smp - d_ren).abs() <= tol)
                keep[k, s + idx[hit]] = True
    return keep.cpu().numpy(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="pf_nonfroz", choices=list(VARIANT))
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--tols", default="0.01,0.02,0.05")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--min-alpha", type=float, default=0.5)
    ap.add_argument("--depth-quantile", type=float, default=0.5)
    ap.add_argument("--out", default="artifacts/scannet/dipole_visibility.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    tols = [float(x) for x in a.tols.split(",")]
    ckpt_dir = f"output/scannet_{a.scene}_{VARIANT[a.recon]}"

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt_dir}/model.pt")
    model.update_vis_cache()
    cameras = dh.cameras

    m = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu", weights_only=False)
    dens = F.softplus(m["density"].float().to(dev), beta=100)
    valid = torch.load(FEATURES[a.recon].format(s=a.scene), map_location=dev,
                       weights_only=True)["valid_mask"].to(dev)
    pts, own, area = extract_surface(m, (dens > DENSITY_THRESH) & valid, a.grid, dev)
    del m
    print(f"[{a.recon}/{a.scene}] {len(pts):,} samples, {area:.1f} m^2, "
          f"{len(cameras)} views", flush=True)

    keep, n_views = visible_mask(model, cameras, pts, tols, a.min_alpha, a.depth_quantile,
                                 a.max_views, dev)

    G = np.load(os.path.join(GT_ROOT, SPLIT[a.scene], a.scene, "coord.npy")).astype(np.float64)
    tg = cKDTree(G)
    rows = []
    for tag, sel in [("all", np.ones(len(pts), bool))] + \
                    [(f"visible@{t*100:g}cm", keep[k]) for k, t in enumerate(tols)]:
        if sel.sum() == 0:
            print(f"  {tag:16s} EMPTY"); continue
        sp = pts[sel]
        dp, _ = tg.query(sp, k=1, workers=-1)
        dg, _ = cKDTree(sp).query(G, k=1, workers=-1)
        rec = {"scene": a.scene, "recon": a.recon, "arm": tag, "n_samples": int(sel.sum()),
               "frac_kept": float(sel.mean()), "area_m2": float(area * sel.mean()),
               "surf2gt_med_cm": float(np.median(dp) * 100),
               "frac_beyond_5cm": float((dp > 0.05).mean()),
               "gt2surf_med_cm": float(np.median(dg) * 100),
               "gt_within_2cm": float((dg < 0.02).mean())}
        rows.append(rec)
        print(f"  {tag:16s} kept={rec['frac_kept']*100:5.1f}%  area={rec['area_m2']:7.1f} m2  "
              f"surf->GT={rec['surf2gt_med_cm']:6.2f} cm  >5cm={rec['frac_beyond_5cm']*100:5.1f}%  "
              f"GT->surf={rec['gt2surf_med_cm']:5.2f} cm  GT<2cm={rec['gt_within_2cm']*100:5.1f}%",
              flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, "w"), indent=1)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
