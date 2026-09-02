"""Are the dipole normals oriented, and does a flipped one put matter in front of the surface?

THE GEOMETRY, read out of the code rather than assumed. `ray_plane_intersect` returns
`dp = dot(n, dir)` (rendering_math.py:133), and the rasterizer clips the cell chord with

    dp >= 0:  t_far  = min(t_surf, t_far)     matter lies BEFORE the surface
    dp <  0:  t_near = max(t_surf, t_near)    matter lies AFTER  the surface

Both branches put the solid half on the -n side. So `n` IS the inside/outside flag: it points out
of the matter, and the dipole plane is the boundary.

THE CONSEQUENCE NOBODY CHECKED. For a ray striking a genuine outward-facing surface, the outward
normal points back toward the camera, so `dp < 0` and the ray ENTERS matter at the dipole. If a
front-hit cell instead has `dp >= 0`, its normal is FLIPPED relative to the viewing ray: the ray
EXITS at the surface, which means that cell's matter sat in front of its own surface. That is
precisely the haze signature measured in OPEN_ISSUES Addendum 3 -- 83.6% of far surface sitting in
front of GT, median gap +6.6 cm.

WHY A FLIP IS EVEN POSSIBLE. Nothing supervises orientation: `normal_supervision: false` in every
ScanNet config, quaternions are initialised randomly (`scene.py`), and the only pressure on a
normal is the rendering gradient. A cell buried in the interior contributes nothing to any image,
so its orientation is unconstrained -- and ~90% of cells are interior non-owners.

WHAT THIS MEASURES, per pixel, using the front-most primitive the rasterizer already reports:
  frac(dp >= 0)     share of visible hits whose normal faces AWAY from the camera (flipped)
  and the same split by whether the hit lands near GT or far from it -- if flips concentrate in the
  far population, the mechanism and the haze are the same thing.

A NULL RESULT IS INFORMATIVE TOO. If flips are rare, or are spread evenly across near and far hits,
then normal orientation is not what produces the stand-off and this line closes.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\Downloads\powerfoam")

import warp as wp

from data_loader import DataHandler
from determinism import enable_determinism
from powerfoam.rasterize import VisOptions
from powerfoam.scene import PowerfoamScene
from run_front_surface import I_ALPHA, I_FRONT_PRIM, I_FRONT_T_ENTRY, VARIANT, load_cargs, ray_maps
from run_percell_masked import SPLIT

GT_ROOT = r"D:\Downloads\scannet_pointcept"


def quat_normal(q):
    """Same normal the renderer uses (scene.py:get_normals)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="pf_nonfroz", choices=list(VARIANT))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--min-alpha", type=float, default=0.5)
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    ckpt = f"output/scannet_{a.scene}_{VARIANT[a.recon]}"

    wp.init()
    cargs = load_cargs(ckpt)
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt}/model.pt")
    model.update_vis_cache()

    m = torch.load(f"{ckpt}/model.pt", map_location="cpu", weights_only=False)
    N = quat_normal(m["quaternions"].float()).to(dev)

    vo = VisOptions()
    vo.transmittance_threshold = 1e-3
    vo.max_intersections = 1024
    vo.depth_quantile = 0.5
    vo.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    G = np.load(os.path.join(GT_ROOT, SPLIT[a.scene], a.scene, "coord.npy")).astype(np.float64)
    tg = cKDTree(G)

    dps, dists = [], []
    n = min(a.max_views, len(dh.cameras))
    for i in range(n):
        cam = dh.cameras[i]
        out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vo)
        alpha, prim, t = out[I_ALPHA], out[I_FRONT_PRIM], out[I_FRONT_T_ENTRY]
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        if t.ndim == 3:
            t = t[..., 0]
        rm = ray_maps(cam, dev)
        sl = (slice(None, None, a.stride), slice(None, None, a.stride))
        o, d = rm[..., 0:3][sl], rm[..., 3:6][sl]
        alpha, prim, t = alpha[sl], prim[sl].long(), t[sl]
        keep = (prim >= 0) & (alpha >= a.min_alpha) & (t > 0)
        if not keep.any():
            continue
        dd = d[keep]
        dd = dd / dd.norm(dim=-1, keepdim=True)
        dp = (N[prim[keep]] * dd).sum(-1)                     # dot(n, ray_dir)
        pts = (o + d * t[..., None])[keep].detach().cpu().numpy().astype(np.float64)
        dps.append(dp.detach().cpu().numpy())
        dists.append(tg.query(pts, k=1, workers=-1)[0])

    dp = np.concatenate(dps)
    dist = np.concatenate(dists)
    far = dist > 0.05
    print(f"\n[{a.recon}/{a.scene}] {len(dp):,} front-hit samples over {n} views\n")
    print(f"  flipped overall (dp >= 0)        {100*(dp >= 0).mean():6.2f}%")
    print(f"    among hits NEAR GT (<5cm)      {100*(dp[~far] >= 0).mean():6.2f}%   n={int((~far).sum()):,}")
    print(f"    among hits FAR from GT (>5cm)  {100*(dp[far] >= 0).mean():6.2f}%   n={int(far.sum()):,}")
    print(f"\n  median |dp| (1 = normal parallel to ray, 0 = grazing) {np.median(np.abs(dp)):.3f}")
    for lo, hi, lab in ((0.05, 1.01, "far"), (0.0, 0.05, "near")):
        s = (dist >= lo) & (dist < hi) if lab == "near" else (dist >= lo)
        if s.sum():
            print(f"  median surf->GT among {lab:4s} flipped={np.median(dist[s & (dp >= 0)])*100:6.2f} cm"
                  f"   aligned={np.median(dist[s & (dp < 0)])*100:6.2f} cm")


if __name__ == "__main__":
    main()
