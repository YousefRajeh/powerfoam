"""Extract the visible dipole surface the way the RENDERER defines it, and measure it.

THE IDEA. Every previous extraction sampled each cell's dipole face on a grid and then tried to
filter the result down to the visible part. All four filters failed the same way -- support,
density, alpha, and a depth-agreement visibility test -- because "is this face real surface" is not
a property of a cell. A cell's face spans its whole power cell; only the patch a ray terminates on
is surface, and which patch that is depends on the view.

So stop filtering and read the surface off the renderer directly. For every pixel the rasterizer
already tracks the primitive with the largest `alpha * trans` (`front_prim_idx`). This adds the two
distances that go with it, both taken from the same iteration that wins that argmax:

    front_t_surf    where the ray meets that primitive's DISPLACED DIPOLE PLANE
    front_t_entry   where the ray ENTERS its matter -- the segment's t_near after both the
                    power-facet clipping and the dipole clipping

They are not the same thing and the difference is the point of reporting both. The dipole clips the
cell's chord into a solid half and an empty half (rasterize.py: `t_far = min(t_surf, t_far)` when
dp >= 0, `t_near = max(t_surf, t_near)` when dp < 0), so the ray enters matter AT the dipole only
when the dipole is the entry face. When the ray instead enters through a power facet, the boundary
it crossed belongs to the neighbouring cell, and `t_surf` then lies somewhere inside solid matter
rather than on the visible surface.

WHY THIS IS NOT THE DEPTH-FUSION DETOUR WE CRITICISE. `depth_out` is a transmittance QUANTILE --
the analytic crossing of `depth_quantile`, which for a cell several centimetres thick lands inside
it. That is what made the previous visibility test mushy. These are closed-form ray/surface
intersections in primitive space, and each carries the id of the primitive it came from, so a point
inherits that primitive's semantic label for free. No TSDF, no marching cubes, no depth image
treated as geometry.

DENSITY IS CONTROLLABLE HERE, which the grid extraction could not manage. Points come out at one
per pixel per view, so `--stride` subsamples pixels on a fixed lattice and the count can be matched
to the reference cloud -- the confound that invalidated the previous surface comparison (a 5-50x
denser predicted set makes mean-distance-to-nearest fall for free).

REPORTED BOTH WAYS, because a filter that throws surface away improves accuracy trivially while
ruining completeness, which is exactly how every previous attempt failed:
    surf->GT   accuracy     does the surface we emit sit on the truth
    GT->surf   completeness does the truth get covered
Both must move in the right direction, or the extraction is just smaller.
"""
import argparse
import json
import os
import sys
import time

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
from run_percell_masked import SPLIT

GT_ROOT = r"D:\Downloads\scannet_pointcept"
VARIANT = {"pf_tfroz": "truefrozen", "pf_nonfroz": "nonfrozen", "pf_expact": "nonfrozen_expact", "pf_exp01": "nonfrozen_exp01", "pf_voro": "nonfrozen_voro"}
FEATURES = {"pf_tfroz": "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt",
            "pf_nonfroz": "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt"}

# indices into the visualize() tuple; the last two are appended by this work
I_ALPHA, I_FRONT_PRIM, I_FRONT_T_SURF, I_FRONT_T_ENTRY = 3, 7, 8, 9


def load_cargs(ckpt_dir):
    """Parse a checkpoint's config.yaml, dropping keys whose value is `null`.

    train.py writes back every Params field, including optional ones it left as None, which YAML
    renders as `null`. configargparse then feeds the literal string "null" to an `int` argument and
    dies (`--max_image_width: invalid int value: 'null'`). Older configs predate those fields, so
    this only bites checkpoints trained recently -- i.e. every one of the exp-activation reruns.
    Dropping the line restores the argparse default, which is what None meant in the first place.

    `parse_known_args` for the same class of reason: the written config also carries trainer-only
    keys (`ckpt_every`, `resume`) that `add_group(Params)` does not register, and which are
    irrelevant to loading a finished model. Unknown keys are ignored rather than fatal.
    """
    import tempfile
    src = os.path.join(ckpt_dir, "config.yaml")
    kept = [ln for ln in open(src, encoding="utf-8").read().splitlines()
            if ln.split(":", 1)[-1].strip() not in ("null", "None")]
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs, _unknown = parser.parse_known_args(["-c", tmp])
    os.unlink(tmp)
    return cargs


def ray_maps(cam, dev):
    rm = cam.ray_maps
    if rm is None:
        rm = cam._build_pinhole_ray_maps()
    return rm.to(dev)


def extract(model, cameras, dev, stride=2, min_alpha=0.5, max_views=None):
    """Returns dict of variant -> (points (S,3) float64, owner primitive ids (S,) int64)."""
    vis_options = VisOptions()
    vis_options.transmittance_threshold = 1e-3
    vis_options.max_intersections = 1024
    vis_options.depth_quantile = 0.5      # unused for these outputs, but must not be left 0.0
    vis_options.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    acc = {"t_surf": ([], []), "t_entry": ([], [])}
    n = len(cameras) if max_views is None else min(max_views, len(cameras))
    for i in range(n):
        cam = cameras[i]
        out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis_options)
        alpha = out[I_ALPHA]
        prim = out[I_FRONT_PRIM]
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        rm = ray_maps(cam, dev)
        o, d = rm[..., 0:3], rm[..., 3:6]
        sl = (slice(None, None, stride), slice(None, None, stride))
        o, d = o[sl], d[sl]
        a, pr = alpha[sl], prim[sl].long()
        for name, idx in (("t_surf", I_FRONT_T_SURF), ("t_entry", I_FRONT_T_ENTRY)):
            t = out[idx]
            if t.ndim == 3:
                t = t[..., 0]
            t = t[sl]
            keep = (pr >= 0) & (a >= min_alpha) & (t > 0)
            if not keep.any():
                continue
            pts = (o + d * t[..., None])[keep]
            acc[name][0].append(pts.detach().cpu().numpy().astype(np.float64))
            acc[name][1].append(pr[keep].detach().cpu().numpy())
    return {k: (np.concatenate(v[0]) if v[0] else np.zeros((0, 3)),
                np.concatenate(v[1]) if v[1] else np.zeros(0, dtype=np.int64))
            for k, v in acc.items()}


def geometry(pts, G, tg, rng, n_match=None):
    """Accuracy/completeness against the GT cloud, optionally at a matched point count."""
    if len(pts) == 0:
        return None
    p = pts
    if n_match is not None and len(p) > n_match:
        p = p[rng.choice(len(p), n_match, replace=False)]
    dp, _ = tg.query(p, k=1, workers=-1)
    dg, _ = cKDTree(p).query(G, k=1, workers=-1)
    return {"n": int(len(p)),
            "surf2gt_med_cm": float(np.median(dp) * 100),
            "surf2gt_mean_cm": float(dp.mean() * 100),
            "frac_beyond_5cm": float((dp > 0.05).mean()),
            "gt2surf_med_cm": float(np.median(dg) * 100),
            "gt_within_2cm": float((dg < 0.02).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="pf_nonfroz", choices=list(VARIANT))
    ap.add_argument("--stride", type=int, default=2, help="pixel subsample lattice")
    ap.add_argument("--min-alpha", type=float, default=0.5)
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--match-gt", action="store_true",
                    help="also report at a point count matched to the GT cloud")
    ap.add_argument("--out", default="artifacts/scannet/front_surface.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    rng = np.random.default_rng(0)
    ckpt_dir = f"output/scannet_{a.scene}_{VARIANT[a.recon]}"

    wp.init()
    cargs = load_cargs(ckpt_dir)
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt_dir}/model.pt")
    model.update_vis_cache()

    t0 = time.time()
    got = extract(model, dh.cameras, dev, a.stride, a.min_alpha, a.max_views)
    G = np.load(os.path.join(GT_ROOT, SPLIT[a.scene], a.scene, "coord.npy")).astype(np.float64)
    tg = cKDTree(G)
    print(f"[{a.recon}/{a.scene}] {len(dh.cameras)} views, stride {a.stride}, "
          f"GT {len(G):,} pts  ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for name, (pts, own) in got.items():
        for tag, nm in ([(name, None)] + ([(name + "@matched", len(G))] if a.match_gt else [])):
            g = geometry(pts, G, tg, rng, nm)
            if g is None:
                print(f"  {tag:18s} EMPTY"); continue
            g.update({"scene": a.scene, "recon": a.recon, "variant": tag, "stride": a.stride,
                      "n_unique_prims": int(np.unique(own).size)})
            rows.append(g)
            print(f"  {tag:18s} n={g['n']:9,d}  surf->GT med={g['surf2gt_med_cm']:6.2f} cm  "
                  f">5cm={g['frac_beyond_5cm']*100:5.1f}%  GT->surf med={g['gt2surf_med_cm']:5.2f} cm"
                  f"  GT<2cm={g['gt_within_2cm']*100:5.1f}%", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, "w"), indent=1)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
