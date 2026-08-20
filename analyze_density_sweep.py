"""Analyze the density-activation x learning-rate sweep.

Measures the three quantities that decide whether sigma = exp(rho) is worth adopting, in
priority order:

1. **INTERIOR-NON-OWNER FRACTION** -- the fraction of valid cells that own ZERO ground-truth
   points. This is the number that closed the whole foam-exclusive clustering direction:
   at ~90% under softplus, the facet graph is a solid volume rather than a surface, so
   geodesic FPS (-5 mIoU), radius-weighted seeding (-5.4) and coherence-gated growing
   (-12.3, catastrophic on 3/10 scenes) all grew blobs through object interiors. If exp
   drops this materially, those methods deserve a retest; if it does not, they stay closed
   and the negative result stands on its own.

2. **PSNR / SSIM** -- read from the run's own metrics.txt. Thin surfaces can always be
   bought by degrading appearance, so a spread improvement that costs image quality is not
   a win. This is the guard against fooling ourselves.

3. **OPACITY BIMODALITY** -- VoroTracing reports that the correct parameterization makes
   density "strongly bimodal: cells are either near-transparent or near-opaque". Per cell we
   form alpha_i = 1 - exp(-sigma_i * 2 r_i), using twice the power radius as the
   characteristic chord through the cell, and report the mass in the tails vs the middle.
   A representation whose cells are decisively empty-or-solid is one where "is this cell on
   a surface" is a trained property instead of a threshold we guess per scene.

The radius exponent k (support ~ r^k, measured 1.98 +/- 0.30 under softplus) is NOT computed
here: it needs a full SAM+CLIP accumulation pass per checkpoint. Run it only for the winning
configuration -- but run it, because if softplus gradients were size-biased then part of that
"surface law" is an optimization artifact and the paper framing has to change.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch

from point_cloud_query import assign_points_to_power_cells
from evaluate_point_cloud_miou import load_scannet_pointcept_gt
from run_cluster_classify_eval import SCENES


def read_metrics(run_dir):
    f = Path(run_dir) / "metrics.txt"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip().replace("Average ", "").lower()] = float(v)
            except ValueError:
                pass
    return out


def load_cells(run_dir, device):
    """Return centers, radii, and RAW density -- we need the raw parameter plus the run's
    own activation to reconstruct sigma, since that is exactly what the sweep varies."""
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{run_dir}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{run_dir}/model.pt")
    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    sigma = model.get_density().detach().cpu().numpy()      # activation applied by the model
    return centers, radii, sigma, getattr(args, "density_activation", "softplus")


def analyze(run_dir, scene, gt_root, device="cuda"):
    centers, radii, sigma, act = load_cells(run_dir, device)
    split = SCENES[scene]
    gt_points, _, _ = load_scannet_pointcept_gt(f"{gt_root}/{split}/{scene}", "segment20")

    valid = np.ones(centers.shape[0], dtype=bool)
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
    owned = assigned[assigned >= 0]
    n_cells = int(centers.shape[0])
    owner_cells = int(np.unique(owned).size)
    non_owner_frac = 1.0 - owner_cells / max(n_cells, 1)

    # per-cell opacity proxy over a chord of 2r (the cell's own scale)
    alpha = 1.0 - np.exp(-np.clip(sigma.reshape(-1) * 2.0 * radii.reshape(-1), 0, 60))
    near_empty = float((alpha < 0.1).mean())
    near_solid = float((alpha > 0.9).mean())
    middle = float(((alpha >= 0.1) & (alpha <= 0.9)).mean())

    m = read_metrics(run_dir)
    return {
        "run": Path(run_dir).name, "activation": act, "n_cells": n_cells,
        "owner_cells": owner_cells,
        "interior_non_owner_frac": non_owner_frac,
        "gt_points_owned_frac": float((assigned >= 0).mean()),
        "alpha_near_empty": near_empty, "alpha_near_solid": near_solid,
        "alpha_middle": middle, "bimodality": near_empty + near_solid,
        "sigma_median": float(np.median(sigma)), "sigma_max": float(sigma.max()),
        "psnr": m.get("psnr"), "ssim": m.get("ssim"), "lpips": m.get("lpips"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--scene", default="scene0062_00")
    p.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rows = []
    for r in a.runs:
        if not (Path(r) / "model.pt").exists():
            print(f"[skip] {r}: no model.pt")
            continue
        try:
            rows.append(analyze(r, a.scene, a.gt_root))
            print(f"[ok] {r}", flush=True)
        except Exception as e:
            print(f"[fail] {r}: {type(e).__name__}: {e}", flush=True)

    rows.sort(key=lambda x: (x["activation"], -(x["psnr"] or 0)))
    hdr = (f"{'run':<30}{'act':<10}{'PSNR':>7}{'SSIM':>7}{'non-owner%':>12}"
           f"{'empty%':>8}{'solid%':>8}{'mid%':>7}{'bimod%':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<30}{r['activation']:<10}"
              f"{(r['psnr'] or 0):>7.2f}{(r['ssim'] or 0):>7.4f}"
              f"{r['interior_non_owner_frac']*100:>12.1f}"
              f"{r['alpha_near_empty']*100:>8.1f}{r['alpha_near_solid']*100:>8.1f}"
              f"{r['alpha_middle']*100:>7.1f}{r['bimodality']*100:>8.1f}")
    print("\nnon-owner% is THE number: ~90% under softplus is what closed the "
          "foam-exclusive clustering direction.")
    json.dump(rows, open(a.output, "w"), indent=2)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
