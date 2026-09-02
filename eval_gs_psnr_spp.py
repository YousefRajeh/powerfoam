"""Reconstruction quality (PSNR/SSIM) for the ScanNet++ refbench 3DGS arm, under foam's protocol.

Companion to eval_gs_psnr.py (ScanNet). Same control, same reason: the ScanNet++ foam arm reports
PSNR 35.26 over 12 scenes and the 3DGS arm reports nothing, so no quality-matched comparison exists.

TWO DIFFERENCES FROM THE SCANNET SCRIPT, both in loading rather than protocol:
  * the reconstruction is a .ply (`scene_point_cloud.ply`), not a ckpt.pt.
  * SH live in the ply as f_dc_{0..2} + f_rest_{0..44}. The 3DGS ply convention stores f_rest
    CHANNEL-MAJOR -- (3, 15) flattened -- so it must be read as (N, 3, 15) and transposed to
    (N, 15, 3) before concatenating with f_dc. Reading it as (N, 15, 3) directly scrambles colour
    across SH bands and produces a plausible-but-wrong render.

Protocol is unchanged and still matched to foam: the ScanNet++ configs also set `eval: false`
(=> split "all", every view) and `downsample: [1, 1]`, and PSNR/SSIM come from powerfoam.metrics.
Scales and opacity are stored raw in the ply and need exp()/sigmoid(), as in the ScanNet arm.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import

import configargparse
import numpy as np
import torch
from plyfile import PlyData

from camera_bridge import K_from_ray_dirs
from configs import Params, add_group
from data_loader import DataHandler
from gsplat import rasterization
from powerfoam.metrics import psnr as pf_psnr, ssim_eval as pf_ssim

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064",
          "27dd4da69e", "3db0a1c8f3", "9071e139d9", "d755b3d9d8", "e7af285f7d", "09c1414f1b"]
GS = r"D:\Downloads\refbench_3dgs_12scenes\output"
RECON = r"D:\Downloads\spp_results\full"
OUT = "artifacts/scannetpp_gs/psnr"


def load_ply(scene):
    p = os.path.join(GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                     "scene_point_cloud.ply")
    v = PlyData.read(p)["vertex"]
    g = lambda k: torch.from_numpy(np.asarray(v[k]).astype(np.float32))
    means = torch.stack([g("x"), g("y"), g("z")], 1)
    scales = torch.exp(torch.stack([g(f"scale_{i}") for i in range(3)], 1))
    quats = torch.stack([g(f"rot_{i}") for i in range(4)], 1)
    quats = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    opac = torch.sigmoid(g("opacity").reshape(-1))
    dc = torch.stack([g(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]      # (N, 1, 3)
    n_rest = len([n for n in v.data.dtype.names if n.startswith("f_rest_")])
    rest = torch.stack([g(f"f_rest_{i}") for i in range(n_rest)], 1)
    # channel-major on disk: (N, 3, n_rest//3) -> transpose to (N, n_rest//3, 3)
    rest = rest.reshape(-1, 3, n_rest // 3).transpose(1, 2)
    sh = torch.cat([dc, rest], 1)
    return means.cuda(), quats.cuda(), scales.cuda(), opac.cuda(), sh.cuda()


def eval_scene(scene):
    cfg = os.path.join(RECON, f"spp_pf_unfroz_{scene}", "config.yaml")
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    split = "test" if getattr(args, "eval", False) else "all"
    dh = DataHandler(args)
    dh.reload(split, downsample=args.downsample[-1])

    means, quats, scales, opac, sh = load_ply(scene)
    sh_deg = int(round(sh.shape[1] ** 0.5)) - 1

    ps, ss = [], []
    for i, cam in enumerate(dh.cameras):
        K, _ = K_from_ray_dirs(cam)
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :4] = dh.c2ws[i].double()
        vm = torch.linalg.inv(c2w).float().cuda()
        W, H = int(cam.width), int(cam.height)
        out, _, _ = rasterization(means, quats, scales, opac, sh,
                                  vm[None], K.cuda()[None], W, H, sh_degree=sh_deg)
        rgb = out[0].clamp(0.0, 1.0)
        gt = dh.rgbs[i].cuda()[..., :3]
        ps.append(float(pf_psnr(rgb, gt)))
        ss.append(float(pf_ssim(rgb, gt)))
        del out, rgb, gt
    return {"scene": scene, "arm": "gs_refbench", "split": split, "views": len(ps),
            "P": int(means.shape[0]), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "psnr_min": float(np.min(ps)), "psnr_max": float(np.max(ps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for scene in a.scenes.split(","):
        dst = f"{OUT}/gs_refbench_{scene}.json"
        if os.path.exists(dst):
            print(f"[skip] {scene}", flush=True)
            continue
        try:
            r = eval_scene(scene)
        except Exception as e:
            print(f"[FAIL] {scene}: {type(e).__name__}: {e}", flush=True)
            continue
        json.dump(r, open(dst, "w"), indent=1)
        print(f"[ok] gs_refbench {scene}  P={r['P']:>9,}  views={r['views']:>4d}  "
              f"PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
