"""PSNR/SSIM for the LERF-OVS 3DGS arm, under PowerFoam's protocol.

Same control as eval_gs_psnr.py (ScanNet) and eval_gs_psnr_spp.py (ScanNet++): the foam arm has
numbers on all three benchmarks and the Gaussian arm has none, so no quality-matched comparison
exists. LERF is the only benchmark where BOTH arms are trained by us -- no public Gaussian
reconstruction of LERF-OVS existed.

Protocol is matched to foam, not chosen: the LERF foam configs set `eval: false` (=> split "all",
every view) and `downsample: [1, 1]`, and PSNR/SSIM come from powerfoam.metrics. The camera bridge
(K from the ray grid, viewmat from the loader's c2ws) is the one verified visually and against
COLMAP at 18.7-21.2 dB.
"""
import glob
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import

import configargparse
import numpy as np
import torch

from camera_bridge import K_from_ray_dirs
from configs import Params, add_group
from data_loader import DataHandler
from gsplat import rasterization
from powerfoam.metrics import psnr as pf_psnr, ssim_eval as pf_ssim

SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
GS = "recon_lerf_gs"
OUT = "artifacts/lerf_ovs/gs_psnr"


def find_ckpt(scene):
    c = sorted(glob.glob(f"{GS}/{scene}/ckpts/ckpt_*_rank0.pt"))
    return c[-1] if c else None


def eval_scene(scene):
    cfg = f"output/lerf_ovs_{scene}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    split = "test" if getattr(args, "eval", False) else "all"
    dh = DataHandler(args)
    dh.reload(split, downsample=args.downsample[-1])

    ck = torch.load(find_ckpt(scene), map_location="cuda", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    means, quats = sp["means"].cuda(), sp["quats"].cuda()
    scales = torch.exp(sp["scales"].cuda())
    opac = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
    sh = torch.cat([sp["sh0"].cuda(), sp["shN"].cuda()], 1)
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
    return {"scene": scene, "arm": "gs", "split": split, "views": len(ps),
            "P": int(means.shape[0]), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def main():
    os.makedirs(OUT, exist_ok=True)
    for scene in SCENES:
        dst = f"{OUT}/gs_{scene}.json"
        if os.path.exists(dst):
            print(f"[skip] {scene}", flush=True)
            continue
        if find_ckpt(scene) is None:
            print(f"[wait] {scene}: no checkpoint yet (training incomplete)", flush=True)
            continue
        try:
            r = eval_scene(scene)
        except Exception as e:
            print(f"[FAIL] {scene}: {type(e).__name__}: {e}", flush=True)
            continue
        json.dump(r, open(dst, "w"), indent=1)
        print(f"[ok] gs {scene:15s} P={r['P']:>9,} views={r['views']:>4d} "
              f"PSNR={r['psnr']:6.2f} SSIM={r['ssim']:.4f}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
