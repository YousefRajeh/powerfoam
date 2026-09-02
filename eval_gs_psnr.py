"""Reconstruction quality (PSNR/SSIM) for the 3DGS ScanNet arms, under PowerFoam's EXACT protocol.

WHY THIS IS NEEDED. A8 reports 14.8x fewer contributing primitives per ray for foam at an identical
primitive budget. That is only meaningful if the two reconstructions are of comparable quality --
otherwise "fewer contributors" could simply mean "sparser, worse field". No 3DGS PSNR exists on any
of the three benchmarks (`recon_remote/gs_*/` holds only ckpt.pt), so the control was missing.

THE PROTOCOL MUST MATCH FOAM'S, NOT A REASONABLE DEFAULT:
  * split = "all", NOT a held-out test split. The ScanNet configs set `eval: false`, which makes
    train.py use test_split="all" -- so foam's published PSNR is over ALL views. Scoring 3DGS on a
    held-out split would compare a generalisation number against a fit number.
  * downsample = args.downsample[-1], the same value foam's test loop uses.
  * the same `powerfoam.metrics.psnr` / `ssim_eval` functions, imported rather than reimplemented.
  * rgb clamped to [0,1] before scoring, as train.py does.

CAMERA BRIDGE. K is derived from the ray grid foam actually traversed (matches COLMAP cameras.bin to
4 dp, 4.3e-05 px pinhole residual); viewmat is the inverse of the loader's own c2ws, which are 3x4
and padded. Verified visually and at 18.7-21.2 dB before use -- reconstructing either by hand gave a
render that scored 5 dB against what turned out to be a BLACK ground truth.

ACTIVATIONS. scales are stored in log space and opacities as logits; both must be activated. Passing
them raw produced near-transparent Gaussians and collapsed the A8 ray counts from 31.5 to 1.07.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import (DLL search path)

import configargparse
import numpy as np
import torch

from camera_bridge import K_from_ray_dirs
from configs import Params, add_group
from data_loader import DataHandler
from gsplat import rasterization
from powerfoam.metrics import psnr as pf_psnr, ssim_eval as pf_ssim

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
OUT = "artifacts/scannet/gs_psnr"


def eval_arm(scene, arm):
    cfg = f"output/scannet_{scene}_nonfrozen/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    # `eval: false` in these configs => foam's own test loop uses split "all"; mirror it exactly
    split = "test" if getattr(args, "eval", False) else "all"
    dh = DataHandler(args)
    dh.reload(split, downsample=args.downsample[-1])

    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cuda", weights_only=False)
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
    return {"scene": scene, "arm": arm, "split": split, "views": len(ps),
            "P": int(means.shape[0]), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "psnr_min": float(np.min(ps)), "psnr_max": float(np.max(ps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gs_froz,gs_unfroz")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for scene in a.scenes.split(","):
        for arm in a.arms.split(","):
            dst = f"{OUT}/{arm}_{scene}.json"
            if os.path.exists(dst):
                print(f"[skip] {arm}/{scene}", flush=True)
                continue
            try:
                r = eval_arm(scene, arm)
            except Exception as e:
                print(f"[FAIL] {arm}/{scene}: {type(e).__name__}: {e}", flush=True)
                continue
            json.dump(r, open(dst, "w"), indent=1)
            print(f"[ok] {arm:10s} {scene}  P={r['P']:>9,}  views={r['views']:>4d}  "
                  f"PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
