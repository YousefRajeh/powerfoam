"""Eval-only PSNR/SSIM/LPIPS for existing PowerFoam checkpoints, writing metrics.txt.

WHY. Ten ScanNet reconstructions finished training but never had their final eval written (9 of 10
`nonfrozen`, 1 `truefrozen`) -- the model.pt exists, metrics.txt does not. train.py has no
eval-only flag, and re-training to recover a number that the checkpoint already determines would be
absurd. This runs train.py's `test_loop(final=True)` path against the saved checkpoint.

TRANSCRIBED FROM train.py:141-196, NOT reimplemented. Every detail that moves the number is kept:
  * split: `test` if args.eval else `all` -- the ScanNet configs set `eval: false`, so foam's
    published PSNR is over ALL views. Silently using a held-out split would make these numbers
    incomparable with the ones already in metrics.txt.
  * downsample = args.downsample[-1] (the test-loop value, not the train value).
  * depth_quantiles = 0.5 * ones(H, W, 1), which train.py passes to model.forward.
  * rgb.clamp(0,1) BEFORE scoring, and psnr/ssim_eval/lpips_eval imported from powerfoam.metrics.
  * the same "Average PSNR:  {:.4f}" formatting, so downstream parsers see one format.

The model is loaded exactly as export_feature_operator.py does -- initialize_from_dataset, then
load_pt, and deliberately no sort_points()/resample() -- so primitive indices stay stable.
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import configargparse
import torch
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.metrics import lpips_eval, psnr, ssim_eval
from powerfoam.scene import PowerfoamScene

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def eval_one(out_dir):
    cfg = os.path.join(out_dir, "config.yaml")
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])

    split = "test" if getattr(args, "eval", False) else "all"
    dh = DataHandler(args)
    dh.reload(split, downsample=args.downsample[-1])

    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(os.path.join(out_dir, "model.pt"))

    ps, ss, lp = [], [], []
    with torch.no_grad():
        for i in range(len(dh.cameras)):
            cam = dh.cameras[i]
            rgb_gt = dh.rgbs[i].cuda()
            dq = 0.5 * torch.ones(*rgb_gt.shape[:-1], 1, device=model.device)
            rgb = model.forward(cam, depth_quantiles=dq)[0].clamp(0.0, 1.0)
            ps.append(psnr(rgb, rgb_gt).item())
            ss.append(ssim_eval(rgb, rgb_gt).item())
            lp.append(lpips_eval(rgb, rgb_gt).item())
            del rgb, rgb_gt, dq
    a_psnr, a_ssim = sum(ps) / len(ps), sum(ss) / len(ss)
    a_lpips = sum(lp) / len(lp)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR:  {a_psnr:.4f}\n")
        f.write(f"Average SSIM:  {a_ssim:.4f}\n")
        f.write(f"Average LPIPS: {a_lpips:.4f}\n")
    return a_psnr, a_ssim, a_lpips, len(ps), split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default="truefrozen,nonfrozen")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    wp.init()
    for recon in a.recons.split(","):
        for scene in a.scenes.split(","):
            out_dir = f"output/scannet_{scene}_{recon}"
            if not os.path.isfile(os.path.join(out_dir, "model.pt")):
                continue
            if os.path.isfile(os.path.join(out_dir, "metrics.txt")):
                print(f"[skip] {recon}/{scene}", flush=True)
                continue
            try:
                pv, sv, lv, n, split = eval_one(out_dir)
            except Exception as e:
                print(f"[FAIL] {recon}/{scene}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"[ok] {recon:11s} {scene}  split={split:5s} views={n:>4d}  "
                  f"PSNR={pv:6.2f}  SSIM={sv:.4f}  LPIPS={lv:.4f}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
