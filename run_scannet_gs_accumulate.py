"""Accumulate feature-lifting stats for the ScanNet 3DGS arms -- the ONLY thing blocking NormLift.

WHY. NormLift's confidence is c_i = ||f_i|| * N_eff/(N_eff + 1) with
N_eff(i) = (sum_s W_i^s)^2 / sum_s (W_i^s)^2, the Kish effective sample size over VIEWS. That needs
per-view weights, i.e. the accumulator state. `artifacts/scannet/*/stats_*.pt` exists for foam only;
no stats_gs_* exists anywhere, so run_baselines.py reports the NormLift arm as blocked rather than
substituting a proxy.

This is the same FFL_STATS_OUT replay path already used for ScanNet++
(run_spp_gs_reaccumulate.py), pointed at the ScanNet data and the recon_remote checkpoints. It
inherits the three fixes that made that path survive: lean stats (drops the four accumulators the
sidecar marks invalid for replay-built stats), in-place per-view reuse, and the compact
columns_unique path -- together they took the largest ScanNet++ scene from OOM at 44 GiB to
completion at ~31 GB, all verified bitwise identical for the solvers we use.

Run:  python run_scannet_gs_accumulate.py [--arms gs_unfroz,gs_froz]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3"
ART = "artifacts/scannet"

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gs_unfroz,gs_froz")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()

    for scene in a.scenes.split(","):
        src = os.path.join(DATA, f"{scene}_colmap")
        if not os.path.isdir(src):
            print(f"[miss] {scene}: no colmap dir at {src}", flush=True)
            continue
        for arm in a.arms.split(","):
            ck = f"recon_remote/{arm}/{scene}/ckpt.pt"
            if not os.path.exists(ck):
                print(f"[miss] {arm}/{scene}: no ckpt", flush=True)
                continue
            out = os.path.abspath(f"{ART}/{scene}/stats_{arm}_ogl3.pt")
            if os.path.exists(out):
                print(f"[skip] {arm}/{scene}: stats exist", flush=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            t0 = time.time()
            print(f"[accumulate] {arm}/{scene}", flush=True)
            env = dict(os.environ, FFL_STATS_OUT=out,
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            r = subprocess.run(
                [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", os.path.abspath(ck),
                 "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"],
                env=env, capture_output=True, text=True, cwd=r"D:\Downloads\splat-distiller")
            if not os.path.exists(out):
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                print(f"[FAIL] {arm}/{scene} rc={r.returncode}: " + " | ".join(tail), flush=True)
                continue
            print(f"[ok] {arm}/{scene} in {time.time()-t0:.0f}s "
                  f"({os.path.getsize(out)/2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
