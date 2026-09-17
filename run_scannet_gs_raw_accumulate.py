"""Accumulate RAW (un-normalised) feature-lifting stats for the ScanNet 3DGS arms.

WHY. The single-query confidence filter needs a per-primitive confidence score, and the strongest
one measured is `||x_j||` of a RAW lift -- magnitude x multi-view agreement -- which scores
AUC 0.738/0.643/0.697 against per-primitive correctness on the foam arm, versus 0.723/0.640/0.675
for the agreement-only form. The 3DGS replication had to use the weaker agreement form because no
raw lift existed for the Gaussian arm. This builds it.

TWO THINGS HAVE TO BE OFF for magnitude to survive to the accumulator:
  1. the EXTRACTION must store un-normalised per-mask embeddings
     -> `openclip_features_sam_l3_nonorm`, produced with LANGSPLAT_NO_NORM=1
        (`splat-distiller/feature_extractor.py:255`)
  2. `distill.py` must not re-normalise the per-pixel feature map
     -> FFL_NO_PIXNORM=1 (`splat-distiller/distill.py:200`)
Both default to the original behaviour, so nothing else in either repo changes.

Mirrors `run_scannet_gs_accumulate.py` exactly apart from those two settings and the output name,
so the raw and normalised stats differ ONLY in whether magnitude was retained.

Run:  python run_scannet_gs_raw_accumulate.py [--arms gs_froz] [--scenes ...]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3_nonorm"
ART = "artifacts/scannet"

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gs_froz")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()

    ok = fail = 0
    for scene in a.scenes.split(","):
        src = os.path.join(DATA, f"{scene}_colmap")
        if not os.path.isdir(src):
            print(f"[miss] {scene}: no colmap dir at {src}", flush=True)
            continue
        if not os.path.isdir(os.path.join(src, FEAT_DIR)):
            print(f"[miss] {scene}: no {FEAT_DIR} -- run the LANGSPLAT_NO_NORM=1 extraction first",
                  flush=True)
            continue
        for arm in a.arms.split(","):
            ck = f"recon_remote/{arm}/{scene}/ckpt.pt"
            if not os.path.exists(ck):
                print(f"[miss] {arm}/{scene}: no ckpt", flush=True)
                continue
            out = os.path.abspath(f"{ART}/{scene}/stats_{arm}_rawpix.pt")
            if os.path.exists(out):
                print(f"[skip] {arm}/{scene}: stats exist", flush=True)
                ok += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            t0 = time.time()
            print(f"[accumulate RAW] {arm}/{scene}", flush=True)
            env = dict(os.environ, FFL_STATS_OUT=out,
                       FFL_NO_PIXNORM="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            r = subprocess.run(
                [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", os.path.abspath(ck),
                 "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"],
                env=env, capture_output=True, text=True, cwd=r"D:\Downloads\splat-distiller")
            if not os.path.exists(out):
                tail = (r.stderr or r.stdout).strip().splitlines()[-4:]
                print(f"[FAIL] {arm}/{scene} rc={r.returncode}: " + " | ".join(tail), flush=True)
                fail += 1
                continue
            ok += 1
            print(f"[ok] {arm}/{scene} in {time.time()-t0:.0f}s "
                  f"({os.path.getsize(out)/2**30:.1f} GB)", flush=True)
    print(f"\nRAW GS ACCUMULATE: {ok} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
