"""Lift CLIP features onto the ScanNet++ Gaussian arm (refbench 3DGS) via splat-distiller.

WHY THIS AND NOT accumulate_feature_stats_sam.py. That script hard-loads `PowerfoamScene`; a
3DGS scene needs the Gaussian rasteriser to produce the per-ray alpha-compositing weights A[r,j].
splat-distiller's `distill.py` is exactly that path and is what produced the ScanNet `gs_*` solves.

NO FORMAT CONVERSION IS NEEDED. refbench ships INRIA-style `scene_point_cloud.ply` rather than a
gsplat `ckpt.pt`, but `GaussianPrimitive.from_file` dispatches on the extension and handles `.ply`
natively (`_load_ply`), so the ply is passed straight through.

SOLVER LABELLING. distill.py accumulates per-Gaussian features then divides by the accumulated
weight -- a WEIGHTED MEAN, not a geometric median. That happens to be exactly NormLift's Eq. 5,
so these artefacts are named `solved_weighted_*` and are the faithful base for the NormLift arm.

VALID MASK. distill.py emits a bare (N,512) tensor with zeros for Gaussians no view ever covered;
`convert` turns that into the {primitive_features, valid_mask} contract the evaluator expects.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\powerfoam")
import torch

from run_gaussian_distill import convert

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\spp_data_1600"
GS = r"D:\Downloads\refbench_3dgs_12scenes\output"
FEAT_DIR = "openclip_features_sam_l3"
SPP = ["0d2ee665be", "3864514494", "27dd4da69e", "c50d2d1d42", "578511c8a9", "5942004064",
       "f9f95681fd", "d755b3d9d8", "3db0a1c8f3", "9071e139d9", "e7af285f7d", "09c1414f1b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    scenes = [s for s in (a.scenes.split(",") if a.scenes else SPP) if s]
    for scene in scenes:
        src = os.path.join(DATA, scene)
        feat = os.path.join(src, FEAT_DIR)
        n_img = len(os.listdir(os.path.join(src, "images")))
        if not os.path.isdir(feat) or len(os.listdir(feat)) < 2 * n_img:
            print(f"[skip] {scene}: features incomplete", flush=True); continue
        ply = os.path.join(GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                           "scene_point_cloud.ply")
        if not os.path.exists(ply):
            print(f"[miss] {scene}: no ply", flush=True); continue
        out = f"artifacts/scannetpp_gs/{scene}/solved_weighted_gs_unfroz_ogl3.pt"
        if os.path.exists(out):
            print(f"[skip] {scene}: solved", flush=True); continue
        raw = os.path.join(os.path.dirname(ply), "scene_point_cloud_features.pt")
        if not os.path.exists(raw):
            t0 = time.time()
            print(f"[distill] {scene} {time.strftime('%H:%M:%S')}", flush=True)
            r = subprocess.run(
                [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", ply,
                 "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"],
                stdout=open(f"logs_spp_gs_distill_{scene}.log", "w"),
                stderr=subprocess.STDOUT, cwd=r"D:\Downloads\splat-distiller")
            if r.returncode != 0 or not os.path.exists(raw):
                print(f"  [FAIL] rc={r.returncode} (logs_spp_gs_distill_{scene}.log)", flush=True)
                continue
            print(f"  distilled {(time.time()-t0)/60:.1f} min", flush=True)
        nv, n = convert(raw, out)
        print(f"  [ok] {scene} {nv:,}/{n:,} valid ({100*nv/max(n,1):.1f}%)", flush=True)
    print("[GS DISTILL DONE]", flush=True)


if __name__ == "__main__":
    main()
