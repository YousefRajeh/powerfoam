"""Lift CLIP features onto the Gaussian arms via splat-distiller, into the ablation's format.

SOLVER LABELLING MATTERS HERE. distill.py accumulates per-Gaussian features and then divides
by the accumulated weight (distill.py:186) -- a WEIGHTED MEAN. It is not a geometric median.
So its output is recorded as solver='weighted', which is directly comparable to the
`weighted` rows now produced for the foam arms from the same solver. Filing it as
geometric_median would have made every gaussian-vs-foam comparison confound representation
with solver.

A MESSAGE TO IGNORE. distill.py prints "No features found, using random features" on every
run. That refers to pre-existing PER-GAUSSIAN features in the checkpoint, of which a fresh
distillation has none; the 2D CLIP features load separately through the Dataset's
feature_folder. Verified on scene0062_00 gs_froz: the output has mean pairwise cosine 0.786
(random 512-d vectors sit near 0.00) and 83.0% of Gaussians carry non-zero features, i.e.
real CLIP structure.

VALID MASK. distill.py emits a bare (N, 512) tensor. A Gaussian that no view ever covered
keeps a zero row, exactly analogous to a foam cell with zero accumulated weight, so
valid_mask is the non-zero-norm rows -- the same criterion the foam solvers use.
"""
import argparse
import os
import subprocess
import time

import torch

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\powerfoam\data\scannet"
RECON = r"D:\Downloads\powerfoam\recon_remote"
FEAT_DIR = "openclip_features_sam_l3"

ARMS = ("gs_froz", "gs_unfroz")
SCENES = ["scene0062_00", "scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00",
          "scene0590_00", "scene0000_00", "scene0097_00", "scene0200_00", "scene0400_00"]


def convert(raw_path, out_path):
    """(N,512) bare tensor -> {primitive_features, valid_mask}, the ablation's contract."""
    x = torch.load(raw_path, map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        x = x.get("primitive_features", next(iter(x.values())))
    x = x.float()
    valid = x.norm(dim=-1) > 1e-6
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"primitive_features": x, "valid_mask": valid}, out_path)
    return int(valid.sum()), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    scenes = [s for s in (a.scenes.split(",") if a.scenes else SCENES) if s]
    arms = [x for x in a.arms.split(",") if x]

    for scene in scenes:
        src = os.path.join(DATA, f"{scene}_colmap")
        feat = os.path.join(src, FEAT_DIR)
        n_img = len(os.listdir(os.path.join(src, "images")))
        if not os.path.isdir(feat) or len(os.listdir(feat)) < 2 * n_img:
            print(f"[skip] {scene}: features incomplete", flush=True)
            continue
        for arm in arms:
            ckpt = os.path.join(RECON, arm, scene, "ckpt.pt")
            if not os.path.exists(ckpt):
                print(f"[miss] {scene}/{arm}: no checkpoint", flush=True)
                continue
            out = f"artifacts/scannet/{scene}/solved_weighted_{arm}_ogl3.pt"
            if os.path.exists(out):
                print(f"[skip] {scene}/{arm}: solved", flush=True)
                continue
            raw = os.path.join(RECON, arm, scene, "ckpt_features.pt")
            if not os.path.exists(raw):
                t0 = time.time()
                print(f"[distill] {scene}/{arm} {time.strftime('%H:%M:%S')}", flush=True)
                r = subprocess.run(
                    [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", ckpt,
                     "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"],
                    stdout=open(f"logs_gs_distill_{scene}_{arm}.log", "w"),
                    stderr=subprocess.STDOUT, cwd=r"D:\Downloads\splat-distiller")
                if r.returncode != 0 or not os.path.exists(raw):
                    print(f"  [FAIL] rc={r.returncode} (see logs_gs_distill_{scene}_{arm}.log)",
                          flush=True)
                    continue
                print(f"  distilled {(time.time()-t0)/60:.1f} min", flush=True)
            nv, n = convert(raw, out)
            print(f"  [ok] {nv:,}/{n:,} valid ({100*nv/max(n,1):.1f}%) -> {out}", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
