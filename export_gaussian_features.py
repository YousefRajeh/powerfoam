"""Package the ScanNet++ 3DGS reconstructions together with their lifted CLIP features.

The two halves live apart in our pipeline and a collaborator should not have to know that:

  * Gaussian parameters -- D:/Downloads/refbench_3dgs_12scenes/output/refbench-{scene}/
                           point_cloud/iteration_30000/scene_point_cloud.ply
  * lifted features     -- artifacts/scannetpp_gs/{scene}/solved_weighted_gs_unfroz_ogl3.pt
                           (primitive_features (N,512) float32, valid_mask (N,) bool)

Row i of the feature tensor corresponds to vertex i of the PLY -- the lifting never reorders or
prunes primitives, which is the same index-stability requirement every export in this project keeps.
This script verifies that correspondence (counts must match exactly) rather than assuming it.

WHAT THE FEATURES ARE. OpenCLIP ViT-B/16 embeddings of SAM level-3 masks, lifted onto each Gaussian
by the volumetric operator and solved per primitive with the WEIGHTED solver
(`solved_weighted_gs_unfroz_ogl3.pt`). Features are NOT unit-normalised on disk; `valid_mask` is
False for primitives no view ever observed, whose feature row is meaningless.

USAGE
    python export_gaussian_features.py --out D:/Downloads/gs_features_export
produces one {scene}.npz per scene with means/scales/quats/opacity/features/valid, plus a README.
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
from plyfile import PlyData

GS = r"D:\Downloads\refbench_3dgs_12scenes\output"
ART = "artifacts/scannetpp_gs"


def load_ply(scene):
    p = os.path.join(GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                     "scene_point_cloud.ply")
    v = PlyData.read(p)["vertex"]
    means = np.stack([np.asarray(v[k]) for k in ("x", "y", "z")], 1).astype(np.float32)
    # scales/rotations are stored in the 3DGS activation domain: exp() and a normalised quaternion,
    # matching run_spp_gs_eval.py::load_gaussians so downstream numbers stay comparable to ours.
    scales = np.exp(np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], 1)).astype(np.float32)
    quats = np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], 1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
    op = None
    if "opacity" in v.data.dtype.names:
        # sigmoid(opacity) is the quantity OpenGaussian thresholds at 0.1; store the RAW logit and
        # let the consumer apply sigmoid, so no convention is baked in here.
        op = np.asarray(v["opacity"]).astype(np.float32)
    return means, scales, quats, op, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\Downloads\gs_features_export")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--fp16", action="store_true", help="store features as float16 (halves size)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenes = ([s for s in a.scenes.split(",") if s] or
              sorted(d for d in os.listdir(ART) if os.path.isdir(os.path.join(ART, d))))
    rows = []
    for scene in scenes:
        fp = os.path.join(ART, scene, "solved_weighted_gs_unfroz_ogl3.pt")
        if not os.path.exists(fp):
            print(f"[skip] {scene}: no solved features"); continue
        d = torch.load(fp, map_location="cpu", weights_only=True)
        feats = d["primitive_features"].numpy()
        valid = d["valid_mask"].numpy()
        try:
            means, scales, quats, op, ply = load_ply(scene)
        except Exception as e:
            print(f"[skip] {scene}: cannot read PLY ({e})"); continue
        if means.shape[0] != feats.shape[0]:
            print(f"[MISMATCH] {scene}: {means.shape[0]:,} gaussians vs "
                  f"{feats.shape[0]:,} feature rows -- NOT written")
            continue
        out = os.path.join(a.out, f"{scene}.npz")
        kw = dict(means=means, scales=scales, quats=quats, valid=valid,
                  features=feats.astype(np.float16) if a.fp16 else feats)
        if op is not None:
            kw["opacity_logit"] = op
        np.savez(out, **kw)
        mb = os.path.getsize(out) / 1e6
        rows.append((scene, means.shape[0], int(valid.sum()), mb))
        print(f"[ok] {scene}: {means.shape[0]:,} gaussians, "
              f"{valid.sum()/len(valid)*100:.1f}% valid, {mb:.0f} MB")

    with open(os.path.join(a.out, "README.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "ScanNet++ 3DGS reconstructions with lifted open-vocabulary CLIP features\n"
            "=======================================================================\n\n"
            "One .npz per scene. Arrays are aligned ROW-WISE: row i is Gaussian i.\n\n"
            "  means         (N,3) float32  centre in world coordinates\n"
            "  scales        (N,3) float32  per-axis std dev, ALREADY exponentiated\n"
            "  quats         (N,4) float32  rotation, normalised, (w,x,y,z) as stored by 3DGS\n"
            "  opacity_logit (N,)  float32  RAW logit; apply sigmoid() to get opacity in [0,1]\n"
            "  features      (N,512)        OpenCLIP ViT-B/16 embedding lifted onto the Gaussian\n"
            "  valid         (N,)  bool     False where no view observed the primitive\n\n"
            "FEATURES ARE NOT UNIT-NORMALISED. Normalise before taking cosines:\n"
            "    f = features[valid]; f /= np.linalg.norm(f, axis=1, keepdims=True)\n\n"
            "Rows where valid==False contain meaningless values -- mask them out, do not zero-fill.\n\n"
            "PROVENANCE\n"
            "  Gaussians : 3DGS, 30k iterations (refbench_3dgs_12scenes).\n"
            "  Features  : SAM level-3 masks -> OpenCLIP ViT-B/16 per mask -> lifted by the\n"
            "              volumetric operator -> per-primitive WEIGHTED solve.\n"
            "  Point->Gaussian queries in our evaluation use Mahalanobis distance\n"
            "              d = (p-mu)^T Sigma^-1 (p-mu) with Sigma from scales/quats (Dr.Splat\n"
            "              convention), plus OpenGaussian's cull of sigmoid(opacity) < 0.1.\n\n"
            "SCENES\n")
        for s, n, v, mb in rows:
            f.write(f"  {s}  {n:>9,} gaussians  {v/n*100:5.1f}% valid  {mb:6.0f} MB\n")
    print(f"\nwrote {len(rows)} scenes to {a.out}")


if __name__ == "__main__":
    main()
