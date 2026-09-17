"""Evaluate PowerFoam's nonfrozen ScanNet reconstructions across all 10 scenes and
OpenGaussian's 3 exact class sets (opengaussian19/15/10), then average mIoU/mAcc across
scenes to compare directly against OpenGaussian's Table 2 (24.73/41.54, 30.13/48.25,
38.29/55.19). Shells out to evaluate_point_cloud_miou.py per (scene, class-set) -- reuses
its already-validated present-class-filtering/correspondence logic verbatim instead of
reimplementing it -- then aggregates the written --output-json files.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON = sys.executable
GT_ROOT = Path(r"D:\Downloads\scannet_pointcept")

SCENES = {
    "scene0000_00": ("train", "output/scannet_scene0000_00_nonfrozen", "artifacts/scannet/scene0000_00/solved_geometric_median_nonfrozen.pt"),
    "scene0062_00": ("train", "output/scannet_scene0062_00_nonfrozen", "artifacts/scannet/scene0062_00/solved_geometric_median_nonfrozen.pt"),
    "scene0070_00": ("train", "output/scannet_scene0070_00_nonfrozen", "artifacts/scannet/scene0070_00/solved_geometric_median_nonfrozen.pt"),
    "scene0097_00": ("train", "output/scannet_scene0097_00_nonfrozen", "artifacts/scannet/scene0097_00/solved_geometric_median_nonfrozen.pt"),
    "scene0140_00": ("train", "output/scannet_scene0140_00_nonfrozen", "artifacts/scannet/scene0140_00/solved_geometric_median_nonfrozen.pt"),
    "scene0200_00": ("train", "output/scannet_scene0200_00_nonfrozen", "artifacts/scannet/scene0200_00/solved_geometric_median_nonfrozen.pt"),
    "scene0347_00": ("train", "output/scannet_scene0347_00_nonfrozen", "artifacts/scannet/scene0347_00/solved_geometric_median_nonfrozen.pt"),
    "scene0400_00": ("train", "output/scannet_scene0400_00_nonfrozen", "artifacts/scannet/scene0400_00/solved_geometric_median_nonfrozen.pt"),
    "scene0590_00": ("train", "output/scannet_scene0590_00_nonfrozen", "artifacts/scannet/scene0590_00/solved_geometric_median_nonfrozen.pt"),
    "scene0645_00": ("val", "output/scannet_scene0645_00_nonfrozen", "artifacts/scannet/scene0645_00/solved_geometric_median_nonfrozen.pt"),
}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

# FEAT_FILE overrides the per-scene solved-feature BASENAME (directory and .pt are added).
# Unset -> byte-identical to the previous behaviour.
#
# WHY. The hard-coded default `solved_geometric_median_nonfrozen.pt` is stale in two independent
# ways: (i) the geometric median measures 0.0545 mIoU WORSE than the plain weighted mean on the
# like-for-like `_l3` path (10 scenes, worse on 8/10), and (ii) it carries no `_l3` suffix, so it
# is the all-levels accumulation rather than the one every current experiment uses. Quoting this
# script's per-primitive numbers against run_cluster_classify_eval.py's cluster-then-classify
# numbers therefore varies BOTH the protocol and the features at once, which is exactly the
# cross-accumulation mistake that invalidated this session's first result table.
# Set FEAT_FILE=solved_normlift_l3 to hold the features fixed and vary only the protocol.
FEAT_FILE = os.environ.get("FEAT_FILE", "").strip()
TAG = os.environ.get("EVAL_TAG", "").strip() or "nonfrozen"

# GT_OPACITY_MASK=1 enables OpenGaussian's low-opacity GT masking (their eval_scannet.py:127-129),
# exposed by evaluate_point_cloud_miou.py as --gt-opacity-mask. This script never passed it, so
# every per-primitive number it produced before now OMITS the culling step the reported protocol
# uses. Default OFF so prior numbers remain reproducible; set 1 for the protocol-faithful run.
# NOTE: PowerFoam has no sigmoid opacity -- the eval converts unbounded density to alpha via
# alpha = 1 - exp(-density * L) for a fixed path length L. See evaluate_point_cloud_miou.py:220.
RECON = os.environ.get("RECON", "nonfrozen").strip() or "nonfrozen"
"""Checkpoint suffix: `output/scannet_<scene>_<RECON>`. The reported config is `truefrozen`
(ablation.sqlite `recon=pf_tfroz`), NOT the `nonfrozen` this script defaults to."""

GT_OPACITY_MASK = os.environ.get("GT_OPACITY_MASK", "0") == "1"
OPACITY_THRESHOLD = os.environ.get("OPACITY_THRESHOLD", "0.1")


def features_for(scene, default_path):
    if not FEAT_FILE:
        return default_path
    name = FEAT_FILE if FEAT_FILE.endswith(".pt") else FEAT_FILE + ".pt"
    return "artifacts/scannet/{}/{}".format(scene, name)


def main():
    results = {cs: {} for cs in CLASS_SETS}

    print("[config] FEAT_FILE=" + (FEAT_FILE or "(default: solved_geometric_median_nonfrozen)")
          + "  TAG=" + TAG
          + "  RECON=" + RECON
          + "  GT_OPACITY_MASK=" + str(GT_OPACITY_MASK)
          + "  OPACITY_THRESHOLD=" + str(OPACITY_THRESHOLD))
    for scene, (split, ckpt_dir, features_path) in SCENES.items():
        # rewrite the hard-coded `_nonfrozen` checkpoint to the selected arm
        ckpt_dir = f"output/scannet_{scene}_{RECON}"
        features_path = features_for(scene, features_path)
        if not Path(ckpt_dir, "model.pt").exists():
            print(f"[SKIP] {scene}: no checkpoint at {ckpt_dir}")
            continue
        if not Path(features_path).exists():
            print(f"[SKIP] {scene}: no solved features at {features_path}")
            continue
        gt_dir = GT_ROOT / split / scene
        for cs in CLASS_SETS:
            out_json = f"artifacts/scannet/{scene}/miou_{cs}_{TAG}.json"
            cmd = [
                PYTHON, "evaluate_point_cloud_miou.py",
                "--gt-points", str(gt_dir),
                "--gt-format", "scannet",
                "--method", "powerfoam",
                "--classes", cs,
                "--powerfoam-checkpoint", ckpt_dir,
                "--powerfoam-features", features_path,
                "--output-json", out_json,
            ]
            if GT_OPACITY_MASK:
                cmd += ["--gt-opacity-mask", "--opacity-threshold", str(OPACITY_THRESHOLD)]
            print(f"\n=== {scene} {cs} ===")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(proc.stdout[-1500:])
            if proc.returncode != 0:
                print(f"[ERROR] {scene} {cs} failed:\n{proc.stderr[-2000:]}")
                continue
            with open(out_json) as f:
                metrics = json.load(f)["powerfoam"]
            results[cs][scene] = metrics

    print("\n\n=== 10-scene averages (nonfrozen PowerFoam, feats="
          + (FEAT_FILE or "solved_geometric_median_nonfrozen")
          + ", opacity_mask=" + str(GT_OPACITY_MASK)
          + ", OpenGaussian classes) ===")
    summary = {
        "_provenance": {
            "feature_file": FEAT_FILE or "solved_geometric_median_nonfrozen",
            "tag": TAG,
            "recon": RECON,
            "protocol": "per-primitive cosine argmax (no clustering)",
            "gt_opacity_mask": GT_OPACITY_MASK,
            "opacity_threshold": OPACITY_THRESHOLD,
        },
    }
    for cs, per_scene in results.items():
        mious = [m["mIoU"] for m in per_scene.values()]
        maccs = [m["mAcc"] for m in per_scene.values()]
        summary[cs] = {
            "num_scenes": len(mious),
            "mean_mIoU": float(np.mean(mious)) if mious else None,
            "mean_mAcc": float(np.mean(maccs)) if maccs else None,
            "per_scene": {s: {"mIoU": m["mIoU"], "mAcc": m["mAcc"]} for s, m in per_scene.items()},
        }
        mi = summary[cs]["mean_mIoU"]
        ma = summary[cs]["mean_mAcc"]
        print(f"{cs}: mean_mIoU={mi:.4f} mean_mAcc={ma:.4f} (n={summary[cs]['num_scenes']})" if mi is not None
              else f"{cs}: no scenes evaluated")

    out_path = "artifacts/scannet/nonfrozen_10scene_avg_geometric_median.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
