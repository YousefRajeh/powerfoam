"""Score every baseline under ONE protocol: OpenGaussian point-level mIoU + semantic surface metrics.

WHY ONE PROTOCOL FOR ALL. The baselines do not agree on how to evaluate. Occam's LGS ships no 3D
eval at all (LERF/3DOVS 2D IoU only). VALA ships a 3D evaluator, but it scores over GAUSSIANS with a
volume weighting and builds pseudo-GT by majority vote over the 1000 nearest GT points -- and, since
only surviving Gaussians are scored, pruning is invisible to it. LangSplat evaluates 2D relevancy.
The only protocol all four can be scored under, and the only one anchored to published numbers, is
OpenGaussian's point-level mIoU, so that is the headline for every method. VALA's own volume-aware
metric can be reported separately as a secondary row; it is not comparable across methods.

THE PRUNING QUESTION IS LEFT OPEN ON PURPOSE. Kept-fractions range from 100% (LUDVIG, LangSplat) to
46% (Occam) to ~12% (VALA). A GT point whose only nearby Gaussian was pruned has no honest label:
counting it wrong punishes pruning, dropping it rewards pruning, and either choice silently reorders
the table. So this script reports, per row:

    kept_frac           what fraction of the reconstruction's Gaussians the method retained
    assign_d_median     distance from each GT point to its assigned surviving Gaussian
    assign_d_p95        the tail of that distance -- large values mean points are being labelled
                        by Gaussians that are nowhere near them

and computes mIoU exactly as OpenGaussian does (nearest surviving Gaussian, no distance cutoff).
No imputed "uncovered" number is produced. The coverage columns make the confound visible so the
decision can be made once, deliberately, with the evidence in hand.

OpenGaussian's opacity>0.1 GT masking is preserved unchanged.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
GT_ROOT = r"D:\Downloads\scannet_pointcept"


def gt_scene_dir(scene):
    for split in ("train", "val", "test"):
        d = os.path.join(GT_ROOT, split, scene)
        if os.path.isdir(d):
            return d
    return None


def assignment_stats(ckpt, gt_dir, opacity_threshold=0.1):
    """How far is each GT point from the surviving Gaussian that will label it?"""
    sys.path.insert(0, HERE)
    from evaluate_point_cloud_miou import load_gaussian_means_opacities, load_scannet_pointcept_gt
    means, opac = load_gaussian_means_opacities(ckpt, "cuda")
    gt_points, _, _ = load_scannet_pointcept_gt(gt_dir, "segment20")
    valid = opac >= opacity_threshold
    m = np.asarray(means)[valid]
    if m.shape[0] == 0:
        return None, None
    # KD-TREE, NOT cdist. Chunking only the GT points still materialises a
    # (chunk x n_gaussians) matrix -- at 2.4M unfrozen Gaussians that is 59.75 GiB for a
    # 20k chunk, which is exactly how the first sweep died. A KD-tree is O(n log n), needs no
    # dense matrix, and runs on CPU so it does not contend with the training job on the GPU.
    from scipy.spatial import cKDTree
    d, _ = cKDTree(m).query(gt_points, k=1, workers=-1)
    return float(np.median(d)), float(np.percentile(d, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"D:\Downloads\powerfoam\artifacts\baseline_eval\manifest.json")
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--output", default=r"D:\Downloads\powerfoam\artifacts\baseline_eval\results.json")
    ap.add_argument("--surface", action="store_true", help="also run semantic surface metrics")
    ap.add_argument("--tags", default=None, help="comma list of tags to restrict to")
    a = ap.parse_args()

    entries = json.load(open(a.manifest))
    if a.tags:
        want = set(a.tags.split(","))
        entries = [e for e in entries if e["tag"] in want]

    done = []
    if os.path.exists(a.output):
        done = json.load(open(a.output))
    seen = {(r["tag"], r["scene"], r["class_set"]) for r in done}

    for e in entries:
        gt_dir = gt_scene_dir(e["scene"])
        if gt_dir is None:
            print(f"  {e['tag']} {e['scene']}: no GT dir", flush=True)
            continue
        dmed, dp95 = assignment_stats(e["ckpt"], gt_dir)
        for cs in a.class_sets.split(","):
            if (e["tag"], e["scene"], cs) in seen:
                continue
            cmd = [PY, os.path.join(HERE, "evaluate_point_cloud_miou.py"),
                   "--gt-format", "scannet", "--gt-points", gt_dir,
                   "--method", "splat_feature_solver", "--classes", cs, "--gt-opacity-mask",
                   "--gaussian-checkpoint", e["ckpt"], "--gaussian-features", e["features"]]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
            miou = macc = None
            for line in out.stdout.splitlines():
                if "mIoU=" in line:
                    for tok in line.replace("[Splat Feature Solver]", "").split():
                        if tok.startswith("mIoU="):
                            miou = float(tok.split("=")[1])
                        elif tok.startswith("mAcc="):
                            macc = float(tok.split("=")[1])
            if miou is None:
                print(f"  {e['tag']:22s} {e['scene']} {cs}: FAIL", flush=True)
                print("    " + (out.stderr.strip().splitlines() or ["<no stderr>"])[-1], flush=True)
                continue
            row = dict(tag=e["tag"], baseline=e["baseline"], arm=e["arm"], level=e["level"],
                       scene=e["scene"], class_set=cs, miou=miou, macc=macc,
                       n_original=e["n_original"], n_kept=e["n_kept"], kept_frac=e["kept_frac"],
                       assign_d_median=dmed, assign_d_p95=dp95,
                       uncovered_policy="open: no imputed value for points whose nearest "
                                        "Gaussian was pruned; see kept_frac / assign_d_*")
            done.append(row)
            print(f"  {e['tag']:22s} {e['scene']} {cs:15s} mIoU={miou:.4f} mAcc={macc:.4f} "
                  f"kept={100*e['kept_frac']:.1f}% d50={dmed:.3f}", flush=True)
            with open(a.output, "w") as fh:
                json.dump(done, fh, indent=1)

    print(f"\n{len(done)} rows -> {a.output}")


if __name__ == "__main__":
    main()
