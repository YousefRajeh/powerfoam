"""Does the POWER WEIGHT earn its place SEMANTICALLY, not just photometrically?

`constant_radii` freezes every radius to one value, which makes the membership rule
`argmin_i ||x-c_i||^2 - r_i^2` reduce to `argmin_i ||x-c_i||^2` -- an exact Voronoi diagram,
i.e. the Radiant Foam / VoroTracing partition, inside our own codebase and pipeline. The
training half of that ablation is done and the reconstruction verdict is near-parity:
PSNR power-minus-Voronoi = +1.90 / +0.33 / +0.19 / +0.32 dB on scene0070/0140/0347/0645.

But PSNR is not what this project is about. Everything SEMANTIC we do -- exact cell
membership, facet adjacency, ray-traversal lifting -- never reads r at all, so the interesting
question is whether the segmentation numbers move. If Voronoi matches power on mIoU too, the
power weight is dead weight for our purposes and the representation simplifies; if power wins
clearly, r earns its place for a reason we have not yet articulated.

This runs the FULL feature-assignment pipeline on each trained checkpoint -- accumulate SAM+CLIP
features through that checkpoint's own geometry, solve by geometric median, cluster
position-aware 64x5, pool-classify-broadcast, score against ScanNet GT -- so the only thing
differing between the two arms is the partition itself.

CAVEAT, recorded so the result is not over-read: radii also scale the texel-site offsets, so
this is not a pure partition-only ablation. A difference could come from appearance
parameterization rather than from the partition.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, pool_classify_broadcast, K_FLAT, SCENES

PY = r"D:\conda\envs\powerfoam\python.exe"


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def accumulate_and_solve(scene, ckpt_dir, tag, sam_level="3"):
    solved = f"artifacts/scannet/{scene}/solved_{tag}.pt"
    if os.path.exists(solved):
        print(f"    [{tag}] reusing existing solve", flush=True)
        return solved
    stats = f"artifacts/scannet/{scene}/stats_{tag}.pt"
    cmd = [PY, "accumulate_feature_stats_sam.py", "--scene", f"{scene}_colmap",
           "--config", f"{ckpt_dir}/config.yaml",
           "--feature-folder", f"artifacts/scannet/{scene}/openclip_features_sam",
           "--output", stats, "--sam-level", sam_level]
    t0 = time.time()
    rc = run(cmd, f"logs_voroeval_{tag}_acc.log")
    if rc != 0 or not os.path.exists(stats):
        print(f"    [{tag}] ACCUMULATION FAILED rc={rc}", flush=True)
        return None
    print(f"    [{tag}] accumulated in {time.time()-t0:.0f}s", flush=True)
    rc = run([PY, "solve_geometric_median.py", "--stats", stats, "--output", solved],
             f"logs_voroeval_{tag}_solve.log")
    try:
        os.remove(stats)
    except OSError:
        pass
    if rc != 0 or not os.path.exists(solved):
        print(f"    [{tag}] SOLVE FAILED rc={rc}", flush=True)
        return None
    return solved


def evaluate(scene, ckpt_dir, solved_path, device="cuda"):
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{split}\{scene}", "segment20")
    centers, radii = load_foam(ckpt_dir, device)
    solved = torch.load(solved_path, map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0

    vi_np = np.where(vm)[0]
    vi = torch.from_numpy(vi_np).to(device)
    unit = F.normalize(feats[vi], dim=-1)
    positions = torch.from_numpy(centers[vi_np]).to(device).float()
    pos_labels = two_level_position_aware(positions, unit, seed=0)

    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    out = {"cells": int(centers.shape[0]),
           "coverage_valid_cells": float(vm.mean()),
           "radius_std": float(np.std(radii)),
           "gt_points_owned": float(owned.mean())}
    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
        text = embed_class_names(tnames, device)
        cls_valid = pool_classify_broadcast(pos_labels, unit, K_FLAT, text).cpu().numpy()
        prim = np.zeros(centers.shape[0], dtype=np.int64)
        prim[vi_np] = cls_valid
        pred = np.zeros(gt_points.shape[0], dtype=np.int64)
        pred[owned] = prim[assigned[owned]] + 1
        _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
        out[cs] = {"mIoU": float(miou), "mAcc": float(macc)}
    return out


def main():
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    p = argparse.ArgumentParser()
    # hardest first by base-protocol mIoU
    p.add_argument("--scenes", default="scene0140_00,scene0645_00,scene0070_00,scene0347_00")
    p.add_argument("--arms", default="power,voronoi")
    p.add_argument("--output", default="artifacts/scannet/voronoi_feature_eval.json")
    a = p.parse_args()

    results = json.load(open(a.output)) if os.path.exists(a.output) else {}
    for scene in a.scenes.split(","):
        print(f"\n===== {scene} =====", flush=True)
        results.setdefault(scene, {})
        for arm in a.arms.split(","):
            tag = f"voro_{scene}_{arm}"
            if arm in results[scene]:
                print(f"  [{arm}] already recorded", flush=True)
                continue
            ckpt = f"output/{tag}"
            if not os.path.exists(f"{ckpt}/model.pt"):
                print(f"  [{arm}] no checkpoint at {ckpt}, skipping", flush=True)
                continue
            print(f"  --- {arm} ---", flush=True)
            solved = accumulate_and_solve(scene, ckpt, tag)
            if solved is None:
                continue
            r = evaluate(scene, ckpt, solved)
            results[scene][arm] = r
            print(f"    cells={r['cells']}  radius_std={r['radius_std']:.6f}  "
                  f"coverage={r['coverage_valid_cells']*100:.1f}%  "
                  f"gt_owned={r['gt_points_owned']*100:.2f}%", flush=True)
            print(f"    19cls={r['opengaussian19']['mIoU']*100:6.2f}  "
                  f"15cls={r['opengaussian15']['mIoU']*100:6.2f}  "
                  f"10cls={r['opengaussian10']['mIoU']*100:6.2f}", flush=True)
            with open(a.output, "w") as f:
                json.dump(results, f, indent=2)

    print("\n\n=== POWER vs VORONOI, feature assignment (mIoU 19/15/10) ===", flush=True)
    d19, d15, d10 = [], [], []
    for scene, arms in results.items():
        if "power" not in arms or "voronoi" not in arms:
            continue
        pw, vo = arms["power"], arms["voronoi"]
        row = [f"{scene:<16}"]
        for cs, acc in (("opengaussian19", d19), ("opengaussian15", d15), ("opengaussian10", d10)):
            dd = (vo[cs]["mIoU"] - pw[cs]["mIoU"]) * 100
            acc.append(dd)
            row.append(f"{pw[cs]['mIoU']*100:6.2f}/{vo[cs]['mIoU']*100:6.2f} ({dd:+5.2f})")
        print("  ".join(row), flush=True)
    if d19:
        print(f"\n  mean Voronoi-minus-power: 19cls {np.mean(d19):+.2f}  "
              f"15cls {np.mean(d15):+.2f}  10cls {np.mean(d10):+.2f}", flush=True)
        print("  (negative = the power weight is buying something semantically)", flush=True)
    print(f"\nwrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
