"""Does lifting features THROUGH occluding surfaces cause our dominant-class failure?

THE HYPOTHESIS, and why it is worth GPU time:

Our per-primitive ScanNet scores barely improve as the class set shrinks (10-scene:
30.98 -> 30.98 -> 34.10 at 19/15/10 classes, +3.1 total) while NormLift's no-refinement
ablation climbs +12.4 over the same axis (32.94 -> 36.30 -> 45.34). The 19->15->10 subsets
REMOVE small classes and KEEP wall/floor/table. So our errors are concentrated in exactly
the classes that never get removed -- consistent with Experiment-F's measured
precision/recall inversion (floor recall 0.35, leaking to sofa 24% / table 21%).

There is a mechanism for that leak sitting in the lifting code. `export_operator_kernel`
walks each ray front-to-back depositing alpha*trans into every cell it crosses, and only
stops when transmittance drops below 1e-3 -- i.e. 99.9% absorbed. But a SAM mask describes
the FIRST surface. So a pixel showing a sofa deposits the SOFA embedding into the floor and
wall cells behind it. Floor is the most-occluded class in an indoor scan (everything stands
on it), which is precisely the class with the worst measured recall.

Measured on a real view of scene0347_00 (verify_surf_truncation.py): truncating at the
median-depth surface keeps 66.6% of total ray weight but only 26.8% of (pixel, cell) pairs.
The median pixel currently deposits its single CLIP embedding into 12 cells; only 3 are in
front of the surface. Roughly three quarters of all deposits are behind it.

Note our own CD-L1 surface extraction already defines the surface at MEDIAN DEPTH (tau=0.5).
The geometry path and the feature path have been disagreeing about where the surface is by
three orders of magnitude in transmittance.

WHAT THIS IS NOT: the already-falsified 'top1' transform, which collapsed each pixel onto a
single cell and lost the soft volumetric weighting we measured to BEAT splat-sharp lifting.
'surf<tau>' keeps every weight in front of the surface untouched and drops only the occluded
tail, so it tests occlusion contamination specifically rather than sharpness in general.

WHAT WOULD FALSIFY IT: per-class IoU for wall/floor/table is printed for every condition.
The hypothesis predicts those specifically recover. If overall mIoU moves but wall/floor/table
do not, the mechanism is wrong even if the number is up, and the honest conclusion is that
something else changed. If floor/wall IoU does not improve on the first two (hardest) scenes,
stop -- the hypothesis is dead and the coarse-class gap is text-side, not lifting-side.

Scenes are ordered HARDEST FIRST by their measured base-protocol mIoU (scene0140_00 22.64,
scene0645_00 27.14, scene0070_00 29.95, scene0347_00 39.84), so a failure shows up early.

SAM granularity is a second swept axis (--sam-levels), not a fixed setting -- see
parse_level_specs. Level 3 (l/whole) is the default because OpenGaussian, NormLift and every
prior result in this project use it, so tau deltas stay comparable to what is already
recorded; level COMBINATIONS are a separate ablation this script can run unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam, spherical_kmeans
from run_cluster_classify_eval import two_level_position_aware, pool_classify_broadcast, K_FLAT, SCENES

PY = r"D:\conda\envs\powerfoam\python.exe"
# the classes the hypothesis makes a specific prediction about: big, occluded, never removed
WATCH = ("wall", "floor", "table", "chair", "sofa", "bed")


def run(cmd, log):
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc


def parse_level_specs(spec):
    """SAM granularity is a swept AXIS here, not a fixed setting.

    The LangSplat hierarchy stored in each `_s.npy` is 0=default, 1=subpart(s), 2=part(m),
    3=whole(l). OpenGaussian ("we only use the large-level mask") and NormLift both use l
    only, and every prior result in this project is l-only, so `l3` stays the default for
    comparability of deltas. But the level choice is a real experimental knob -- summing
    several levels blends up to 4 mask embeddings per pixel, which is a candidate
    contamination source of exactly the same kind as the occlusion tail this script tests --
    so combinations are expressible directly.

    Spec syntax (';'-separated so ',' can join levels):
        "3"        -> level 3 only            tag l3
        "2,3"      -> levels 2 and 3 summed   tag l23
        "all"      -> every level summed (the splat-distiller loader default)   tag lall
        "3;2,3;all" -> three conditions
    Returns [(tag, value_for_--sam-level or None), ...].
    """
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            out.append(("lall", None))
        else:
            levels = [x.strip() for x in item.split(",") if x.strip() != ""]
            out.append(("l" + "".join(levels), ",".join(levels)))
    return out


def accumulate_and_solve(scene, tag, transform, sam_level, force=False):
    """One accumulation pass + geometric-median solve. Stats are deleted afterwards
    (they are ~3GB per large scene and reconstructible)."""
    solved = f"artifacts/scannet/{scene}/solved_surfabl_{tag}.pt"
    if os.path.exists(solved) and not force:
        print(f"    [{tag}] solved features already present, reusing", flush=True)
        return solved
    stats = f"artifacts/scannet/{scene}/stats_surfabl_{tag}.pt"
    cfg = f"output/scannet_{scene}_nonfrozen/config.yaml"
    feat = f"artifacts/scannet/{scene}/openclip_features_sam"

    cmd = [PY, "accumulate_feature_stats_sam.py", "--scene", f"{scene}_colmap",
           "--config", cfg, "--feature-folder", feat, "--output", stats]
    if sam_level is not None:          # None = sum every level (loader default)
        cmd += ["--sam-level", str(sam_level)]
    if transform is not None:
        cmd += ["--weight-transform", transform]
    t0 = time.time()
    rc = run(cmd, f"logs_surfabl_{scene}_{tag}_acc.log")
    if rc != 0 or not os.path.exists(stats):
        print(f"    [{tag}] ACCUMULATION FAILED rc={rc}, see logs_surfabl_{scene}_{tag}_acc.log",
              flush=True)
        return None
    print(f"    [{tag}] accumulated in {time.time()-t0:.0f}s", flush=True)

    rc = run([PY, "solve_geometric_median.py", "--stats", stats, "--output", solved],
             f"logs_surfabl_{scene}_{tag}_solve.log")
    try:
        os.remove(stats)
    except OSError:
        pass
    if rc != 0 or not os.path.exists(solved):
        print(f"    [{tag}] SOLVE FAILED rc={rc}", flush=True)
        return None
    return solved


def evaluate(scene, solved_path, device="cuda"):
    """Same pool-classify-broadcast protocol as every other result in this project,
    plus per-class IoU so the mechanism can be checked, not just the headline number."""
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{split}\{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_nonfrozen", device)
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
    out = {"coverage_valid_cells": float(vm.mean())}
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
        ious, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                  len(tids) + 1)
        per_class = {n: float(ious[i + 1]) for i, n in enumerate(tnames)}
        out[cs] = {"mIoU": float(miou), "mAcc": float(macc), "per_class_IoU": per_class}
    return out


def fmt_watch(res, cs="opengaussian19"):
    pc = res[cs]["per_class_IoU"]
    return "  ".join(f"{n}={pc[n]*100:.1f}" for n in WATCH if n in pc)


def main():
    p = argparse.ArgumentParser()
    # hardest first by measured base-protocol mIoU
    p.add_argument("--scenes", default="scene0140_00,scene0645_00,scene0070_00,scene0347_00")
    p.add_argument("--taus", default="none,0.5,0.25",
                   help="'none' = current behaviour (full ray to trans<1e-3)")
    p.add_argument("--sam-levels", default="3",
                   help="';'-separated SAM level specs, ',' joins levels within one spec. "
                        "e.g. '3' | '3;2,3;all' | 'all'. Default 3 (l-level) matches "
                        "OpenGaussian/NormLift and every prior result here.")
    p.add_argument("--output", default="artifacts/scannet/surface_truncation_ablation.json")
    a = p.parse_args()

    scenes = a.scenes.split(",")
    taus = a.taus.split(",")
    levels = parse_level_specs(a.sam_levels)
    print(f"levels: {[t for t, _ in levels]}   taus: {taus}   scenes: {scenes}", flush=True)

    results = {}
    if os.path.exists(a.output):
        results = json.load(open(a.output))

    for scene in scenes:
        print(f"\n===== {scene} =====", flush=True)
        results.setdefault(scene, {})
        for ltag, sam_level in levels:
            base_tag = f"{ltag}_base"
            for tau in taus:
                tag = base_tag if tau == "none" else f"{ltag}_surf{tau}"
                transform = None if tau == "none" else f"surf{tau}"
                if tag in results[scene]:
                    print(f"  [{tag}] already recorded, skipping", flush=True)
                    continue
                print(f"  --- {tag} ---", flush=True)
                solved = accumulate_and_solve(scene, tag, transform, sam_level)
                if solved is None:
                    continue
                res = evaluate(scene, solved)
                results[scene][tag] = res
                b = results[scene].get(base_tag)
                d19 = (f" (delta {(res['opengaussian19']['mIoU'] - b['opengaussian19']['mIoU'])*100:+.2f})"
                       if b and tag != base_tag else "")
                print(f"    19cls mIoU={res['opengaussian19']['mIoU']*100:.2f}{d19}  "
                      f"15cls={res['opengaussian15']['mIoU']*100:.2f}  "
                      f"10cls={res['opengaussian10']['mIoU']*100:.2f}", flush=True)
                print(f"    per-class IoU (19cls): {fmt_watch(res)}", flush=True)
                with open(a.output, "w") as f:
                    json.dump(results, f, indent=2)

            # the decisive readout: did the WATCHED classes move, within this level?
            if base_tag in results[scene]:
                print(f"  --- {scene}/{ltag} verdict on the mechanism ---", flush=True)
                bpc = results[scene][base_tag]["opengaussian19"]["per_class_IoU"]
                for tag, res in results[scene].items():
                    if tag == base_tag or not tag.startswith(ltag + "_"):
                        continue
                    pc = res["opengaussian19"]["per_class_IoU"]
                    moved = {n: (pc[n] - bpc[n]) * 100 for n in WATCH if n in pc and n in bpc}
                    s = "  ".join(f"{n}{v:+.1f}" for n, v in moved.items())
                    print(f"    {tag}: {s}", flush=True)

    print("\n\n=== SUMMARY (mIoU 19/15/10, delta vs same-level base) ===", flush=True)
    for scene, per_tag in results.items():
        for tag, res in sorted(per_tag.items()):
            b = per_tag.get(tag.split("_")[0] + "_base")
            line = (f"{scene:<16}{tag:<16} "
                    f"{res['opengaussian19']['mIoU']*100:6.2f}"
                    f"{res['opengaussian15']['mIoU']*100:7.2f}"
                    f"{res['opengaussian10']['mIoU']*100:7.2f}")
            if b and not tag.endswith("_base"):
                line += ("   delta "
                         + " ".join(f"{(res[c]['mIoU']-b[c]['mIoU'])*100:+.2f}"
                                    for c in ("opengaussian19", "opengaussian15", "opengaussian10")))
            print(line, flush=True)
    print(f"\nwrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
