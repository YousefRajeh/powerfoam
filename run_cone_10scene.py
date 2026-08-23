"""The 10-scene, 3-seed verdict on the cone-constrained solver.

scene0347_00 alone gave +0.04 / +0.81 / +3.35 at 19/15/10 classes, with the 10-class delta the
first in this campaign to clear the ~1.5 mIoU evidence threshold. Eight single-scene results have
reversed under multi-scene confirmation in this project, so this is what decides it.

Protocol, matching everything else recorded here:
  - all 10 OpenGaussian scan_list scenes, HARDEST FIRST by measured base-protocol mIoU so a
    collapse shows up early
  - both arms produced from the SAME accumulation pass, so the ONLY difference is the solve
  - 3 clustering seeds per arm, reporting mean and spread (per-arm seed std was measured at 0.71
    mIoU average and 1.64 worst, which is why single-seed numbers are not evidence)
  - deterministic evaluation path (verified bitwise, spread 0.0 across processes)

Kill criterion: if the 10-class gain does not survive on the first four scenes, the cone solve is
a single-scene artifact and the "solve the coupling" direction closes for good -- unconstrained
loses by 9 mIoU, constrained would then be null, and the diagonal stands as correct rather than
merely convenient.
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

PY = r"D:\conda\envs\powerfoam\python.exe"
# hardest first by measured base-protocol mIoU, then the rest
SCENES = ["scene0140_00", "scene0645_00", "scene0070_00", "scene0347_00",
          "scene0000_00", "scene0590_00", "scene0400_00", "scene0200_00",
          "scene0097_00", "scene0062_00"]
CLASS_SETS = ("opengaussian19", "opengaussian15", "opengaussian10")


def solve(scene, iters, force=False):
    """One run produces BOTH arms: the cone solution and its own diagonal baseline, from the
    same accumulation, so the comparison cannot drift."""
    cone = f"artifacts/scannet/{scene}/solved_cone10.pt"
    diag = f"artifacts/scannet/{scene}/solved_cone10_diag.pt"
    if os.path.exists(cone) and os.path.exists(diag) and not force:
        print(f"  [{scene}] reusing existing solves", flush=True)
        return cone, diag
    cmd = [PY, "solve_cone_fast.py", "--scene", scene,
           "--config", f"output/scannet_{scene}_nonfrozen/config.yaml",
           "--feature-folder", f"artifacts/scannet/{scene}/openclip_features_sam",
           "--output", cone, "--sam-level", "3", "--iters", str(iters),
           "--precond", "1", "--fallback", "1", "--save-diagonal", diag]
    t0 = time.time()
    with open(f"logs_cone10_{scene}.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(cone):
        print(f"  [{scene}] SOLVE FAILED rc={rc}, see logs_cone10_{scene}.log", flush=True)
        return None, None
    print(f"  [{scene}] solved in {time.time()-t0:.0f}s", flush=True)
    return cone, diag


def main():
    enable_determinism()
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--output", default="artifacts/scannet/cone_10scene.json")
    a = p.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("voro", "run_voronoi_feature_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    seeds = [int(x) for x in a.seeds.split(",")]
    results = json.load(open(a.output)) if os.path.exists(a.output) else {}

    for scene in a.scenes.split(","):
        if scene in results:
            print(f"===== {scene} (already recorded) =====", flush=True)
            continue
        print(f"\n===== {scene} =====", flush=True)
        if not os.path.exists(f"output/scannet_{scene}_nonfrozen/model.pt"):
            print(f"  no checkpoint, skipping", flush=True)
            continue
        cone, diag = solve(scene, a.iters)
        if cone is None:
            continue
        ckpt = f"output/scannet_{scene}_nonfrozen"
        row = {}
        for tag, path in (("diagonal", diag), ("cone", cone)):
            per_seed = [m.evaluate(scene, ckpt, path, seed=s) for s in seeds]
            row[tag] = {cs: {"mean": float(np.mean([r[cs]["mIoU"] for r in per_seed])),
                             "std": float(np.std([r[cs]["mIoU"] for r in per_seed])),
                             "per_seed": [float(r[cs]["mIoU"]) for r in per_seed]}
                        for cs in CLASS_SETS}
        results[scene] = row
        d = [(row["cone"][cs]["mean"] - row["diagonal"][cs]["mean"]) * 100 for cs in CLASS_SETS]
        print(f"  diagonal " + "  ".join(
            f"{cs[12:]}={row['diagonal'][cs]['mean']*100:6.2f}+/-{row['diagonal'][cs]['std']*100:4.2f}"
            for cs in CLASS_SETS), flush=True)
        print(f"  cone     " + "  ".join(
            f"{cs[12:]}={row['cone'][cs]['mean']*100:6.2f}+/-{row['cone'][cs]['std']*100:4.2f}"
            for cs in CLASS_SETS), flush=True)
        print(f"  DELTA    19cls {d[0]:+.2f}   15cls {d[1]:+.2f}   10cls {d[2]:+.2f}", flush=True)
        with open(a.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\n=== {len(results)}-SCENE VERDICT (3 seeds, cone minus diagonal) ===", flush=True)
    hdr = f"{'scene':<16}" + "".join(f"{cs[12:]+'cls':>22}" for cs in CLASS_SETS)
    print(hdr); print("-" * len(hdr))
    acc = {cs: [] for cs in CLASS_SETS}
    for scene, row in results.items():
        line = f"{scene:<16}"
        for cs in CLASS_SETS:
            dd = (row["cone"][cs]["mean"] - row["diagonal"][cs]["mean"]) * 100
            acc[cs].append(dd)
            line += f"{row['diagonal'][cs]['mean']*100:7.2f}/{row['cone'][cs]['mean']*100:6.2f} ({dd:+5.2f})"
        print(line, flush=True)
    print("-" * len(hdr))
    line = f"{'MEAN DELTA':<16}"
    for cs in CLASS_SETS:
        line += f"{np.mean(acc[cs]):+22.2f}"
    print(line, flush=True)
    line = f"{'scenes improved':<16}"
    for cs in CLASS_SETS:
        line += f"{sum(1 for x in acc[cs] if x > 0):>12}/{len(acc[cs]):<10}"
    print(line, flush=True)
    print(f"\nwrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
