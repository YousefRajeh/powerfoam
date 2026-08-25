"""Re-lift the powerfoam arms KEEPING the stats, then solve under every solver.

WHY RE-LIFT AT ALL. The existing powerfoam artifacts are geometric-median only: the earlier
pipeline deleted each scene's per-view statistics right after solving, because they are
1-3 GB apiece. Every other solver needs those statistics, so `solver` was never actually an
ablation axis for the foams -- and that matters now, because splat-distiller lifts Gaussians
with a WEIGHTED MEAN (distill.py:186 divides by accumulated weight). Comparing a
weighted-mean Gaussian row against a geometric-median foam row would confound representation
with solver. Solving both representations under both solvers separates them.

The stats are KEPT this time (user's call, disk is fine), so adding ridge / inverse-variance
later costs a solve rather than another lift.

Level 0 is correct and is NOT a typo for 3: the single-level artifact written by
SAM_ONLY_LEVEL=l holds its one granularity (level 3, whole-object) at index 0.
"""
import argparse
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3"

ARMS = {                     # recon tag -> (config dir, artifact suffix)
    "pf_nonfroz": ("output/scannet_{scene}_nonfrozen", "nonfrozen"),
    "pf_tfroz":   ("output/scannet_{scene}_truefrozen", "truefrozen"),
}

# Hardest-first, matching every other sweep in this project.
SCENES = ["scene0062_00", "scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00",
          "scene0590_00", "scene0000_00", "scene0097_00", "scene0200_00", "scene0400_00"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--arms", default="pf_nonfroz,pf_tfroz")
    ap.add_argument("--solvers", default="geometric_median,weighted")
    a = ap.parse_args()

    scenes = [s for s in (a.scenes.split(",") if a.scenes else SCENES) if s]
    arms = [x for x in a.arms.split(",") if x]

    for scene in scenes:
        feat = os.path.join(DATA, f"{scene}_colmap", FEAT_DIR)
        n_img = len(os.listdir(os.path.join(DATA, f"{scene}_colmap", "images")))
        if not os.path.isdir(feat) or len(os.listdir(feat)) < 2 * n_img:
            print(f"[skip] {scene}: features incomplete", flush=True)
            continue
        for arm in arms:
            cfg_dir, suffix = ARMS[arm]
            cfg = os.path.join(cfg_dir.format(scene=scene), "config.yaml")
            if not os.path.exists(cfg):
                print(f"[miss] {scene}/{arm}: no config", flush=True)
                continue
            art = f"artifacts/scannet/{scene}"
            os.makedirs(art, exist_ok=True)
            stats = f"{art}/stats_{suffix}_ogl3.pt"

            wanted = a.solvers.split(",")
            if all(os.path.exists(f"{art}/solved_{sv}_{suffix}_ogl3.pt") for sv in wanted):
                print(f"[skip] {scene}/{arm}: all solvers present", flush=True)
                continue

            if not os.path.exists(stats):
                t0 = time.time()
                print(f"[lift] {scene}/{arm} {time.strftime('%H:%M:%S')}", flush=True)
                r = subprocess.run(
                    [PY, "accumulate_feature_stats_sam.py", "--scene", scene,
                     "--config", cfg, "--feature-folder", feat,
                     "--output", stats, "--sam-level", "0"],
                    stdout=open(f"logs_ms_lift_{scene}_{arm}.log", "w"),
                    stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    print(f"  [FAIL] lift rc={r.returncode}", flush=True)
                    continue
                print(f"  lifted {(time.time()-t0)/60:.1f} min", flush=True)

            r = subprocess.run(
                [PY, "solve_from_stats.py", "--stats", stats,
                 "--out-template", f"{art}/solved_{{solver}}_{suffix}_ogl3.pt",
                 "--solvers", a.solvers],
                capture_output=True, text=True)
            for line in r.stdout.strip().splitlines():
                print(f"  {line}", flush=True)
            if r.returncode != 0:
                print(f"  [FAIL] solve rc={r.returncode}: {r.stderr[-300:]}", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
