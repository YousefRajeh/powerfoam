"""Re-lift PowerFoam from the TRAIN SPLIT ONLY, matching the baselines' view condition.

WHY. OpenGaussian trains with `--eval`, so it lifts from the train split (i % 8 != 0) while
its 3D evaluation scores the FULL GT point cloud -- `scripts/eval_scannet.py` reads the whole
labels.ply with no visibility or split filter. LUDVIG inherits that eval verbatim and uses the
same data. We have been lifting from ALL views and scoring the same target, i.e. building each
cell's feature from ~12% more observations than any baseline had. That is an advantage we were
taking silently, and it has to come out before our numbers are set against their table.

EXPECT THIS TO COST US. Fewer views per cell means a noisier lifted feature and a lower
classifiable fraction. The point is not to improve the number, it is to make the number
comparable. If the drop is large, that is itself worth reporting: it would mean a meaningful
part of our margin was view count rather than method.

ALIGNMENT HAZARD, already fixed in accumulate_feature_stats_sam.py. `cameras` comes from the
DataHandler and is split-filtered; the feature loader's `image_names` came from a plain
directory listing and was not. Under --split train that is 37 names against 32 cameras: the
assert catches it, but had the counts happened to match, every view would have been fed the
WRONG image's features with no error at all. Both are now filtered by the same i % 8 rule.

Writes `solved_geometric_median_nonfrozen_ts3.pt`, so the existing scorer picks it up with
FEAT_SUFFIX=_ts3 and the all-views result stays intact for the paired comparison.
"""
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3"

# hardest-first, as everywhere else in this project
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
          "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]


def main():
    for scene in SCENES:
        art = f"artifacts/scannet/{scene}"
        out = f"{art}/solved_geometric_median_nonfrozen_ts3.pt"
        if os.path.exists(out):
            print(f"[skip] {scene}", flush=True)
            continue
        cfg = f"output/scannet_{scene}_nonfrozen/config.yaml"
        feat = os.path.join(DATA, f"{scene}_colmap", FEAT_DIR)
        if not (os.path.exists(cfg) and os.path.isdir(feat)):
            print(f"[miss] {scene}", flush=True)
            continue
        stats = f"{art}/stats_ts3.pt"
        t0 = time.time()
        print(f"[lift] {scene} (train split) {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run(
            [PY, "accumulate_feature_stats_sam.py", "--scene", scene, "--config", cfg,
             "--feature-folder", feat, "--output", stats,
             "--sam-level", "0", "--split", "train"],
            stdout=open(f"logs_ts3_lift_{scene}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"  [FAIL] lift rc={r.returncode}", flush=True)
            continue
        nv = ""
        try:
            with open(f"logs_ts3_lift_{scene}.log") as f:
                for ln in f:
                    if "num_views=" in ln:
                        nv = ln.split("num_views=")[1].split()[0]
        except OSError:
            pass
        r = subprocess.run(
            [PY, "solve_from_stats.py", "--stats", stats,
             "--out-template", f"{art}/solved_{{solver}}_nonfrozen_ts3.pt",
             "--solvers", "geometric_median,weighted"],
            capture_output=True, text=True)
        for ln in r.stdout.strip().splitlines():
            print(f"  {ln}", flush=True)
        # keep the stats: they are the input to any further solver, and disk is not tight here
        print(f"[done] {scene} views={nv} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL LIFTED]", flush=True)

    env = dict(os.environ, FEAT_SUFFIX="_ts3")
    print("\n=== scoring the train-split lift, all 10 scenes ===", flush=True)
    r = subprocess.run([PY, "-u", "run_cluster_classify_eval.py"], env=env,
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if any(k in ln for k in ("scene averages", "feat_kmeans320 ", "pos_aware_64x5 ", "Wrote")):
            print(ln, flush=True)


if __name__ == "__main__":
    main()
