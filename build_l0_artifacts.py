"""Build level-0 SAM feature artifacts for all 10 ScanNet scenes.

WHY LEVEL 0. OpenGaussian's ScanNet script uses `--sam_level 0` (`scripts/train_scannet.sh:35`,
with `train.py:398` commenting `# sam_level, leaf:3, scannet:0`); their LeRF scripts use level 3.
We have only ever scored level 3 and the all-levels sum, so their actual ScanNet configuration has
never been measured on our pipeline.

Note level 0 is NOT a granularity. The vendored SAM fork
(`segment-anything-langsplat/automatic_mask_generator.py:342-361`) splits SAM's 3-way multimask
head into s/m/l and then builds `default` by concatenating ALL THREE and running box NMS over the
union -- so level 0 is a mixed-granularity set. Measured on our own `_s.npy`: level 0 has 38.7
masks/image at 2,816 px median, versus level 3's 15.5 masks at 24,382 px.

Measured context this is being compared against (10-scene, protocol-correct plain argmax,
nonfrozen, best clustering per class set):
    all-levels  27.64 / 28.62 / 34.47 mIoU
    level 3     32.84 / 34.38 / 41.13 mIoU
"""
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
SCENES = ["scene0062_00", "scene0347_00", "scene0097_00", "scene0000_00", "scene0200_00",
          "scene0070_00", "scene0400_00", "scene0590_00", "scene0645_00", "scene0140_00"]


def main():
    for s in SCENES:
        out = f"artifacts/scannet/{s}/solved_geometric_median_nonfrozen_l0.pt"
        if os.path.exists(out):
            print(f"[SKIP] {s}", flush=True)
            continue
        cfg = f"output/scannet_{s}_nonfrozen/config.yaml"
        if not os.path.exists(cfg):
            print(f"[MISS] {s}: no nonfrozen config", flush=True)
            continue
        stats = f"artifacts/scannet/{s}/stats_l0.pt"
        t0 = time.time()
        print(f"[START] {s}", flush=True)
        r = subprocess.run(
            [PY, "accumulate_feature_stats_sam.py", "--scene", s, "--config", cfg,
             "--feature-folder", f"artifacts/scannet/{s}/openclip_features_sam",
             "--output", stats, "--sam-level", "0"],
            stdout=open(f"logs_l0_{s}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[FAIL] {s} lift rc={r.returncode}", flush=True)
            continue
        r = subprocess.run([PY, "solve_geometric_median.py", "--stats", stats, "--output", out],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # the per-scene stats file is ~GB-scale and only needed for the solve
        try:
            os.remove(stats)
        except OSError:
            pass
        print(f"[DONE ] {s} rc={r.returncode} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
