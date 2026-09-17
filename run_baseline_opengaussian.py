"""Train OpenGaussian on our 10 ScanNet scenes, with THEIR hyperparameters.

Every flag below is copied from `baselines/OpenGaussian/scripts/train_scannet.sh`, not chosen by
us -- 90k total steps, stage boundaries at 30k/50k/70k, k1=64 root x k2=5 leaf, half resolution,
`--frozen_init_pts` (one Gaussian per ground-truth vertex, no densification), `--sam_level 0`.
The point is to reproduce their method, so their settings win wherever they differ from ours,
including `--eval` holding out every eighth view.

`--sam_level 0` is theirs AND correct for our artifacts: these are single-level SAM extractions
that store the chosen granularity at index 0. Asking for 3 would select nothing and train on
silence, which is how a previous run in this project wasted hours.

DATA. `prepare_baseline_data.py` has already aliased each scene's `openclip_features_sam_l3` to
the `language_features/` name their reader expects (junction, no copy); the COLMAP layout is
already what they read. Verified per scene: image count equals feature-pair count.

ARM. `--frozen_init_pts` makes this the `gs_froz` counterpart of our rows -- one primitive per GT
vertex on both sides -- so its numbers belong beside our frozen arm, not the unfrozen one.

RESUMABLE. A scene whose output already holds the final checkpoint is skipped, so an interrupted
night continues instead of restarting. Their trainer writes `point_cloud/iteration_90000/`.
"""
import argparse
import os
import subprocess
import sys
import time

PY = r"C:\Users\rajehyl\AppData\Local\miniconda3\envs\gaussian_splatting\python.exe"
REPO = r"D:\Downloads\baselines\OpenGaussian"
DATA = r"D:\Downloads\powerfoam\data\scannet"
OUT = r"D:\Downloads\powerfoam\baseline_out\opengaussian"
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--iterations", type=int, default=90000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    done, failed = [], []
    for s in a.scenes.split(","):
        src = os.path.join(DATA, f"{s}_colmap")
        dst = os.path.join(OUT, s)
        final = os.path.join(dst, "point_cloud", f"iteration_{a.iterations}")
        if os.path.isdir(final):
            print(f"[skip] {s} already trained", flush=True)
            done.append(s)
            continue
        if not os.path.isdir(os.path.join(src, "language_features")):
            print(f"[MISS] {s}: no language_features (run prepare_baseline_data.py)", flush=True)
            failed.append(s)
            continue
        cmd = [PY, "train.py", "-s", src, "-m", dst, "-r", "2", "--frozen_init_pts",
               "--iterations", str(a.iterations),
               "--start_ins_feat_iter", "30000", "--start_root_cb_iter", "50000",
               "--start_leaf_cb_iter", "70000", "--sam_level", "0",
               "--root_node_num", "64", "--leaf_node_num", "5", "--pos_weight", "1.0",
               "--test_iterations", "30000", "--eval"]
        print(f"\n=== {s} === {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, errors="replace")
        dt = (time.time() - t0) / 60
        if r.returncode == 0 and os.path.isdir(final):
            print(f"[ok] {s} ({dt:.1f} min)", flush=True)
            done.append(s)
        else:
            print(f"[FAIL] {s} rc={r.returncode} ({dt:.1f} min)", flush=True)
            print("\n".join((r.stdout + r.stderr).splitlines()[-15:]), flush=True)
            failed.append(s)
    print(f"\ndone={len(done)} failed={len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
