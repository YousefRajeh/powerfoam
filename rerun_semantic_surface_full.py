"""Full per-scene rerun: re-extract SAM+CLIP -> accumulate L3 stats -> solve -> eval.

WHY THIS EXISTS
---------------
The semantic-surface metric needs the accumulator's reliability vector R. The per-scene
stats (~1.9GB each) and the SAM+CLIP feature caches (~5.3GB each) were deleted for 9 of the
10 scenes under the disk-pressure policy, so a first pass used a uniform-R fallback. The
uniform substitution measured cheap on scene0000_00 (mIoU within ~1 point) but it is still
a different configuration from the validated champion stack, so this script regenerates
everything properly.

It also RE-SOLVES rather than reusing the existing solved_*_l3.pt files. That is
deliberate: pairing a freshly-extracted R with features solved from the ORIGINAL extraction
would silently mix two extraction runs. Re-solving keeps each scene's stats, features and
reliability internally consistent, and costs only a few minutes per scene on top of the
extraction that dominates the runtime anyway.

DISK DISCIPLINE
Intermediates are KEPT by default (the user asked for them: the SAM+CLIP caches and the
accumulator stats are needed for downstream work, and re-extraction is the expensive step
we are here to avoid repeating). That means the footprint is CUMULATIVE, ~7.2GB per scene
(5.3GB features + 1.9GB stats) -- about 65GB for all nine. Free space is therefore checked
BEFORE each scene and the run stops cleanly, mid-list, rather than filling the disk; the
remaining scenes can be resumed once space is freed. Pass --purge to restore the old
delete-after-each-scene behaviour if space runs short.

Resumable: any scene whose per-scene result JSON already exists is skipped, so an
interrupted run continues where it stopped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\powerfoam")

SPLAT_DISTILLER = r"D:\Downloads\splat-distiller"
SAM_CKPT = r"D:\Downloads\powerfoam\checkpoints\sam_vit_h_4b8939.pth"
PF_PY = r"D:\conda\envs\powerfoam\python.exe"
SD_PY = r"D:\conda\envs\splat-distiller\python.exe"

SCENES = ["scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00", "scene0200_00",
          "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]

# Intermediates are kept, so each scene ADDS ~7.2GB rather than reusing one slot. Stop with
# real headroom left: a disk that fills mid-write corrupts a torch save, which already cost
# this project a set of artifacts once.
MIN_FREE_GB = 15.0


def free_gb(path="D:\\"):
    return shutil.disk_usage(path).free / 1e9


def run(cmd, cwd, tag, env=None):
    t0 = time.time()
    print(f"    [{tag}] {' '.join(str(c) for c in cmd[:4])} ...", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace")
    dt = time.time() - t0
    if r.returncode != 0:
        # print the tail rather than the whole log -- extraction logs are enormous
        print(f"    [{tag}] FAILED rc={r.returncode} after {dt/60:.1f}min", flush=True)
        print("\n".join(r.stdout.splitlines()[-25:]), flush=True)
        return False
    print(f"    [{tag}] ok ({dt/60:.1f}min)", flush=True)
    return True


def process_scene(scene, variant, purge):
    base = Path("artifacts/scannet") / scene
    base.mkdir(parents=True, exist_ok=True)
    feat_dir = base / "openclip_features_sam"
    stats_path = base / f"train_stats_sam_{variant}_l3.pt"
    solved_path = base / f"solved_geometric_median_{variant}_l3.pt"
    result_path = base / f"semantic_surface_{variant}.json"
    config = f"output/scannet_{scene}_{variant}/config.yaml"
    data_dir = f"data/scannet/{scene}_colmap"

    if result_path.exists():
        print(f"[{scene}] already done ({result_path}), skipping", flush=True)
        return True
    if not Path(config).exists():
        print(f"[{scene}] MISSING checkpoint config {config} -- skipping", flush=True)
        return False

    print(f"\n[{scene}] starting (free {free_gb():.1f}GB)", flush=True)

    # 1. SAM + OpenCLIP extraction. Reuse an existing cache ONLY if it is COMPLETE: an
    #    interrupted run leaves a partial directory, and treating "some files present" as
    #    "cached" would silently accumulate features over a subset of views -- a wrong
    #    result that looks like a successful run. Compare against the actual image count.
    n_img = len(list((Path(data_dir) / "images").iterdir()))
    n_npy = len(list(feat_dir.glob("*_s.npy"))) if feat_dir.exists() else 0
    if n_npy < n_img:
        if n_npy:
            print(f"    [{scene} extract] cache INCOMPLETE ({n_npy}/{n_img}), re-extracting",
                  flush=True)
        env = dict(os.environ, PYTHONPATH=SPLAT_DISTILLER)
        ok = run([SD_PY, "feature_extractor.py", "-s", str(Path(data_dir).resolve()),
                  "--model", "SAMOpenCLIP", "--ouput-dir", str(feat_dir.resolve()),
                  "--sam_ckpt_path", SAM_CKPT],
                 cwd=SPLAT_DISTILLER, tag=f"{scene} extract", env=env)
        if not ok:
            return False
    else:
        print(f"    [{scene} extract] cache complete ({n_npy}/{n_img}), reusing", flush=True)

    # 2. Accumulate at SAM level 3 only (the l/whole level -- the root-cause fix that gained
    #    ~10 mIoU; summing all 4 levels contaminates the pixel weights).
    if not stats_path.exists():
        if not run([PF_PY, "accumulate_feature_stats_sam.py", "--scene", scene,
                    "--config", config, "--feature-folder", str(feat_dir),
                    "--output", str(stats_path), "--sam-level", "3"],
                   cwd=".", tag=f"{scene} accumulate"):
            return False

    # 3. Re-solve, so R and the features come from the SAME extraction.
    if not run([PF_PY, "solve_geometric_median.py", "--stats", str(stats_path),
                "--output", str(solved_path)], cwd=".", tag=f"{scene} solve"):
        return False

    # 4. Semantic surface eval with the TRUE reliability (no --uniform-reliability).
    if not run([PF_PY, "eval_semantic_surface.py", "--scenes", scene,
                "--variant", variant, "--output", str(result_path)],
               cwd=".", tag=f"{scene} eval"):
        return False

    # 5. Working set is KEPT by default -- re-extraction is the expensive step and the
    #    caches/stats are wanted downstream. --purge restores the old free-as-you-go mode.
    if purge:
        if stats_path.exists():
            stats_path.unlink()
        if feat_dir.exists():
            shutil.rmtree(feat_dir, ignore_errors=True)
        print(f"[{scene}] DONE, purged working set (free {free_gb():.1f}GB)", flush=True)
    else:
        print(f"[{scene}] DONE, working set kept (free {free_gb():.1f}GB)", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--purge", action="store_true",
                   help="delete each scene's SAM cache + stats after its eval. OFF by "
                        "default: the intermediates are wanted downstream and re-extraction "
                        "is expensive. Turn on only if disk space runs short.")
    p.add_argument("--output", default="artifacts/scannet/semantic_surface_10scene_trueR.json")
    args = p.parse_args()

    scenes = args.scenes.split(",")
    done, failed = [], []
    for scene in scenes:
        if free_gb() < MIN_FREE_GB:
            print(f"STOPPING: only {free_gb():.1f}GB free (floor {MIN_FREE_GB}GB). "
                  f"Completed {len(done)}/{len(scenes)}; remaining: "
                  f"{[s for s in scenes if s not in done and s not in failed]}. "
                  f"Free space and re-run -- finished scenes are skipped.", flush=True)
            break
        (done if process_scene(scene, args.variant, args.purge) else failed).append(scene)

    # aggregate every per-scene result that exists, including scene0000_00's
    import numpy as np
    per_scene = {}
    for scene in ["scene0000_00"] + scenes:
        rp = Path("artifacts/scannet") / scene / f"semantic_surface_{args.variant}.json"
        if rp.exists():
            d = json.load(open(rp))
            for cs, sc in d["per_scene"].items():
                per_scene.setdefault(cs, {}).update(sc)

    summary = {}
    keys = ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "mIoU", "mAcc")
    for cs, per in per_scene.items():
        agg = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}
        agg["n_scenes"] = len(per)
        agg["mean_missed_per_scene"] = float(np.mean([v["n_missed"] for v in per.values()]))
        summary[cs] = agg
        print(f"{cs}: mIoU={agg['mIoU']*100:.2f} semanticCD={agg['scd']*100:.2f}cm "
              f"(p->g {agg['mae_pred2gt']*100:.2f} / g->p {agg['mae_gt2pred']*100:.2f}) "
              f"HD95={agg['hd95']*100:.2f}cm bF1={agg['boundary_f1']:.3f} (n={agg['n_scenes']})")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_scene": per_scene,
                   "uniform_reliability": False, "variant": args.variant}, f, indent=2)
    print(f"\nwrote {args.output}; done={len(done)} failed={failed}")


if __name__ == "__main__":
    main()
