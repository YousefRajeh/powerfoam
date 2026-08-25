"""Lift -> solve -> score every scene whose OpenGaussian-protocol L3 features are complete.

FEATURES THIS CONSUMES. `openclip_features_sam_l3`, extracted with the LangSplat/OpenGaussian
configuration (black mask fill, black crop pad) at SAM level 3 ONLY. Two consequences:

  * The artifact holds ONE level, stored at index 0 -- so the lift runs with `--sam-level 0`,
    NOT 3. Reading it with 3 would select an empty slice; accumulate_feature_stats_sam.py now
    raises on that rather than silently lifting zero features.
  * Level 3 is byte-identical to the level 3 of a full four-level run (verified: masks,
    embeddings and seg map all exactly equal), so these numbers are comparable to the
    four-level `_bb3` artifacts, just cheaper to produce.

WHY THIS CONFIGURATION. The white mask fill was an inherited splat-distiller deviation from
LangSplat, whose pipeline OpenGaussian consumes. Reverting it on scene0062_00 (nonfrozen,
geometric-median, plain argmax):

    white fill + black pad (previous)   27.27 / 27.27 / 38.61 mIoU (19/15/10cls)
    white fill + white pad              26.31 / 26.31 / 37.27
    black fill + black pad (protocol)   34.71 / 34.71 / 50.65

+7.4 to +12.2, 6/6 cells positive. That is a pilot on ONE scene -- the smallest of the ten --
and this script exists to turn it into a ten-scene number, because eleven single-scene results
in this project have reversed at ten-scene scale.

Idempotent: skips scenes already solved, skips scenes whose features are incomplete, and
deletes the multi-GB per-scene stats file after each solve.
"""
import argparse
import glob
import json
import os
import subprocess
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3"
SUFFIX = "_ogl3"          # solved_geometric_median_nonfrozen_ogl3.pt

# HARDEST FIRST. scene0347/0070/0140 are where coherence-gated growing collapsed (1.84 /
# 0.42 / 3.67 mIoU against a ~40 baseline) and 0645/0590 carry the lowest baseline mIoU and
# the largest cell counts. Scoring these early means a reversal of the scene0062 pilot shows
# up in the first few scenes rather than the last. scene0062_00 leads only because it is the
# reproduction check against the already-measured 34.71/34.71/50.65.
SCENES = ["scene0062_00", "scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00",
          "scene0590_00", "scene0000_00", "scene0097_00", "scene0200_00", "scene0400_00"]


def complete(scene):
    src = os.path.join(DATA, f"{scene}_colmap")
    feat = os.path.join(src, FEAT_DIR)
    if not os.path.isdir(feat):
        return False, 0, 0
    n_img = len(os.listdir(os.path.join(src, "images")))
    have = len(os.listdir(feat))
    return have >= 2 * n_img, have, 2 * n_img


def process(scene):
    art = f"artifacts/scannet/{scene}"
    solved = f"{art}/solved_geometric_median_nonfrozen{SUFFIX}.pt"
    if os.path.exists(solved):
        print(f"[SKIP ] {scene} already solved", flush=True)
        return True
    cfg = f"output/scannet_{scene}_nonfrozen/config.yaml"
    if not os.path.exists(cfg):
        print(f"[MISS ] {scene}: no nonfrozen config", flush=True)
        return False

    stats = f"{art}/stats{SUFFIX}.pt"
    t0 = time.time()
    print(f"[LIFT ] {scene}", flush=True)
    r = subprocess.run(
        [PY, "accumulate_feature_stats_sam.py", "--scene", scene, "--config", cfg,
         "--feature-folder", os.path.join(DATA, f"{scene}_colmap", FEAT_DIR),
         "--output", stats,
         # index 0 -- the single-level artifact's only row IS level 3
         "--sam-level", "0"],
        stdout=open(f"logs_ogl3_lift_{scene}.log", "w"), stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"[FAIL ] {scene} lift rc={r.returncode} (see logs_ogl3_lift_{scene}.log)",
              flush=True)
        return False

    r = subprocess.run([PY, "solve_geometric_median.py", "--stats", stats,
                        "--output", solved],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        os.remove(stats)
    except OSError:
        pass
    if r.returncode != 0:
        print(f"[FAIL ] {scene} solve rc={r.returncode}", flush=True)
        return False
    print(f"[SOLVE] {scene} done {(time.time()-t0)/60:.1f} min", flush=True)
    return True


def score(scene):
    env = dict(os.environ, ONLY_SCENES=scene, FEAT_SUFFIX=SUFFIX)
    r = subprocess.run([PY, "run_cluster_classify_eval.py"], env=env,
                       capture_output=True, text=True)
    out = r.stdout
    with open(f"logs_ogl3_eval_{scene}.log", "w") as f:
        f.write(out + "\n" + r.stderr)
    for line in out.splitlines():
        if "opengaussian" in line and "mIoU" in line:
            print("   " + line.strip(), flush=True)
    return r.returncode == 0


def aggregate():
    """Average whatever per-scene results exist so far, labelled with the true count."""
    rows = {}
    for p in glob.glob(f"artifacts/scannet/cluster_classify_1scene_*_avg{SUFFIX}.json"):
        d = json.load(open(p))
        for method, per_cs in d.items():
            for cs, v in per_cs.items():
                for sc, m in v["per_scene"].items():
                    rows.setdefault((method, cs), {})[sc] = m
    if not rows:
        return
    scenes = sorted({s for v in rows.values() for s in v})
    print(f"\n=== running average over {len(scenes)} scene(s): {', '.join(scenes)} ===")
    print("NormLift reference: 35.77 / 39.62 / 48.93 mIoU (19/15/10cls)")
    for method in sorted({m for m, _ in rows}):
        parts = []
        for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
            v = rows.get((method, cs), {})
            if v:
                mi = sum(x["mIoU"] for x in v.values()) / len(v)
                ma = sum(x["mAcc"] for x in v.values()) / len(v)
                parts.append(f"{cs} {mi*100:.2f}/{ma*100:.2f}")
        print(f"  {method:<16} " + "  ".join(parts))
    if len(scenes) < len(SCENES):
        print(f"  [partial: {len(scenes)}/{len(SCENES)} scenes -- NOT the final number]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true",
                   help="Keep polling for scenes as extraction finishes them.")
    p.add_argument("--poll", type=int, default=180)
    a = p.parse_args()

    while True:
        pending = []
        for s in SCENES:
            ok, have, need = complete(s)
            if not ok:
                pending.append(f"{s}({have}/{need})")
                continue
            # Score ONCE. process() returns True for an already-solved scene, so without this
            # guard every poll re-ran the eval for every finished scene -- pure GPU waste that
            # grows with each scene and contends with the extraction still in flight.
            result = f"artifacts/scannet/cluster_classify_1scene_{s}_avg{SUFFIX}.json"
            if os.path.exists(result):
                continue
            if process(s):
                score(s)
        aggregate()
        if not a.watch or not pending:
            if pending:
                print(f"\nnot yet extracted: {', '.join(pending)}")
            else:
                print("\n[ALL SCENES SCORED]")
            return
        print(f"\nwaiting on: {', '.join(pending)}", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
