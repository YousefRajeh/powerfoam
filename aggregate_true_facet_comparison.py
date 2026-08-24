"""Aggregate run_true_facet_comparison.py per-scene JSONs into the arm-vs-arm tables.

Reports, per class set:
  - per-scene mIoU for true-facet growing, Cech growing (both at every swept threshold)
    and the k-means pos-aware reference (per-seed, mean, std)
  - the n-scene mean for every arm
  - the decisive comparison at a SINGLE threshold chosen once across all scenes (picking
    the best threshold per scene would be selection on the test set and inflates the
    growing arms; the per-scene-best column is printed but flagged as an upper bound)

Noise band: the measured per-arm clustering-seed std in this project is ~0.71 mIoU (up to
1.64). Deltas below ~1.5 mIoU are NOT evidence and are labelled as such.
"""
import argparse
import glob
import json
import os

import numpy as np

NOISE = 1.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="artifacts/scannet/tfg_compare")
    p.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    p.add_argument("--output", default="artifacts/scannet/tfg_compare/summary.json")
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.json")))
    files = [f for f in files if not f.endswith("summary.json")]
    scenes = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        scenes[d["scene"]] = d
    if not scenes:
        print("no per-scene results found")
        return

    thrs = sorted({float(k.split("thr")[1]) for d in scenes.values()
                   for k in d["arms"] if k.startswith("grow_true_facet_thr")})
    seeds = sorted({int(k.split("seed")[1]) for d in scenes.values()
                    for k in d["arms"] if k.startswith("kmeans_pos_seed")})
    names = list(scenes.keys())
    print(f"scenes ({len(names)}): {', '.join(names)}")
    print(f"thresholds: {thrs}   kmeans seeds: {seeds}\n")

    summary = {"scenes": names, "thresholds": thrs, "seeds": seeds, "class_sets": {}}

    for cs in a.class_sets.split(","):
        def g(scene, key):
            arm = scenes[scene]["arms"].get(key)
            return None if arm is None else arm["metrics"][cs]["mIoU"] * 100

        cols = ([f"tf@{t}" for t in thrs] + [f"ce@{t}" for t in thrs]
                + [f"km_s{s}" for s in seeds] + ["km_mean", "km_std"])
        print(f"===== {cs} (mIoU %) =====")
        print("scene         " + "".join(f"{c:>9}" for c in cols))
        table = {}
        for scene in names:
            row = {}
            for t in thrs:
                row[f"tf@{t}"] = g(scene, f"grow_true_facet_thr{t}")
                row[f"ce@{t}"] = g(scene, f"grow_cech_thr{t}")
            km = [g(scene, f"kmeans_pos_seed{s}") for s in seeds]
            for s, v in zip(seeds, km):
                row[f"km_s{s}"] = v
            kmv = [v for v in km if v is not None]
            row["km_mean"] = float(np.mean(kmv)) if kmv else None
            row["km_std"] = float(np.std(kmv)) if kmv else None
            row["km_best"] = float(np.max(kmv)) if kmv else None
            table[scene] = row
            print(f"{scene:<14}" + "".join(
                f"{row[c]:>9.2f}" if row.get(c) is not None else f"{'--':>9}" for c in cols))

        means = {}
        for c in cols + ["km_best"]:
            vals = [table[s][c] for s in names if table[s].get(c) is not None]
            means[c] = float(np.mean(vals)) if len(vals) == len(names) else None
        print(f"{'MEAN':<14}" + "".join(
            f"{means[c]:>9.2f}" if means.get(c) is not None else f"{'--':>9}" for c in cols))

        # per-scene-best-threshold column (upper bound, selection on the test set)
        tf_best_per_scene = [max(table[s][f"tf@{t}"] for t in thrs) for s in names]
        ce_best_per_scene = [max(table[s][f"ce@{t}"] for t in thrs) for s in names]

        # single threshold chosen once, on the n-scene mean
        tf_thr = max(thrs, key=lambda t: means[f"tf@{t}"])
        ce_thr = max(thrs, key=lambda t: means[f"ce@{t}"])
        tf = means[f"tf@{tf_thr}"]
        ce = means[f"ce@{ce_thr}"]
        km = means["km_mean"]
        kmb = means["km_best"]

        print(f"\n  best single threshold, true facet : {tf_thr}  -> {tf:.2f}")
        print(f"  best single threshold, Cech       : {ce_thr}  -> {ce:.2f}")
        print(f"  k-means mean over {len(seeds)} seeds        : {km:.2f}"
              f"  (mean of per-scene best seed: {kmb:.2f})")
        print(f"  per-scene-best-thr UPPER BOUND    : tf {np.mean(tf_best_per_scene):.2f}"
              f"   ce {np.mean(ce_best_per_scene):.2f}   (selection on test set, not a result)")

        def verdict(delta, what):
            if abs(delta) < NOISE:
                return f"{delta:+.2f}  INSIDE the ~{NOISE} noise band -- NOT evidence ({what})"
            return f"{delta:+.2f}  outside the noise band ({what})"

        print(f"\n  true facet vs k-means MEAN : {verdict(tf - km, 'tf - km_mean')}")
        print(f"  true facet vs k-means BEST : {verdict(tf - kmb, 'tf - km_best')}")
        print(f"  true facet vs Cech         : {verdict(tf - ce, 'tf - ce')}")

        # per-scene sign test at the chosen threshold
        wins = sum(1 for s in names if table[s][f"tf@{tf_thr}"] > table[s]["km_mean"])
        print(f"  per-scene: true facet beats k-means mean on {wins}/{len(names)} scenes\n")

        summary["class_sets"][cs] = {
            "per_scene": table, "means": means,
            "best_thr_true_facet": tf_thr, "best_thr_cech": ce_thr,
            "true_facet_mean": tf, "cech_mean": ce,
            "kmeans_mean": km, "kmeans_bestseed_mean": kmb,
            "delta_tf_minus_kmeans_mean": tf - km,
            "delta_tf_minus_kmeans_best": tf - kmb,
            "delta_tf_minus_cech": tf - ce,
            "scenes_true_facet_beats_kmeans": wins,
        }

    with open(a.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
