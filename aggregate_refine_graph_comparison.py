"""Aggregate run_refine_graph_comparison.py per-scene JSONs.

Both arms are DETERMINISTIC (mode-voting refinement is a fixed function of features,
reliability and graph; classification is a plain cosine argmax), and both are evaluated
against the SAME per-scene state.  So this is a paired comparison and the per-scene delta
carries no seed noise -- unlike the k-means arms elsewhere in this project, a sub-1-point
mean delta here is a real, reproducible difference rather than sampling.  The reported
significance is therefore a sign test / paired mean, not a comparison against the 1.5-point
clustering noise band.
"""
import argparse
import glob
import json
import os

import numpy as np

CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="artifacts/scannet/refine_graph")
    p.add_argument("--output", default="artifacts/scannet/refine_graph/summary.json")
    a = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.indir, "*.json"))
                   if not f.endswith("summary.json"))
    scenes = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        scenes[d["scene"]] = d
    if not scenes:
        print("no per-scene results")
        return
    names = sorted(scenes)
    print(f"scenes ({len(names)}): {', '.join(names)}\n")

    print("graph structure (per scene)")
    print(f"{'scene':<14}{'P':>10}{'deg_cech':>10}{'deg_true':>10}"
          f"{'maxd_cech':>11}{'maxd_true':>11}{'chg_cech':>10}{'chg_true':>10}")
    for s in names:
        g = scenes[s]["graph"]
        print(f"{s:<14}{scenes[s]['num_primitives']:>10}"
              f"{g['cech']['mean_degree']:>10.2f}{g['true_facet']['mean_degree']:>10.2f}"
              f"{g['cech']['max_degree']:>11}{g['true_facet']['max_degree']:>11}"
              f"{g['cech']['changed_frac']*100:>9.1f}%{g['true_facet']['changed_frac']*100:>9.1f}%")
    md = {k: float(np.mean([scenes[s]["graph"][k]["mean_degree"] for s in names]))
          for k in ("cech", "true_facet")}
    print(f"{'MEAN':<14}{'':>10}{md['cech']:>10.2f}{md['true_facet']:>10.2f}\n")

    summary = {"scenes": names, "class_sets": {}}
    for cs in CLASS_SETS:
        def v(s, arm):
            return scenes[s]["arms"][arm][cs]["mIoU"] * 100
        print(f"===== {cs} (mIoU %) =====")
        print(f"{'scene':<14}{'base':>9}{'ref_cech':>10}{'ref_true':>10}"
              f"{'true-cech':>11}{'true-base':>11}{'cech-base':>11}")
        rows = []
        for s in names:
            b, c, t = v(s, "base"), v(s, "refined_cech"), v(s, "refined_true_facet")
            rows.append((b, c, t))
            print(f"{s:<14}{b:>9.2f}{c:>10.2f}{t:>10.2f}"
                  f"{t-c:>+11.2f}{t-b:>+11.2f}{c-b:>+11.2f}")
        arr = np.array(rows)
        mb, mc, mt = arr.mean(0)
        d = arr[:, 2] - arr[:, 1]
        wins = int((d > 0).sum())
        print(f"{'MEAN':<14}{mb:>9.2f}{mc:>10.2f}{mt:>10.2f}"
              f"{mt-mc:>+11.2f}{mt-mb:>+11.2f}{mc-mb:>+11.2f}")
        print(f"  true-facet beats Cech on {wins}/{len(names)} scenes; "
              f"paired delta mean {d.mean():+.2f}, std {d.std(ddof=1):.2f}, "
              f"min {d.min():+.2f}, max {d.max():+.2f}\n")
        summary["class_sets"][cs] = {
            "mean_base": mb, "mean_refined_cech": mc, "mean_refined_true_facet": mt,
            "delta_true_minus_cech": float(mt - mc),
            "delta_true_minus_base": float(mt - mb),
            "delta_cech_minus_base": float(mc - mb),
            "paired_delta_mean": float(d.mean()),
            "paired_delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "wins_true_over_cech": wins, "n_scenes": len(names),
            "per_scene": {s: {"base": rows[i][0], "refined_cech": rows[i][1],
                              "refined_true_facet": rows[i][2]}
                          for i, s in enumerate(names)},
        }
    with open(a.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
