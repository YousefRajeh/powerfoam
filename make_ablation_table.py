"""Render the ablation as three class-set tables -- every reconstruction always shown.

A METHOD WITH NO NUMBER IS STILL A ROW. Filtering the table to whatever happens to be
computed makes an unfinished arm look like a method that was never in the study, and makes a
FAILED arm indistinguishable from an unfinished one. Every (recon, solver) cell of the design
is therefore printed, and where there is no result the reason is taken from the `failures`
table and shown in place of the numbers.

LIKE-FOR-LIKE IS NOT ASSUMED, IT IS LABELLED. Arms are scored on different numbers of scenes
while the sweep is in progress, and a mean over 2 easy scenes is not comparable with a mean
over 10. Every row therefore carries its own scene count, and --common restricts all arms to
the scenes they share so the comparison is exact at the cost of coverage.

The NormLift row is their PUBLISHED number, not a re-run here: it carries protocol
assumptions the within-table comparisons do not, and is separated for that reason.
"""
import argparse
import glob
import json
from collections import defaultdict

import ablation_db as DB

RECONS = ["pf_nonfroz", "pf_tfroz", "rf_froz", "rf_unfroz", "gs_froz", "gs_unfroz"]
SOLVERS = ["geometric_median", "weighted"]
KIND = {"pf": "PowerFoam", "rf": "RadFoam", "gs": "Gaussian"}
NORMLIFT = {"opengaussian19": 35.77, "opengaussian15": 39.62, "opengaussian10": 48.93}
LABEL = {"opengaussian19": "19 CLASSES", "opengaussian15": "15 CLASSES",
         "opengaussian10": "10 CLASSES"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", action="store_true",
                    help="restrict every arm to the scenes ALL scored arms share")
    a = ap.parse_args()
    con = DB.connect()

    rows = list(con.execute("SELECT recon,solver,grouping,class_set,scene,miou,macc,scd,"
                            "boundary_f1,n_missed FROM results"))
    scenes_by = defaultdict(set)
    for r in rows:
        scenes_by[(r["recon"], r["solver"])].add(r["scene"])

    keep = None
    if a.common and scenes_by:
        keep = set.intersection(*scenes_by.values())

    # Why is a CELL missing? Keyed by (recon, solver) -- keying by recon alone reported the
    # last failure of ANY solver, so an arm whose weighted solve simply had not run yet was
    # labelled with an unrelated inverse_variance failure. A misattributed reason is worse
    # than none.
    why = {}
    for f in con.execute("SELECT recon, stage, detail FROM failures ORDER BY id"):
        stage = f["stage"] or ""
        if stage.startswith("solve:"):
            why[(f["recon"], stage.split(":", 1)[1])] = stage
        else:
            why[(f["recon"], None)] = stage

    agg = defaultdict(list)
    for r in rows:
        if keep is not None and r["scene"] not in keep:
            continue
        agg[(r["recon"], r["solver"], r["grouping"], r["class_set"])].append(r)

    # 10-scene scorer numbers (different assignment convention -- stated, not merged)
    scorer = defaultdict(dict)
    for p in glob.glob("artifacts/scannet/cluster_classify_1scene_*_avg_ogl3.json"):
        for m, per in json.load(open(p)).items():
            for cs, v in per.items():
                for sc, x in v["per_scene"].items():
                    scorer[(m, cs)][sc] = x["mIoU"]

    if keep is not None:
        print(f"restricted to {len(keep)} scenes common to all scored arms: "
              f"{sorted(s[5:9] for s in keep)}")
    print()

    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        print(f"############  {LABEL[cs]}  ############")
        print(f"{'repr':<11}{'arm':<12}{'solver':<10}{'best grouping':<22}"
              f"{'n':>3}{'mIoU':>8}{'mAcc':>7}{'scd cm':>8}{'bF1':>7}")
        print("-" * 89)
        for recon in RECONS:
            for solver in SOLVERS:
                cand = [(k, v) for k, v in agg.items()
                        if k[0] == recon and k[1] == solver and k[3] == cs]
                kind = KIND[recon[:2]]
                if not cand:
                    reason = why.get((recon, solver)) or why.get((recon, None)) or "not run"
                    reason = {"no_features": "not lifted yet",
                              "load": "checkpoint missing"}.get(reason, reason)
                    if reason.startswith("solve:"):
                        reason = "not solved yet"
                    print(f"{kind:<11}{recon:<12}{solver[:9]:<10}"
                          f"{'-- ' + reason:<22}{'-':>3}{'--':>8}{'--':>7}{'--':>8}{'--':>7}")
                    continue
                best = max(cand, key=lambda t: sum(x["miou"] for x in t[1]) / len(t[1]))
                k, v = best
                n = len({x["scene"] for x in v})
                mi = sum(x["miou"] for x in v) / len(v) * 100
                ma = sum(x["macc"] for x in v) / len(v) * 100
                sc = sum((x["scd"] or 0) for x in v) / len(v) * 100
                bf = sum((x["boundary_f1"] or 0) for x in v) / len(v)
                print(f"{kind:<11}{recon:<12}{solver[:9]:<10}{k[2]:<22}"
                      f"{n:>3}{mi:>8.2f}{ma:>7.2f}{sc:>8.2f}{bf:>7.3f}")
        print("-" * 89)
        for m, lbl in (("pos_aware_64x5", "pos_aware"), ("feat_kmeans320", "kmeans320")):
            v = scorer.get((m, cs), {})
            if v:
                print(f"{'PowerFoam':<11}{'pf_nonfroz':<12}{'geom*':<10}{lbl + ' (scorer)':<22}"
                      f"{len(v):>3}{sum(v.values())/len(v)*100:>8.2f}{'--':>7}{'--':>8}{'--':>7}")
        print(f"{'--':<11}{'NormLift':<12}{'--':<10}{'published, not re-run':<22}"
              f"{10:>3}{NORMLIFT[cs]:>8.2f}{'--':>7}{'--':>8}{'--':>7}")
        print()

    print("* scorer rows use the OTHER assignment convention: points in cells no camera ever")
    print("  saw are reassigned to the nearest observed cell, whereas the ablation leaves them")
    print("  unpredicted. Worth 1-3 mIoU; the two blocks are not interchangeable.")


if __name__ == "__main__":
    main()
