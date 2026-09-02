"""Export `results_unified` to a multi-tab .xlsx: drop it in Drive and it becomes a Sheet.

LAYOUT
  BEST          one row per REPRESENTATION (its best config), plus NormLift's published row
  <arm>         one tab per reconstruction arm: pf_nonfroz, pf_tfroz, rf_froz, rf_unfroz,
                gs_froz, gs_unfroz -- every config, sortable
  PILOTS        configs measured on FEWER than 10 scenes, quarantined so they cannot be
                mistaken for full results
  ALL           the long-format union, for pivoting

WIDE FORMAT. One row per config with six metric columns (19/15/10 x mIoU/mAcc) rather than
one row per class set, so a config can be compared at a glance. Surface metrics follow.

THREE THINGS THIS GUARDS AGAINST, each of which had already produced a wrong "best":

1. OPACITY MASKING IS NOT COMPARABLE. OpenGaussian's mask deletes GT points below opacity
   0.1 from the metric, so a masked arm is scored on FEWER points -- on scene0062 it scores
   RadFoam on 39% of GT and PowerFoam on 93%. The DB's raw top row at 19cls is a masked
   config. Masked and unmasked are therefore never mixed in one ranking: `masked` is its own
   column and BEST reports unmasked by default.

2. SEED NOISE. 2,565 key groups in the source table have repeat runs at different k-means
   seeds, spread up to 18.5 mIoU (rf_froz/kmeans320/10cls on scene0347: 32.57 vs 51.10).
   Every aggregated row therefore carries n_runs and spread19 (max-min at 19cls) so a lucky
   seed is visible rather than hidden inside a mean.

3. SCENE COUNT. Only 234 of 297 source configs cover all 10 scenes, and scene0347_00 scores
   ~4 mIoU above the 10-scene average. Sorting per-scene rows by mIoU yields a scene-difficulty
   leaderboard, not a method ranking. n_scenes is on every row and <10 goes to PILOTS.

NormLift's published numbers are included as a reference row, marked as published rather than
measured here. They use OpenGaussian's released `language_features`, so their comparison
carries zero extraction variance while ours does not -- the row is a target, not a like-for-like
measurement.
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

DB = "artifacts/ablation.sqlite"
ARMS = ["pf_nonfroz", "pf_tfroz", "rf_froz", "rf_unfroz", "gs_froz", "gs_unfroz"]
ARM_LABEL = {"pf_nonfroz": "PowerFoam (unfrozen)", "pf_tfroz": "PowerFoam (frozen)",
             "rf_froz": "RadFoam (frozen)", "rf_unfroz": "RadFoam (unfrozen)",
             "gs_froz": "3DGS (frozen)", "gs_unfroz": "3DGS (unfrozen)"}
SURF = ["scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed"]
NORMLIFT = {"mIoU_19": 35.77, "mIoU_15": 39.62, "mIoU_10": 48.93,
            "mAcc_19": 54.02, "mAcc_15": 59.26, "mAcc_10": 68.83}


def wide(con):
    """Long -> wide: one row per config, six metric columns, surface metrics averaged."""
    df = pd.read_sql_query("SELECT * FROM results_unified", con)
    key = ["recon", "solver", "grouping", "complex", "method", "masked", "assignment",
           "features", "source"]
    for k in ("grouping", "complex", "assignment"):
        df[k] = df[k].fillna("")
    # per (config, class_set): mean over scenes, plus run-to-run spread
    g = df.groupby(key + ["n_classes"], dropna=False)
    agg = g.agg(miou=("miou", "mean"), macc=("macc", "mean"),
                n_scenes=("scene", "nunique"), n_runs=("miou", "size"),
                lo=("miou", "min"), hi=("miou", "max"),
                **{s: (s, "mean") for s in SURF}).reset_index()

    out = []
    for kv, sub in agg.groupby(key, dropna=False):
        row = dict(zip(key, kv))
        for nc in (19, 15, 10):
            r = sub[sub.n_classes == nc]
            row[f"mIoU_{nc}"] = round(float(r.miou.iloc[0]), 2) if len(r) else None
            row[f"mAcc_{nc}"] = (round(float(r.macc.iloc[0]), 2)
                                 if len(r) and pd.notna(r.macc.iloc[0]) else None)
        r19 = sub[sub.n_classes == 19]
        row["n_scenes"] = int(sub.n_scenes.max())
        row["n_runs"] = int(r19.n_runs.iloc[0]) if len(r19) else int(sub.n_runs.max())
        row["spread19"] = (round(float(r19.hi.iloc[0] - r19.lo.iloc[0]), 2)
                           if len(r19) else None)
        for s in SURF:
            v = sub[s].mean()
            row[s] = round(float(v), 4) if pd.notna(v) else None
        out.append(row)

    w = pd.DataFrame(out)
    cols = (["recon", "method", "assignment", "grouping", "complex", "solver", "masked"]
            + [f"mIoU_{n}" for n in (19, 15, 10)] + [f"mAcc_{n}" for n in (19, 15, 10)]
            + SURF + ["n_scenes", "n_runs", "spread19", "features", "source"])
    w = w[[c for c in cols if c in w.columns]]
    w.insert(0, "id", range(1, len(w) + 1))
    return w.sort_values("mIoU_19", ascending=False, na_position="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/ablation_results.xlsx")
    ap.add_argument("--min-scenes", type=int, default=10)
    a = ap.parse_args()
    con = sqlite3.connect(DB)
    w = wide(con)
    con.close()

    full = w[w.n_scenes >= a.min_scenes]
    pilots = w[w.n_scenes < a.min_scenes]

    # ---- BEST: the best UNMASKED config per representation.
    # Ranked by the MEAN over the three class sets, not by 19cls alone: several configs tie
    # at 19cls while differing by >2 mIoU at 15/10, and a 19cls-only sort broke those ties
    # arbitrarily (it picked a clustering arm over the modevote+diffusion stack that beats it
    # at both other class sets).
    cand = w[(w.masked == 0) & w.mIoU_19.notna()].copy()
    cand["mIoU_mean3"] = cand[["mIoU_19", "mIoU_15", "mIoU_10"]].mean(axis=1).round(2)
    # Prefer a 10-scene config; fall back to the best available so a representation is never
    # SILENTLY missing -- n_scenes on the row carries the caveat.
    # One row per (representation x ASSIGNMENT PROTOCOL). The two protocols score different
    # point sets (nearest_valid ~100% coverage vs geometric ~88%) and are worth ~4 mIoU, so
    # they are never collapsed into one ranking -- filter the `assignment` column to pick.
    picks = []
    for arm in ARMS:
        for prot in ("nearest_valid", "geometric"):
            sub = cand[(cand.recon == arm) & (cand.assignment == prot)]
            if not len(sub):
                continue
            full_sub = sub[sub.n_scenes >= a.min_scenes]
            pick = (full_sub if len(full_sub) else sub).sort_values(
                "mIoU_mean3", ascending=False).head(1)
            picks.append(pick)
    best = (pd.concat(picks).sort_values(["assignment", "mIoU_mean3"],
                                        ascending=[True, False])
            if picks else cand.head(0))
    best.insert(1, "representation", best.recon.map(ARM_LABEL).fillna(best.recon))
    nl = {c: None for c in best.columns}
    nl.update({"id": 0, "representation": "NormLift (published)", "recon": "—",
               "method": "published, OpenGaussian released language_features",
               "n_scenes": 10, "assignment": "frozen 1:1 (no assignment needed)",
               "source": "NormLift paper"}, **NORMLIFT)
    best = pd.concat([best, pd.DataFrame([nl])], ignore_index=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with pd.ExcelWriter(a.out, engine="openpyxl") as xl:
        best.to_excel(xl, sheet_name="BEST", index=False)
        # Arm tabs carry EVERY config for that arm, not just 10-scene ones: RadFoam has no
        # 10-scene coverage yet (4 scenes only), and an empty tab hides that rather than
        # showing it. n_scenes is the column to filter on; rows are sorted so 10-scene
        # configs come first.
        for arm in ARMS:
            sub = w[w.recon == arm].sort_values(
                ["n_scenes", "mIoU_19"], ascending=[False, False], na_position="last")
            if len(sub):
                sub.to_excel(xl, sheet_name=arm[:31], index=False)
        if len(pilots):
            pilots.to_excel(xl, sheet_name="PILOTS", index=False)
        con2 = sqlite3.connect(DB)
        pd.read_sql_query("SELECT * FROM results_unified", con2).to_excel(
            xl, sheet_name="ALL", index=False)
        con2.close()
        for ws in xl.book.worksheets:              # freeze header, size columns
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ln = max((len(str(c.value)) for c in col[:60] if c.value is not None),
                         default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(ln + 2, 9), 46)

    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"  BEST   {len(best)} rows (best unmasked config per representation + NormLift)")
    for arm in ARMS:
        n10 = int(((w.recon == arm) & (w.n_scenes >= a.min_scenes)).sum())
        print(f"  {arm:<12} {int((w.recon == arm).sum()):>5} configs "
              f"({n10} at {a.min_scenes} scenes)")
    print(f"  PILOTS {len(pilots):>5} configs measured on <{a.min_scenes} scenes")
    print("\n=== BEST tab ===")
    show = ["representation", "assignment", "method", "mIoU_19", "mIoU_15", "mIoU_10",
            "mIoU_mean3", "n_scenes"]
    print(best[[c for c in show if c in best.columns]].to_string(index=False, max_colwidth=44))


if __name__ == "__main__":
    main()
