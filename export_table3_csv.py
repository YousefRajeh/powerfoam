"""Export Table 3 (tab:3dseg) as CSV: per-method aggregate + per-scene detail.

Two files:
  table3_summary.csv  one row per method, mirroring the paper table -- mIoU/mAcc at 19/15/10
                      classes plus the 19-class surface columns.
  table3_per_scene.csv one row per (method, scene, class_set), the underlying 10-scene detail.

PROVENANCE IS A COLUMN, not a footnote. The prior-work mIoU/mAcc are published values transcribed
from the paper; their surface columns and every baseline number in the per-scene file are our own
re-runs. Mixing the two silently would make the table unreadable later, so `source` marks each.

`coverage_pct` is the fraction of the reconstruction's Gaussians the method retained. VALA and
Occam's prune to 9.3% and 36.6%; the surface columns for those rows therefore describe a much
thinner support than the 100% rows, and the two are not matched. Left as a column rather than
folded into the metrics, for the same reason the paper leaves the scoring of pruned points open.
"""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SURF = os.path.join(HERE, "artifacts", "baseline_eval", "surface.json")
OUT_SUM = os.path.join(HERE, "artifacts", "baseline_eval", "table3_summary.csv")
OUT_SCENE = os.path.join(HERE, "artifacts", "baseline_eval", "table3_per_scene.csv")

# Rows already in the paper. Prior-work mIoU/mAcc as published; our own rows measured by us.
# surface = (scd, hd95, bf1, del_pct) where known.
PAPER = [
    # name,            train_free, m19, a19, m15, a15, m10, a10, surface, source
    ("OpenGaussian",   False, 24.73, 41.54, 30.13, 48.25, 38.29, 55.19, None, "published"),
    ("LAGA",           False, 32.50, 49.10, 35.50, 53.50, 42.60, 63.20, None, "published"),
    ("THGS",           True,  34.39, 50.74, 39.61, 57.07, 46.38, 64.74, None, "published"),
    ("SFS",            True,  33.33, 51.35, 36.43, 55.38, 44.74, 63.53, None, "published"),
    ("NormLift",       True,  35.77, 54.02, 39.62, 59.26, 48.93, 68.83, None, "published"),
    ("3DGS, unfrozen", True,  32.29, 52.16, 34.65, 55.00, 41.71, 60.70, (0.4477, 2.1550, 44.75, 23.7), "ours"),
    ("3DGS, frozen",   True,  35.31, 57.23, 37.59, 59.72, 45.25, 66.09, (0.4326, 2.0991, 47.24, 22.7), "ours"),
    ("Ours, unfrozen", True,  36.31, 58.43, 39.02, 61.19, 46.35, 67.35, (0.3961, 2.0128, 47.41, 11.0), "ours"),
    ("Ours, frozen",   True,  37.60, 59.38, 40.48, 62.43, 48.16, 68.32, (0.4139, 1.9853, 48.29, 4.4), "ours"),
]
# published mIoU/mAcc for the four we re-ran (surface columns come from our runs)
PUBLISHED_BASELINE = {
    "LangSplat":   (False, 3.78, 9.11, 5.35, 13.20, 8.40, 22.06),
    "VALA":        (True, 32.11, 50.05, 35.10, 54.77, 46.21, 65.61),
    "Occam's LGS": (True, 31.93, 48.93, 34.25, 53.71, 45.16, 64.39),
    "LUDVIG":      (True, 33.90, 51.40, 37.40, 57.20, 46.40, 66.20),
}
# which run of ours backs each baseline's surface columns (its own paper's arm / level)
TAG_FOR = {"LangSplat": "langsplat_frozen_l3", "VALA": "vala_unfrozen",
           "Occam's LGS": "occam_unfrozen", "LUDVIG": "ludvig_frozen"}

rows = json.load(open(SURF))
by = defaultdict(list)
for r in rows:
    by[(r["tag"], r["class_set"])].append(r)


def mean(tag, cs, key):
    v = by.get((tag, cs), [])
    return (sum(r[key] for r in v) / len(v)) if v else None


HDR = ["method", "train_free", "arm", "level", "coverage_pct",
       "mIoU_19", "mAcc_19", "mIoU_15", "mAcc_15", "mIoU_10", "mAcc_10",
       "SCD_m", "HD95_m", "BF1_pct", "del_pct",
       "mIoU_19_ours_rerun", "n_scenes", "source"]

out = []
for name, tf, m19, a19, m15, a15, m10, a10, surf, src in PAPER:
    d = dict.fromkeys(HDR, "")
    d.update(method=name, train_free=int(tf), arm="", level="", coverage_pct=100.0,
             mIoU_19=m19, mAcc_19=a19, mIoU_15=m15, mAcc_15=a15, mIoU_10=m10, mAcc_10=a10,
             n_scenes=10, source=src)
    if surf:
        d["SCD_m"], d["HD95_m"], d["BF1_pct"], d["del_pct"] = surf
    out.append(d)

for name, tag in TAG_FOR.items():
    tf, m19, a19, m15, a15, m10, a10 = PUBLISHED_BASELINE[name]
    arm = "frozen" if "frozen" in tag and "unfrozen" not in tag else "unfrozen"
    lvl = tag.rsplit("_l", 1)[1] if "_l" in tag else ""
    d = dict.fromkeys(HDR, "")
    d.update(method=name, train_free=int(tf), arm=arm, level=lvl,
             coverage_pct=round(100 * (mean(tag, "opengaussian19", "kept_frac") or 0), 1),
             mIoU_19=m19, mAcc_19=a19, mIoU_15=m15, mAcc_15=a15, mIoU_10=m10, mAcc_10=a10,
             SCD_m=round(mean(tag, "opengaussian19", "scd"), 4),
             HD95_m=round(mean(tag, "opengaussian19", "hd95"), 4),
             BF1_pct=round(100 * mean(tag, "opengaussian19", "boundary_f1"), 2),
             del_pct="",
             mIoU_19_ours_rerun=round(100 * mean(tag, "opengaussian19", "mIoU"), 2),
             n_scenes=len(by[(tag, "opengaussian19")]),
             source="mIoU published / surface ours")
    out.append(d)

with open(OUT_SUM, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HDR)
    w.writeheader()
    w.writerows(out)

PH = ["tag", "baseline", "arm", "level", "scene", "class_set", "kept_frac", "n_kept", "n_original",
      "mIoU", "mAcc", "scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1",
      "n_missed", "n_classes_present"]
with open(OUT_SCENE, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=PH, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["tag"], r["class_set"], r["scene"])):
        w.writerow(r)

print(f"{len(out)} rows -> {OUT_SUM}")
print(f"{len(rows)} rows -> {OUT_SCENE}")
