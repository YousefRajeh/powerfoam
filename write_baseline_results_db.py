"""Insert the baseline IoU + surface results into results_unified.

UNIT CONVENTIONS ARE TAKEN FROM THE EXISTING ROWS, not assumed: miou/macc are stored as PERCENT
(e.g. 32.58) while scd/mae/hd95 are stored in METRES (e.g. 0.462) and boundary_f1 as a fraction.
surface.json holds mIoU as a fraction, so it is scaled by 100 here; the distances are already in
metres and are inserted unchanged.

`coverage` carries kept_frac -- the fraction of the reconstruction's Gaussians the method retained.
For our own rows coverage is 100; for the baselines it ranges from 100 (LUDVIG, LangSplat) down to
~9 (VALA unfrozen). It is stored per row rather than folded into the metrics because the pruning
methods trade the two surface directions against each other -- better completeness (g->p), worse
accuracy (p->g) -- and a symmetric CD alone hides that. How to score points whose nearest Gaussian
was pruned remains deliberately open; this column is what makes the question answerable later.

Idempotent: rows are keyed on (scene, recon, method, class_set, source) and replaced, so re-running
after more baselines land will not duplicate.
"""
import json
import os
import sqlite3
from datetime import datetime

DB = r"D:\Downloads\powerfoam\artifacts\ablation.sqlite"
SURF = r"D:\Downloads\powerfoam\artifacts\baseline_eval\surface.json"
SOURCE = "baseline_eval:surface.json"

# feature level each method uses, from its own paper/scripts (not our default):
#   LangSplat  levels 1/2/3, per-level fields               (their 3-level protocol)
#   Occam      level 2, the code default in arguments/__init__.py
#   VALA       level 0, specified in their run_scannet.sh for ScanNet
#   LUDVIG     level 0, ScanNetDataset reads s_map[0]
FEATURES = {"langsplat": lambda lv: f"ae3_l{lv}", "occam": lambda lv: "l2",
            "vala": lambda lv: "l0", "ludvig": lambda lv: "l0"}
RECON = {"frozen": "gs_froz", "unfrozen": "gs_unfroz"}
NCLS = {"opengaussian19": 19, "opengaussian15": 15, "opengaussian10": 10}

rows = json.load(open(SURF))
con = sqlite3.connect(DB)
cur = con.cursor()
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

ins = rep = 0
for r in rows:
    method = r["baseline"] + (f"_l{r['level']}" if r["level"] else "")
    recon = RECON[r["arm"]]
    key = (r["scene"], recon, method, r["class_set"], SOURCE)
    cur.execute("DELETE FROM results_unified WHERE scene=? AND recon=? AND method=? "
                "AND class_set=? AND source=?", key)
    rep += cur.rowcount
    cur.execute(
        "INSERT INTO results_unified (scene, recon, features, solver, method, family, class_set, "
        "n_classes, miou, macc, coverage, scd, mae_pred2gt, mae_gt2pred, hd95, boundary_f1, "
        "n_missed, grouping, complex, assignment, masked, source, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (r["scene"], recon, FEATURES[r["baseline"]](r["level"]), "baseline", method, "baseline",
         r["class_set"], NCLS[r["class_set"]],
         100.0 * r["mIoU"], 100.0 * r["mAcc"],
         100.0 * r["kept_frac"] if r["kept_frac"] is not None else None,
         r["scd"], r["mae_pred2gt"], r["mae_gt2pred"], r["hd95"], r["boundary_f1"],
         r["n_missed"], None, None, "nearest_center", 1, SOURCE, now))
    ins += 1

con.commit()
print(f"inserted {ins} rows (replaced {rep} existing) into results_unified")
for m, c, mi, cov in cur.execute(
        "SELECT method, COUNT(*), ROUND(AVG(miou),2), ROUND(AVG(coverage),1) "
        "FROM results_unified WHERE source=? AND class_set='opengaussian19' "
        "GROUP BY method ORDER BY 3 DESC", (SOURCE,)):
    print(f"  {m:16s} n={c:3d}  mIoU19={mi:6.2f}  coverage={cov:5.1f}%")
con.close()
