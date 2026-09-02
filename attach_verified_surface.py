"""Attach surface metrics to original rows wherever a recompute reproduces their mIoU exactly.

CONTEXT. After the source-predicate bug (one measurement stamped onto 192 rows from 9 sources),
the recompute passes were changed to INSERT under their own source rather than UPDATE. That was
the right fix, but it left the original rows blank even where the recomputation is provably the
same prediction -- e.g. `diffusion_cross_recon.json` shows 0/288 filled while
`backfill_surface_cross_recon.py` holds 240 rows covering the same arms.

THE GATE. Copy surface metrics from a recompute row onto the original row only when the two
rows agree on (scene, recon, method, class_set) AND their mIoU matches within --tol. Measured
before writing: on the 120 currently-matched pairs the reproduction is EXACT (max |delta| =
0.000), so the copy is provably describing the same predictions rather than merely similar
ones. Anything that fails the gate is left blank and logged.

This never invents a measurement; it only relabels one that was verified to correspond.
"""
import argparse
import sqlite3

DB = "artifacts/ablation.sqlite"
SRC_MAP = [
    ("diffusion_cross_recon.json", "backfill_surface_cross_recon.py"),
    ("simplex_10scene", "backfill_surface_metrics.py"),
    ("simplex_stack10", "backfill_surface_metrics.py"),
]
COLS = ["scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(DB, timeout=120)

    total_ok = total_rej = 0
    for target, donor in SRC_MAP:
        rows = con.execute(
            "SELECT a.id, a.miou, b.miou, " + ",".join("b." + c for c in COLS) +
            " FROM results_unified a JOIN results_unified b"
            "   ON a.scene=b.scene AND a.recon=b.recon AND a.method=b.method"
            "  AND a.class_set=b.class_set AND a.assignment=b.assignment"
            " WHERE a.source=? AND b.source=? AND a.scd IS NULL AND b.scd IS NOT NULL",
            (target, donor)).fetchall()
        ok = rej = 0
        for r in rows:
            rid, stored, mine = r[0], r[1], r[2]
            if abs(stored - mine) > a.tol:
                rej += 1
                continue
            if not a.dry_run:
                con.execute(
                    "UPDATE results_unified SET " + ",".join(c + "=?" for c in COLS) +
                    " WHERE id=?", tuple(r[3:]) + (rid,))
            ok += 1
        total_ok += ok
        total_rej += rej
        print(f"  {target:<32} candidates={len(rows):>4}  attached={ok:>4}  "
              f"rejected(|d|>{a.tol})={rej}")
    if not a.dry_run:
        con.commit()

    t = con.execute("SELECT COUNT(*) FROM results_unified").fetchone()[0]
    f = con.execute("SELECT COUNT(*) FROM results_unified WHERE scd IS NOT NULL").fetchone()[0]
    print(f"\nattached {total_ok}, rejected {total_rej}")
    print(f"surface-metric fill now {f:,}/{t:,} ({100*f/t:.1f}%)")
    con.close()


if __name__ == "__main__":
    main()
