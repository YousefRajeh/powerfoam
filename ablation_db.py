"""SQLite store for the full ScanNet segmentation ablation.

WHY A DB. The ablation is a cross product -- 10 scenes x 6 reconstructions x
{adjacency} x {solver} x {grouping} x 3 class sets -- and the expensive pieces (GT->primitive
assignment, the adjacency graph, the lifted per-view statistics) are shared by many cells of
it. Recomputing them per cell would dominate the runtime, and keeping them in loose .pt files
is how we previously ended up with artifacts nobody could match back to the run that made
them. Every row here records the exact inputs that produced it, so a result can be traced or
recomputed, and `results` can be queried directly for the paper's tables.

CACHING CONTRACT. `assignments` and `adjacency` are keyed by (scene, recon) and are written
ONCE. In particular the Mahalanobis GT->Gaussian assignment for the unfrozen Gaussian arm is
computed a single time per scene and every downstream cell reads that stored assignment, so
the correspondence can never drift between methods within a scene.

PROVENANCE. `runs` stamps each invocation with git HEAD and start time; every result carries
run_id, so two sweeps months apart stay distinguishable and a bad sweep can be deleted
wholesale without touching the good one.
"""
import json
import os
import sqlite3
import subprocess
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "artifacts", "ablation.sqlite")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    git_head    TEXT,
    note        TEXT
);

-- One row per (scene, recon). Records the primitive count and where the checkpoint came
-- from, so a result can always be tied back to a specific reconstruction.
CREATE TABLE IF NOT EXISTS reconstructions (
    scene           TEXT NOT NULL,
    recon           TEXT NOT NULL,     -- pf_tfroz|pf_nonfroz|rf_froz|rf_unfroz|gs_froz|gs_unfroz
    kind            TEXT NOT NULL,     -- powerfoam|radfoam|gaussian
    n_primitives    INTEGER,
    ckpt_path       TEXT,
    created_at      REAL,
    PRIMARY KEY (scene, recon)
);

-- GT point -> primitive index. Computed ONCE per (scene, recon).
-- method: power_cell (powerfoam, exact argmin |x-c|^2 - r^2)
--         nearest_center (radfoam; identical to power_cell at r=0)
--         mahalanobis (gaussian; argmin over (x-mu)^T S^-1 (x-mu) + logdet)
CREATE TABLE IF NOT EXISTS assignments (
    scene        TEXT NOT NULL,
    recon        TEXT NOT NULL,
    method       TEXT NOT NULL,
    n_points     INTEGER,
    n_owned      INTEGER,
    path         TEXT NOT NULL,        -- .npy of int64 primitive index per GT point (-1 = unowned)
    seconds      REAL,
    created_at   REAL,
    PRIMARY KEY (scene, recon)
);

-- Adjacency graph over primitives. Computed ONCE per (scene, recon, complex).
-- complex: delaunay (exact power/Voronoi facet dual), alpha (delaunay filtered by
--          |xi-xj| < ri+rj), cech (AABB overlap superset, powerfoam only)
CREATE TABLE IF NOT EXISTS adjacency (
    scene        TEXT NOT NULL,
    recon        TEXT NOT NULL,
    complex      TEXT NOT NULL,
    n_edges      INTEGER,
    mean_degree  REAL,
    max_degree   INTEGER,
    path         TEXT NOT NULL,        -- CSR .pt with adjacent/offsets
    seconds      REAL,
    created_at   REAL,
    PRIMARY KEY (scene, recon, complex)
);

-- Solved per-primitive features. Keyed by (scene, recon, features, solver).
CREATE TABLE IF NOT EXISTS solves (
    scene        TEXT NOT NULL,
    recon        TEXT NOT NULL,
    features     TEXT NOT NULL,        -- feature artifact tag, e.g. ogl3
    solver       TEXT NOT NULL,        -- geometric_median|weighted|ridge|inverse_variance|consensus
    n_valid      INTEGER,
    path         TEXT NOT NULL,
    seconds      REAL,
    created_at   REAL,
    PRIMARY KEY (scene, recon, features, solver)
);

-- The actual ablation cell.
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    scene         TEXT NOT NULL,
    recon         TEXT NOT NULL,
    features      TEXT NOT NULL,
    solver        TEXT NOT NULL,
    grouping      TEXT NOT NULL,       -- kmeans320|pos_aware_64x5|grow_<complex>_thr<t>|none
    complex       TEXT,                -- adjacency used by the grouping (NULL if unused)
    class_set     TEXT NOT NULL,       -- opengaussian19|15|10
    n_classes     INTEGER,             -- classes actually present in this scene
    miou          REAL,
    macc          REAL,
    overall_acc   REAL,
    per_class     TEXT,                -- JSON {class_name: iou}
    seconds       REAL,
    created_at    REAL,
    UNIQUE (scene, recon, features, solver, grouping, class_set)
);

CREATE INDEX IF NOT EXISTS idx_results_scene   ON results(scene);
CREATE INDEX IF NOT EXISTS idx_results_recon   ON results(recon);
CREATE INDEX IF NOT EXISTS idx_results_group   ON results(grouping);
CREATE INDEX IF NOT EXISTS idx_results_cls     ON results(class_set);
CREATE INDEX IF NOT EXISTS idx_results_combo   ON results(recon, solver, grouping, class_set);

-- Anything that failed, so a gap in `results` is never silently mistaken for "not run yet".
CREATE TABLE IF NOT EXISTS failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,
    scene       TEXT, recon TEXT, stage TEXT,
    detail      TEXT,
    created_at  REAL
);
"""


def connect(path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=60.0)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def start_run(con, note=""):
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        head = None
    cur = con.execute("INSERT INTO runs (started_at, git_head, note) VALUES (?,?,?)",
                      (time.time(), head, note))
    con.commit()
    return cur.lastrowid


def record_failure(con, run_id, scene, recon, stage, detail):
    con.execute("INSERT INTO failures (run_id,scene,recon,stage,detail,created_at) "
                "VALUES (?,?,?,?,?,?)", (run_id, scene, recon, stage, str(detail)[:4000],
                                         time.time()))
    con.commit()


def have_result(con, scene, recon, features, solver, grouping, class_set):
    r = con.execute("SELECT 1 FROM results WHERE scene=? AND recon=? AND features=? AND "
                    "solver=? AND grouping=? AND class_set=?",
                    (scene, recon, features, solver, grouping, class_set)).fetchone()
    return r is not None


def put_result(con, run_id, scene, recon, features, solver, grouping, complex_, class_set,
               n_classes, miou, macc, acc, per_class, seconds):
    con.execute(
        "INSERT OR REPLACE INTO results (run_id,scene,recon,features,solver,grouping,complex,"
        "class_set,n_classes,miou,macc,overall_acc,per_class,seconds,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, scene, recon, features, solver, grouping, complex_, class_set, n_classes,
         miou, macc, acc, json.dumps(per_class), seconds, time.time()))
    con.commit()


def summary(con, class_set="opengaussian19", grouping=None):
    """Mean mIoU per (recon, solver, grouping) with the scene count -- the paper's Table 1."""
    q = ("SELECT recon, solver, grouping, COUNT(*) n, AVG(miou)*100 miou, AVG(macc)*100 macc "
         "FROM results WHERE class_set=? ")
    args = [class_set]
    if grouping:
        q += "AND grouping=? "
        args.append(grouping)
    q += "GROUP BY recon, solver, grouping ORDER BY miou DESC"
    return con.execute(q, args).fetchall()


if __name__ == "__main__":
    con = connect()
    print(f"schema ready at {DB_PATH}")
    for t in ("runs", "reconstructions", "assignments", "adjacency", "solves", "results",
              "failures"):
        n = con.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        print(f"  {t:<16} {n} rows")
