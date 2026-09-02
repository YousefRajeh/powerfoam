"""SQLite tracking for the full hyperparameter sweep and ablation lattice.

WHY A DB. The sweep is (representations x solvers x hyperparameters x scenes x class sets) and will
run over many sessions. A DB gives three things a log file cannot: RESUMABILITY (a UNIQUE key on the
configuration means an interrupted run skips what it already did), QUERYABILITY (the paper plot is a
GROUP BY, not a re-run), and an audit trail of which constants produced which number -- which is
precisely the reporting gap the coauthor raised.

SCHEMA NOTE. Every knob that can change a number is a column, including the ones we currently hold
fixed, so a later sweep over them does not silently collide with earlier rows. `cfg_hash` is the
UNIQUE key over the full configuration; adding a knob later changes the hash and re-runs cleanly
rather than overwriting.
"""
import hashlib
import json
import os
import sqlite3

DB_PATH = os.path.join("artifacts", "sweep.db")

# Every field that can change a result. Order matters for the hash, so append -- never reorder.
CFG_FIELDS = [
    "representation",   # pf_unfroz | gs_unfroz | rf_unfroz
    "solver",           # geometric_median | weighted | inverse_variance | ridge
    "dataset",          # scannetpp | scannet
    "scene",
    "class_set",        # spp_top100 | spp_top20 | opengaussian19 | ...
    "lam",              # centering strength; 0 = off
    "csls_k",           # absolute top-K over primitives; 0 = off
    "csls_frac",        # if >0, csls_k was derived as frac * n_valid (recorded for the plot)
    "graph_k",          # kNN K for the primitive graph
    "alpha",            # diffusion fidelity
    "iters",            # diffusion iterations
    "rank_s",           # rank-encode template softness
    "use_consensus",    # neighbourhood feature consensus on/off
    "use_diffusion",    # rank-encode + diffuse on/off
    "text_transform",   # none | lowdin | whiten
    "text_alpha",       # whitening exponent when text_transform=whiten
    "coverage_k",       # evaluation-side coverage filter
    # APPENDED (never reorder -- cfg_hash depends on order). The neighbourhood construction is a
    # lattice axis, not an implementation detail: it feeds BOTH feature consensus and diffusion.
    # Without it in the hash, two different groupings at otherwise identical settings would collide
    # on the same cfg_hash and the second would be silently skipped as "already done".
    "grouping",         # knn_pos | knn_maha | knn_feat | radius | delaunay | kmeans | codebook
    # APPENDED. Which reliability fed feature consensus. `norm_proxy_constant` means the weighting
    # was INERT (see OPEN_ISSUES K) -- such rows are NOT comparable to `stats.reliability` rows, so
    # the field is part of the hash rather than a note.
    "reliability_source",   # stats.reliability | norm_proxy_live | norm_proxy_constant
    # APPENDED after a silent collision: the opacity-culled lattice inherited four cells from the
    # UNCULLED grid because the threshold was not part of the hash, so two different evaluation
    # POPULATIONS shared a cfg_hash and the second was skipped as "already done". Any knob that
    # changes WHICH GT POINTS ARE SCORED belongs here, not just knobs that change the prediction.
    "opacity_mask",         # 0.0 = no culling; 0.1 = OpenGaussian/NormLift rule
]

_TEXT_FIELDS = ("representation", "solver", "dataset", "scene", "class_set", "text_transform",
                "grouping", "reliability_source")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    cfg_hash   TEXT PRIMARY KEY,
    {os.linesep.join(f'    {f} TEXT,' if f in _TEXT_FIELDS else f'    {f} REAL,' for f in CFG_FIELDS)}
    n_classes  INTEGER,
    n_valid    INTEGER,
    miou       REAL,
    macc       REAL,
    phase      TEXT,
    script     TEXT,
    ts         REAL
);
CREATE INDEX IF NOT EXISTS idx_lookup ON runs(representation, solver, class_set, phase);
"""


def cfg_hash(cfg):
    """Stable hash over the full configuration. Missing fields default, so old rows stay valid."""
    payload = json.dumps([f"{k}={cfg.get(k)}" for k in CFG_FIELDS], sort_keys=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def connect(path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=60.0)
    con.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS will not add a column to a table that predates it, so migrate
    # explicitly. Rows written before `grouping` existed used the default builder, and their
    # cfg_hash was computed WITHOUT the field -- so they are backfilled to 'knn_pos' for
    # readability but will not collide with new rows, whose hashes include it.
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for f in CFG_FIELDS:
        if f not in have:
            con.execute(f"ALTER TABLE runs ADD COLUMN {f} "
                        f"{'TEXT' if f in _TEXT_FIELDS else 'REAL'}")
            if f == "grouping":
                con.execute("UPDATE runs SET grouping='knn_pos' WHERE grouping IS NULL")
            if f == "opacity_mask":
                # Everything written before this field existed was UNCULLED -- tag it so, and DELETE
                # the partial culled lattice rows: their hashes were computed without the field, so
                # on resume they would neither match nor be recognised as stale.
                con.execute("UPDATE runs SET opacity_mask=0.0 WHERE opacity_mask IS NULL")
                con.execute("DELETE FROM runs WHERE phase='lattice'")
    con.commit()
    return con


def already_done(con, cfg):
    return con.execute("SELECT 1 FROM runs WHERE cfg_hash=?", (cfg_hash(cfg),)).fetchone() is not None


def record(con, cfg, miou, macc, n_classes, n_valid, phase, script):
    import time
    h = cfg_hash(cfg)
    cols = ["cfg_hash"] + CFG_FIELDS + ["n_classes", "n_valid", "miou", "macc", "phase", "script", "ts"]
    vals = [h] + [cfg.get(f) for f in CFG_FIELDS] + [n_classes, n_valid, miou, macc,
                                                     phase, script, time.time()]
    con.execute(f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                vals)
    con.commit()


def summary(con, where="1=1"):
    """Mean mIoU per configuration, averaged over scenes -- the shape a paper plot needs."""
    return con.execute(f"""
        SELECT representation, solver, grouping, class_set, phase,
               lam, csls_k, csls_frac, graph_k, alpha, iters, rank_s,
               use_consensus, use_diffusion, text_transform,
               COUNT(*) n_scenes, AVG(miou) mean_miou
        FROM runs WHERE {where}
        GROUP BY representation, solver, grouping, class_set, phase,
                 lam, csls_k, csls_frac, graph_k, alpha, iters, rank_s,
                 use_consensus, use_diffusion, text_transform
        ORDER BY mean_miou DESC""").fetchall()
