"""Backfill every result that was written to JSON/logs into the ablation DB, under ONE
self-describing method name per row.

WHY THIS EXISTS. Results have been accumulating in `artifacts/scannet/**.json` and log files
instead of `ablation.sqlite`. The consequence was concrete: the DB contains ZERO rows for
mode-voting or posterior diffusion, so the DB's best 10-scene means (38.44/40.99/48.24) are
below our actual documented best (38.95/41.76/49.54, mode-vote + diffusion on the true facet
graph). Anything built from the DB alone therefore pointed at the wrong method.

NAMING. One `method` string that says what the pipeline WAS, rather than a row of booleans:

    percell-argmax                                  bare per-primitive cosine argmax
    kmeans320 / pos_aware_64x5 / grow_delaunay@0.95 clustering-then-pool arms
    modevote(truefacet)                             NormLift reliability-guided mode voting
    diffusion(truefacet,s1000,a0.9)                 posterior simplex diffusion
    modevote(truefacet)+diffusion(truefacet,...)    the stack
    dipole-coplanar(tn0.98,td1.0)                   segments grown on dipole macro-geometry
    +dipfill                                        suffix: blind cells borrow the segment label

Sorting by this column groups a family; the parenthesised parts carry the settings that would
otherwise need their own columns.

SCALE HAZARD, handled explicitly. `diffusion_cross_recon.json` stores mIoU as a FRACTION
(0.4247) while `simplex_*` and `dipole_stack.json` store PERCENT (47.18). A magnitude
heuristic would misread a genuinely bad arm (LangSplat publishes 3.78 mIoU), so the scale is
declared per source, never inferred.

PROVENANCE. Every row records `source` (the file it came from) and `n_scenes` is computed at
aggregation time, so a 1-scene pilot can never be mistaken for a 10-scene result.

Writes a NEW table `results_unified`, leaving `results` untouched.
"""
import glob
import json
import re
import os
import sqlite3
import sys
import time

DB = "artifacts/ablation.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS results_unified (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scene TEXT, recon TEXT, features TEXT, solver TEXT,
  method TEXT,          -- self-describing pipeline name; see module docstring
  family TEXT,          -- coarse bucket for filtering: percell/cluster/modevote/diffusion/dipole/...
  class_set TEXT, n_classes INT,
  miou REAL, macc REAL, coverage REAL,
  scd REAL, mae_pred2gt REAL, mae_gt2pred REAL, hd95 REAL, boundary_f1 REAL, n_missed REAL,
  grouping TEXT, complex TEXT,   -- kept as their own columns for filtering, as well as in `method`
  assignment TEXT,      -- GT-point -> primitive protocol. THE TWO ARE NOT COMPARABLE:
                        --   nearest_valid  assign to the nearest cell THAT HAS A FEATURE
                        --                  (assign_points_to_power_cells(valid=valid_mask));
                        --                  every owned point is classifiable, coverage ~100%
                        --   geometric      assign to the nearest cell regardless of feature
                        --                  (valid=None), then drop points whose owner is
                        --                  featureless -- coverage ~88% on scene0347
                        -- Worth ~4 mIoU on scene0347 (46.25 vs 42.47), consistent with the
                        -- coverage law (mIoU ~= 0.53 x classifiable fraction).
  masked INT,           -- 1 = OpenGaussian opacity mask applied (scores FEWER GT points)
  source TEXT, created_at TEXT,
  UNIQUE(scene, recon, features, solver, method, class_set, source, assignment)
)
"""


def norm_cs(cs):
    return cs if cs.startswith("opengaussian") else f"opengaussian{cs}"


def family_of(method):
    m = method.lower()
    if "dipole" in m or "dipfill" in m:
        return "dipole"
    if "diffusion" in m and "modevote" in m:
        return "modevote+diffusion"
    if "diffusion" in m:
        return "diffusion"
    if "modevote" in m:
        return "modevote"
    if "percell" in m:
        return "percell"
    return "cluster"


class Sink:
    def __init__(self, con):
        self.con, self.n, self.skipped = con, 0, []

    def add(self, scene, recon, method, class_set, miou, macc, source,
            features="ogl3", solver="geometric_median", coverage=None, scale=1.0,
            surf=None, grouping=None, complex=None):
        if miou is None:
            return
        cs = norm_cs(class_set)
        self.con.execute(
            "INSERT OR IGNORE INTO results_unified (scene,recon,features,solver,method,family,"
            "class_set,n_classes,miou,macc,coverage,masked,"
            "scd,mae_pred2gt,mae_gt2pred,hd95,boundary_f1,n_missed,grouping,complex,"
            "assignment,source,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scene, recon, features, solver, method, family_of(method), cs,
             int(cs.replace("opengaussian", "")), miou * scale,
             (macc * scale) if macc is not None else None, coverage,
             1 if "opacitymask" in method else 0,
             *(surf or (None,) * 6), grouping, complex,
             # simplex_* passes valid=valid_mask into the assignment; everything else routes
             # through ablation_assign.py, which passes valid=None (verified at line 125).
             "nearest_valid" if str(source).startswith("simplex") else "geometric", source,
             time.strftime("%Y-%m-%d %H:%M:%S")))
        self.n += 1


def ingest_existing(sink):
    """The 5,130 rows already in `results`, renamed into the method scheme."""
    cur = sink.con.execute(
        "SELECT scene,recon,features,solver,grouping,complex,class_set,miou,macc,"
        "mask_opacity,kept_fraction,scd,mae_pred2gt,mae_gt2pred,hd95,boundary_f1,n_missed "
        "FROM results")
    for (scene, recon, feats, solver, grouping, cx, cs, miou, macc, mo, kf,
         scd, m1, m2, hd, bf, nm) in cur.fetchall():
        g = (grouping or "?").replace("_thr", "@").replace("grow_", "grow-")
        method = g if cx in (None, "", "none") else f"{g}[{cx}]"
        if mo is not None:
            method += f"+opacitymask@{mo}"
        sink.add(scene, recon, method, cs, miou, macc, "ablation.sqlite:results",
                 features=feats, solver=solver, coverage=(kf * 100 if kf else None),
                 scale=100.0,                       # `results` stores fractions
                 surf=(scd, m1, m2, hd, bf, nm), grouping=grouping, complex=cx)


def ingest_simplex(sink):
    """artifacts/scannet/simplex_*/scene*.json -- percent scale, arms keyed by graph+params."""
    for d in sorted(glob.glob("artifacts/scannet/simplex_*")):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)
        stacked = "stack" in tag
        for f in sorted(glob.glob(os.path.join(d, "scene*.json"))):
            try:
                j = json.load(open(f))
            except Exception as e:
                sink.skipped.append(f"{f}: {e}")
                continue
            scene = j.get("scene") or os.path.basename(f)[:-5]
            for arm, per in (j.get("arms") or {}).items():
                if not isinstance(per, dict):
                    continue
                if arm == "base":
                    method = "modevote(truefacet)" if stacked else "percell-argmax"
                else:
                    graph = "truefacet" if arm.startswith("true_facet") else \
                            ("cech" if arm.startswith("cech") else arm.split("_")[0])
                    params = arm.split("_", 2)[-1] if "_" in arm else ""
                    method = f"diffusion({graph},{params})"
                    if stacked:
                        method = f"modevote(truefacet)+{method}"
                for cs, r in per.items():
                    if isinstance(r, dict) and "mIoU" in r:
                        sf = tuple(r.get(k) if isinstance(r.get(k), (int, float)) else None
                                   for k in ("scd", "mae_pred2gt", "mae_gt2pred", "hd95",
                                             "boundary_f1", "n_missed"))
                        sink.add(scene, "pf_nonfroz", method, cs, r["mIoU"], r.get("mAcc"),
                                 tag,   # dir, NOT the per-scene file: a run spans scenes
                                 surf=sf if any(v is not None for v in sf) else None)


def ingest_dipole_stack(sink):
    """artifacts/scannet/dipole_stack.json -- percent, keys '<arm>|<class_set>'."""
    f = "artifacts/scannet/dipole_stack.json"
    if not os.path.exists(f):
        return
    NAME = {"base": "percell-argmax",
            "base+diff": "diffusion(truefacet,s1000,a0.9)",
            "modevote": "modevote(truefacet)",
            "modevote+diff": "modevote(truefacet)+diffusion(truefacet,s1000,a0.9)",
            "dipole_pool": "dipole-coplanar(tn0.98,td1.0)-pool"}
    for key, per in json.load(open(f)).items():
        arm, cs = key.split("|")
        base = arm[:-len("+dipfill")] if arm.endswith("+dipfill") else arm
        method = NAME.get(base, base)
        if arm.endswith("+dipfill"):
            method += "+dipfill(tn0.98,td1.0)"
        for scene, r in per.items():
            sink.add(scene, "pf_nonfroz", method, cs, r["mIoU"], r.get("mAcc"),
                     "dipole_stack.json", coverage=r.get("cov"))


def ingest_cross_recon(sink):
    """artifacts/scannet/diffusion_cross_recon.json -- FRACTION scale, all six arms."""
    f = "artifacts/scannet/diffusion_cross_recon.json"
    if not os.path.exists(f):
        return
    for key, per in json.load(open(f)).items():
        recon, cs = key.split("|")
        for scene, r in per.items():
            for tag, method in (("base", "percell-argmax"),
                                ("diffused", "diffusion(truefacet,s1000,a0.9)")):
                if tag in r:
                    sink.add(scene, recon, method, cs, r[tag]["mIoU"], r[tag].get("mAcc"),
                             "diffusion_cross_recon.json", scale=100.0)


def ingest_dipole_macro(sink):
    """artifacts/scannet/dipole_macro_10scene.json -- percent, keys '<arm>|<class_set>'."""
    f = "artifacts/scannet/dipole_macro_10scene.json"
    if not os.path.exists(f):
        return
    for key, per in json.load(open(f)).items():
        arm, cs = key.split("|")
        if arm == "percell":
            method = "percell-argmax"
        elif arm.startswith("coplanar"):
            t = arm.split()[-1].replace("/", ",td")
            method = f"dipole-coplanar(tn{t})-pool"
        elif arm.startswith("kmeans"):
            method = f"kmeans-matchedK({arm.split()[-1]})"
        else:
            method = arm
        for scene, v in per.items():
            sink.add(scene, "pf_nonfroz", method, cs,
                     v if isinstance(v, (int, float)) else v.get("mIoU"),
                     None if isinstance(v, (int, float)) else v.get("mAcc"),
                     "dipole_macro_10scene.json")


SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
CS_RE = re.compile(r"(?:opengaussian)?(19|15|10)(?:cls)?$", re.I)
RECONS = ("pf_nonfroz", "pf_tfroz", "rf_froz", "rf_unfroz", "gs_froz", "gs_unfroz",
          "nonfrozen", "frozen", "truefrozen")
RECON_ALIAS = {"nonfrozen": "pf_nonfroz", "frozen": "pf_tfroz", "truefrozen": "pf_tfroz"}


def _find(tokens, pool):
    for t in tokens:
        for p in pool:
            if p == t or p in str(t):
                return p
    return None


def _class_set(tokens):
    for t in tokens:
        m = CS_RE.search(str(t))
        if m:
            return f"opengaussian{m.group(1)}"
    return None


def ingest_generic(sink, root="artifacts/scannet"):
    """ONE recursive walker for all remaining JSON shapes (14 distinct nestings were found).

    Descends any nesting, and emits a row wherever it finds either a dict carrying an mIoU
    key or a float whose key path identifies a class set. scene / class_set / recon are read
    from the key path first and the filename second; whatever is left of the path becomes the
    method name, prefixed by the file stem so provenance is legible when sorting.

    Files already covered by an explicit ingester are skipped -- those produce better method
    names than a generic walker can infer.
    """
    handled = ("simplex_", "dipole_stack.json", "diffusion_cross_recon.json",
               "dipole_macro_10scene.json")
    for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(f, root).replace("\\", "/")
        if any(h in rel for h in handled):
            continue
        try:
            doc = json.load(open(f))
        except Exception as e:
            sink.skipped.append(f"{rel}: unreadable ({type(e).__name__})")
            continue
        stem = os.path.basename(f)[:-5]
        rows = []

        def walk(o, path):
            if isinstance(o, dict):
                if any(k in o for k in ("mIoU", "miou", "mIou")):
                    mi = o.get("mIoU", o.get("miou", o.get("mIou")))
                    ma = o.get("mAcc", o.get("macc"))
                    if isinstance(mi, (int, float)):
                        # Surface metrics usually sit in the SAME leaf as mIoU (these files
                        # were written by the semantic-surface evaluators). Harvesting them
                        # here fills ~1,100 rows with zero recomputation -- the first version
                        # of this walker read only mIoU/mAcc and left them null.
                        surf = tuple(o.get(k) if isinstance(o.get(k), (int, float)) else None
                                     for k in ("scd", "mae_pred2gt", "mae_gt2pred", "hd95",
                                               "boundary_f1", "n_missed"))
                        rows.append((path, mi, ma, o.get("cov"),
                                     surf if any(v is not None for v in surf) else None))
                    return
                for k, v in o.items():
                    walk(v, path + [k])
            elif isinstance(o, (int, float)) and path:
                if CS_RE.search(str(path[-1])):          # e.g. {"opengaussian19": 41.7}
                    rows.append((path, float(o), None, None, None))

        walk(doc, [])
        if not rows:
            continue
        # per-FILE scale decision, never per-value: a genuinely bad arm can score < 1.5
        # (LangSplat publishes 3.78 mIoU), so a magnitude heuristic on one value is unsafe.
        vals = [r[1] for r in rows]
        scale = 100.0 if max(vals) <= 1.5 else 1.0
        for path, mi, ma, cov, surf in rows:
            tok = list(path) + [stem] + rel.split("/")
            scene = _find(tok, [t for t in tok if SCENE_RE.fullmatch(str(t))]) or \
                (SCENE_RE.search(rel).group(0) if SCENE_RE.search(rel) else None)
            cs = _class_set(list(path) + [stem])
            if not (scene and cs):
                continue                                  # not a per-scene scored result
            rc = _find(tok, RECONS)
            rc = RECON_ALIAS.get(rc, rc) or "pf_nonfroz"
            rest = [str(p) for p in path
                    if str(p) != scene and not CS_RE.search(str(p)) and str(p) != rc]
            method = f"{stem}:{'/'.join(rest)}" if rest else stem
            src = SCENE_RE.sub("<scene>", rel)   # so a per-scene file set is ONE run
            sink.add(scene, rc, method, cs, mi, ma, src, coverage=cov, scale=scale,
                     surf=surf)


def main():
    if not os.path.exists(DB):
        sys.exit(f"missing {DB}")
    con = sqlite3.connect(DB)
    con.execute(SCHEMA)
    sink = Sink(con)
    for fn in (ingest_existing, ingest_simplex, ingest_dipole_stack,
               ingest_cross_recon, ingest_dipole_macro, ingest_generic):
        before = sink.n
        try:
            fn(sink)
        except Exception as e:
            sink.skipped.append(f"{fn.__name__}: {type(e).__name__} {e}")
        print(f"  {fn.__name__:<22} +{sink.n - before:,} rows offered")
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM results_unified").fetchone()[0]
    print(f"\nresults_unified now holds {total:,} unique rows "
          f"({sink.n:,} offered, duplicates ignored)")
    if sink.skipped:
        print("SKIPPED (logged, never silent):")
        for s in sink.skipped[:20]:
            print("  ", s)
    print("\n=== best 10-scene mean per class set, AFTER backfill ===")
    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        r = con.execute(
            "SELECT method,recon,ROUND(AVG(miou),2),COUNT(DISTINCT scene) FROM results_unified "
            "WHERE class_set=? GROUP BY method,recon,solver HAVING COUNT(DISTINCT scene)=10 "
            "ORDER BY AVG(miou) DESC LIMIT 1", (cs,)).fetchone()
        if r:
            print(f"  {cs.replace('opengaussian',''):>2}cls: {r[2]:6.2f}  {r[1]:<12} {r[0]}")
    con.close()


if __name__ == "__main__":
    main()
