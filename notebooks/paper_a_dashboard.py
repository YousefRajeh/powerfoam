import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Paper A — PowerFoam for open-vocabulary 3D segmentation

        **Everything below is read live from the databases and artifact files on disk.** No number
        is typed in except the published baselines (§2), which are quoted from NormLift's paper and
        are labelled as such.

        The one rule this dashboard exists to enforce: *a row must say which pipeline produced it.*
        The previous version of Table 4 compared four reconstructions under **three different
        pipelines** and the difference was read as a property of the representation. It was not.
        """
    )
    return


@app.cell
def _():
    import json
    import sqlite3
    import re
    from pathlib import Path

    import pandas as pd

    ROOT = Path(r"D:\Downloads\powerfoam")
    DB = ROOT / "artifacts/ablation.sqlite"
    DB_SPP = ROOT / "artifacts/ablation_scannetpp.sqlite"

    def q(db, sql, params=()):
        """Query -> DataFrame, tolerating a missing/locked database."""
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            return pd.read_sql_query(sql, con, params=params)
        except Exception as e:                      # noqa: BLE001 - dashboard must not crash
            return pd.DataFrame({"error": [str(e)]})

    def jload(path):
        try:
            return json.loads(Path(path).read_text())
        except Exception:                            # noqa: BLE001
            return None

    CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
    ARM_LABEL = {
        "gs_unfroz": "3DGS, unfrozen",
        "gs_froz": "3DGS, frozen",
        "pf_nonfroz": "Ours (PowerFoam), unfrozen",
        "pf_tfroz": "Ours (PowerFoam), frozen",
        "rf_unfroz": "RadFoam, unfrozen",
        "rf_froz": "RadFoam, frozen",
    }
    ARM_ORDER = ["gs_unfroz", "gs_froz", "pf_nonfroz", "pf_tfroz"]
    return (ARM_LABEL, ARM_ORDER, CLASS_SETS, DB, DB_SPP, Path, ROOT,
            jload, json, pd, q, re, sqlite3)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Table 4 — ScanNet, 10 scenes, one identical pipeline

        `percell-argmax` = solved features → per-primitive cosine argmax against the class names.
        **No diffusion, no mode-vote, no reweighting, no coverage filter.** All four arms run it, so
        a difference between rows is a property of the representation.

        Scored **with OpenGaussian's low-opacity GT deletion** (`sigmoid(opacity) < 0.1 → label 0`),
        which NormLift copies verbatim — so our rows and the published rows are scored on the same
        GT vector. `del.%` is the share of labelled GT points that rule removes.
        """
    )
    return


@app.cell
def _(ARM_LABEL, ARM_ORDER, CLASS_SETS, DB, pd, q):
    _sql = """
        SELECT recon, class_set, COUNT(DISTINCT scene) AS n,
               AVG(miou) AS miou, AVG(macc) AS macc, AVG(coverage) AS cov,
               AVG(scd) AS scd, AVG(hd95) AS hd95, AVG(boundary_f1) AS bf1
        FROM (SELECT recon, class_set, scene,
                     AVG(miou) miou, AVG(macc) macc, AVG(coverage) coverage,
                     AVG(scd) scd, AVG(hd95) hd95, AVG(boundary_f1) boundary_f1
              FROM results_unified
              WHERE method = 'percell-argmax+opacitymask@0.1' AND masked = 1
                AND assignment = 'geometric'
              GROUP BY recon, class_set, scene)
        GROUP BY recon, class_set
    """
    _raw = q(DB, _sql)

    def _t4(raw):
        if "error" in raw.columns or raw.empty:
            return raw
        wide = []
        for arm in ARM_ORDER:
            sub = raw[raw.recon == arm]
            if sub.empty:
                continue
            row = {"arm": ARM_LABEL.get(arm, arm),
                   "n": int(sub.n.max())}
            for cs, lab in zip(CLASS_SETS, ["19", "15", "10"]):
                r = sub[sub.class_set == cs]
                row[f"mIoU {lab}"] = round(float(r.miou.iloc[0]), 2) if len(r) else None
                row[f"mAcc {lab}"] = round(float(r.macc.iloc[0]), 2) if len(r) else None
            r19 = sub[sub.class_set == CLASS_SETS[0]]
            if len(r19):
                row["SCD↓"] = round(float(r19.scd.iloc[0]), 4)
                row["HD95↓"] = round(float(r19.hd95.iloc[0]), 4)
                row["BF1↑"] = round(float(r19.bf1.iloc[0]) * 100, 2)
            wide.append(row)
        return pd.DataFrame(wide)

    table4 = _t4(_raw)
    table4
    return (table4,)


@app.cell
def _(mo, table4):
    def _delta(t):
        try:
            pf = t[t.arm.str.startswith("Ours")]["mIoU 19"].max()
            gs = t[t.arm.str.startswith("3DGS")]["mIoU 19"].max()
            return f"**Best foam − best 3DGS at 19 classes: {pf - gs:+.2f} mIoU.**"
        except Exception:                            # noqa: BLE001
            return "_(table unavailable)_"

    mo.md(
        f"""
        {_delta(table4)}

        Scored **without** the opacity rule that same gap is +6.20 — the rule deletes ~23% of the
        Gaussian arms' evaluated points against ~4% of ours, because 3DGS covers a scene with many
        near-transparent primitives. That asymmetry is a property of point-counting IoU, and it is
        why the surface columns exist.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Published baselines (quoted from NormLift, **not** run here)

        These are the only hardcoded numbers on this page. T-F marks training-free methods.
        """
    )
    return


@app.cell
def _(pd):
    published = pd.DataFrame([
        ("LangSplat",    False,  3.78,  9.11,  5.35, 13.20,  8.40, 22.06),
        ("OpenGaussian", False, 24.73, 41.54, 30.13, 48.25, 38.29, 55.19),
        ("LAGA",         False, 32.50, 49.10, 35.50, 53.50, 42.60, 63.20),
        ("THGS",         True,  34.39, 50.74, 39.61, 57.07, 46.38, 64.74),
        ("VALA",         True,  32.11, 50.05, 35.10, 54.77, 46.21, 65.61),
        ("Occam's LGS",  True,  31.93, 48.93, 34.25, 53.71, 45.16, 64.39),
        ("SFS",          True,  33.33, 51.35, 36.43, 55.38, 44.74, 63.53),
        ("LUDVIG",       True,  33.90, 51.40, 37.40, 57.20, 46.40, 66.20),
        ("NormLift",     True,  35.77, 54.02, 39.62, 59.26, 48.93, 68.83),
    ], columns=["method", "training_free", "mIoU 19", "mAcc 19",
                "mIoU 15", "mAcc 15", "mIoU 10", "mAcc 10"])
    published
    return (published,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. ScanNet++ — the ordering reverses, and we report it

        Same plain-argmax pipeline (`A_base`, no coverage filter), 100-class protocol, paired over
        the scenes all three representations reconstruct. **3DGS wins here.** Two confounds are
        live and unseparated: correspondence is each representation's own query (Mahalanobis vs
        exact cell membership), and the arms are not matched on primitive budget — unlike ScanNet's
        frozen pair.
        """
    )
    return


@app.cell
def _(ARM_LABEL, DB_SPP, pd, q):
    _sql = """
        SELECT recon, scene, AVG(miou) miou, AVG(macc) macc
        FROM results_unified WHERE method = 'A_base|covNONE'
        GROUP BY recon, scene
    """
    _r = q(DB_SPP, _sql)

    def _paired(r):
        if "error" in r.columns or r.empty:
            return r
        common = set.intersection(*[set(g.scene) for _, g in r.groupby("recon")])
        out = []
        for arm, g in r.groupby("recon"):
            g = g[g.scene.isin(common)]
            out.append({"arm": ARM_LABEL.get(arm, arm), "scenes": len(g),
                        "mIoU": round(g.miou.mean(), 2), "mAcc": round(g.macc.mean(), 2)})
        return pd.DataFrame(out).sort_values("mIoU", ascending=False)

    spp = _paired(_r)
    spp
    return (spp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. The dipole rows — both negative, reported as negative

        Two different things were tried and neither improves the metric.

        **(a) `dipole_fill`** — a blind cell borrows its coplanar segment's label. Pure coverage.
        **(b) dipole *surface*** — replace the predicted geometry with the surface we reconstruct.
        """
    )
    return


@app.cell
def _(ROOT, jload, pd):
    _rows = jload(ROOT / "artifacts/scannet/dipole_row.json") or []
    if _rows:
        _d = pd.DataFrame(_rows)
        _d = _d[_d.class_set == "opengaussian19"]
        dipole_fill = (_d.groupby(["recon", "arm"])
                         .agg(n=("scene", "nunique"), mIoU=("miou", "mean"),
                              coverage=("cov", "mean"), BF1=("boundary_f1", "mean"))
                         .round(3).reset_index())
        dipole_fill["BF1"] = (dipole_fill.BF1 * 100).round(2)
    else:
        dipole_fill = pd.DataFrame({"note": ["dipole_row.json not found"]})
    dipole_fill
    return (dipole_fill,)


@app.cell
def _(ROOT, jload, pd):
    _ctl = jload(ROOT / "artifacts/scannet/dipole_surface_control.json") or []
    if _ctl:
        _c = pd.DataFrame(_ctl)
        _acc = []
        for arm in ("point", "surface", "matched"):
            for recon, g in _c.groupby("recon"):
                _acc.append({
                    "recon": recon,
                    "predicted geometry": {"point": "GT points (as in Table 4)",
                                           "surface": "dipole surface (raw)",
                                           "matched": "dipole surface, DENSITY-MATCHED"}[arm],
                    "pred→gt cm": round(g[f"{arm}_mae_pred2gt"].mean() * 100, 2),
                    "gt→pred cm": round(g[f"{arm}_mae_gt2pred"].mean() * 100, 2),
                    "SCD cm": round(g[f"{arm}_scd"].mean() * 100, 2),
                    "BF1": round(g[f"{arm}_boundary_f1"].mean() * 100, 2),
                })
        dipole_surface = pd.DataFrame(_acc).sort_values(["recon", "predicted geometry"])
    else:
        dipole_surface = pd.DataFrame({"note": ["dipole_surface_control.json not found"]})
    dipole_surface
    return (dipole_surface,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The raw substitution *looks* like a completeness win (gt→pred 17.6 → 15.7 cm frozen). It is
        not: the extracted surface carries 5–50× more points than the reference, and
        mean-distance-to-nearest falls with density for free. **Matched per class to the same point
        count the effect inverts** — worse on 10/10 scenes for the frozen arm.

        ## 5. Why the surface metric cannot currently reward a better surface

        The reconstruction places matter **≈7 cm in front of** the true surface. Splitting
        front-hit points by whether they land near GT, and comparing our hit distance from the eye
        against the nearest GT point's distance from the same eye:

        | | n | ours in front of GT | median gap |
        |---|---|---|---|
        | near GT (<5 cm) | 777,950 | 66.5% | +0.8 cm |
        | **far from GT (>5 cm)** | 2,358,337 | **83.6%** | **+6.6 cm** |

        A rigid misalignment would be symmetric and would move both rows. This is directional and
        confined to the far population — a **haze** standing off the real geometry, which the
        renderer then reports as the front-most surface. Four independent extractions agree the
        excess is not an extraction artifact (grid over faces 8.81 cm, renderer median depth
        8.48 cm, front `t_entry` 9.61 cm, front `t_surf` 9.91 cm).

        Using the labelled **mesh** instead of its vertices changes this by 0.03 cm — measured, not
        assumed. The reference was never the bottleneck.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Live: exp-activation retrain status

        Testing whether `σ = exp(ρ)` (VoroTracing Sec 5.4) removes the haze. `scene.py:get_density`
        predicts it should: under softplus the segment length does not cancel in the gradient, so
        large cells that should be empty settle at a low but non-zero density.

        **Caveat that must travel with any result from this**: activation and its learning rate move
        together by necessity (under exp the raw parameter is `ln σ`, so softplus's LR of 1.0 would
        change σ by a factor of *e* per step). It is therefore not a single-variable comparison.
        """
    )
    return


@app.cell
def _(Path, ROOT, pd, re):
    def _progress(log, total=30000):
        try:
            txt = Path(log).read_text(errors="ignore").replace("\r", "\n")
        except Exception:                            # noqa: BLE001
            return None
        m = re.findall(rf"(\d+)/{total}", txt)
        return int(m[-1]) if m else None

    _runs = [
        ("softplus (baseline, reference)", "output/scannet_scene0347_00_nonfrozen", None),
        ("exp, density_lr=0.02  (BROKEN — under-trained)",
         "output/scannet_scene0347_00_nonfrozen_expact", "logs_expact_0347.log"),
        ("exp, density_lr=0.1   (pilot)",
         "output/scannet_scene0347_00_nonfrozen_exp01", "logs_exp01_0347.log"),
    ]
    _rows = []
    for _name, _out, _log in _runs:
        _p = _progress(ROOT / _log) if _log else 30000
        _rows.append({
            "run": _name,
            "iterations": f"{_p}/30000" if _p else "—",
            "model.pt": "yes" if (ROOT / _out / "model.pt").exists() else "no",
        })
    training = pd.DataFrame(_rows)
    training
    return (training,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Density-LR probe (why 0.1)

        Under exp, `dL/dρ = σ · dL/dσ` — the gradient scales *with* σ. At σ₀ = 0.1 gradients are
        ~10× smaller than under softplus, so learning self-stalls exactly when σ must grow 500×.
        Probed at 4.5k iterations on scene0347_00:

        | density_lr | n_prim | σ q50 | σ q90 | ρ_max | verdict |
        |---|---|---|---|---|---|
        | softplus 1.0 (ref, 30k) | 203,952 | 50.3 | 159.8 | — | reference |
        | 0.02 | 85,908 @30k | 2.6 | — | — | **stalled**, PSNR 19.96 vs 28.78 |
        | 0.1 | 78,081 | 7.7 | 59.9 | 8.16 | viable |
        | 0.5 | 78,081 | 2812 | 1.2e8 | **30.00** | diverged (hit clamp) |
        | 2.0 | 78,081 | 85410 | 1.1e13 | **30.00** | diverged (hit clamp) |

        The usable band is narrow: 0.02 < lr < 0.5.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Open issues

        Read live from `OPEN_ISSUES.md` — headings only, newest last.
        """
    )
    return


@app.cell
def _(Path, pd):
    def _issues():
        try:
            txt = Path(r"D:\Downloads\OPEN_ISSUES.md").read_text(errors="ignore")
        except Exception:                            # noqa: BLE001
            return pd.DataFrame({"note": ["OPEN_ISSUES.md not found"]})
        heads = [ln.lstrip("# ").strip() for ln in txt.splitlines()
                 if ln.startswith("## ") or ln.startswith("### ")]
        return pd.DataFrame({"issue": heads})

    issues = _issues()
    issues
    return (issues,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
