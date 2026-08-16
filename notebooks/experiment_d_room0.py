import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # Experiment D: room_0 -- Feature Foam vs. Splat Feature Solver / gsplat

        Comparing Feature Foam (CLIP features lifted onto PowerFoam primitives) against the
        Splat Feature Solver baseline (`splat-distiller`, lifting onto a real 3DGS
        reconstruction) on Replica `room_0`. Same 900 train / 113 test views for every method.

        Re-run this notebook any time metrics files are updated (e.g. once the real
        Splat Feature Solver mIoU is computed) -- it reads straight from disk, no manual
        number-copying.
        """
    )
    return


@app.cell
def _():
    import json
    import pandas as pd
    from pathlib import Path

    ROOT = Path(r"D:\Downloads\powerfoam")

    def read_json(path):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            return None

    def read_psnr_log(path, key="Average PSNR"):
        try:
            for line in Path(path).read_text(errors="ignore").splitlines():
                if key in line:
                    return float(line.split(":")[-1].strip())
        except FileNotFoundError:
            return None
        return None

    feature_foam_miou = read_json(ROOT / "artifacts/replica_room0/miou_powerfoam.json")
    gsplat_baseline_miou = read_json(ROOT / "artifacts/replica_room0_gsplat/miou_gsplat.json")
    splatdistiller_val = read_json(ROOT / "artifacts/room0_splatdistiller/stats/val_step29999.json")
    # Computed with the SAME OpenCLIP features Feature Foam used (ViT-B-16-quickgelu/openai,
    # dense patch tokens) fed into the Splat Feature Solver's own weighted-average solve step
    # (distill.py) -- apples-to-apples: only the 3D representation + lifting algorithm differ.
    splatdistiller_miou = read_json(ROOT / "artifacts/room0_splatdistiller/miou_splatdistiller_featurefoamfeatures.json")

    feature_foam_render = {}
    try:
        for line in (ROOT / "output/room_0/metrics.txt").read_text().splitlines():
            k, v = line.split(":")
            feature_foam_render[k.strip()] = float(v.strip())
    except FileNotFoundError:
        pass

    rows = [
        {
            "method": "Feature Foam (PowerFoam)",
            "psnr": feature_foam_render.get("Average PSNR"),
            "ssim": feature_foam_render.get("Average SSIM"),
            "lpips": feature_foam_render.get("Average LPIPS"),
            "mIoU": (feature_foam_miou or {}).get("mean_iou"),
        },
        {
            "method": "Splat Feature Solver (real, room0_splatdistiller)",
            "psnr": (splatdistiller_val or {}).get("psnr"),
            "ssim": (splatdistiller_val or {}).get("ssim"),
            "lpips": (splatdistiller_val or {}).get("lpips"),
            "mIoU": (splatdistiller_miou or {}).get("mean_iou"),  # pending
        },
        {
            "method": "gsplat_baseline (simplified, in-repo -- NOT Splat Feature Solver)",
            "psnr": read_psnr_log(ROOT / "gsplat_baseline/eval_psnr_gsplat_baseline.log"),
            "ssim": None,
            "lpips": None,
            "mIoU": (gsplat_baseline_miou or {}).get("mean_iou"),
        },
    ]
    df = pd.DataFrame(rows)
    df
    return ROOT, df, json, pd, read_json, read_psnr_log


@app.cell
def _(df, mo):
    mo.md(
        f"""
        {'**Splat Feature Solver mIoU is still pending** -- the feature extraction/mIoU rebuild ' if df["mIoU"].isna().any() else ''}
        Values read live from the artifact files on disk. `None` means that metric hasn't been
        computed/saved yet for that method.
        """
    )
    return


@app.cell
def _(df, mo):
    import altair as alt

    plot_df = df.melt(id_vars="method", value_vars=["psnr", "mIoU"], var_name="metric", value_name="value").dropna()
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("value:Q"),
            color="method:N",
            column="metric:N",
        )
        .properties(width=280, height=280)
    )
    mo.ui.altair_chart(chart)
    return alt, chart, plot_df


@app.cell
def _(mo):
    mo.md("## Notes / next experiments")
    return


@app.cell
def _(mo):
    mo.md(
        """
        - [ ] Rebuild `test_operator`/`solved_weighted`/segmentation/mIoU for `room0_splatdistiller`
          once its OpenCLIP features finish extracting (see
          `D:\\Downloads\\claude_logs\\experiment_d_progress.txt` for live progress).
        - [ ] Fill in `miou_splatdistiller.json` at the path this notebook reads from.
        - [ ] Extend this notebook to other scenes once room_0 is closed out.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
