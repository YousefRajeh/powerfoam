"""Tier 1 idea #3: sweep attention_map2auto_threshold's hyperparameters (min_bin,
min_peak_count, use_rescale, percentile_fallback) against ALREADY-RENDERED AttentionMap.pt
files -- zero re-render, zero re-solve, just re-scoring with the REAL, unmodified
LERFMetrics.compute_metrics (only its default keyword args to attention_map2auto_threshold
are monkeypatched via functools.partial, the algorithm code itself is untouched).

Note: the BACKGROUND_WORDS *set* cannot be swept this way -- it's baked into the channel
count of the already-rendered attention maps (channels = len(prompts) + len(BACKGROUND_WORDS)
at render time), so changing the word set needs a cheap re-render (dot different background
text embeddings against already-solved features), not covered by this script.

Run in splat-distiller env: D:\\conda\\envs\\splat-distiller\\python.exe
"""
import functools
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")

from metrics import LERFMetrics

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")

SCENES_AND_METHODS = [
    ("ramen", "eval_sam_featurefoam/rendered"),
    ("ramen", "eval_sam_splatfeaturesolver"),
    ("waldo_kitchen", "eval_sam_featurefoam/rendered"),
    ("waldo_kitchen", "eval_sam_splatfeaturesolver"),
]

GRID = [
    {"min_bin": 40, "min_peak_count": 600, "use_rescale": False, "percentile_fallback": 0.75},  # default
    {"min_bin": 40, "min_peak_count": 300, "use_rescale": False, "percentile_fallback": 0.75},
    {"min_bin": 40, "min_peak_count": 900, "use_rescale": False, "percentile_fallback": 0.75},
    {"min_bin": 40, "min_peak_count": 600, "use_rescale": True, "percentile_fallback": 0.75},
    {"min_bin": 40, "min_peak_count": 600, "use_rescale": False, "percentile_fallback": 0.60},
    {"min_bin": 40, "min_peak_count": 600, "use_rescale": False, "percentile_fallback": 0.90},
    {"min_bin": 20, "min_peak_count": 600, "use_rescale": False, "percentile_fallback": 0.75},
    {"min_bin": 80, "min_peak_count": 600, "use_rescale": False, "percentile_fallback": 0.75},
]


def run_one(scene, rendered_subdir, params):
    label_dir = LERF_ROOT / "label" / scene
    rendered_folder = Path(r"D:\Downloads\powerfoam\artifacts\lerf_ovs") / scene / rendered_subdir
    metrics = LERFMetrics(
        label_folder=label_dir,
        rendered_folder=rendered_folder,
        text_encoder="SAMOpenCLIP",
        enable_pca=None,
    )
    metrics.attention_map2auto_threshold = functools.partial(
        LERFMetrics.attention_map2auto_threshold, metrics, **params,
    )
    result = metrics.compute_metrics(Path(r"D:\Downloads\claude_logs\_sweep_tmp") / scene / rendered_subdir.replace("/", "_"),
                                      mode="attention_map")
    return result["scene_mean"]["mIoU"], result["scene_mean"].get("mAcc", float("nan"))


if __name__ == "__main__":
    for scene, subdir in SCENES_AND_METHODS:
        print(f"\n=== {scene} / {subdir} ===")
        best = None
        for params in GRID:
            miou, macc = run_one(scene, subdir, params)
            tag = "DEFAULT" if params == GRID[0] else ""
            print(f"  {params}  mIoU={miou:.4f}  mAcc={macc:.4f}  {tag}")
            if best is None or miou > best[0]:
                best = (miou, macc, params)
        print(f"  BEST: mIoU={best[0]:.4f} mAcc={best[1]:.4f} params={best[2]}")
