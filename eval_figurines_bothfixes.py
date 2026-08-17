"""Evaluate figurines' 500k checkpoint trained with BOTH data-quality fixes (floater-init fix,
commit 6f8106e, and the mislocalized-camera re-registration, commit 920924e) against the
original pre-fix 500k baseline (0.5180/0.927). Same final_points (isolated variable), real
unmodified LERFMetrics, fixed bridge text-encoder defaults.

Note: these two fixes cannot be isolated from each other for figurines specifically (the floater
fix affects all 4 scenes; only figurines has the camera bug) -- this measures their combined
effect, not the camera fix alone.

Run in splat-distiller env: D:\\conda\\envs\\splat-distiller\\python.exe
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")
from metrics import LERFMetrics

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")

scene = "figurines"
rendered_folder = Path(r"D:\Downloads\powerfoam\artifacts\lerf_ovs") / scene / "eval_sam_featurefoam_500k_bothfixes"
label_dir = LERF_ROOT / "label" / scene

metrics = LERFMetrics(
    label_folder=label_dir,
    rendered_folder=rendered_folder,
    text_encoder="SAMOpenCLIP",
    enable_pca=None,
)
result = metrics.compute_metrics(Path(r"D:\Downloads\claude_logs\_figurines_bothfixes_tmp"), mode="attention_map")
print(f"figurines 500k bothfixes: mIoU={result['scene_mean']['mIoU']:.4f} mAcc={result['scene_mean'].get('mAcc', float('nan')):.4f}")
