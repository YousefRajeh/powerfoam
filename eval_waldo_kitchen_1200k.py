"""Evaluate waldo_kitchen's 1.2M-primitive Feature Foam checkpoint with the real,
unmodified LERFMetrics, using the now-fixed featurefoam_lerf_bridge.py text-encoder
defaults. Mirrors the pattern used for figurines/ramen/teatime's final_points sweep.

Run in splat-distiller env: D:\\conda\\envs\\splat-distiller\\python.exe
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")
from metrics import LERFMetrics

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")

scene = "waldo_kitchen"
rendered_folder = Path(r"D:\Downloads\powerfoam\artifacts\lerf_ovs") / scene / "eval_sam_featurefoam_1200k_fixed"
label_dir = LERF_ROOT / "label" / scene

metrics = LERFMetrics(
    label_folder=label_dir,
    rendered_folder=rendered_folder,
    text_encoder="SAMOpenCLIP",
    enable_pca=None,
)
result = metrics.compute_metrics(Path(r"D:\Downloads\claude_logs\_waldo_1200k_fixed_tmp"), mode="attention_map")
print(f"waldo_kitchen 1.2M: mIoU={result['scene_mean']['mIoU']:.4f} mAcc={result['scene_mean'].get('mAcc', float('nan')):.4f}")
