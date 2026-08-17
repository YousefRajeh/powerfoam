"""Evaluate teatime's 800k-primitive Feature Foam checkpoint with the real, unmodified
LERFMetrics, using the fixed featurefoam_lerf_bridge.py text-encoder defaults. Part of the
final_points sweep (500k/800k/1.2M) for the LERF-OVS 4-scene comparison.

Run in splat-distiller env: D:\\conda\\envs\\splat-distiller\\python.exe
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")
from metrics import LERFMetrics

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")

scene = "teatime"
rendered_folder = Path(r"D:\Downloads\powerfoam\artifacts\lerf_ovs") / scene / "eval_sam_featurefoam_800k_fixed"
label_dir = LERF_ROOT / "label" / scene

metrics = LERFMetrics(
    label_folder=label_dir,
    rendered_folder=rendered_folder,
    text_encoder="SAMOpenCLIP",
    enable_pca=None,
)
result = metrics.compute_metrics(Path(r"D:\Downloads\claude_logs\_teatime_800k_fixed_tmp"), mode="attention_map")
print(f"teatime 800k: mIoU={result['scene_mean']['mIoU']:.4f} mAcc={result['scene_mean'].get('mAcc', float('nan')):.4f}")
