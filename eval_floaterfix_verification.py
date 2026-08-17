"""Direct before/after verification of the floater-init-primitive fix (powerfoam/scene.py
::init_points_sfm, commit 6f8106e) for ramen and waldo_kitchen at final_points=500000, held
constant against the existing (pre-fix) 500k baselines. Uses the real, unmodified LERFMetrics
and the fixed featurefoam_lerf_bridge.py text-encoder defaults, exactly as every other number
in this comparison.

Run in splat-distiller env: D:\\conda\\envs\\splat-distiller\\python.exe
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")
from metrics import LERFMetrics

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")

SCENES = [
    ("ramen", "eval_sam_featurefoam_500k_floaterfix"),
    ("waldo_kitchen", "eval_sam_featurefoam_500k_floaterfix"),
]

for scene, subdir in SCENES:
    rendered_folder = Path(r"D:\Downloads\powerfoam\artifacts\lerf_ovs") / scene / subdir
    label_dir = LERF_ROOT / "label" / scene
    metrics = LERFMetrics(
        label_folder=label_dir,
        rendered_folder=rendered_folder,
        text_encoder="SAMOpenCLIP",
        enable_pca=None,
    )
    result = metrics.compute_metrics(Path(r"D:\Downloads\claude_logs\_floaterfix_tmp") / scene, mode="attention_map")
    print(f"{scene} 500k floaterfix: mIoU={result['scene_mean']['mIoU']:.4f} mAcc={result['scene_mean'].get('mAcc', float('nan')):.4f}")
