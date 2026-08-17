"""Phase 2 (splat-distiller env): verify prompt-ordering equivalence against the REAL
json_parser, then run the REAL, unmodified LERFMetrics.compute_metrics on Phase 1's
rendered output. Run with D:\\conda\\envs\\splat-distiller\\python.exe from
D:\\Downloads\\splat-distiller (needs metrics.py/evaluator_loader.py on path)."""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\splat-distiller")
sys.path.insert(0, r"D:\Downloads\claude_logs")

from featurefoam_lerf_bridge import build_prompts_for_frame, BACKGROUND_WORDS
from metrics import LERFMetrics

SCENE = "figurines"
LABEL_DIR = Path(rf"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs\label\{SCENE}")
RESULT_FOLDER = Path(rf"D:\Downloads\claude_logs\_bridge_test_{SCENE}")


def test_prompt_ordering_matches_json_parser():
    dummy = LERFMetrics.__new__(LERFMetrics)
    failures = []
    label_files = sorted(LABEL_DIR.glob("*.json"))
    for label_file in label_files:
        _, masks_categories = dummy.json_parser(label_file)
        their_order = [cat for cat, _ in masks_categories]
        my_positives, my_full_prompts = build_prompts_for_frame(label_file)
        assert my_full_prompts[len(my_positives):] == BACKGROUND_WORDS
        if my_positives != their_order:
            failures.append((label_file.name, my_positives, their_order))
    if failures:
        for name, mine, theirs in failures:
            print(f"MISMATCH {name}: mine={mine} theirs={theirs}")
        raise AssertionError(f"{len(failures)} prompt-ordering mismatches")
    print(f"PASS: prompt ordering matches json_parser exactly for all {len(label_files)} labeled frames")


def test_end_to_end_runs_through_real_lerfmetrics():
    metrics = LERFMetrics(
        label_folder=LABEL_DIR,
        rendered_folder=RESULT_FOLDER / "rendered",
        text_encoder="SAMOpenCLIP",
        enable_pca=None,
    )
    assert len(metrics.labels) > 0, "no labeled frames were matched -- file-naming contract is broken"
    print(f"matched {len(metrics.labels)} labeled frames: {list(metrics.labels.keys())}")
    result = metrics.compute_metrics(RESULT_FOLDER / "metrics_out", mode="attention_map")
    scene_mean = result["scene_mean"]
    print("scene_mean:\n", scene_mean)
    assert "mIoU" in scene_mean.index
    assert 0.0 <= scene_mean["mIoU"] <= 1.0
    print(f"PASS: real LERFMetrics.compute_metrics ran end-to-end, scene mIoU={scene_mean['mIoU']:.4f}")


if __name__ == "__main__":
    test_prompt_ordering_matches_json_parser()
    test_end_to_end_runs_through_real_lerfmetrics()
    print("\nALL BRIDGE VERIFICATION TESTS PASSED")
