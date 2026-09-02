"""Cache the EXPENSIVE half of run_cluster_classify_eval.py (foam load, power-cell
assignment, the two clusterings) so that many DECISION RULES can be A/B'd on the exact
same partition for free.

Nothing here changes the pipeline; it is byte-for-byte the same code path as
run_cluster_classify_eval.py up to the point where `pool_classify_broadcast` is called.
Cache lands in the scratchpad, not in artifacts/.
"""
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam, spherical_kmeans
from run_cluster_classify_eval import SCENES, K_FLAT, two_level_position_aware

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"


def main():
    enable_determinism()
    device = "cuda"
    os.makedirs(CACHE, exist_ok=True)
    suffix = os.environ.get("FEAT_SUFFIX", "_ogl3")
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = {k: v for k, v in SCENES.items() if k in only} if only else SCENES

    for scene, split in scenes.items():
        out = os.path.join(CACHE, f"{scene}{suffix}.pt")
        if os.path.exists(out):
            print(f"skip {scene} (cached)", flush=True)
            continue
        ckpt_dir = f"output/scannet_{scene}_nonfrozen"
        features_path = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen{suffix}.pt"
        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
        print(f"\n===== {scene} =====", flush=True)
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        centers, radii = load_foam(ckpt_dir, device)
        solved = torch.load(features_path, map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)

        valid_idx_np = np.where(valid_mask)[0]
        valid_idx = torch.from_numpy(valid_idx_np).to(device)
        unit = F.normalize(feats[valid_idx], dim=-1)
        positions = torch.from_numpy(centers[valid_idx_np]).to(device).float()

        flat_labels, _ = spherical_kmeans(unit, K_FLAT, seed=0)
        pos_labels = two_level_position_aware(positions, unit, seed=0)

        # point -> row in `unit` (or -1). Collapses assigned/valid_idx into one array so the
        # rule driver never needs the foam again.
        g2v = np.full(centers.shape[0], -1, dtype=np.int64)
        g2v[valid_idx_np] = np.arange(valid_idx_np.shape[0])
        point_row = np.where(assigned >= 0, g2v[assigned.clip(min=0)], -1)

        torch.save({
            "unit": unit.half().cpu(),
            "positions": positions.cpu(),
            "flat_labels": flat_labels.cpu(),
            "pos_labels": pos_labels.cpu(),
            "point_row": torch.from_numpy(point_row),
            "raw_labels": torch.from_numpy(raw_labels),
            "all_names": all_names,
        }, out)
        print(f"  wrote {out}  N={unit.shape[0]} pts={raw_labels.shape[0]} "
              f"owned={(point_row >= 0).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
