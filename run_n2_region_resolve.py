"""N2: region-level multi-view resolve ("cluster-then-resolve").

Regions come from the existing clustering (kmeansFPS 64x5, or the feature-coherent
grower). The streaming solve is then re-run with operator columns remapped cell->region,
so each region's feature is aggregated over VIEWS directly: a region contaminated by a
few bad views can have those views rejected wholesale by the geometric median, which
per-cell medians cannot do (each cell sees a different view subset, so no cell-level
median ever sees the region's full view distribution).

Motivated by the measured error structure: 83% of errors are interior cells of coherent
regions (whole-region misclassification), so the fix has to change what the REGION
believes, not how boundaries are drawn. Raw class names + plain cosine, no templates.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.feature_operator import accumulate_feature_stats_for_views
from feature_foam_lifting.operator import (AccumulatedFeatureStats,
    solve_geometric_median_from_stats, solve_weighted_from_stats)
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt, calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from run_normlift_refine_eval import mode_vote_refine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--regions", choices=["kmeansFPS", "grower"], default="kmeansFPS")
    p.add_argument("--refine-passes", type=int, default=3)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    results = {}
    for scene in args.scenes.split(","):
        split = SCENES[scene]
        ckpt_dir = f"output/scannet_{scene}_{args.variant}"

        wp.init()
        parser = configargparse.ArgParser()
        add_group(parser, Params)
        parser.add_argument("-c", "--config", is_config_file=True)
        cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
        dh = DataHandler(cargs)
        dh.reload("all", downsample=cargs.downsample[-1])
        model = PowerfoamScene(cargs)
        model.initialize_from_dataset(dh, device=device)
        model.load_pt(f"{ckpt_dir}/model.pt")
        cameras = dh.cameras
        images_dir = Path(cargs.data_path) / cargs.scene / "images"
        image_names = sorted(q.stem for q in images_dir.iterdir())
        centers = model.points.detach().cpu().numpy()
        radii = model.get_radii().detach().cpu().numpy()

        solved = torch.load(
            f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}_l3.pt",
            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        vm = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(vm).to(device)
        vi = torch.where(vm_t)[0]
        unit_full = torch.zeros_like(feats)
        unit_full[vi] = F.normalize(feats[vi], dim=-1)

        stats0 = AccumulatedFeatureStats.load(
            f"artifacts/scannet/{scene}/train_stats_sam_{args.variant}_l3.pt")
        R = stats0.reliability()["reliability"].to(device).float() * vm_t
        del stats0
        torch.cuda.empty_cache()

        adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{args.variant}.pt",
                         map_location=device, weights_only=True)
        adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        positions_full = torch.from_numpy(centers).to(device).float()

        ref = unit_full
        for _ in range(args.refine_passes):
            ref = mode_vote_refine(ref, R, positions_full, adjacent, offsets)
        unit = ref[vi]

        if args.regions == "grower":
            from run_region_grow_eval import batched_region_grow
            gl, gn = batched_region_grow(adjacent, offsets, ref, vm_t, 0.95)
            region_of_cell = gl.clone()
            num_regions = gn
        else:
            leaf = two_level_position_aware(positions_full[vi], unit, seed=0, leaf_init="fps")
            region_of_cell = torch.full((centers.shape[0],), -1, dtype=torch.long, device=device)
            region_of_cell[vi] = leaf
            num_regions = K_FLAT
        print(f"  [{scene}] regions={num_regions} ({args.regions})", flush=True)

        feature_dir = Path(f"artifacts/scannet/{scene}/openclip_features_sam")

        def load_feature_map(view_id):
            cam = cameras[view_id]
            return load_image_feature_from_SAMOpenCLIP(
                feature_dir, image_names[view_id], height=cam.height, width=cam.width, sam_level=3)

        rstats = accumulate_feature_stats_for_views(
            model, cameras, list(range(len(cameras))), load_feature_map, batch_size=1,
            column_map=region_of_cell, num_columns=num_regions)
        x_gm = solve_geometric_median_from_stats(rstats)[0]
        x_w = solve_weighted_from_stats(rstats)[0]

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{args.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
        owned = assigned >= 0
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        roc = region_of_cell.cpu().numpy()

        for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            tids = [i for i, _ in kept]
            tnames = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(tnames, device)

            def score(region_cls_t, tag):
                rc = region_cls_t.cpu().numpy()
                pc = np.where(roc >= 0, rc[np.clip(roc, 0, None)], 0)
                pred = np.zeros(len(gt_t), dtype=np.int64)
                pred[owned] = pc[assigned[owned]] + 1
                _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
                results.setdefault(f"{cs}|{tag}", {})[scene] = (mi, ma)
                print(f"  {scene} {cs} {tag}: {mi:.4f}/{ma:.4f}", flush=True)

            pooled = torch.zeros(num_regions, unit.shape[1], device=device)
            pooled.index_add_(0, region_of_cell[vi], unit)
            pooled = F.normalize(pooled, dim=-1)
            score((pooled @ text.T).argmax(-1), "base_cellmedian_pool")
            score((F.normalize(x_gm.to(device).float(), dim=-1) @ text.T).argmax(-1), "N2_region_geomedian")
            score((F.normalize(x_w.to(device).float(), dim=-1) @ text.T).argmax(-1), "N2_region_weighted")

        del unit_full, feats, ref, R, adjacent, offsets, rstats, model
        torch.cuda.empty_cache()

    print("\n=== averages ===")
    out = {}
    for key, per in sorted(results.items()):
        mi = float(np.mean([v[0] for v in per.values()]))
        ma = float(np.mean([v[1] for v in per.values()]))
        out[key] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per)}
        print(f"{key}: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
