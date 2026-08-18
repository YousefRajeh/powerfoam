"""Mask-association feature solver: OpenGaussian-Stage-3-style cluster<->SAM-mask
association, done TRAINING-FREE via PowerFoam's exact rendering operator.

Why: every prior protocol pools pixel-space feature reconstructions, i.e. averages of
averages of CLIP vectors -- off-manifold blends whose cosine geometry vs text is
distorted (measured: plain argmax collapses, hubness correction needed, wall/floor
recall 0.26-0.35 under every clustering). OpenGaussian's Stage 3 instead SELECTS, per
cluster per view, the single best-overlapping SAM mask and uses that mask's ORIGINAL
CLIP embedding -- on-manifold, no blending, plain argmax works. They need a training
stage for the 2D-3D association; we get exact pixel<->primitive ownership for free from
export_feature_operator.

Pipeline per scene:
  1. Cluster primitives (position-aware 64x5 k-means, seed 0 -- the best protocol).
  2. Per training view: export the exact sparse operator; per-pixel DOMINANT cluster
     (argmax over per-cluster summed rendering weights; pixels with total weight <
     min_pixel_weight stay unassigned). IoU every cluster footprint against every SAM
     segment (all granularity levels); select the best mask per cluster if
     IoU >= min_iou; record (cluster, mask CLIP embedding, IoU weight).
  3. Per cluster: weighted geometric median (Weiszfeld) over its selected mask
     embeddings -- robust to occasional wrong selections, stays on-manifold.
     Clusters that never matched a mask fall back to their pixel-pooled mean feature.
  4. Classify clusters with template-ensemble text embeddings + PLAIN cosine argmax
     (the winning text-side rule; no hubness correction), broadcast, score.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES, CLASS_SETS
from run_text_side_eval import embed_prompts

MAX_HITS = 64


def load_model_and_cameras(scene, device):
    wp.init()
    ckpt_dir = f"output/scannet_{scene}_nonfrozen"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    data_handler = DataHandler(args)
    data_handler.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device=device)
    model.load_pt(f"{ckpt_dir}/model.pt")
    images_dir = Path(args.data_path) / args.scene / "images"
    image_names = sorted(p.stem for p in images_dir.iterdir())
    assert len(image_names) == len(data_handler.cameras)
    return model, data_handler.cameras, image_names


def per_pixel_dominant_cluster(out_col, out_val, slot_counter, prim_cluster, num_pixels,
                               num_clusters, min_pixel_weight, device):
    """Exact per-pixel argmax over per-cluster SUMMED rendering weights."""
    slots_used = slot_counter.clamp(max=MAX_HITS)
    keep = (torch.arange(MAX_HITS, device=device)[None, :] < slots_used[:, None]).reshape(-1)
    cols = out_col.reshape(-1)[keep].long()
    vals = out_val.reshape(-1)[keep].float()
    pix = torch.arange(num_pixels, device=device).repeat_interleave(MAX_HITS)[keep]
    cl = prim_cluster[cols]  # (nnz,) cluster id or -1 for invalid primitives
    ok = cl >= 0
    pix, vals, cl = pix[ok], vals[ok], cl[ok]

    total_w = torch.zeros(num_pixels, device=device)
    total_w.index_add_(0, pix, vals)

    key = pix * num_clusters + cl
    order = torch.argsort(key)
    key_s, val_s = key[order], vals[order]
    uniq, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    seg_sum = torch.zeros(uniq.numel(), device=device)
    seg_sum.index_add_(0, inverse, val_s)
    seg_pix = (uniq // num_clusters).long()
    seg_cl = (uniq % num_clusters).long()

    best = torch.zeros(num_pixels, device=device)
    best.scatter_reduce_(0, seg_pix, seg_sum, reduce="amax")
    win = seg_sum >= best[seg_pix] - 1e-12
    dom = torch.full((num_pixels,), -1, dtype=torch.long, device=device)
    dom[seg_pix[win]] = seg_cl[win]
    dom[total_w < min_pixel_weight] = -1
    return dom


def weighted_geomedian(vectors, weights, iters=25):
    """Weiszfeld on unit-normalized inputs; returns unit vector."""
    v = F.normalize(vectors, dim=-1)
    x = F.normalize((v * weights[:, None]).sum(0), dim=0)
    for _ in range(iters):
        d = (v - x[None, :]).norm(dim=-1).clamp_min(1e-6)
        w = weights / d
        x_new = F.normalize((v * w[:, None]).sum(0), dim=0)
        if (x_new - x).norm() < 1e-7:
            x = x_new
            break
        x = x_new
    return x


def solve_scene(scene, split, device, min_iou, min_pixel_weight, feature_dir_root):
    features_path = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen.pt"
    solved = torch.load(features_path, map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(valid_mask).to(device)
    vi = torch.where(vm_t)[0]
    unit = F.normalize(feats[vi], dim=-1)

    model, cameras, image_names = load_model_and_cameras(scene, device)
    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    positions = torch.from_numpy(centers[vi.cpu().numpy()]).to(device).float()

    leaf = two_level_position_aware(positions, unit, seed=0)
    prim_cluster = torch.full((centers.shape[0],), -1, dtype=torch.long, device=device)
    prim_cluster[vi] = leaf

    # pixel-pooled fallback features (current protocol's pools)
    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
    pooled.index_add_(0, leaf, unit)
    pooled = F.normalize(pooled, dim=-1)

    feature_dir = Path(feature_dir_root) / scene / "openclip_features_sam"
    picked_cl, picked_feat, picked_w = [], [], []
    n_views_used = 0
    for view_id, camera in enumerate(cameras):
        fpath = feature_dir / f"{image_names[view_id]}_f.npy"
        spath = feature_dir / f"{image_names[view_id]}_s.npy"
        if not fpath.exists():
            continue
        H, W = camera.height, camera.width
        num_pixels = H * W
        out_col, out_val, slot_counter, _, _ = model.export_feature_operator(
            camera, transmittance_threshold=1e-3, max_intersections=1024,
            max_hits_per_pixel=MAX_HITS)
        dom = per_pixel_dominant_cluster(out_col, out_val, slot_counter, prim_cluster,
                                         num_pixels, K_FLAT, min_pixel_weight, device)
        del out_col, out_val, slot_counter

        seg = torch.from_numpy(np.load(spath)).to(device).long()  # (L, H, W), -1 = none
        mask_feats = torch.from_numpy(np.load(fpath)).to(device).float()  # (M, 512)
        M = mask_feats.shape[0]
        if seg.dim() == 2:
            seg = seg.unsqueeze(0)
        seg = seg.reshape(seg.shape[0], -1)  # (L, num_pixels)

        cluster_size = torch.bincount(dom[dom >= 0], minlength=K_FLAT).float()
        inter = torch.zeros(K_FLAT * M, device=device)
        mask_size = torch.zeros(M, device=device)
        for l in range(seg.shape[0]):
            sl = seg[l]
            okm = sl >= 0
            mask_size.index_add_(0, sl[okm], torch.ones(int(okm.sum()), device=device))
            both = okm & (dom >= 0)
            if both.any():
                k = dom[both] * M + sl[both]
                inter.index_add_(0, k, torch.ones(int(both.sum()), device=device))
        inter = inter.reshape(K_FLAT, M)
        union = cluster_size[:, None] + mask_size[None, :] - inter
        iou = inter / union.clamp_min(1.0)
        best_iou, best_m = iou.max(dim=1)
        sel = best_iou >= min_iou
        if sel.any():
            cl_ids = torch.where(sel)[0]
            picked_cl.append(cl_ids)
            picked_feat.append(mask_feats[best_m[sel]])
            picked_w.append(best_iou[sel])
        n_views_used += 1
        del seg, mask_feats, dom

    if picked_cl:
        picked_cl = torch.cat(picked_cl)
        picked_feat = torch.cat(picked_feat)
        picked_w = torch.cat(picked_w)
    matched = torch.zeros(K_FLAT, dtype=torch.bool, device=device)
    cluster_feats = pooled.clone()
    if len(picked_cl) > 0:
        for c in torch.unique(picked_cl).tolist():
            m = picked_cl == c
            cluster_feats[c] = weighted_geomedian(picked_feat[m], picked_w[m])
            matched[c] = True
    print(f"  [{scene}] views used={n_views_used}, mask-matched clusters="
          f"{int(matched.sum())}/{K_FLAT} (fallback=pixel-pooled for the rest), "
          f"selections={len(picked_cl)}", flush=True)
    return cluster_feats, prim_cluster, centers, radii, valid_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="all")
    p.add_argument("--class-sets", default="all")
    p.add_argument("--min-iou", type=float, default=0.25)
    p.add_argument("--min-pixel-weight", type=float, default=0.3)
    p.add_argument("--feature-root", default="artifacts/scannet")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scenes = SCENES if args.scenes == "all" else {s: SCENES[s] for s in args.scenes.split(",")}
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")

    results = {cs: {} for cs in class_sets}
    for scene, split in scenes.items():
        cluster_feats, prim_cluster, centers, radii, valid_mask = solve_scene(
            scene, split, device, args.min_iou, args.min_pixel_weight, args.feature_root)

        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())

        for cs in class_sets:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
            target_ids = [i for i, _ in kept]
            target_names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
            text_feats = embed_prompts(target_names, "templates", device)
            cls = (cluster_feats @ text_feats.T).argmax(dim=-1)  # PLAIN argmax
            prim_class = torch.where(prim_cluster >= 0, cls[prim_cluster.clamp_min(0)],
                                     torch.zeros_like(prim_cluster)).cpu().numpy()
            pred = np.zeros(gt_points.shape[0], dtype=np.int64)
            pred[owned] = prim_class[assigned[owned]] + 1
            _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                   len(target_ids) + 1)
            results[cs][scene] = {"mIoU": miou, "mAcc": macc}
            print(f"  {scene} {cs} mask-assoc: mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)

    print("\n=== mask-association averages ===")
    summary = {}
    for cs, per_scene in results.items():
        if not per_scene:
            continue
        mi = float(np.mean([m["mIoU"] for m in per_scene.values()]))
        ma = float(np.mean([m["mAcc"] for m in per_scene.values()]))
        summary[cs] = {"mean_mIoU": mi, "mean_mAcc": ma, "per_scene": per_scene}
        print(f"{cs}: {mi*100:.2f}/{ma*100:.2f} (n={len(per_scene)})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
