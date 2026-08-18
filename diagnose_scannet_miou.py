"""Diagnose why point-mIoU is low despite high mAcc on ScanNet.

Three hypotheses, each measured directly on one scene (default scene0000_00 nonfrozen):
  1. Coverage loss: GT points that end up with pred=0 (unassigned / owned by a support=0
     primitive) are automatic misses -- quantifies the "did we throw away points" worry.
  2. Precision collapse: per-class precision vs recall. mAcc only sees recall; mIoU also
     pays for false positives. Per-primitive CLIP argmax is noisy, so precision is the
     suspected casualty (especially wall/floor, the dominant structural classes).
  3. No-grouping penalty: OpenGaussian/NormLift-style protocols classify POOLED cluster
     features (OpenGaussian: 64x5=320 leaf codebook) and broadcast -- pooling denoises.
     We replicate that here (spherical k-means k=320 on the solved features, mean-pool,
     classify each cluster once, broadcast to members) and re-score, isolating how much
     mIoU the direct per-primitive protocol costs vs a cluster-then-classify protocol.
"""
import argparse
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, SCANNET20_CLASS_NAMES,
    calculate_metrics, remap_gt_labels, embed_class_names, classify_primitives,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells


def load_foam(checkpoint_dir, device, return_normals=False):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{checkpoint_dir}/config.yaml"])
    data_handler = DataHandler(args)
    data_handler.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device=device)
    model.load_pt(f"{checkpoint_dir}/model.pt")
    if return_normals:
        return (model.points.detach().cpu().numpy(), model.get_radii().detach().cpu().numpy(),
                model.get_normals().detach().cpu().numpy())
    return model.points.detach().cpu().numpy(), model.get_radii().detach().cpu().numpy()


def fps_features(x, k, seed=0):
    """Greedy farthest-point sampling in cosine distance on unit features: seeds spread
    across feature modes instead of following density (user-proposed init)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.empty(k, dtype=torch.long, device=x.device)
    idx[0] = torch.randint(0, x.shape[0], (1,), generator=g, device=x.device)
    min_sim = x @ x[idx[0]]
    for i in range(1, k):
        idx[i] = min_sim.argmin()
        min_sim = torch.maximum(min_sim, x @ x[idx[i]])
    return idx


def spherical_kmeans(x, k, iters=25, seed=0, init="randperm"):
    """x: (N, C) unit-normalized. Returns (labels, centroids)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    if init == "fps":
        centroids = x[fps_features(x, k, seed=seed)].clone()
    else:
        centroids = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        sim = x @ centroids.T
        labels = sim.argmax(dim=1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).clamp_min(1).unsqueeze(1)
        new_centroids /= counts
        norms = new_centroids.norm(dim=1, keepdim=True)
        dead = norms.squeeze(1) < 1e-8
        if dead.any():
            new_centroids[dead] = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:int(dead.sum())]]
            norms = new_centroids.norm(dim=1, keepdim=True)
        centroids = new_centroids / norms.clamp_min(1e-8)
    return (x @ centroids.T).argmax(dim=1), centroids


def report(tag, gt_t, pred_t, num_classes, target_names):
    ious, miou, acc, macc = calculate_metrics(gt_t, pred_t, num_classes + 1)
    print(f"\n[{tag}] mIoU={miou:.4f} mAcc={macc:.4f} overall_acc={acc:.4f}")
    gt_np, pred_np = gt_t.numpy(), pred_t.numpy()
    print(f"{'class':<16} {'gt%':>6} {'pred%':>6} {'recall':>7} {'prec':>7} {'IoU':>7}")
    valid = gt_np != 0
    n_valid = valid.sum()
    for i, name in enumerate(target_names):
        cls = i + 1
        gt_c = (gt_np == cls)
        pred_c = (pred_np == cls) & valid
        tp = (gt_c & pred_c).sum()
        rec = tp / max(gt_c.sum(), 1)
        prec = tp / max(pred_c.sum(), 1)
        print(f"{name:<16} {gt_c.sum()/n_valid*100:>5.1f}% {pred_c.sum()/n_valid*100:>5.1f}% "
              f"{rec:>7.3f} {prec:>7.3f} {float(ious[cls]):>7.3f}")
    return miou, macc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--split", default="train")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--classes", default="opengaussian19")
    p.add_argument("--k", type=int, default=320)
    args = p.parse_args()

    device = "cuda"
    gt_dir = rf"D:\Downloads\scannet_pointcept\{args.split}\{args.scene}"
    ckpt_dir = f"output/scannet_{args.scene}_{args.variant}"
    features_path = f"artifacts/scannet/{args.scene}/solved_geometric_median_{args.variant}.pt"

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
    wanted = OPENGAUSSIAN_CLASS_SETS[args.classes]
    name_to_id = {n: i for i, n in enumerate(all_names)}
    target_ids = [name_to_id[n] for n in wanted]
    present = set(np.unique(raw_labels).tolist())
    kept = [(i, n) for i, n in zip(target_ids, wanted) if i in present]
    target_ids = [i for i, _ in kept]
    target_names = [n for _, n in kept]
    num_classes = len(target_ids)
    gt_remapped = remap_gt_labels(raw_labels, target_ids)
    gt_t = torch.from_numpy(gt_remapped).long()

    centers, radii = load_foam(ckpt_dir, device)
    solved = torch.load(features_path, map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)

    # --- Hypothesis 1: coverage ---
    gt_valid = gt_remapped != 0
    unowned = (assigned < 0) & gt_valid
    print(f"=== {args.scene} {args.variant} {args.classes} ===")
    print(f"primitives: {centers.shape[0]}, valid (support>0): {valid_mask.sum()} "
          f"({valid_mask.mean()*100:.1f}%)")
    print(f"GT points with non-ignore labels: {gt_valid.sum()}")
    print(f"  ...of which UNOWNED by any valid primitive (auto-miss, pred=0): "
          f"{unowned.sum()} ({unowned.sum()/max(gt_valid.sum(),1)*100:.2f}%)")

    text_feats = embed_class_names(target_names, device)

    # --- Hypothesis 2: per-primitive protocol (current) ---
    prim_class = classify_primitives(feats, text_feats).cpu().numpy()
    pred = np.zeros(gt_points.shape[0], dtype=np.int64)
    owned = assigned >= 0
    pred[owned] = prim_class[assigned[owned]] + 1
    report("per-primitive argmax (current protocol)", gt_t, torch.from_numpy(pred).long(),
           num_classes, target_names)

    # no hubness correction ablation
    prim_class_nh = classify_primitives(feats, text_feats, hubness_correct=False).cpu().numpy()
    pred_nh = np.zeros(gt_points.shape[0], dtype=np.int64)
    pred_nh[owned] = prim_class_nh[assigned[owned]] + 1
    report("per-primitive argmax, NO hubness correction", gt_t, torch.from_numpy(pred_nh).long(),
           num_classes, target_names)

    # --- Hypothesis 3: cluster-then-classify (OpenGaussian-style protocol) ---
    valid_idx = np.where(valid_mask)[0]
    unit = F.normalize(feats[valid_idx], dim=-1)
    labels, _ = spherical_kmeans(unit, k=args.k)
    pooled = torch.zeros(args.k, unit.shape[1], device=device)
    pooled.index_add_(0, labels, unit)
    pooled = F.normalize(pooled, dim=-1)
    cluster_class = classify_primitives(pooled, text_feats).cpu().numpy()
    prim_class_cl = np.zeros(centers.shape[0], dtype=np.int64)
    prim_class_cl[valid_idx] = cluster_class[labels.cpu().numpy()]
    pred_cl = np.zeros(gt_points.shape[0], dtype=np.int64)
    pred_cl[owned] = prim_class_cl[assigned[owned]] + 1
    report(f"cluster({args.k})-pool-classify-broadcast (OpenGaussian-style)", gt_t,
           torch.from_numpy(pred_cl).long(), num_classes, target_names)

    # cluster + no hubness
    cluster_class_nh = classify_primitives(pooled, text_feats, hubness_correct=False).cpu().numpy()
    prim_class_cl_nh = np.zeros(centers.shape[0], dtype=np.int64)
    prim_class_cl_nh[valid_idx] = cluster_class_nh[labels.cpu().numpy()]
    pred_cl_nh = np.zeros(gt_points.shape[0], dtype=np.int64)
    pred_cl_nh[owned] = prim_class_cl_nh[assigned[owned]] + 1
    report(f"cluster({args.k})-pool, NO hubness correction", gt_t,
           torch.from_numpy(pred_cl_nh).long(), num_classes, target_names)


if __name__ == "__main__":
    main()
