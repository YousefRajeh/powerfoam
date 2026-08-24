"""Emulate OpenGaussian's codebook consumption path on OUR solved per-cell features.

Audit question (Agent 4): OpenGaussian never classifies a per-primitive feature. It
classifies 320 codebook LEAF features (`train.py:904` per_leaf_feat), zeroes leaves seen in
<2 views (`scripts/eval_scannet.py:140`), does a PLAIN (no hubness correction) cosine argmax
(`eval_scannet.py:155-159`), and broadcasts the leaf class to member points
(`eval_scannet.py:160`). This script runs that exact SHAPE of protocol on our solved
geometric-median features for scene0347_00 and reports both the metric and the attractor
statistics, versus classifying every cell directly.

CPU-only by design (all GPUs busy). Single scene -> every number here is PROVISIONAL.
"""
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, remap_gt_labels, embed_class_names,
    classify_primitives, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import spherical_kmeans
from run_cluster_classify_eval import two_level_position_aware

SCENE = "scene0347_00"
CKPT = rf"D:\Downloads\powerfoam\output\scannet_{SCENE}_nonfrozen\model.pt"
FEATS = rf"D:\Downloads\powerfoam\artifacts\scannet\{SCENE}\solved_geometric_median_nonfrozen_l3.pt"
GT_DIR = rf"D:\Downloads\scannet_pointcept\train\{SCENE}"
K = 320  # OpenGaussian: root_node_num 64 x leaf_node_num 5 (scripts/train_scannet.sh:36-37)


def plain_argmax(unit_feats, text_feats):
    """OpenGaussian eval_scannet.py:155-159 exactly: normalize both, cosine, argmax over
    classes. A zeroed feature row has cosine 0 to every class, so argmax returns index 0."""
    sim = text_feats @ F.normalize(unit_feats, dim=-1).T  # (K_cls, N), their orientation
    return sim.argmax(dim=0)


def attractor_stats(unit_feats, text_feats, names, tag):
    sim = F.normalize(unit_feats, dim=-1) @ text_feats.T  # (N, K_cls) raw cosine
    means = sim.mean(dim=0)
    top2 = sim.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    win = sim.argmax(dim=-1)
    counts = torch.bincount(win, minlength=len(names)).float() / len(win)
    order = means.argsort(descending=True)
    print(f"\n--- attractor stats [{tag}] (N={len(sim)}) ---")
    print(f"  mean-cosine range over classes: {means.min():.4f} .. {means.max():.4f} "
          f"(spread {means.max()-means.min():.4f})")
    print(f"  top1-top2 margin: mean {margin.mean():.4f}  median {margin.median():.4f}  "
          f"frac<0.01 {(margin < 0.01).float().mean()*100:.2f}%")
    print(f"  {'class':<16} {'meancos':>8} {'win%':>7}")
    for i in order.tolist():
        print(f"  {names[i]:<16} {means[i]:>8.4f} {counts[i]*100:>6.2f}%")
    return dict(mean_min=float(means.min()), mean_max=float(means.max()),
                spread=float(means.max() - means.min()), margin_mean=float(margin.mean()),
                frac_margin_lt_01=float((margin < 0.01).float().mean()))


def pool(labels, unit, k):
    pooled = torch.zeros(k, unit.shape[1])
    pooled.index_add_(0, labels, unit)
    counts = torch.bincount(labels, minlength=k)
    return pooled / counts.clamp_min(1).unsqueeze(1).float(), counts


def main():
    enable_determinism()
    torch.set_num_threads(8)

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(GT_DIR, "segment20")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    centers = ck["points"].numpy().astype(np.float64)
    radii = F.softplus(ck["radii"], beta=100).numpy().astype(np.float64)  # scene.py:360-361
    solved = torch.load(FEATS, map_location="cpu", weights_only=True)
    feats = solved["primitive_features"].float()
    valid_mask = solved["valid_mask"].numpy()
    print(f"[setup] cells={centers.shape[0]} valid={valid_mask.sum()} gt_points={gt_points.shape[0]}")

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
    owned = assigned >= 0
    valid_idx_np = np.where(valid_mask)[0]
    unit = F.normalize(feats[torch.from_numpy(valid_idx_np)], dim=-1)
    positions = torch.from_numpy(centers[valid_idx_np]).float()

    print("[cluster] spherical k-means k=320 on features ...", flush=True)
    flat_labels, _ = spherical_kmeans(unit, K, seed=0)
    print("[cluster] two-level 64 position roots x 5 feature leaves ...", flush=True)
    pos_labels = two_level_position_aware(positions, unit, seed=0)

    flat_pooled, flat_counts = pool(flat_labels, unit, K)
    pos_pooled, pos_counts = pool(pos_labels, unit, K)
    print(f"[cluster] flat: {int((flat_counts > 0).sum())} nonempty, "
          f"{int(((flat_counts > 0) & (flat_counts < 2)).sum())} singleton")
    print(f"[cluster] pos : {int((pos_counts > 0).sum())} nonempty, "
          f"{int(((pos_counts > 0) & (pos_counts < 2)).sum())} singleton")

    # support-based occupancy proxy for the stronger variant
    support = torch.zeros(K)
    stats_support = torch.load(
        rf"D:\Downloads\powerfoam\artifacts\scannet\{SCENE}\train_stats_sam_nonfrozen_l3.pt",
        map_location="cpu", weights_only=False)["support"][torch.from_numpy(valid_idx_np)].float()
    support.index_add_(0, flat_labels, stats_support)

    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    results = {}
    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
        target_ids = [i for i, _ in kept]
        target_names = [n for _, n in kept]
        nc = len(target_ids)
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
        text_feats = embed_class_names(target_names, "cpu")
        print(f"\n================ {cs} ({nc} classes present) ================")

        if cs == "opengaussian19":
            attractor_stats(unit, text_feats, target_names, "per-cell (ours)")
            nz = flat_counts > 0
            attractor_stats(flat_pooled[nz], text_feats, target_names, "flat-kmeans320 pooled")
            nzp = pos_counts > 0
            attractor_stats(pos_pooled[nzp], text_feats, target_names, "pos-aware 64x5 pooled")

        def score(prim_cls_valid, tag):
            prim_class = np.zeros(centers.shape[0], dtype=np.int64)
            prim_class[valid_idx_np] = prim_cls_valid.numpy()
            pred = np.zeros(gt_points.shape[0], dtype=np.int64)
            pred[owned] = prim_class[assigned[owned]] + 1
            _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nc + 1)
            print(f"  {tag:<44} mIoU={miou*100:6.2f}  mAcc={macc*100:6.2f}  oAcc={acc*100:6.2f}")
            results.setdefault(cs, {})[tag] = dict(mIoU=miou, mAcc=macc, oAcc=acc)

        # ---- A/B: per-cell (no codebook) ----
        score(plain_argmax(unit, text_feats), "per-cell, plain argmax (OG rule)")
        score(classify_primitives(unit, text_feats), "per-cell, hubness argmax (ours)")

        # ---- C..F: codebook variants ----
        for cname, labels, pooled, counts in (
                ("flat-kmeans320", flat_labels, flat_pooled, flat_counts),
                ("pos-aware 64x5", pos_labels, pos_pooled, pos_counts)):
            score(plain_argmax(pooled, text_feats)[labels], f"{cname}, plain argmax")
            score(classify_primitives(pooled, text_feats)[labels], f"{cname}, hubness argmax")
            zeroed = pooled.clone()
            zeroed[counts < 2] = 0.0  # OG eval_scannet.py:140 analogue (occu_count < 2)
            score(plain_argmax(zeroed, text_feats)[labels], f"{cname}, plain argmax + occu<2 zero")

        # stronger occupancy filter on the flat codebook: bottom-10% by accumulated support
        thr = torch.quantile(support[flat_counts > 0], 0.10)
        zeroed = flat_pooled.clone()
        kill = (support < thr) | (flat_counts == 0)
        zeroed[kill] = 0.0
        print(f"  [support filter zeroes {int(kill.sum())}/{K} clusters, "
              f"{(torch.isin(flat_labels, torch.nonzero(kill).squeeze(1))).float().mean()*100:.2f}% of cells]")
        score(plain_argmax(zeroed, text_feats)[flat_labels], "flat-kmeans320, plain argmax + support-p10 zero")

    import json
    with open(rf"D:\Downloads\powerfoam\artifacts\scannet\{SCENE}\codebook_emulation_audit.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
