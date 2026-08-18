"""Adjacency-graph region-growing cluster-then-classify (user-proposed idea, tested as the
next single idea after position-aware k-means).

Clustering: instead of any k-means, grow regions over PowerFoam's REAL power-diagram
adjacency graph (export_adjacency_graph.py's CSR structure -- facet-sharing neighbors from
bvh.py::build_cech_complex):
  1. Seed selection: the unassigned primitive with the highest mean cosine similarity to
     its unassigned graph neighbors (most locally-coherent point -- a wall interior beats
     a boundary primitive).
  2. BFS flood-fill from the seed: a frontier neighbor joins the region iff
     cos(feature, running region mean) >= threshold. The mean is updated after every BFS
     level (online spherical mean), so the region tracks its own average semantics rather
     than the (possibly noisy) seed alone.
  3. Repeat until k=320 regions (OpenGaussian's codebook size, same as the k-means runs
     for a controlled comparison) or no unassigned primitives remain.
  4. Leftover primitives (unreached or rejected everywhere): iterative label propagation
     -- each takes the label of its most-feature-similar already-labeled graph neighbor
     (5 rounds), then any still-orphaned fall back to the globally nearest region centroid.

Classification: identical to run_cluster_classify_eval.py (mean-pool unit features per
region, hubness-corrected argmax vs class-set text embeddings, broadcast), so any metric
delta is attributable to the clustering alone.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS,
    calculate_metrics, remap_gt_labels, embed_class_names,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import pool_classify_broadcast, SCENES, CLASS_SETS

K_REGIONS = 320


def region_grow(adjacent, offsets, unit_feats, valid_mask_t, threshold, k=K_REGIONS):
    """CSR graph region growing. Returns (labels, num_regions); labels[i] = -1 for
    primitives never reached (handled by caller's propagation), region id otherwise.
    All tensors on the same device. unit_feats: (P, C) unit-normalized (invalid rows
    are zeros and never join any region since valid_mask_t excludes them)."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    assignable = valid_mask_t.clone()

    # Per-node local coherence (mean cosine to graph neighbors), computed once.
    # Edge similarities are computed in chunks: materializing (E, 512) gathers whole
    # (scene0140_00: ~21M edges -> ~40GB per gather) OOMs; 2M-edge chunks stay <5GB.
    src = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    coherence = torch.zeros(P, device=device)
    E = adjacent.numel()
    for s in range(0, E, 2_000_000):
        e = min(s + 2_000_000, E)
        chunk_sim = (unit_feats[src[s:e]] * unit_feats[adjacent[s:e]]).sum(-1)
        coherence.index_add_(0, src[s:e], chunk_sim)
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    coherence /= deg

    for region_id in range(k):
        cand = torch.where(assignable)[0]
        if cand.numel() == 0:
            return labels, region_id
        seed = cand[coherence[cand].argmax()]
        labels[seed] = region_id
        assignable[seed] = False
        region_sum = unit_feats[seed].clone()
        frontier = seed.unsqueeze(0)
        while frontier.numel() > 0:
            starts, ends = offsets[frontier], offsets[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            # flat CSR gather: for each frontier node, adjacent[start:end], fully vectorized
            flat = torch.repeat_interleave(starts, counts) + (
                torch.arange(total, device=device)
                - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts))
            neigh = torch.unique(adjacent[flat])
            neigh = neigh[assignable[neigh]]
            if neigh.numel() == 0:
                break
            mean = F.normalize(region_sum, dim=0)
            sim = unit_feats[neigh] @ mean
            accepted = neigh[sim >= threshold]
            if accepted.numel() == 0:
                break
            labels[accepted] = region_id
            assignable[accepted] = False
            region_sum += unit_feats[accepted].sum(0)
            frontier = accepted
    return labels, k


def propagate_leftovers(labels, adjacent, offsets, unit_feats, valid_mask_t, num_regions, rounds=5):
    """Attach labels[i]==-1 primitives: most-similar labeled graph neighbor (iterated),
    then global nearest region centroid for anything still orphaned."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    src_all = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    for _ in range(rounds):
        todo_mask = (labels == -1) & valid_mask_t
        if not todo_mask.any():
            break
        # every edge (i -> j) with i unlabeled+valid and j labeled; per-i argmax over sim
        emask = todo_mask[src_all] & (labels[adjacent] >= 0)
        src, dst = src_all[emask], adjacent[emask]
        if src.numel() == 0:
            break
        sim = torch.empty(src.numel(), device=device)
        for s in range(0, src.numel(), 2_000_000):
            e = min(s + 2_000_000, src.numel())
            sim[s:e] = (unit_feats[src[s:e]] * unit_feats[dst[s:e]]).sum(-1)
        best = torch.full((P,), float("-inf"), device=device)
        best.scatter_reduce_(0, src, sim, reduce="amax")
        winners = sim >= best[src] - 1e-7
        new_labels = labels.clone()
        new_labels[src[winners]] = labels[dst[winners]]
        if (new_labels == labels).all():
            break
        labels = new_labels
    todo = torch.where((labels == -1) & valid_mask_t)[0]
    if todo.numel() > 0:
        centroids = torch.zeros(num_regions, unit_feats.shape[1], device=device)
        labeled = labels >= 0
        centroids.index_add_(0, labels[labeled], unit_feats[labeled])
        centroids = F.normalize(centroids, dim=-1)
        labels[todo] = (unit_feats[todo] @ centroids.T).argmax(dim=1)
    return labels


def eval_scene(scene, split, threshold, device, class_sets, text_cache):
    ckpt_dir = f"output/scannet_{scene}_nonfrozen"
    features_path = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen.pt"
    adjacency_path = f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt"
    gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
    centers, radii = load_foam(ckpt_dir, device)
    solved = torch.load(features_path, map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()
    adj = torch.load(adjacency_path, map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

    unit_all = torch.zeros_like(feats)
    vm_t = torch.from_numpy(valid_mask).to(device)
    unit_all[vm_t] = F.normalize(feats[vm_t], dim=-1)

    labels, num_regions = region_grow(adjacent, offsets, unit_all, vm_t, threshold)
    grown = int((labels >= 0).sum())
    labels = propagate_leftovers(labels, adjacent, offsets, unit_all, vm_t, num_regions)
    print(f"  [{scene} thr={threshold}] regions={num_regions}, grown={grown}, "
          f"propagated={int((labels >= 0).sum()) - grown}, valid={int(vm_t.sum())}", flush=True)

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
    owned = assigned >= 0

    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    out = {}
    valid_idx = torch.where(vm_t)[0]
    for cs in class_sets:
        kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
        target_ids = [i for i, _ in kept]
        target_names = [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
        key = (cs, tuple(target_names))
        if key not in text_cache:
            text_cache[key] = embed_class_names(target_names, device)
        text_feats = text_cache[key]

        prim_cls_valid = pool_classify_broadcast(
            labels[valid_idx], unit_all[valid_idx], K_REGIONS, text_feats).cpu().numpy()
        prim_class = np.zeros(centers.shape[0], dtype=np.int64)
        prim_class[valid_idx.cpu().numpy()] = prim_cls_valid
        pred = np.zeros(gt_points.shape[0], dtype=np.int64)
        pred[owned] = prim_class[assigned[owned]] + 1
        _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(target_ids) + 1)
        out[cs] = {"mIoU": miou, "mAcc": macc, "overall_acc": acc}
        print(f"  {scene} {cs} thr={threshold}: mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="all", help="'all' or comma-separated scene names")
    p.add_argument("--thresholds", default="0.85", help="comma-separated cosine thresholds")
    p.add_argument("--class-sets", default="all", help="'all' or comma-separated class set names")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scenes = SCENES if args.scenes == "all" else {s: SCENES[s] for s in args.scenes.split(",")}
    thresholds = [float(t) for t in args.thresholds.split(",")]
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
    text_cache = {}

    summary = {}
    for thr in thresholds:
        results = {cs: {} for cs in class_sets}
        for scene, split in scenes.items():
            per_cs = eval_scene(scene, split, thr, device, class_sets, text_cache)
            for cs, m in per_cs.items():
                results[cs][scene] = m
        summary[str(thr)] = {}
        line = [f"thr={thr}"]
        for cs in class_sets:
            mious = [m["mIoU"] for m in results[cs].values()]
            maccs = [m["mAcc"] for m in results[cs].values()]
            summary[str(thr)][cs] = {
                "num_scenes": len(mious),
                "mean_mIoU": float(np.mean(mious)),
                "mean_mAcc": float(np.mean(maccs)),
                "per_scene": results[cs],
            }
            line.append(f"{cs} {np.mean(mious)*100:.2f}/{np.mean(maccs)*100:.2f}")
        print("\n== " + "  ".join(line) + " ==\n", flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
