"""Is the clustering SPATIALLY INCOHERENT, and is that what costs us mIoU?

The qualitative region panels showed the failure directly: on scene0000_00 the entire "desk"
prediction came from ONE region out of 320, while the best classes were covered by 7-18.
A region is pooled into a single feature and its single label is broadcast to every cell it
contains -- so a region spanning disconnected geometry paints one label across unrelated
parts of the room, and no per-cell feature is ever consulted for those points.

This measures the effect instead of eyeballing it, using the REAL facet adjacency:

  components_per_region  how many connected pieces a region breaks into on the Cech complex.
                         1 = spatially coherent. Large = the region is scattered.
  largest_component_frac what share of the region lives in its biggest piece. Low = the label
                         is decided by cells that are mostly somewhere else.
  spatial_diameter       extent of the region in metres, against the scene diagonal.

Then the decisive test, which needs no retraining: SPLIT every region into its connected
components on the facet graph, re-pool each component independently, and re-classify. If
mIoU rises, spatial incoherence was costing us accuracy and the fix belongs in the
clustering. If it does not move, the regions were incoherent but the labels happened to be
right anyway, and the effort belongs elsewhere.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from run_normlift_refine_eval import mode_vote_refine

LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def connected_components(labels, adjacent, offsets, n_labels, device):
    """Label-constrained connected components over the facet graph.

    Two cells join a component only if they share a facet AND carry the same region id, so
    the result counts how many disjoint pieces each region actually occupies. Implemented as
    iterated min-propagation over edges: each cell takes the smallest component id among
    same-region neighbours until nothing changes. Vectorized, so the cost is O(rounds * E)
    with rounds ~ the diameter of the largest component.
    """
    P = labels.numel()
    deg = offsets[1:] - offsets[:-1]
    src = torch.repeat_interleave(torch.arange(P, device=device), deg)
    dst = adjacent
    same = labels[src] == labels[dst]          # only traverse within a region
    src, dst = src[same], dst[same]
    comp = torch.arange(P, device=device, dtype=torch.long)
    for _ in range(400):
        cand = torch.full((P,), P, device=device, dtype=torch.long)
        cand.scatter_reduce_(0, dst, comp[src], reduce="amin", include_self=True)
        new = torch.minimum(comp, cand)
        if bool((new == comp).all()):
            break
        comp = new
    return comp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    scene, variant = a.scene, a.variant
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats)
    unit_full[vi] = F.normalize(feats[vi], dim=-1)
    import os
    sp = f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt"
    if os.path.exists(sp):
        st = AccumulatedFeatureStats.load(sp)
        R = st.reliability()["reliability"].to(device).float() * vm_t
        del st
    else:
        R = vm_t.float()
    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt",
                     map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    positions = torch.from_numpy(centers).to(device).float()

    ref = unit_full
    for _ in range(a.refine):
        ref = mode_vote_refine(ref, R, positions, adjacent, offsets)
    unit = ref[vi]
    leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")

    # region id per FULL cell array (-1 where invalid), so components run on the real graph
    region_full = torch.full((centers.shape[0],), -1, dtype=torch.long, device=device)
    region_full[vi] = leaf
    comp = connected_components(region_full, adjacent, offsets, K_FLAT, device)

    diag = positions.max(0).values - positions.min(0).values
    scene_diag = float(diag.norm())
    stats = []
    for r in range(K_FLAT):
        m = region_full == r
        n = int(m.sum())
        if n == 0:
            continue
        cs, counts = torch.unique(comp[m], return_counts=True)
        pos = positions[m]
        extent = float((pos.max(0).values - pos.min(0).values).norm())
        stats.append({"region": r, "cells": n, "components": int(cs.numel()),
                      "largest_frac": float(counts.max()) / n,
                      "diameter_m": extent, "diameter_rel": extent / scene_diag})
    ncomp = np.array([s["components"] for s in stats])
    lfrac = np.array([s["largest_frac"] for s in stats])
    drel = np.array([s["diameter_rel"] for s in stats])
    print(f"\n=== {scene} ({variant}): {len(stats)} non-empty regions of {K_FLAT} ===")
    print(f"  components per region : median={np.median(ncomp):.0f}  mean={ncomp.mean():.1f} "
          f"p90={np.percentile(ncomp,90):.0f}  max={ncomp.max()}")
    print(f"  regions that are a SINGLE connected piece: {100*(ncomp==1).mean():.1f}%")
    print(f"  largest-component share: median={np.median(lfrac):.3f} "
          f"p10={np.percentile(lfrac,10):.3f}")
    print(f"  region diameter / scene diagonal: median={np.median(drel):.3f} "
          f"p90={np.percentile(drel,90):.3f}  (1.0 = spans the whole room)")

    # ---- decisive test: split regions into connected components, then re-classify ----
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    Rv = R[vi]
    comp_v = comp[vi]
    _, split_leaf = torch.unique(comp_v, return_inverse=True)
    n_split = int(split_leaf.max()) + 1
    print(f"\n  splitting by connected component: {K_FLAT} regions -> {n_split} regions")

    out = {"scene": scene, "n_regions": len(stats), "n_split_regions": n_split,
           "components_median": float(np.median(ncomp)),
           "single_piece_frac": float((ncomp == 1).mean()), "results": {}}
    for cs_name in ("opengaussian19", "opengaussian15", "opengaussian10"):
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs_name] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
        text = embed_class_names(tnames, device)
        percell = unit @ text.T
        lam = LAMBDA[cs_name]
        lab = (percell - lam * percell.mean(0, keepdim=True)).argmax(-1)
        row = {}
        for tag, lf, nreg in (("baseline", leaf, K_FLAT), ("split", split_leaf, n_split)):
            hist = torch.zeros(nreg, len(tids), device=device)
            hist.index_put_((lf, lab), Rv, accumulate=True)
            vcls = hist.argmax(-1)
            pc = np.zeros(centers.shape[0], dtype=np.int64)
            pc[vi.cpu().numpy()] = vcls[lf].cpu().numpy()
            pred = np.zeros(len(gt_t), dtype=np.int64)
            pred[owned] = pc[assigned[owned]] + 1
            _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                 len(tids) + 1)
            row[tag] = {"mIoU": float(miou), "mAcc": float(macc)}
        d = (row["split"]["mIoU"] - row["baseline"]["mIoU"]) * 100
        print(f"  {cs_name:<16} baseline mIoU={row['baseline']['mIoU']*100:5.2f} "
              f"-> split={row['split']['mIoU']*100:5.2f}  ({d:+.2f})")
        out["results"][cs_name] = row

    if a.output:
        json.dump(out, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
