"""Growing strategies for the clustering-locality failure, tested on cached artifacts.

MEASURED PROBLEM (diagnose_region_locality.py, scene0347_00): only 6.6% of the 320 regions
are a single connected piece on the facet graph; the median region is scattered across 15.5
disjoint fragments, and for the worst tenth the largest fragment holds under 26% of the
region's cells. One pooled feature and one label are then broadcast across all of it -- which
is exactly the "desk predicted on a distant wall" artifact seen in the qualitative panels.

MEASURED NON-FIX: splitting every region into its connected components (320 -> 11,261)
helps fine classes (+0.70 / +0.90 mIoU at 19/15cls) but badly hurts coarse ones
(-3.48 at 10cls). Fragments become too small to pool a stable feature, and coarse classes
(wall, floor, large furniture) depend on that aggregation. So the fix cannot be "always
split"; it has to split only where fragmentation is actually pathological.

STRATEGIES (all post-hoc on existing features -- no retraining, no re-lifting):

  drop_tail    Keep each region's LARGEST component; re-assign the stray fragments to the
               most similar neighbouring region in feature space. Removes the long tail of
               orphan cells that hijack a region's label without changing region count.

  split_bad    Split ONLY regions whose largest component holds less than `--frac` of the
               cells (the p10 = 0.252 pathological cases). Coherent regions are untouched, so
               coarse-class pooling survives while the "desk"-type regions get separated.

  min_size     Split by component, then merge any component below `--min-cells` back into its
               most similar neighbouring component. Directly targets the noise that made the
               naive split lose at 10cls.

  relabel      Keep the 320 regions for POOLING (so features stay well-averaged) but assign
               labels per connected COMPONENT using the region's own per-cell votes. Decouples
               "what feature describes this" from "where that label applies" -- the cheapest
               strategy, and the one that most directly targets broadcast-across-disconnected
               geometry.
"""
import argparse
import json
import os
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
from diagnose_region_locality import connected_components

LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def reassign_to_neighbour(target_mask, keep_mask, unit, out_labels, chunk=4096):
    """Give every cell in `target_mask` the label of the most similar KEPT cell."""
    idx_t = torch.where(target_mask)[0]
    idx_k = torch.where(keep_mask)[0]
    if idx_t.numel() == 0 or idx_k.numel() == 0:
        return out_labels
    for c in idx_t.split(chunk):
        sim = unit[c] @ unit[idx_k].T
        out_labels[c] = out_labels[idx_k[sim.argmax(-1)]]
    return out_labels


def build_variant(mode, leaf, comp_v, unit, args, device):
    """Return (region_ids, n_regions, label_groups) for a strategy.

    `label_groups` is the grouping used for LABEL assignment; it may differ from the pooling
    grouping (that is the whole point of `relabel`).
    """
    n_cells = leaf.numel()
    if mode == "baseline":
        return leaf, K_FLAT, leaf

    _, comp_idx = torch.unique(comp_v, return_inverse=True)
    n_comp = int(comp_idx.max()) + 1

    if mode == "split":
        return comp_idx, n_comp, comp_idx

    if mode == "relabel":
        # pool over the ORIGINAL regions, but assign labels per connected component
        return leaf, K_FLAT, comp_idx

    # component sizes and, per region, which component is largest
    sizes = torch.bincount(comp_idx, minlength=n_comp)
    if mode == "drop_tail":
        keep = torch.zeros(n_cells, dtype=torch.bool, device=device)
        for r in torch.unique(leaf):
            m = leaf == r
            cs, cnt = torch.unique(comp_idx[m], return_counts=True)
            keep |= m & (comp_idx == cs[cnt.argmax()])
        out = leaf.clone()
        out = reassign_to_neighbour(~keep, keep, unit, out)
        return out, K_FLAT, out

    if mode == "split_bad":
        out = leaf.clone()
        nxt = int(leaf.max()) + 1
        for r in torch.unique(leaf):
            m = leaf == r
            n = int(m.sum())
            cs, cnt = torch.unique(comp_idx[m], return_counts=True)
            if float(cnt.max()) / n >= args.frac:
                continue                      # coherent enough -- leave it alone
            for c in cs:                      # pathological: separate every component
                sel = m & (comp_idx == c)
                out[sel] = nxt
                nxt += 1
        _, out = torch.unique(out, return_inverse=True)
        return out, int(out.max()) + 1, out

    if mode == "min_size":
        small = sizes[comp_idx] < args.min_cells
        out = comp_idx.clone()
        out = reassign_to_neighbour(small, ~small, unit, out)
        _, out = torch.unique(out, return_inverse=True)
        return out, int(out.max()) + 1, out

    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--modes", default="baseline,split,relabel,drop_tail,split_bad,min_size")
    p.add_argument("--frac", type=float, default=0.5,
                   help="split_bad: split a region if its largest component holds < this")
    p.add_argument("--min-cells", type=int, default=50,
                   help="min_size: merge components smaller than this")
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    scene, variant = a.scene, a.variant
    split_name = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{split_name}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats)
    unit_full[vi] = F.normalize(feats[vi], dim=-1)
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

    region_full = torch.full((centers.shape[0],), -1, dtype=torch.long, device=device)
    region_full[vi] = leaf
    comp_v = connected_components(region_full, adjacent, offsets, K_FLAT, device)[vi]

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    Rv = R[vi]
    vi_np = vi.cpu().numpy()

    results = {}
    for mode in a.modes.split(","):
        pool_ids, n_pool, label_ids = build_variant(mode, leaf, comp_v, unit, a, device)
        n_lab = int(label_ids.max()) + 1
        row = {"n_pool_regions": int(n_pool), "n_label_groups": int(n_lab)}
        for cs_name in ("opengaussian19", "opengaussian15", "opengaussian10"):
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs_name] if n2i[n] in present]
            tids, tnames = [i for i, _ in kept], [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(tnames, device)
            percell = unit @ text.T
            lam = LAMBDA[cs_name]
            lab = (percell - lam * percell.mean(0, keepdim=True)).argmax(-1)
            hist = torch.zeros(n_lab, len(tids), device=device)
            hist.index_put_((label_ids, lab), Rv, accumulate=True)
            vcls = hist.argmax(-1)
            pc = np.zeros(centers.shape[0], dtype=np.int64)
            pc[vi_np] = vcls[label_ids].cpu().numpy()
            pred = np.zeros(len(gt_t), dtype=np.int64)
            pred[owned] = pc[assigned[owned]] + 1
            _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                 len(tids) + 1)
            row[cs_name] = {"mIoU": float(miou), "mAcc": float(macc)}
        results[mode] = row
        print(f"  {mode:<11} pool={n_pool:>6} labels={n_lab:>6}  "
              f"mIoU 19/15/10 = {row['opengaussian19']['mIoU']*100:5.2f} / "
              f"{row['opengaussian15']['mIoU']*100:5.2f} / "
              f"{row['opengaussian10']['mIoU']*100:5.2f}", flush=True)

    if "baseline" in results:
        b = results["baseline"]
        print("\n  deltas vs baseline (19/15/10 cls):")
        for m, r in results.items():
            if m == "baseline":
                continue
            d = [(r[c]["mIoU"] - b[c]["mIoU"]) * 100
                 for c in ("opengaussian19", "opengaussian15", "opengaussian10")]
            print(f"    {m:<11} {d[0]:+6.2f} {d[1]:+6.2f} {d[2]:+6.2f}"
                  f"   mean {np.mean(d):+.2f}")
    if a.output:
        json.dump({"scene": scene, "results": results, "frac": a.frac,
                   "min_cells": a.min_cells}, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
