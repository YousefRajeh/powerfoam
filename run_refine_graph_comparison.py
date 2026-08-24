"""PAIRED comparison of the Cech adjacency vs the TRUE power-diagram facet adjacency
inside NormLift's reliability-guided mode-voting refinement.

WHY
---
`powerfoam/bvh.py::count_adjacent` builds the graph stored in model.pt (and re-exported to
artifacts/scannet/<scene>/adjacency_<variant>.pt) by testing `d < r_i + r_j` with
`r = 0.5 * (max.x - min.x)`.  That is bounding-ball overlap (a Cech complex) driven by half
the AABB's X-EXTENT ONLY -- it is not facet sharing.  Measured on scene0347_00: 54.17% facet
recall, 37.77% precision, 36.77% facet-AREA recall, mean degree 21.84 vs a true 15.23.
The correct graph is the regular (weighted Delaunay) triangulation, built by
build_true_facet_graph.py and stored as adjacency_true_facet.pt.

`run_normlift_refine_eval.py` hardcoded `adjacency_{variant}.pt`, so the whole project's
KNN-refinement path has been running on the wrong neighbourhood.  This script measures the
correction.

DESIGN
------
Both arms are fully DETERMINISTIC (no clustering, no seeds: mode_vote_refine is a fixed
function of the features/reliability/graph, and classification is a plain cosine argmax).
So this is a genuinely paired comparison -- the per-scene delta carries NO seed noise and a
sub-1-point mean delta is still meaningful, unlike the k-means arms elsewhere in the project.

Each scene is loaded ONCE and all three arms (base, refined-on-Cech, refined-on-true-facet)
are evaluated against that single state, so the point->cell assignment (the expensive part)
is paid once per scene rather than three times.  Per-scene JSONs are flushed as each scene
finishes so a partial sweep is usable.

Scene order is HARDEST-FIRST (the canonical order from run_cluster_classify_eval.py).

Usage:
  python run_refine_graph_comparison.py --suffix _l3 --outdir artifacts/scannet/refine_graph
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from run_normlift_refine_eval import mode_vote_refine
from build_true_facet_graph import load_points_radii

HARDEST_FIRST = ["scene0140_00", "scene0645_00", "scene0070_00", "scene0347_00",
                 "scene0590_00", "scene0400_00", "scene0200_00", "scene0000_00",
                 "scene0097_00", "scene0062_00"]


def load_csr(path, device):
    adj = torch.load(path, map_location="cpu", weights_only=True)
    return adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()


def safe_chunk(offsets, budget_rows=200_000):
    """mode_vote_refine materialises (B, D, C) with C=512; cap B*D so the gather stays small.
    Chunk size does not change the arithmetic (rows are independent), only the memory."""
    D = int((offsets[1:] - offsets[:-1]).max()) + 1
    return max(256, budget_rows // max(D, 1)), D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_l3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--outdir", default="artifacts/scannet/refine_graph")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene} already done", flush=True)
            continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        ckpt_dir = f"output/scannet_{scene}_{a.variant}"

        centers, radii = load_points_radii(ckpt_dir)
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(valid_mask).to(device)
        unit = torch.zeros_like(feats)
        unit[vm_t] = F.normalize(feats[vm_t], dim=-1)
        del feats, solved
        stats = AccumulatedFeatureStats.load(f"{art}/train_stats_sam_{a.variant}{a.suffix}.pt")
        R = stats.reliability()["reliability"].to(device).float() * vm_t
        del stats
        positions = torch.from_numpy(centers).to(device).float()
        print(f"[{scene}] P={centers.shape[0]} valid={int(valid_mask.sum())} "
              f"(load {time.time()-t0:.0f}s)", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        print(f"[{scene}] assigned {gt_points.shape[0]} GT points "
              f"({time.time()-t0:.0f}s)", flush=True)

        arms = {"base": unit}
        graph_info = {}
        for tag, path in (("cech", f"{art}/adjacency_{a.variant}.pt"),
                          ("true_facet", f"{art}/adjacency_true_facet.pt")):
            adjacent, offsets = load_csr(path, device)
            chunk, D = safe_chunk(offsets)
            deg = (offsets[1:] - offsets[:-1]).float()
            graph_info[tag] = {"mean_degree": float(deg.mean()),
                               "max_degree": int(deg.max()),
                               "num_directed_edges": int(adjacent.numel())}
            r = unit
            for _ in range(a.passes):
                r = mode_vote_refine(r, R, positions, adjacent, offsets,
                                     tau=a.tau, gamma=a.gamma, delta=a.delta,
                                     chunk=chunk)
            changed = float(((r - unit).abs().sum(-1) > 1e-6).float().mean())
            graph_info[tag]["changed_frac"] = changed
            arms[f"refined_{tag}"] = r
            del adjacent, offsets
            torch.cuda.empty_cache()
            print(f"[{scene}] {tag}: mean_deg={graph_info[tag]['mean_degree']:.2f} "
                  f"max_deg={D-1} changed={changed*100:.1f}% ({time.time()-t0:.0f}s)",
                  flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "graph": graph_info, "num_primitives": int(centers.shape[0]),
               "arms": {}}
        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names([n for _, n in kept], device)
            for tag, u in arms.items():
                cls = (u @ text.T).argmax(-1).cpu().numpy()
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(
                    gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou),
                                                       "mAcc": float(macc)}
                print(f"  {scene} {cs} [{tag}]: mIoU={miou*100:.2f} mAcc={macc*100:.2f}",
                      flush=True)

        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[{scene}] done in {time.time()-t0:.0f}s -> {out_path}\n", flush=True)
        del unit, arms, R, positions
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
