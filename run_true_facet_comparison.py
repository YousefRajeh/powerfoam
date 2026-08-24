"""Three-arm comparison on the TRUE power-diagram facet graph vs the historical Cech graph
vs the k-means pos-aware reference, over all ScanNet scenes.

Arms (all classify identically -- mean-pool unit features per region, hubness-corrected
argmax vs class-set text embeddings, broadcast -- so every delta is attributable to the
clustering alone):
  A. batched mean-anchored region growing on adjacency_true_facet.pt   (deterministic)
  B. batched mean-anchored region growing on adjacency_nonfrozen.pt    (deterministic)
     ...both swept over --thresholds
  C. two-level position-aware k-means (64 pos roots x 5 feature leaves = 320) over --seeds

WHY THIS EXISTS RATHER THAN LOOPING run_region_grow_eval.py
-----------------------------------------------------------
run_region_grow_eval.py rebuilds the whole per-scene state (warp PowerfoamScene
construction + assign_points_to_power_cells + GT load) once per (scene, threshold, seed,
mode). The full matrix is 4 thr x 2 graphs + 3 seeds = 11 evals per scene, i.e. 110 scene
loads, and the load dominates the run (a single k-means arm did not finish one 1.1M-cell
scene in 40 minutes). Here each scene is loaded ONCE and all 11 evaluations run against
that state, so the load cost is paid 10 times instead of 110.

Per-scene results are flushed to <outdir>/<scene>.json as each scene finishes, so a
partial sweep is still usable.
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

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS,
    calculate_metrics, remap_gt_labels, embed_class_names,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import (
    pool_classify_broadcast, SCENES, CLASS_SETS, two_level_position_aware, K_FLAT,
)
from run_region_grow_eval import batched_region_grow

# Hardest-first: the scenes that would FALSIFY the lead go first (see the ordering note in
# run_cluster_classify_eval.py). scene0140_00 / scene0645_00 are also the two 1M-cell scenes.
HARDEST_FIRST = ["scene0140_00", "scene0645_00", "scene0070_00", "scene0347_00",
                 "scene0590_00", "scene0400_00", "scene0200_00", "scene0000_00",
                 "scene0097_00", "scene0062_00"]


def load_adjacency(path, device):
    adj = torch.load(path, map_location="cpu", weights_only=True)
    return (adj["adjacent"].to(device).long(), adj["offsets"].to(device).long())


class SceneState:
    """Everything an arm needs, built once per scene."""

    def __init__(self, scene, split, device, suffix=""):
        t0 = time.time()
        ckpt_dir = f"output/scannet_{scene}_nonfrozen"
        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
        self.scene = scene
        self.device = device

        self.gt_points, self.raw_labels, self.all_names = load_scannet_pointcept_gt(
            gt_dir, "segment20")
        self.centers, self.radii = load_foam(ckpt_dir, device)
        solved = torch.load(
            f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen{suffix}.pt",
            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        self.vm_t = torch.from_numpy(valid_mask).to(device)

        # unit features; invalid rows stay zero. Built in 200k-row chunks -- a boolean
        # mask on an (N, 512) tensor materialises a full copy and has OOMed this project
        # repeatedly at N ~ 1.1M.
        self.unit_all = torch.zeros_like(feats)
        P = feats.shape[0]
        for s in range(0, P, 200_000):
            e = min(s + 200_000, P)
            m = self.vm_t[s:e]
            if m.any():
                self.unit_all[s:e][m] = F.normalize(feats[s:e][m], dim=-1)
        del feats, solved
        torch.cuda.empty_cache()

        self.assigned = assign_points_to_power_cells(
            self.gt_points, self.centers, self.radii, valid=valid_mask, k=64)
        self.owned = self.assigned >= 0
        self.valid_idx = torch.where(self.vm_t)[0]
        # (V, 512) slice used by every scoring call -- taken once, not 11 times
        self.unit_valid = self.unit_all[self.valid_idx].contiguous()
        self.valid_idx_np = self.valid_idx.cpu().numpy()
        self.P = P
        self.n_valid = int(self.vm_t.sum())
        print(f"  [{scene}] loaded P={P} valid={self.n_valid} "
              f"gt_points={self.gt_points.shape[0]} in {time.time()-t0:.1f}s", flush=True)

    def score(self, labels, num_regions, class_sets, text_cache):
        """labels over all P (-1 outside valid). Returns {class_set: metrics}."""
        name_to_id = {n: i for i, n in enumerate(self.all_names)}
        present = set(np.unique(self.raw_labels).tolist())
        lab_valid = labels[self.valid_idx]
        out = {}
        for cs in class_sets:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            target_ids = [i for i, _ in kept]
            target_names = [n for _, n in kept]
            key = (cs, tuple(target_names))
            if key not in text_cache:
                text_cache[key] = embed_class_names(target_names, self.device)
            text_feats = text_cache[key]
            gt_t = torch.from_numpy(remap_gt_labels(self.raw_labels, target_ids)).long()

            prim_cls_valid = pool_classify_broadcast(
                lab_valid, self.unit_valid, num_regions, text_feats).cpu().numpy()
            prim_class = np.zeros(self.P, dtype=np.int64)
            prim_class[self.valid_idx_np] = prim_cls_valid
            pred = np.zeros(self.gt_points.shape[0], dtype=np.int64)
            pred[self.owned] = prim_class[self.assigned[self.owned]] + 1
            _, miou, acc, macc = calculate_metrics(
                gt_t, torch.from_numpy(pred).long(), len(target_ids) + 1)
            out[cs] = {"mIoU": miou, "mAcc": macc, "overall_acc": acc}
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="all")
    p.add_argument("--thresholds", default="0.95,0.97,0.98,0.99")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--class-sets", default="all")
    p.add_argument("--graphs", default="true_facet,cech")
    p.add_argument("--arms", default="grow,kmeans")
    p.add_argument("--suffix", default="")
    p.add_argument("--outdir", default="artifacts/scannet/tfg_compare")
    args = p.parse_args()

    from determinism import enable_determinism
    enable_determinism()      # measured drift 40.93 vs 40.30 without it
    device = "cuda"
    os.makedirs(args.outdir, exist_ok=True)

    if args.scenes == "all":
        scenes = [s for s in HARDEST_FIRST if s in SCENES]
    else:
        scenes = args.scenes.split(",")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
    graphs = args.graphs.split(",")
    arms = args.arms.split(",")
    text_cache = {}

    for scene in scenes:
        out_path = os.path.join(args.outdir, f"{scene}.json")
        print(f"\n===== {scene} =====", flush=True)
        st = SceneState(scene, SCENES[scene], device, suffix=args.suffix)
        res = {"scene": scene, "num_cells": st.P, "num_valid": st.n_valid, "arms": {}}

        if "grow" in arms:
            for gname in graphs:
                path = (f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt"
                        if gname == "cech"
                        else f"artifacts/scannet/{scene}/adjacency_true_facet.pt")
                if not os.path.exists(path):
                    print(f"  [{scene}] MISSING {path} -- skipping arm {gname}", flush=True)
                    continue
                adjacent, offsets = load_adjacency(path, device)
                mean_deg = adjacent.numel() / (offsets.numel() - 1)
                print(f"  [{scene}] graph={gname} E={adjacent.numel()} "
                      f"mean_deg={mean_deg:.2f}", flush=True)
                res.setdefault("graphs", {})[gname] = {
                    "directed_entries": int(adjacent.numel()), "mean_degree": mean_deg}
                for thr in thresholds:
                    t0 = time.time()
                    labels, num_regions = batched_region_grow(
                        adjacent, offsets, st.unit_all, st.vm_t, thr)
                    nonsing = int((torch.bincount(labels[labels >= 0],
                                                  minlength=num_regions) > 1).sum())
                    m = st.score(labels, num_regions, class_sets, text_cache)
                    key = f"grow_{gname}_thr{thr}"
                    res["arms"][key] = {
                        "num_regions": num_regions, "non_singleton": nonsing,
                        "metrics": m, "seconds": time.time() - t0}
                    print(f"  {scene} {key}: regions={num_regions} ({nonsing} nonsing) "
                          + "  ".join(f"{cs}={m[cs]['mIoU']*100:.2f}" for cs in class_sets)
                          + f"  [{time.time()-t0:.1f}s]", flush=True)
                    del labels
                    torch.cuda.empty_cache()
                del adjacent, offsets
                torch.cuda.empty_cache()

        if "kmeans" in arms:
            pos_v = torch.from_numpy(st.centers[st.valid_idx_np]).to(device).float()
            for sd in seeds:
                t0 = time.time()
                leaf = two_level_position_aware(pos_v, st.unit_valid, seed=sd)
                labels = torch.full((st.P,), -1, dtype=torch.long, device=device)
                labels[st.valid_idx] = leaf
                m = st.score(labels, K_FLAT, class_sets, text_cache)
                key = f"kmeans_pos_seed{sd}"
                res["arms"][key] = {"num_regions": K_FLAT, "metrics": m,
                                    "seconds": time.time() - t0}
                print(f"  {scene} {key}: "
                      + "  ".join(f"{cs}={m[cs]['mIoU']*100:.2f}" for cs in class_sets)
                      + f"  [{time.time()-t0:.1f}s]", flush=True)
                del labels, leaf
                torch.cuda.empty_cache()
            del pos_v

        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {out_path}", flush=True)
        del st
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
