"""COVERAGE-GATE PILOT: does invalidating cells with low within-view SAM-mask purity help?

`sam_purity[c]` (from diagnose_lifting_rays.py) is, per view, the weighted share of cell c's
incoming rendering weight that came from the DOMINANT SAM mask in that view, averaged over
views by total weight. It uses no GT and no class names -- only the operator A and the SAM
masks the lift already consumes.

The intervention acts on the COVERAGE SET S / the assignment `a` (Theorem 1's levers), NOT on
per-view weights: a gated cell is marked INVALID, and the standard `nearest_valid` protocol
re-assigns its GT points to the nearest cell that survived. No feature is reweighted.

PRE-REGISTERED FALSIFIER: >= +0.5 mIoU at 19cls, positive on all three class sets, on the
three hardest scenes, at some threshold that is fixed across scenes.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, remap_gt_labels, embed_class_names,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import SCENES

SCR = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad"
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
STAT = os.environ.get("STAT", "sam_purity")


def stat_from_rays(scene, name):
    d = np.load(os.path.join(SCR, f"rays_{scene}.npz"), allow_pickle=True)
    eps = 1e-12
    if name == "sam_purity":
        return d["sam_top"] / np.maximum(d["sam_tot"], eps)
    if name == "sam_groups_per_view":
        return -(d["sam_groups"] / np.maximum(d["n_views"], 1))
    if name == "random":
        # CONTROL. If a random gate of the same size gains as much, the gain is the
        # nearest_valid re-assignment itself and has nothing to do with SAM purity.
        return np.random.default_rng(0).random(d["n_rays"].shape[0])
    if name == "n_rays":
        return d["n_rays"].astype(np.float64)
    if name == "mean_slot":
        return -(d["w_slot"] / np.maximum(d["w_sum"], eps))
    if name == "combo":
        # Does SAM purity add anything ON TOP of the reliability the stats file already has?
        # Rank-average of the two, so the two scales are commensurable.
        a = stat_from_rays(scene, "sam_purity")
        b = stat_from_rays(scene, "reliability_ctrl")
        rk = lambda x: np.argsort(np.argsort(x)) / (len(x) - 1.0)
        return 0.5 * rk(a) + 0.5 * rk(b)
    if name == "reliability_ctrl":
        st = torch.load(f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt",
                        map_location="cpu", weights_only=False)
        sup = st["support"].numpy()
        n_eff = sup ** 2 / np.maximum(st["sum_view_weight_sq"].numpy(), eps)
        nf = st["numerator"].norm(dim=-1).numpy() / np.maximum(sup, eps)
        return nf * n_eff / (n_eff + 1.0)
    raise KeyError(name)


def main():
    enable_determinism()
    device = "cuda"
    scenes = [s for s in os.environ.get("ONLY", "scene0347_00,scene0070_00,scene0140_00").split(",") if s]
    thresholds = [float(x) for x in os.environ.get("THR", "0,0.05,0.10,0.20,0.30").split(",")]
    out = {}
    for scene in scenes:
        split = SCENES[scene]
        ckpt = f"output/scannet_{scene}_nonfrozen"
        solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        centers, radii = load_foam(ckpt, device)
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{split}\{scene}", "segment20")
        # CONTROL: low SAM purity may simply mean BIG cell (a big cell spans more masks), in
        # which case the gate is a size gate and nothing about SAM masks is load-bearing.
        stat = -np.asarray(radii) if STAT == "radius" else stat_from_rays(scene, STAT)
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        print(f"\n===== {scene} =====", flush=True)
        out[scene] = {}
        for drop in thresholds:
            gated = valid_mask.copy()
            if drop > 0:
                s = stat[valid_mask]
                thr = np.quantile(s, drop)
                gated = valid_mask & (stat > thr)
            assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=gated, k=64)
            unit = F.normalize(feats, dim=-1)
            pred_cell = np.asarray(assigned)
            row = {}
            for cs in CLASS_SETS:
                kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                names = [n for _, n in kept]
                gt_t = torch.from_numpy(remap_gt_labels(raw_labels, [i for i, _ in kept])).long()
                text = embed_class_names(names, device)
                cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1     # per PRIMITIVE
                pred = np.where(pred_cell >= 0, cls[np.maximum(pred_cell, 0)], 0)
                _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(kept) + 1)
                row[cs] = {"mIoU": miou * 100, "mAcc": macc * 100}
            out[scene][f"drop{drop}"] = {"kept_cells": int(gated.sum()), **row}
            print(f"  drop {drop:>5}: kept {gated.sum():>7} cells  " + "  ".join(
                f"{cs[13:]}={row[cs]['mIoU']:.2f}" for cs in CLASS_SETS), flush=True)

    print(f"\n=== {len(scenes)}-scene mean, gate stat = {STAT} ===")
    b0 = f"drop{thresholds[0]}"
    base = [float(np.mean([out[s][b0][cs]["mIoU"] for s in scenes])) for cs in CLASS_SETS]
    for drop in thresholds:
        cells = [float(np.mean([out[s][f"drop{drop}"][cs]["mIoU"] for s in scenes])) for cs in CLASS_SETS]
        print(f"drop {drop:>5}: " + "  ".join(f"{c:6.2f}" for c in cells) +
              "   delta " + "  ".join(f"{c-b:+6.2f}" for c, b in zip(cells, base)))
    with open(os.path.join(SCR, f"gate_{STAT}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
