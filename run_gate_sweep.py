"""GATE SWEEP, two protocols, GT-free statistics only.

The previous pass (`run_sam_purity_gate.py`) called this a COVERAGE move. It is not: it passes
`valid=gated` to `assign_points_to_power_cells`, which deletes the gated cell from the KD-tree,
so its GT points are RE-ASSIGNED to the nearest survivor and still get a prediction. Coverage
stays exactly 100%. That is the ASSIGNMENT lever `a`, not the coverage lever `S`.

This script measures both, on the same statistics and thresholds:

  mode=reassign  gated cells removed from the assignment KD-tree  -> lever a, coverage 100%
  mode=abstain   assignment unchanged; a point whose owner is gated scores 0 (no prediction)
                 -> lever S, coverage falls, Theorem 2 ceiling falls with it

`random` and `radius` are the negative controls: reassign alone reshuffles ownership, and any
gain a random gate of the same size produces is protocol artifact, not signal.
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
EPS = 1e-12


def _rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64) / max(len(x) - 1.0, 1.0)


def stat(scene, name, radii):
    d = np.load(os.path.join(SCR, f"rays_{scene}.npz"), allow_pickle=True)
    ws = np.maximum(d["w_sum"], EPS)
    if name == "sam_purity":
        return d["sam_top"] / np.maximum(d["sam_tot"], EPS)
    if name == "n_rays":
        return d["n_rays"].astype(np.float64)
    if name == "n_views":
        return d["n_views"].astype(np.float64)
    if name == "support":                      # column sum A^T 1
        return d["w_sum"].astype(np.float64)
    if name == "front05_share":                # depth order: share of mass arriving with T>=0.5
        return d["w_front05"] / ws
    if name == "firstsig_share":
        return d["w_firstsig"] / ws
    if name == "mean_trans":
        return d["w_trans"] / ws
    if name == "neg_mean_slot":
        return -(d["w_slot"] / ws)
    if name == "radius":
        return -np.asarray(radii, dtype=np.float64)
    if name == "random":
        return np.random.default_rng(0).random(d["n_rays"].shape[0])
    if name == "reliability":
        st = torch.load(f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt",
                        map_location="cpu", weights_only=False)
        sup = st["support"].numpy()
        n_eff = sup ** 2 / np.maximum(st["sum_view_weight_sq"].numpy(), EPS)
        nf = st["numerator"].norm(dim=-1).numpy() / np.maximum(sup, EPS)
        return nf * n_eff / (n_eff + 1.0)
    if name == "n_eff":
        st = torch.load(f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt",
                        map_location="cpu", weights_only=False)
        sup = st["support"].numpy()
        return sup ** 2 / np.maximum(st["sum_view_weight_sq"].numpy(), EPS)
    if "+" in name:                            # rank-average combination
        parts = name.split("+")
        return np.mean([_rank(stat(scene, p, radii)) for p in parts], 0)
    raise KeyError(name)


def main():
    enable_determinism()
    device = "cuda"
    scenes = [s for s in os.environ.get(
        "ONLY", "scene0347_00,scene0070_00,scene0140_00").split(",") if s]
    stats = [s for s in os.environ.get(
        "STATS", "random,radius,reliability,n_eff,support,n_views,n_rays,sam_purity,"
                 "front05_share,firstsig_share,mean_trans,"
                 "sam_purity+reliability,sam_purity+reliability+front05_share").split(",") if s]
    thr = [float(x) for x in os.environ.get(
        "THR", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7").split(",")]
    modes = [m for m in os.environ.get("MODES", "reassign,abstain").split(",") if m]
    tag = os.environ.get("TAG", "sweep")
    out = {}

    for scene in scenes:
        split = SCENES[scene]
        ckpt = f"output/scannet_{scene}_nonfrozen"
        solved = torch.load(
            f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
            map_location=device, weights_only=True)
        feats = F.normalize(solved["primitive_features"].to(device).float(), dim=-1)
        valid_mask = solved["valid_mask"].cpu().numpy()
        centers, radii = load_foam(ckpt, device)
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{split}\{scene}", "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        cls_cache = {}
        for cs in CLASS_SETS:
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt_t = torch.from_numpy(
                remap_gt_labels(raw_labels, [i for i, _ in kept])).long()
            text = embed_class_names([n for _, n in kept], device)
            cls_cache[cs] = ((feats @ text.T).argmax(-1).cpu().numpy() + 1, gt_t, len(kept) + 1)

        # baseline assignment (nearest_valid over the unmodified valid set)
        base_assign = np.asarray(
            assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64))
        print(f"\n===== {scene}  valid={valid_mask.sum()} gtpts={len(gt_points)} =====",
              flush=True)
        out[scene] = {}
        for name in stats:
            s = stat(scene, name, radii)
            for drop in thr:
                gated = valid_mask.copy()
                if drop > 0:
                    # RANK-based, not quantile: sam_purity has a huge atom at exactly 1.0, so
                    # `s > quantile` drops far more (or all) cells than requested and the
                    # kept-count is not comparable across statistics. Ranking gives the exact
                    # requested fraction for every statistic, which the controls require.
                    vi = np.nonzero(valid_mask)[0]
                    order = vi[np.argsort(s[vi], kind="stable")]
                    gated = np.zeros_like(valid_mask)
                    gated[order[int(round(drop * len(order))):]] = True
                for mode in modes:
                    if drop == 0 and mode == "abstain":
                        continue
                    if mode == "reassign":
                        pc = np.asarray(assign_points_to_power_cells(
                            gt_points, centers, radii, valid=gated, k=64))
                        alive = np.ones(len(pc), dtype=bool)
                    else:
                        pc = base_assign
                        alive = gated[np.maximum(pc, 0)] & (pc >= 0)
                    row = {}
                    for cs in CLASS_SETS:
                        cls, gt_t, nlab = cls_cache[cs]
                        pred = np.where((pc >= 0) & alive, cls[np.maximum(pc, 0)], 0)
                        _, miou, acc, macc = calculate_metrics(
                            gt_t, torch.from_numpy(pred).long(), nlab)
                        row[cs] = {"mIoU": miou * 100, "mAcc": macc * 100}
                    # classifiable fraction over GT points with a real label
                    lab = (gt_t.numpy() > 0)
                    cov = float((alive & lab).sum() / max(lab.sum(), 1))
                    key = f"{name}|{mode}|{drop}"
                    out[scene][key] = {"kept_cells": int(gated.sum()),
                                       "coverage": cov, **row}
                    print(f"  {name:<38} {mode:<9} drop{drop:<4} cov={cov:.3f} " +
                          "  ".join(f"{cs[13:]}={row[cs]['mIoU']:.2f}"
                                    for cs in CLASS_SETS), flush=True)
    with open(os.path.join(SCR, f"gatesweep_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote gatesweep_{tag}.json")


if __name__ == "__main__":
    main()
