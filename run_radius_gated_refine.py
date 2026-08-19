"""Radius-gated mode-voting refinement (Feature-Foam-only).

Measured motivation (diagnose_radius_accuracy.py): per-point accuracy rises monotonically
with power radius (0.518 -> 0.729), the smallest-radius decile carries 13.7% of all errors
vs 7.7% for the largest, and radius is ORTHOGONAL to reliability (rank-corr -0.011) while
still separating accuracy inside every reliability decile. The power radius is foam's own
partition weight -- it decides how much volume a cell wins -- so this signal has no
Gaussian analogue.

Simple radius WEIGHTING was already shown not to convert into mIoU. This tests radius as a
TARGETING variable inside NormLift's Eq. 9-10 refinement instead, via two knobs:

  alpha: per-cell replacement margin  delta_i = delta * (r_i / median_r)^alpha
         alpha > 0 makes small cells cheap to overwrite and protects large ones.
  beta:  voter/candidate authority     R_eff(k) = R(k) * (r_k / median_r)^beta
         beta > 0 lets larger, better-evidenced neighbours dominate the vote.

alpha=beta=0 reproduces the existing uniform refinement exactly.
"""
import argparse
import itertools
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt, calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES


def radius_gated_refine(unit, R, radius, positions, adjacent, offsets, sigma_d=None,
                        tau=0.8, gamma=0.05, delta=0.1, alpha=0.0, beta=0.0, chunk=8192):
    device = unit.device
    P = unit.shape[0]
    deg = offsets[1:] - offsets[:-1]
    D = int(deg.max()) + 1
    if sigma_d is None:
        src = torch.repeat_interleave(torch.arange(P, device=device), deg)
        sigma_d = float((positions[src] - positions[adjacent]).norm(dim=-1).median())
    med_r = float(radius[radius > 0].median())
    rrel = (radius / med_r).clamp_min(1e-6)
    R_eff = R * rrel.pow(beta) if beta != 0.0 else R
    delta_i = delta * rrel.pow(alpha) if alpha != 0.0 else torch.full_like(radius, delta)

    refined = unit.clone()
    arangeD = torch.arange(D, device=device)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        B = e - s
        rows = torch.arange(s, e, device=device)
        cand = torch.full((B, D), -1, dtype=torch.long, device=device)
        cand[:, 0] = rows
        dg = deg[s:e]
        mask = arangeD[None, 1:] <= dg[:, None]
        flat = torch.repeat_interleave(offsets[s:e], dg) + (
            torch.arange(int(dg.sum()), device=device)
            - torch.repeat_interleave(torch.cumsum(dg, 0) - dg, dg))
        cand[:, 1:][mask] = adjacent[flat]
        valid = cand >= 0
        cidx = cand.clamp_min(0)

        U = unit[cidx]
        Rj = R_eff[cidx] * valid
        d_ik = torch.exp(-((positions[cidx] - positions[rows][:, None, :]) ** 2).sum(-1)
                         / (2 * sigma_d ** 2)) * valid
        a = Rj * d_ik
        g = torch.sigmoid(((U @ U.transpose(1, 2)) - tau) / gamma)
        S = Rj * (g @ a.unsqueeze(-1)).squeeze(-1)
        S[~valid] = float("-inf")
        S_self = S[:, 0]
        S_neigh = S.clone()
        S_neigh[:, 0] = float("-inf")
        best_val, best_j = S_neigh.max(dim=1)
        take = best_val > S_self + delta_i[rows]
        refined[rows[take]] = U[take, best_j[take]]
    return refined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--alphas", default="0,0.5,1.0,2.0")
    p.add_argument("--betas", default="0,0.5,1.0")
    p.add_argument("--class-sets", default="opengaussian19,opengaussian10")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    results = {}
    for scene in args.scenes.split(","):
        split = SCENES[scene]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{args.gt_root}/{split}/{scene}", "segment20")
        centers, radii = load_foam(f"output/scannet_{scene}_{args.variant}", device)
        solved = torch.load(
            f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}_l3.pt",
            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        vm = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(vm).to(device)
        vi = torch.where(vm_t)[0]
        unit_full = torch.zeros_like(feats)
        unit_full[vi] = F.normalize(feats[vi], dim=-1)
        stats = AccumulatedFeatureStats.load(
            f"artifacts/scannet/{scene}/train_stats_sam_{args.variant}_l3.pt")
        R = stats.reliability()["reliability"].to(device).float() * vm_t
        del stats
        torch.cuda.empty_cache()
        adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{args.variant}.pt",
                         map_location=device, weights_only=True)
        adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        positions = torch.from_numpy(centers).to(device).float()
        rad_t = torch.from_numpy(radii).to(device).float()

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
        owned = assigned >= 0
        P = centers.shape[0]
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())

        for alpha, beta in itertools.product(
                [float(x) for x in args.alphas.split(",")],
                [float(x) for x in args.betas.split(",")]):
            ref = unit_full
            for _ in range(args.passes):
                ref = radius_gated_refine(ref, R, rad_t, positions, adjacent, offsets,
                                          alpha=alpha, beta=beta)
            changed = float(((ref - unit_full).abs().sum(-1) > 1e-6).float().mean())
            unit = ref[vi]
            leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")
            Rv = R[vi]
            for cs in args.class_sets.split(","):
                kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                tids = [i for i, _ in kept]
                tnames = [n for _, n in kept]
                K = len(tids)
                gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
                text = embed_class_names(tnames, device)

                pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
                pooled.index_add_(0, leaf, unit)
                pooled = F.normalize(pooled, dim=-1)
                sim = pooled @ text.T
                cell_pool = (sim - args.lam * sim.mean(0, keepdim=True)).argmax(-1)[leaf]

                pcs = unit @ text.T
                pc_lab = (pcs - args.lam * pcs.mean(0, keepdim=True)).argmax(-1)
                hist = torch.zeros(K_FLAT, K, device=device)
                hist.index_put_((leaf, pc_lab), Rv, accumulate=True)
                cell_vote = hist.argmax(-1)[leaf]

                for tag, cc in [("pool", cell_pool), ("voteR", cell_vote)]:
                    pc = np.zeros(P, dtype=np.int64)
                    pc[vi.cpu().numpy()] = cc.cpu().numpy()
                    pred = np.zeros(len(gt_t), dtype=np.int64)
                    pred[owned] = pc[assigned[owned]] + 1
                    _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), K + 1)
                    key = f"{cs}|{tag}|a{alpha}_b{beta}"
                    results.setdefault(key, {})[scene] = (mi, ma)
                    print(f"  {scene} {cs} {tag} alpha={alpha} beta={beta} "
                          f"(changed {changed*100:.0f}%): {mi:.4f}/{ma:.4f}", flush=True)
            del ref, unit
            torch.cuda.empty_cache()

    print("\n=== averages ===")
    out = {}
    for key, per in sorted(results.items()):
        mi = float(np.mean([v[0] for v in per.values()]))
        ma = float(np.mean([v[1] for v in per.values()]))
        out[key] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per),
                    "per_scene": {s: list(v) for s, v in per.items()}}
        print(f"{key}: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
