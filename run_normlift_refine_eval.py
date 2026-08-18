"""NormLift's reliability-guided KNN mode-voting refinement (their Eqs. 9-10),
implemented on PowerFoam with two substitutions that are ours by construction:
neighbors come from the EXACT power-diagram adjacency (facet-sharing cells) instead of
Euclidean KNN, and per-primitive features/reliability come from our geometric-median
solve + the already-ported NormLift Eq. 6-8 reliability in AccumulatedFeatureStats.

Refinement rule (verbatim from their paper):
  S_ij = R(j) * sum_k R(k) * d_ik * g_jk   over candidates j,k in N+(i) = neighbors + self
    d_ik = exp(-||mu_i - mu_k||^2 / (2 sigma_d^2))      spatial decay
    g_jk = sigmoid((<u_k, u_j> - tau) / gamma)          soft semantic agreement
  replace u_i <- u_j* iff S_ij* > S_ii + Delta (j* = best neighbor excluding i)
Copying an existing feature keeps everything on the CLIP manifold -- no averaging.
Their sensitivity analysis: all six hyperparameters vary mIoU < 2 points across wide
ranges (Appendix C), so literature-sane defaults are used here; sigma_d defaults to the
scene's median neighbor distance.

Evaluation follows THEIR protocol exactly: every primitive queried INDEPENDENTLY
(no clustering), raw class names (byte-identical to OpenGaussian's shipped embeddings),
plain cosine argmax.
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
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import SCENES, CLASS_SETS


def mode_vote_refine(unit, R, positions, adjacent, offsets, sigma_d=None,
                     tau=0.8, gamma=0.05, delta=0.1, chunk=8192):
    """One conservative mode-voting pass. unit: (P,C) unit features (invalid rows zero);
    R: (P,) reliability; returns refined unit features."""
    device = unit.device
    P = unit.shape[0]
    deg = (offsets[1:] - offsets[:-1])
    D = int(deg.max()) + 1  # +1 for self

    if sigma_d is None:
        src = torch.repeat_interleave(torch.arange(P, device=device), deg)
        nd = (positions[src] - positions[adjacent]).norm(dim=-1)
        sigma_d = float(nd.median())

    refined = unit.clone()
    arangeD = torch.arange(D, device=device)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        B = e - s
        rows = torch.arange(s, e, device=device)
        # padded candidate table: slot 0 = self, slots 1.. = graph neighbors
        cand = torch.full((B, D), -1, dtype=torch.long, device=device)
        cand[:, 0] = rows
        dg = deg[s:e]
        mask = arangeD[None, 1:] <= dg[:, None]  # (B, D-1)
        flat = torch.repeat_interleave(offsets[s:e], dg) + (
            torch.arange(int(dg.sum()), device=device)
            - torch.repeat_interleave(torch.cumsum(dg, 0) - dg, dg))
        cand[:, 1:][mask] = adjacent[flat]
        valid = cand >= 0
        cidx = cand.clamp_min(0)

        U = unit[cidx]                       # (B, D, C)
        Rj = R[cidx] * valid                 # (B, D)
        d_ik = torch.exp(-((positions[cidx] - positions[rows][:, None, :]) ** 2).sum(-1)
                         / (2 * sigma_d ** 2)) * valid
        a = Rj * d_ik                        # (B, D) voter weights
        g = torch.sigmoid(((U @ U.transpose(1, 2)) - tau) / gamma)  # (B, D, D)
        S = Rj * (g @ a.unsqueeze(-1)).squeeze(-1)                  # (B, D)
        S[~valid] = float("-inf")

        S_self = S[:, 0]
        S_neigh = S.clone()
        S_neigh[:, 0] = float("-inf")
        best_val, best_j = S_neigh.max(dim=1)
        take = best_val > S_self + delta
        refined[rows[take]] = U[take, best_j[take]]
    return refined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen", help="nonfrozen | truefrozen | frozen_v2")
    p.add_argument("--class-sets", default="all")
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
    results = {}
    for scene in args.scenes.split(","):
        split = SCENES[scene]
        ckpt_dir = f"output/scannet_{scene}_{args.variant}"
        stats_path = f"artifacts/scannet/{scene}/train_stats_sam_{args.variant}.pt"
        adjacency_path = f"artifacts/scannet/{scene}/adjacency_{args.variant}.pt"
        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"

        stats = AccumulatedFeatureStats.load(stats_path)
        solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(valid_mask).to(device)
        unit = torch.zeros_like(feats)
        unit[vm_t] = F.normalize(feats[vm_t], dim=-1)
        R = stats.reliability()["reliability"].to(device).float()
        R = R * vm_t  # invalid primitives cast no votes and get no support

        centers, radii = load_foam(ckpt_dir, device)
        positions = torch.from_numpy(centers).to(device).float()
        import os
        if not os.path.exists(adjacency_path):
            import subprocess
            subprocess.run([sys.executable, "export_adjacency_graph.py", "-c",
                            f"{ckpt_dir}/config.yaml", "--output", adjacency_path], check=True)
        adj = torch.load(adjacency_path, map_location=device, weights_only=True)
        adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

        refined = unit
        for _ in range(args.passes):
            refined = mode_vote_refine(refined, R, positions, adjacent, offsets,
                                       tau=args.tau, gamma=args.gamma, delta=args.delta)
        changed = float(((refined - unit).abs().sum(-1) > 1e-6).float().mean())
        print(f"  [{scene} {args.variant}] refinement changed {changed*100:.1f}% of primitives", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        for cs in class_sets:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            tnames = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(tnames, device)  # RAW names, plain argmax below
            for tag, u in (("base", unit), ("refined", refined)):
                cls = (u @ text.T).argmax(-1).cpu().numpy()
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
                results.setdefault((cs, tag), {})[scene] = (miou, macc)
                print(f"  {scene} {cs} per-primitive raw argmax [{tag}]: "
                      f"mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)

    print("\n=== averages ===")
    out = {}
    for (cs, tag), per in sorted(results.items()):
        mi = float(np.mean([v[0] for v in per.values()]))
        ma = float(np.mean([v[1] for v in per.values()]))
        out[f"{cs}|{tag}"] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per)}
        print(f"{cs} [{tag}]: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
