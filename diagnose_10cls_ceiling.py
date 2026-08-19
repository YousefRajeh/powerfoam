"""The 10cls lifting-side question, two measurements.

T1. ORACLE CEILING. Our cells own many GT points each (78-91% of points share a cell,
    max 270). Any per-cell method is therefore capped: give every cell its MAJORITY GT
    label and score -- that is the best any labeling of this tessellation can do.
    NormLift/OpenGaussian run frozen 1:1 (one Gaussian per GT point), so their ceiling is
    100% by construction. If our 10cls ceiling is materially lower than our 19cls ceiling,
    the coarse-class gap is a RESOLUTION limit of the representation, not a feature bug.

T2. SOFT POWER QUERY. Hard membership (argmin power distance) gives every point in a cell
    the same label. Foam offers a natural softening: a point's owner cell plus the owner's
    FACET NEIGHBOURS, weighted by softmax(-power_distance / T). Points near a cell boundary
    then inherit from the neighbour they sit closest to, recovering sub-cell resolution
    without changing the reconstruction. Tested in feature space (blend then classify) and
    label space (weighted vote).
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
    remap_gt_labels, load_scannet_pointcept_gt, calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from run_normlift_refine_eval import mode_vote_refine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--scenes", default=None, help="comma list; overrides --scene")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    variant = args.variant
    results = {}
    for scene in (args.scenes.split(",") if args.scenes else [args.scene]):
      split = SCENES[scene]
      if True:
      gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
          f"{args.gt_root}/{split}/{scene}", "segment20")
      centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
      solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                          map_location=device, weights_only=True)
      feats = solved["primitive_features"].to(device).float()
      vm = solved["valid_mask"].cpu().numpy()
      vm_t = torch.from_numpy(vm).to(device)
      vi = torch.where(vm_t)[0]
      unit_full = torch.zeros_like(feats)
      unit_full[vi] = F.normalize(feats[vi], dim=-1)

      stats = AccumulatedFeatureStats.load(
          f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt")
      R = stats.reliability()["reliability"].to(device).float() * vm_t
      del stats
      torch.cuda.empty_cache()
      adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt",
                       map_location=device, weights_only=True)
      adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
      positions = torch.from_numpy(centers).to(device).float()
      radii_t = torch.from_numpy(radii).to(device).float()

      ref = unit_full
      for _ in range(3):
          ref = mode_vote_refine(ref, R, positions, adjacent, offsets)
      unit = ref[vi]
      leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")

      assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
      owned = assigned >= 0
      P = centers.shape[0]
      n2i = {n: i for i, n in enumerate(all_names)}
      present = set(np.unique(raw_labels).tolist())

      pts_t = torch.from_numpy(gt_points).to(device).float()
      own_idx = torch.from_numpy(np.where(owned)[0]).to(device)
      own_cell = torch.from_numpy(assigned[owned]).to(device).long()

      # padded candidate table: owner + owner's facet neighbours
      deg = (offsets[1:] - offsets[:-1])
      D = int(deg.max()) + 1
      N = own_cell.numel()
      cand = torch.full((N, D), -1, dtype=torch.long, device=device)
      cand[:, 0] = own_cell
      dg = deg[own_cell]
      mask = torch.arange(D - 1, device=device)[None, :] < dg[:, None]
      flat = torch.repeat_interleave(offsets[own_cell], dg) + (
          torch.arange(int(dg.sum()), device=device)
          - torch.repeat_interleave(torch.cumsum(dg, 0) - dg, dg))
      cand[:, 1:][mask] = adjacent[flat]
      valid_c = (cand >= 0) & vm_t[cand.clamp_min(0)]
      cidx = cand.clamp_min(0)
      # power distance ||x - c||^2 - r^2 (the same quantity the ray-traversal kernels use)
      pw = ((pts_t[own_idx][:, None, :] - positions[cidx]) ** 2).sum(-1) - radii_t[cidx] ** 2
      pw = torch.where(valid_c, pw, torch.full_like(pw, float("inf")))

      print(f"=== {scene} ({variant}) ===")
      for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
          kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
          tids = [i for i, _ in kept]
          tnames = [n for _, n in kept]
          K = len(tids)
          gt = remap_gt_labels(raw_labels, tids)
          gt_t = torch.from_numpy(gt).long()
          text = embed_class_names(tnames, device)

          # ---- T1: oracle ceiling ----
          sel = gt[owned] > 0
          cop = assigned[owned][sel]
          lop = gt[owned][sel]
          vote = np.zeros((P, K + 1), dtype=np.int64)
          np.add.at(vote, (cop, lop), 1)
          maj = vote.argmax(1)
          pred_or = np.zeros(len(gt), dtype=np.int64)
          pred_or[owned] = maj[assigned[owned]]
          _, mi_or, _, ma_or = calculate_metrics(gt_t, torch.from_numpy(pred_or).long(), K + 1)
          purity = vote.max(1)[maj > 0].sum() / max(vote.sum(1)[maj > 0].sum(), 1)
          print(f"  {cs} ORACLE ceiling (cell majority label): {mi_or:.4f}/{ma_or:.4f}  "
                f"label purity of owner cells: {purity*100:.1f}%")

          # ---- baseline: hard membership, pooled+centered, raw names ----
          pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
          pooled.index_add_(0, leaf, unit)
          pooled = F.normalize(pooled, dim=-1)
          sim = pooled @ text.T
          cell_cls = (sim - args.lam * sim.mean(0, keepdim=True)).argmax(-1)
          pc = np.zeros(P, dtype=np.int64)
          pc[vi.cpu().numpy()] = cell_cls[leaf].cpu().numpy()
          pred = np.zeros(len(gt), dtype=np.int64)
          pred[owned] = pc[assigned[owned]] + 1
          _, mi_b, _, ma_b = calculate_metrics(gt_t, torch.from_numpy(pred).long(), K + 1)
          print(f"  {cs} hard membership (current): {mi_b:.4f}/{ma_b:.4f}", flush=True)

          # ---- T2: soft power query ----
          cell_lab_t = torch.from_numpy(pc).to(device)          # per-cell class 0..K-1
          cell_feat = torch.zeros_like(unit_full)
          cell_feat[vi] = F.normalize(pooled[leaf], dim=-1)     # each cell's region feature
          for T in [0.001, 0.005, 0.02]:
              # chunked over points: the (N, D, C) blend is far too large to materialize
              simf = torch.empty(N, K, device=device)
              hist = torch.zeros(N, K, device=device)
              CH = 20000
              for s0 in range(0, N, CH):
                  e0 = min(s0 + CH, N)
                  wc = torch.softmax(-pw[s0:e0] / T, dim=1)
                  blend = F.normalize((cell_feat[cidx[s0:e0]] * wc[..., None]).sum(1), dim=-1)
                  simf[s0:e0] = blend @ text.T
                  hist[s0:e0].scatter_add_(1, cell_lab_t[cidx[s0:e0]], wc * valid_c[s0:e0].float())
              cls_f = (simf - args.lam * simf.mean(0, keepdim=True)).argmax(-1) + 1
              pf = np.zeros(len(gt), dtype=np.int64)
              pf[owned] = cls_f.cpu().numpy()
              _, mi_f, _, ma_f = calculate_metrics(gt_t, torch.from_numpy(pf).long(), K + 1)
              cls_l = hist.argmax(1) + 1
              pl = np.zeros(len(gt), dtype=np.int64)
              pl[owned] = cls_l.cpu().numpy()
              _, mi_l, _, ma_l = calculate_metrics(gt_t, torch.from_numpy(pl).long(), K + 1)
              print(f"    soft power query T={T}: feat={mi_f:.4f}/{ma_f:.4f}  vote={mi_l:.4f}/{ma_l:.4f}", flush=True)
              results.setdefault(f"{cs}|softfeat_T{T}", {})[scene] = (mi_f, ma_f)
              results.setdefault(f"{cs}|softvote_T{T}", {})[scene] = (mi_l, ma_l)
              del simf, hist


    if args.output:
        with open(args.output, "w") as f:
            json.dump({k: v for k, v in results.items()}, f, indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
