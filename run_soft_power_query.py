"""Soft power query, multi-scene runner.

Hard membership (argmin power distance) gives every GT point in a cell the same label.
Foam offers a natural softening that costs no re-solve: a point's owner cell plus the
owner's FACET NEIGHBOURS, weighted by softmax(-power_distance / T). Points near a cell
boundary then inherit from the neighbour they actually sit closest to, recovering
sub-cell resolution.

Pilot (scene0000_00) found this flat at 19/15cls but a consistent win at 10cls
(+0.68 mIoU / +1.1 mAcc, monotone in T) -- coarse classes are large smooth surfaces
where interpolating across a facet is informative. This runner confirms on all 10 scenes
with a wider T sweep. Classification is the validated raw-only stack: 3-pass reliability
mode-voting refinement -> FPS-seeded 64x5 clustering -> pooled features -> raw class
names + partial centering. No templates.

Also reports the ORACLE ceiling (cell-majority labeling) per scene as context.
"""
import argparse
import json
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
from run_normlift_refine_eval import mode_vote_refine

TEMPS = [0.005, 0.02, 0.05, 0.1, 0.3]


def run_scene(scene, variant, gt_root, lam, results, device="cuda"):
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{gt_root}/{split}/{scene}", "segment20")
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

    deg = offsets[1:] - offsets[:-1]
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
    pw = ((pts_t[own_idx][:, None, :] - positions[cidx]) ** 2).sum(-1) - radii_t[cidx] ** 2
    pw = torch.where(valid_c, pw, torch.full_like(pw, float("inf")))

    print(f"=== {scene} ({variant}) N={N} pts, D={D} ===", flush=True)
    for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids = [i for i, _ in kept]
        tnames = [n for _, n in kept]
        K = len(tids)
        gt = remap_gt_labels(raw_labels, tids)
        gt_t = torch.from_numpy(gt).long()
        text = embed_class_names(tnames, device)

        # oracle ceiling
        sel = gt[owned] > 0
        vote = np.zeros((P, K + 1), dtype=np.int64)
        np.add.at(vote, (assigned[owned][sel], gt[owned][sel]), 1)
        maj = vote.argmax(1)
        pred_or = np.zeros(len(gt), dtype=np.int64)
        pred_or[owned] = maj[assigned[owned]]
        _, mi_or, _, ma_or = calculate_metrics(gt_t, torch.from_numpy(pred_or).long(), K + 1)
        results.setdefault(f"{cs}|oracle", {})[scene] = (mi_or, ma_or)

        # hard membership baseline (validated raw-only stack)
        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, leaf, unit)
        pooled = F.normalize(pooled, dim=-1)
        sim = pooled @ text.T
        cell_cls = (sim - lam * sim.mean(0, keepdim=True)).argmax(-1)
        pc = np.zeros(P, dtype=np.int64)
        pc[vi.cpu().numpy()] = cell_cls[leaf].cpu().numpy()
        pred = np.zeros(len(gt), dtype=np.int64)
        pred[owned] = pc[assigned[owned]] + 1
        _, mi_b, _, ma_b = calculate_metrics(gt_t, torch.from_numpy(pred).long(), K + 1)
        results.setdefault(f"{cs}|hard", {})[scene] = (mi_b, ma_b)
        print(f"  {cs} oracle={mi_or:.4f} hard={mi_b:.4f}/{ma_b:.4f}", flush=True)

        cell_lab_t = torch.from_numpy(pc).to(device)
        cell_feat = torch.zeros_like(unit_full)
        cell_feat[vi] = F.normalize(pooled[leaf], dim=-1)
        for T in TEMPS:
            simf = torch.empty(N, K, device=device)
            hist = torch.zeros(N, K, device=device)
            CH = 20000
            for s0 in range(0, N, CH):
                e0 = min(s0 + CH, N)
                wc = torch.softmax(-pw[s0:e0] / T, dim=1)
                blend = F.normalize((cell_feat[cidx[s0:e0]] * wc[..., None]).sum(1), dim=-1)
                simf[s0:e0] = blend @ text.T
                hist[s0:e0].scatter_add_(1, cell_lab_t[cidx[s0:e0]], wc * valid_c[s0:e0].float())
            cls_f = (simf - lam * simf.mean(0, keepdim=True)).argmax(-1) + 1
            pf = np.zeros(len(gt), dtype=np.int64)
            pf[owned] = cls_f.cpu().numpy()
            _, mi_f, _, ma_f = calculate_metrics(gt_t, torch.from_numpy(pf).long(), K + 1)
            cls_l = hist.argmax(1) + 1
            pl = np.zeros(len(gt), dtype=np.int64)
            pl[owned] = cls_l.cpu().numpy()
            _, mi_l, _, ma_l = calculate_metrics(gt_t, torch.from_numpy(pl).long(), K + 1)
            results.setdefault(f"{cs}|softfeat_T{T}", {})[scene] = (mi_f, ma_f)
            results.setdefault(f"{cs}|softvote_T{T}", {})[scene] = (mi_l, ma_l)
            print(f"    T={T}: feat={mi_f:.4f}/{ma_f:.4f} vote={mi_l:.4f}/{ma_l:.4f}", flush=True)
            del simf, hist

    del unit_full, feats, ref, R, adjacent, offsets, pw, cand, cidx, valid_c
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    results = {}
    for scene in args.scenes.split(","):
        run_scene(scene, args.variant, args.gt_root, args.lam, results)

    print("\n=== averages ===")
    out = {}
    for key, per in sorted(results.items()):
        mi = float(np.mean([v[0] for v in per.values()]))
        ma = float(np.mean([v[1] for v in per.values()]))
        out[key] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per),
                    "per_scene": {s: list(v) for s, v in per.items()}}
        print(f"{key}: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
