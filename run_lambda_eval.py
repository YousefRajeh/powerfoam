"""10-scene validation of partial centering (sim - lambda*columnmean) on the raw-only
stack: 3-pass adjacency mode-voting refinement -> kmeansFPS 64x5 -> pooled raw-name
cosine, lambda in {0, 0.25, 0.5}. No templates anywhere."""
import argparse, json, sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src"); sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src"); sys.path.insert(0, "/home/rajehyl/powerfoam")
import numpy as np, torch, torch.nn.functional as F
from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt, calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
import json
from run_normlift_refine_eval import mode_vote_refine

p = argparse.ArgumentParser()
p.add_argument("--scenes", required=True)
p.add_argument("--variant", default="nonfrozen")
p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
p.add_argument("--output", required=True)
args = p.parse_args()

device = "cuda"
results = {}
for scene in args.scenes.split(","):
    split = SCENES[scene]
    variant = args.variant
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{args.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt", map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device); vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats); unit_full[vi] = F.normalize(feats[vi], dim=-1)
    stats = AccumulatedFeatureStats.load(f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt")
    R = stats.reliability()["reliability"].to(device).float() * vm_t
    del stats; torch.cuda.empty_cache()
    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt", map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    positions_full = torch.from_numpy(centers).to(device).float()
    ref = unit_full
    for _ in range(3):
        ref = mode_vote_refine(ref, R, positions_full, adjacent, offsets)
    unit = ref[vi]
    leaf = two_level_position_aware(positions_full[vi], unit, seed=0, leaf_init="fps")
    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device); pooled.index_add_(0, leaf, unit)
    pooled = F.normalize(pooled, dim=-1)
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids = [i for i, _ in kept]; tnames = [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
        text = embed_class_names(tnames, device)
        sim0 = pooled @ text.T
        percell_sim = unit @ text.T
        Rv = R[vi]
        K = len(tids)
        for lam in [0.4, 0.5, 0.6, 0.7]:
            # A) pooled features, centered similarities (validated path)
            ccls = (sim0 - lam * sim0.mean(0, keepdim=True)).argmax(-1)
            pc = np.zeros(centers.shape[0], dtype=np.int64); pc[vi.cpu().numpy()] = ccls[leaf].cpu().numpy()
            pred = np.zeros(len(gt_t), dtype=np.int64); pred[owned] = pc[assigned[owned]] + 1
            _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
            results.setdefault(f"{cs}|pool_lam{lam}", {})[scene] = (mi, ma)
            print(f"  {scene} {cs} pool_lam={lam}: {mi:.4f}/{ma:.4f}", flush=True)
            # B) N1 label-space voting on centered PER-CELL similarities, R-weighted
            pc_lab = (percell_sim - lam * percell_sim.mean(0, keepdim=True)).argmax(-1)
            hist = torch.zeros(K_FLAT, K, device=device)
            hist.index_put_((leaf, pc_lab), Rv, accumulate=True)
            vcls = hist.argmax(-1)
            pc = np.zeros(centers.shape[0], dtype=np.int64); pc[vi.cpu().numpy()] = vcls[leaf].cpu().numpy()
            pred = np.zeros(len(gt_t), dtype=np.int64); pred[owned] = pc[assigned[owned]] + 1
            _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
            results.setdefault(f"{cs}|voteR_lam{lam}", {})[scene] = (mi, ma)
            print(f"  {scene} {cs} voteR_lam={lam}: {mi:.4f}/{ma:.4f}", flush=True)
    del unit_full, feats, ref, R, adjacent, offsets; torch.cuda.empty_cache()

with open(args.output, "w") as f:
    json.dump({k: {s: v for s, v in per.items()} for k, per in results.items()}, f, indent=2)
print("wrote", args.output)
