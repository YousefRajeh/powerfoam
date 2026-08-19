"""N1: reliability-weighted pooling and reliability-weighted label voting, vs the
unweighted-pooling baseline. Raw class names + plain cosine throughout (no templates).
Cells contribute proportionally to reliability instead of being gated (hard gating was
already measured negative)."""
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
from run_normlift_refine_eval import mode_vote_refine

p = argparse.ArgumentParser()
p.add_argument("--scenes", default="scene0000_00")
p.add_argument("--variant", default="nonfrozen")
p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
p.add_argument("--refine-passes", type=int, default=3)
p.add_argument("--output", default=None)
args = p.parse_args()

device = "cuda"
results = {}
for scene in args.scenes.split(","):
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{args.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{args.variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device); vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats); unit_full[vi] = F.normalize(feats[vi], dim=-1)
    stats = AccumulatedFeatureStats.load(f"artifacts/scannet/{scene}/train_stats_sam_{args.variant}_l3.pt")
    R = stats.reliability()["reliability"].to(device).float() * vm_t
    del stats; torch.cuda.empty_cache()
    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{args.variant}.pt", map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    positions_full = torch.from_numpy(centers).to(device).float()

    ref = unit_full
    for _ in range(args.refine_passes):
        ref = mode_vote_refine(ref, R, positions_full, adjacent, offsets)
    unit = ref[vi]; Rv = R[vi]
    leaf = two_level_position_aware(positions_full[vi], unit, seed=0, leaf_init="fps")
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())

    def score(cell_cls, tag, cs, tids, gt_t):
        pc = np.zeros(centers.shape[0], dtype=np.int64)
        pc[vi.cpu().numpy()] = cell_cls.cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64); pred[owned] = pc[assigned[owned]] + 1
        _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
        results.setdefault(f"{cs}|{tag}", {})[scene] = (mi, ma)
        print(f"  {scene} {cs} {tag}: {mi:.4f}/{ma:.4f}", flush=True)

    for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids = [i for i, _ in kept]; tnames = [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
        text = embed_class_names(tnames, device)  # RAW names
        K = len(tids)

        # baseline: unweighted mean pooling
        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, leaf, unit)
        pooled = F.normalize(pooled, dim=-1)
        score((pooled @ text.T).argmax(-1)[leaf], "pool_plain", cs, tids, gt_t)

        # N1a: reliability-weighted pooling (R and R^2)
        for pw, tag in [(Rv, "pool_R"), (Rv * Rv, "pool_R2")]:
            pw_pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
            pw_pooled.index_add_(0, leaf, unit * pw[:, None])
            pw_pooled = F.normalize(pw_pooled, dim=-1)
            score((pw_pooled @ text.T).argmax(-1)[leaf], tag, cs, tids, gt_t)

        # N1b: reliability-weighted majority vote in LABEL space
        percell = (unit @ text.T).argmax(-1)
        for vw, tag in [(torch.ones_like(Rv), "vote_plain"), (Rv, "vote_R"), (Rv * Rv, "vote_R2")]:
            hist = torch.zeros(K_FLAT, K, device=device)
            hist.index_put_((leaf, percell), vw, accumulate=True)
            score(hist.argmax(-1)[leaf], tag, cs, tids, gt_t)

    del unit_full, feats, ref, R, adjacent, offsets; torch.cuda.empty_cache()

print("\n=== averages ===")
out = {}
for key, per in sorted(results.items()):
    mi = float(np.mean([v[0] for v in per.values()])); ma = float(np.mean([v[1] for v in per.values()]))
    out[key] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per)}
    print(f"{key}: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
if args.output:
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
