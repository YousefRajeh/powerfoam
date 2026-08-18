"""Task #20: combination grid on L3 features. Axes: refinement (0/3 passes, applied to
per-primitive features BEFORE pooling) x clustering (kmeans-FPS-leaves / grower@0.95 /
flat320-FPS) x text (raw/templates) x calibration (none/center). Pilot on one scene via
--scenes, 10-scene via --scenes all."""
import argparse, json, sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")
import numpy as np, torch, torch.nn.functional as F
from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt, calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES, CLASS_SETS
from run_text_side_eval import embed_prompts, classify
from run_region_grow_eval import batched_region_grow
from run_normlift_refine_eval import mode_vote_refine

p = argparse.ArgumentParser()
p.add_argument("--scenes", default="scene0000_00")
p.add_argument("--variant", default="nonfrozen")
p.add_argument("--scene0000-variant", default="nonfrozen")
p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
p.add_argument("--class-sets", default="opengaussian19")
p.add_argument("--output", default=None)
args = p.parse_args()

device = "cuda"
scenes = list(SCENES.keys()) if args.scenes == "all" else args.scenes.split(",")
class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
results = {}
for scene in scenes:
    split = SCENES[scene]
    variant = args.scene0000_variant if scene == "scene0000_00" else args.variant
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{args.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats); unit_full[vi] = F.normalize(feats[vi], dim=-1)
    positions_full = torch.from_numpy(centers).to(device).float()

    stats = AccumulatedFeatureStats.load(f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt")
    R = stats.reliability()["reliability"].to(device).float() * vm_t
    del stats
    torch.cuda.empty_cache()
    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt", map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

    feature_sets = {"norefine": unit_full}
    ref = unit_full
    for _ in range(3):
        ref = mode_vote_refine(ref, R, positions_full, adjacent, offsets)
    feature_sets["refine3"] = ref

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())

    for fs_tag, uf in feature_sets.items():
        unit = uf[vi]
        pos = positions_full[vi]
        clusterings = {
            "kmeansFPS": (two_level_position_aware(pos, unit, seed=0, leaf_init="fps"), K_FLAT, True),
        }
        gl, gn = batched_region_grow(adjacent, offsets, uf, vm_t, 0.95)
        clusterings["grower95"] = (gl[vi], gn, True)
        from diagnose_scannet_miou import spherical_kmeans
        fl, _ = spherical_kmeans(unit, K_FLAT, seed=0, init="fps")
        clusterings["flatFPS"] = (fl, K_FLAT, True)

        for cl_tag, (leaf, K, _) in clusterings.items():
            pooled = torch.zeros(K, unit.shape[1], device=device)
            pooled.index_add_(0, leaf, unit)
            pooled = F.normalize(pooled, dim=-1)
            for cs in class_sets:
                kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                tids = [i for i, _ in kept]; tnames = [n for _, n in kept]
                gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
                for txt_tag, text in [("raw", embed_class_names(tnames, device)),
                                      ("tpl", embed_prompts(tnames, "templates", device))]:
                    for cal in ["none", "center"]:
                        ccls = classify(pooled, text, cal)
                        pc = np.zeros(centers.shape[0], dtype=np.int64)
                        pc[vi.cpu().numpy()] = ccls[leaf].cpu().numpy()
                        pred = np.zeros(len(gt_t), dtype=np.int64)
                        pred[owned] = pc[assigned[owned]] + 1
                        _, mi, _, ma = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
                        key = f"{fs_tag}|{cl_tag}|{txt_tag}|{cal}|{cs}"
                        results.setdefault(key, {})[scene] = (mi, ma)
                        print(f"  {scene} {key}: {mi:.4f}/{ma:.4f}", flush=True)

print("\n=== averages ===")
out = {}
for key, per in sorted(results.items()):
    mi = float(np.mean([v[0] for v in per.values()]))
    ma = float(np.mean([v[1] for v in per.values()]))
    out[key] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per)}
    print(f"{key}: {mi*100:.2f}/{ma*100:.2f} (n={len(per)})")
if args.output:
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
