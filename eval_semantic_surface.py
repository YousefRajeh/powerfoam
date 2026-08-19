"""SEMANTIC surface metrics: boundary quality of the 3D SEGMENTATION.

Why this exists, and how it differs from eval_surface_chamfer*.py
-----------------------------------------------------------------
`eval_surface_chamfer.py` measures RECONSTRUCTION geometry -- extract a mesh, compare it to
the ScanNet mesh. That answers a defensive question ("is the geometry any good?"), not this
project's thesis, which is SEGMENTATION quality.

This script measures the segmentation itself. mIoU reports what FRACTION of points agree
with the ground truth; it says nothing about how far off the disagreements are. An mIoU of
0.40 is consistent with two completely different failure modes:

  (a) every class roughly right, boundaries slopping a couple of cm into the neighbour --
      the errors sit within ~2cm of the correct region, or
  (b) whole objects confidently assigned to the wrong class -- errors are metres deep.

Those are not the same result and a segmentation paper should distinguish them. Per-class
Chamfer between the PREDICTED region and the GT region does exactly that, in cm.

No mesh, no TSDF, no reconstruction is involved: predictions and GT labels both live on the
SAME ScanNet point cloud, so distances are measured between two subsets of one point set.
That also makes the metric independent of every reconstruction hyperparameter (point
budget, densification) that confounded the geometry table.

Definitions (per class c, computed only over classes PRESENT in that scene's GT, matching
the mIoU averaging convention):
  GT_c   = points whose GT label is c
  PRED_c = points whose predicted label is c
  mae_pred2gt(c)  = mean over PRED_c of distance to nearest GT_c   -- "how far outside the
                    true region does the predicted region reach" (false-positive depth)
  mae_gt2pred(c)  = mean over GT_c of distance to nearest PRED_c   -- "how far from the
                    predicted region are the true points we missed" (false-negative depth)
  scd(c)          = (mae_pred2gt + mae_gt2pred) / 2                -- semantic Chamfer-L1
  hd95(c)         = max of the two 95th percentiles                -- worst-case, outlier-trimmed
  boundary_f1(c)  = F1 under the criterion "nearest same-class point within tau" (default 2cm)

NAMING follows the rule in surface_metrics.py: mean distances are MAE (cm), never
"accuracy"; precision/recall/F1 appear only with the criterion stated.

MISSED CLASSES ARE COUNTED, NOT SILENTLY DROPPED. If PRED_c is empty the class was never
predicted anywhere and its distances are undefined. Excluding those silently would reward a
method for predicting nothing, so they are excluded from the distance means AND reported as
`n_missed`. Read the distance means together with that count, always.

A LOW semantic Chamfer with a LOW mIoU is the interesting outcome: it means the errors are
boundary slop rather than semantic confusion.
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
from scipy.spatial import cKDTree

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from run_normlift_refine_eval import mode_vote_refine

# per-class-set lambda for partial centering, as validated over 10 scenes (see
# Experiment-F-scannet: 0.5 for 19/15cls, 0.4 for 10cls; the surface is flat so this
# choice is not delicate)
LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def semantic_surface_metrics(points, gt, pred, n_classes, tau=0.02):
    """Per-class Chamfer between predicted and GT regions of the SAME point cloud.

    `gt` and `pred` are 1-based class ids with 0 = ignore/unlabelled, matching
    calculate_metrics' convention. Classes absent from the GT are skipped entirely (the
    mIoU protocol averages over present classes only).
    """
    per_class = {}
    for c in range(1, n_classes):
        gm, pm = gt == c, pred == c
        n_gt, n_pred = int(gm.sum()), int(pm.sum())
        if n_gt == 0:
            continue                      # class not present in this scene -- not our business
        if n_pred == 0:
            # predicted nowhere: distances undefined. Recorded, not silently dropped.
            per_class[c] = {"n_gt": n_gt, "n_pred": 0, "missed": True}
            continue
        gpts, ppts = points[gm], points[pm]
        d_p2g, _ = cKDTree(gpts).query(ppts, k=1)      # predicted -> true region
        d_g2p, _ = cKDTree(ppts).query(gpts, k=1)      # true -> predicted region
        prec = float((d_p2g <= tau).mean())
        rec = float((d_g2p <= tau).mean())
        per_class[c] = {
            "n_gt": n_gt, "n_pred": n_pred, "missed": False,
            "mae_pred2gt": float(d_p2g.mean()),
            "mae_gt2pred": float(d_g2p.mean()),
            "scd": float((d_p2g.mean() + d_g2p.mean()) / 2),
            "median_pred2gt": float(np.median(d_p2g)),
            "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
            "boundary_precision": prec, "boundary_recall": rec,
            "boundary_f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        }
    live = [m for m in per_class.values() if not m["missed"]]
    n_missed = sum(1 for m in per_class.values() if m["missed"])
    if not live:
        return {"n_classes_present": len(per_class), "n_missed": n_missed, "per_class": per_class}
    agg = {k: float(np.mean([m[k] for m in live])) for k in
           ("mae_pred2gt", "mae_gt2pred", "scd", "median_pred2gt", "hd95",
            "boundary_precision", "boundary_recall", "boundary_f1")}
    agg.update({"n_classes_present": len(per_class), "n_missed": n_missed,
                "n_scored": len(live), "tau": tau, "per_class": per_class})
    return agg


def predict_labels(scene, variant, gt_root, device, class_sets, uniform_R=False,
                   protocol="champion"):
    """Return per-class-set labels under one of two protocols.

    protocol="champion" -- the validated raw-only stack (no templates): L3 features ->
        3-pass adjacency mode-vote refinement -> position-aware two-level clustering with
        FPS leaf init -> R-weighted label voting on partially-centered per-cell
        similarities. This is what the headline mIoU table reports.

    protocol="opengaussian" -- the PROTOCOL-MATCHED baseline: OpenGaussian's own recipe,
        i.e. the two-level 64x5=320 codebook (random leaf init, seed 0), pooled cluster
        features, raw class names, plain argmax. No refinement, no partial centering, no
        reliability weighting. This exists so foam and the Gaussian baseline can be
        compared under ONE pipeline instead of foam's full stack against the baseline's
        simple one -- the asymmetry that otherwise makes the comparison unreadable as a
        statement about representations. It needs no reliability vector at all, so it runs
        on scenes whose accumulator stats are gone.
    """
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats)
    unit_full[vi] = F.normalize(feats[vi], dim=-1)
    # Reliability weights. The per-scene accumulator stats are ~1.9GB each and were deleted
    # after use under the disk-pressure policy, and regenerating them means re-running
    # SAM+CLIP extraction. R is only ever used as a WEIGHT (in mode_vote_refine and in the
    # vote histogram), so a uniform fallback is a valid -- but DIFFERENT -- configuration.
    # It is never silently substituted: the caller must pass --uniform-reliability, the
    # choice is recorded in the output JSON, and its cost is measured on scene0000_00 where
    # both are available.
    stats_path = f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt"
    if protocol == "opengaussian":
        uniform_R = True          # unused by this path; kept uniform so nothing is loaded
    if uniform_R:
        R = vm_t.float()
        used_uniform = True
    else:
        stats = AccumulatedFeatureStats.load(stats_path)
        R = stats.reliability()["reliability"].to(device).float() * vm_t
        del stats
        used_uniform = False
    torch.cuda.empty_cache()
    positions_full = torch.from_numpy(centers).to(device).float()

    adjacent = offsets = None
    if protocol == "champion":
        adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt",
                         map_location=device, weights_only=True)
        adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        ref = unit_full
        for _ in range(3):
            ref = mode_vote_refine(ref, R, positions_full, adjacent, offsets)
        unit = ref[vi]
        leaf = two_level_position_aware(positions_full[vi], unit, seed=0, leaf_init="fps")
    else:
        # OpenGaussian's recipe: no refinement, random (not FPS) leaf init, seed 0 --
        # the same seed the point-mIoU eval uses, so the clustering is identical.
        ref = unit_full
        unit = unit_full[vi]
        leaf = two_level_position_aware(positions_full[vi], unit, seed=0, leaf_init="randperm")

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    Rv = R[vi]
    out = {}
    for cs in class_sets:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
        text = embed_class_names(tnames, device)
        if protocol == "champion":
            percell = unit @ text.T
            lam = LAMBDA[cs]
            lab = (percell - lam * percell.mean(0, keepdim=True)).argmax(-1)
            hist = torch.zeros(K_FLAT, len(tids), device=device)
            hist.index_put_((leaf, lab), Rv, accumulate=True)
            vcls = hist.argmax(-1)
        else:
            # pool cluster features, then plain argmax against raw class names
            pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
            pooled.index_add_(0, leaf, unit)
            vcls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1)
        pc = np.zeros(centers.shape[0], dtype=np.int64)
        pc[vi.cpu().numpy()] = vcls[leaf].cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = pc[assigned[owned]] + 1
        out[cs] = (gt_points, gt_t.numpy(), pred, len(tids) + 1, tnames)
    del unit_full, feats, ref, R
    torch.cuda.empty_cache()
    out["_uniform_reliability"] = used_uniform
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--tau", type=float, default=0.02, help="boundary criterion (m)")
    p.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    p.add_argument("--uniform-reliability", action="store_true",
                   help="use R=1 on valid primitives instead of the accumulator's "
                        "reliability. Needed for the 9 scenes whose ~1.9GB stats were "
                        "deleted under the disk policy; cost measured on scene0000_00.")
    p.add_argument("--protocol", choices=["champion", "opengaussian"], default="champion",
                   help="champion = validated raw-only stack (headline numbers); "
                        "opengaussian = protocol-matched baseline (two-level 64x5 codebook, "
                        "pooled features, raw names, plain argmax) for comparison against "
                        "the Gaussian baseline under one pipeline.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    class_sets = args.class_sets.split(",")
    results = {}
    for scene in args.scenes.split(","):
        preds = predict_labels(scene, args.variant, args.gt_root, device, class_sets,
                               uniform_R=args.uniform_reliability,
                               protocol=args.protocol)
        uniform_used = preds.pop("_uniform_reliability", False)
        for cs, (pts, gt, pred, ncls, tnames) in preds.items():
            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                                 torch.from_numpy(pred).long(), ncls)
            m = semantic_surface_metrics(pts, gt, pred, ncls, tau=args.tau)
            m["mIoU"], m["mAcc"] = float(miou), float(macc)
            m["class_names"] = tnames
            results.setdefault(cs, {})[scene] = m
            print(f"  {scene} {cs}: mIoU={miou*100:.2f} | semantic CD={m['scd']*100:.2f}cm "
                  f"MAE p->g={m['mae_pred2gt']*100:.2f} g->p={m['mae_gt2pred']*100:.2f} "
                  f"HD95={m['hd95']*100:.2f} bF1@{args.tau*100:.0f}cm={m['boundary_f1']:.3f} "
                  f"missed={m['n_missed']}/{m['n_classes_present']}", flush=True)

    print("\n=== 10-scene averages (semantic surface quality) ===")
    summary = {}
    for cs, per in results.items():
        keys = ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "mIoU", "mAcc")
        agg = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}
        agg["n_scenes"] = len(per)
        agg["mean_missed_per_scene"] = float(np.mean([v["n_missed"] for v in per.values()]))
        agg["mean_classes_present"] = float(np.mean([v["n_classes_present"] for v in per.values()]))
        summary[cs] = agg
        print(f"{cs}: mIoU={agg['mIoU']*100:.2f} semanticCD={agg['scd']*100:.2f}cm "
              f"(p->g {agg['mae_pred2gt']*100:.2f} / g->p {agg['mae_gt2pred']*100:.2f}) "
              f"HD95={agg['hd95']*100:.2f}cm bF1={agg['boundary_f1']:.3f} "
              f"missed {agg['mean_missed_per_scene']:.1f}/{agg['mean_classes_present']:.1f} "
              f"(n={agg['n_scenes']})")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_scene": results, "tau": args.tau,
                   "variant": args.variant,
                   "uniform_reliability": bool(args.uniform_reliability),
                   "protocol": args.protocol}, f, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
