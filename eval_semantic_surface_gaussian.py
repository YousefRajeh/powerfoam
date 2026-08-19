"""Semantic-surface metrics for the Splat Feature Solver (3DGS) baseline.

Same metric as eval_semantic_surface.py -- per-class Chamfer between the PREDICTED region
and the GT region of the same ScanNet point cloud -- so the two are directly comparable.
Nothing here is re-derived: the prediction path is imported from
evaluate_point_cloud_miou.py, the same code that produced the published mIoU table for this
baseline (nearest-Gaussian-center correspondence, opacity>=0.1 validity, raw class names,
per-primitive argmax).

Two protocols, so the comparison against foam can be made on equal terms:

  --protocol argmax        per-primitive argmax on the lifted features (this baseline's
                           established recipe, matching the published mIoU table).
  --protocol opengaussian  OpenGaussian's own recipe: two-level 64x5=320 codebook over the
                           Gaussians (Euclidean k-means on centers -> spherical k-means on
                           features within each root, random init, seed 0), pooled cluster
                           features, raw class names, plain argmax.

The second exists because comparing foam's full champion stack against a per-primitive
Gaussian baseline confounds representation with pipeline. Run
`eval_semantic_surface.py --protocol opengaussian` and this script with
`--protocol opengaussian` and the two sides differ ONLY in the representation and its
point<->primitive correspondence (exact power-cell membership vs nearest-center), which is
the comparison the paper actually wants to make.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch

import torch.nn.functional as F

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics, classify_primitives,
                                       load_gaussian_means_opacities)
from point_cloud_query import assign_points_to_nearest_center
from run_cluster_classify_eval import SCENES, two_level_position_aware, K_FLAT
from eval_semantic_surface import semantic_surface_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--ckpt-root", default="/home/rajehyl/gaussian_baseline_scannet",
                   help="expects {root}/{scene}/ckpts/ckpt_29999_rank0{,_features}.pt")
    p.add_argument("--ckpt", default=None, help="override checkpoint path (single scene)")
    p.add_argument("--features", default=None, help="override features path (single scene)")
    p.add_argument("--opacity-threshold", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    p.add_argument("--protocol", choices=["argmax", "opengaussian"], default="argmax",
                   help="argmax = this baseline's established per-primitive recipe; "
                        "opengaussian = two-level 64x5 codebook + pooled features, so the "
                        "comparison against foam differs only in the representation.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda"
    class_sets = args.class_sets.split(",")
    results = {}
    for scene in args.scenes.split(","):
        split = SCENES[scene]
        ckpt = args.ckpt or f"{args.ckpt_root}/{scene}/ckpts/ckpt_29999_rank0.pt"
        featp = args.features or f"{args.ckpt_root}/{scene}/ckpts/ckpt_29999_rank0_features.pt"
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{args.gt_root}/{split}/{scene}", "segment20")

        means, opacities = load_gaussian_means_opacities(ckpt, device)
        valid = opacities >= args.opacity_threshold
        feats = torch.load(featp, map_location=device, weights_only=False).float()
        assert feats.shape[0] == means.shape[0], (feats.shape, means.shape)
        assigned = assign_points_to_nearest_center(gt_points, means, valid=valid)
        owned = assigned >= 0
        print(f"[{scene}] {means.shape[0]} gaussians, {int(valid.sum())} valid, "
              f"{owned.mean()*100:.1f}% of GT points owned", flush=True)

        # OpenGaussian-protocol clustering is class-set-independent, so it is built once:
        # Euclidean k-means on Gaussian CENTERS for the 64 roots, spherical k-means on the
        # lifted features for the 5 leaves within each root -- the same function and seed
        # the foam side uses, with means standing in for power-cell centers.
        if args.protocol == "opengaussian":
            vi = np.where(valid)[0]
            unit = F.normalize(feats[torch.from_numpy(vi).to(device)], dim=-1)
            pos = torch.from_numpy(means[vi] if isinstance(means, np.ndarray)
                                   else means[vi].cpu().numpy()).to(device).float()
            leaf = two_level_position_aware(pos, unit, seed=0, leaf_init="randperm")
            print(f"  clustered {len(vi)} valid gaussians into {K_FLAT} leaves", flush=True)

        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        for cs in class_sets:
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            tids, tnames = [i for i, _ in kept], [n for _, n in kept]
            gt_t = remap_gt_labels(raw_labels, tids)
            text = embed_class_names(tnames, device)
            if args.protocol == "opengaussian":
                pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
                pooled.index_add_(0, leaf, unit)
                vcls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1)
                pcls = np.zeros(means.shape[0], dtype=np.int64)
                pcls[vi] = vcls[leaf].cpu().numpy()
            else:
                pcls = classify_primitives(feats, text).cpu().numpy()
            pred = np.zeros(gt_points.shape[0], dtype=np.int64)
            pred[owned] = pcls[assigned[owned]] + 1

            ncls = len(tids) + 1
            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_t).long(),
                                                 torch.from_numpy(pred).long(), ncls)
            m = semantic_surface_metrics(gt_points, gt_t, pred, ncls, tau=args.tau)
            m["mIoU"], m["mAcc"] = float(miou), float(macc)
            results.setdefault(cs, {})[scene] = m
            print(f"  {scene} {cs}: mIoU={miou*100:.2f} | semantic CD={m['scd']*100:.2f}cm "
                  f"MAE p->g={m['mae_pred2gt']*100:.2f} g->p={m['mae_gt2pred']*100:.2f} "
                  f"HD95={m['hd95']*100:.2f} bF1@{args.tau*100:.0f}cm={m['boundary_f1']:.3f} "
                  f"missed={m['n_missed']}/{m['n_classes_present']}", flush=True)

    print("\n=== averages (GAUSSIAN / splat feature solver) ===")
    summary = {}
    keys = ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "mIoU", "mAcc")
    for cs, per in results.items():
        agg = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}
        agg["n_scenes"] = len(per)
        agg["mean_missed_per_scene"] = float(np.mean([v["n_missed"] for v in per.values()]))
        summary[cs] = agg
        print(f"{cs}: mIoU={agg['mIoU']*100:.2f} semanticCD={agg['scd']*100:.2f}cm "
              f"(p->g {agg['mae_pred2gt']*100:.2f} / g->p {agg['mae_gt2pred']*100:.2f}) "
              f"HD95={agg['hd95']*100:.2f}cm bF1={agg['boundary_f1']:.3f} "
              f"missed {agg['mean_missed_per_scene']:.1f} (n={agg['n_scenes']})")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_scene": results, "tau": args.tau,
                   "method": "splat_feature_solver", "protocol": args.protocol,
                   "opacity_threshold": args.opacity_threshold}, f, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
