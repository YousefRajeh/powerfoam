"""Where does rank truncation actually cost accuracy -- in the CLUSTERING or in the POOLED
FEATURES?

CONTEXT. run_rank_compress.py measured rank-32 truncation as slightly BETTER than the full 512-d
field under the bare per-cell argmax (+0.31 mIoU, 10 scenes). Re-scored through the headline
cluster-then-classify pipeline it went the other way on the pilot scene (32.79 vs 37.95 on
scene0000_00, feat_kmeans320/19cls). Both cannot be the whole story, and the difference between the
two protocols is exactly one thing: spherical k-means runs between the features and the decision.

THE 2x2. Cluster on one field, pool-and-classify on the other:

              pool/classify: full        pool/classify: R=32
  cluster full     full/full (baseline)      full/r32
  cluster R=32     r32/full                  r32/r32   (what the headline rerun measured)

  * if r32/full ~ full/full and full/r32 ~ r32/r32, the loss is in the POOLED FEATURES;
  * if full/r32 ~ full/full and r32/full ~ r32/r32, the loss is in the CLUSTERING -- the truncated
    space puts cluster boundaries somewhere else, and the 480 discarded directions matter for
    where k-means splits even though they do not matter for a single argmax;
  * if only r32/r32 drops, the two interact and neither half is separately to blame.

This is cheap enough to be worth resolving because it decides whether 10e's dictionary is usable at
all in the pipeline we actually report, or only under the per-cell rule.

Uses load_points_radii rather than PowerfoamScene: this needs centres and radii, not a dataset, and
constructing the full warp/DataHandler stack loads every image of the scene for nothing.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from build_true_facet_graph import load_points_radii
from determinism import enable_determinism
from diagnose_scannet_miou import spherical_kmeans
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, pool_classify_broadcast

K_FLAT = 320
POINTCEPT = r"D:\Downloads\scannet_pointcept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="solved_geometric_median_nonfrozen_ogl3.pt")
    ap.add_argument("--trunc", default="solved_geometric_median_nonfrozen_ogl3r32.pt")
    ap.add_argument("--out", default="artifacts/rank_cluster_split.json")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    enable_determinism()
    dev = args.device
    res = {}

    for scene in args.scenes:
        ck = f"output/scannet_{scene}_nonfrozen"
        pf = f"artifacts/scannet/{scene}/{args.full}"
        pt = f"artifacts/scannet/{scene}/{args.trunc}"
        if not (os.path.isdir(ck) and os.path.exists(pf) and os.path.exists(pt)):
            print(f"[skip] {scene}")
            continue
        centers, radii = load_points_radii(ck)
        a = torch.load(pf, map_location="cpu", weights_only=True)
        b = torch.load(pt, map_location="cpu", weights_only=True)
        vm = a["valid_mask"].numpy()
        vi = np.where(vm)[0]
        full = F.normalize(a["primitive_features"][vi].to(dev).float(), dim=-1)
        trunc = F.normalize(b["primitive_features"][vi].to(dev).float(), dim=-1)

        gt, rawl, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SCENES[scene], scene), "segment20")
        assigned = assign_points_to_power_cells(gt, centers, radii, valid=vm, k=64)
        owned = assigned >= 0
        n2i = {n: i for i, n in enumerate(all_names)}
        pres = set(np.unique(rawl).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
        gt_lab = remap_gt_labels(rawl, [n2i[n] for n in names])
        text = embed_class_names(names, dev)
        nc = len(names) + 1

        # same seed for both clusterings, so the arms differ only by the space k-means sees
        lab = {"full": spherical_kmeans(full, K_FLAT, seed=0)[0],
               "r32": spherical_kmeans(trunc, K_FLAT, seed=0)[0]}
        feat = {"full": full, "r32": trunc}

        row = {}
        for ck_name, labels in lab.items():
            for fk, u in feat.items():
                cls_valid = pool_classify_broadcast(labels, u, K_FLAT, text).cpu().numpy()
                prim = np.zeros(centers.shape[0], dtype=np.int64)
                prim[vi] = cls_valid
                pred = np.zeros(gt.shape[0], dtype=np.int64)
                pred[owned] = prim[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                                    torch.from_numpy(pred).long(), nc)
                row[f"cluster_{ck_name}/feat_{fk}"] = [float(miou) * 100, float(macc) * 100]
        res[scene] = row
        print(f"{scene}: " + "  ".join(f"{k} {v[0]:.2f}" for k, v in row.items()), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    if res:
        print("\n=== mean over scenes ===")
        for k in next(iter(res.values())):
            print(f"{k:<26} {np.mean([v[k][0] for v in res.values()]):6.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
