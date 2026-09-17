"""Score OpenGaussian: point mIoU + semantic surface metrics, 19/15/10 classes.

Their protocol, ported from scripts/eval_scannet.py lines 131-161 without alteration:

    leaf_lang_feat[leaf_occu_count < 2] *= 0    # under-supported leaves contribute nothing
    leaf_ind = leaf_ind.clamp(max=319)          # 64 roots x 5 leaves
    normalize both sides -> cosine -> argmax over classes -> broadcast via leaf_ind

The prediction is therefore per-LEAF, not per-Gaussian: 320 codebook entries carry the language
features and every point inherits its leaf's class. That is the whole point of their method, so it
is reproduced rather than replaced by a per-point argmax.

RESOLUTION. These artifacts come from a re-run at `-r 2`, which is OpenGaussian's own documented
setting ("We use half-resolution data for training", scripts/train_scannet.sh). Our earlier pass
used `-r 1`; at full resolution the two largest scenes (scene0000_00, scene0140_00) exhaust 47 GB
inside mask_feature_mean/process_in_chunks, whose buffers scale with image size, and so never wrote
the cluster_lang.npz that this evaluator needs -- reproducibly, in both the original run and a
retry. The whole set was redone at -r 2 so that all ten scenes share one resolution.

INDEX IDENTITY, verified per scene before scoring: leaf_ind has exactly one entry per GT vertex
(81,369 / 51,610 / ... ), because ScanNet training uses --frozen_init_pts, one Gaussian per GT
point. So point i of the prediction IS GT point i and no assignment step exists to go wrong.

TEXT BANK. Their evaluator reads precomputed embeddings from assets/text_features.json. We embed
the same class names with the same CLIP model our whole pipeline uses (ViT-B-16 / laion2b_s34b_b88k)
rather than shipping their json, and the names are the present-classes subset, matching every other
baseline row here. Their own convention differs only in using the full bank.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

ART = r"D:\Downloads\powerfoam\artifacts\opengaussian_r2"
POINTCEPT = r"D:\Downloads\scannet_pointcept"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
OPACITY_THRESH = 0.1


def ply_opacity(path):
    """sigmoid(opacity) per Gaussian, straight from their saved ply."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"].data
    # np.ascontiguousarray, not np.asarray: a plyfile vertex field is a strided view into the
    # structured record array, and torch.from_numpy rejects strides that are not a multiple of the
    # element size.
    op = np.ascontiguousarray(v["opacity"], dtype=np.float32)
    return torch.sigmoid(torch.from_numpy(op)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART)
    ap.add_argument("--out", default="artifacts/baseline_eval/opengaussian_surface.json")
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    enable_determinism()
    dev = a.device

    rows = []
    for scene, split in SPLIT.items():
        npz = os.path.join(a.art, scene, "cluster_lang.npz")
        ply = os.path.join(a.art, scene, "point_cloud.ply")
        if not (os.path.exists(npz) and os.path.exists(ply)):
            print(f"[miss] {scene}", flush=True)
            continue
        d = np.load(npz)
        leaf_feat = torch.from_numpy(d["leaf_feat"]).float().to(dev)      # [320, 512]
        occu = torch.from_numpy(d["occu_count"]).to(dev)                  # [320]
        leaf_ind = torch.from_numpy(d["leaf_ind"]).long().to(dev)         # [num_pts]
        leaf_feat[occu < 2] *= 0.0                                        # their line 140
        leaf_ind = leaf_ind.clamp(max=319)                                # their line 141

        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, split, scene), "segment20")
        if gt_pts.shape[0] != leaf_ind.shape[0]:
            print(f"[MISALIGNED] {scene}: {leaf_ind.shape[0]:,} points vs "
                  f"{gt_pts.shape[0]:,} GT -- skipping", flush=True)
            continue
        pts = np.asarray(gt_pts, dtype=np.float64)
        opacity = ply_opacity(ply)
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())

        for cs in a.class_sets.split(","):
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = F.normalize(embed_class_names(names, dev), dim=1, p=2)
            lf = F.normalize(leaf_feat, dim=1, p=2)
            max_id = torch.argmax(text @ lf.T, dim=0)                     # [320]
            # A ZEROED LEAF MUST NOT VOTE. Their line 140 zeroes the language feature of any leaf
            # seen by fewer than two views, but a zero row has identical (zero) cosine to every
            # class, so argmax returns index 0 and every such point is labelled with the FIRST
            # class -- "wall" for these sets, and 41.8% of all points on average land in such a
            # leaf (up to 53.3% on scene0200_00). Those points are frequently right by accident.
            # Marking them unpredicted (-1 -> label 0) scores them as misses, which is the same
            # treatment our own arms give a cell with no feature.
            max_id[occu < 2] = -1
            pred = (max_id[leaf_ind] + 1).cpu().numpy()                   # their line 160

            gt_m = gt.copy()
            dropped_pct = 0.0
            if not a.no_mask:
                scored_before = int((gt_m != 0).sum())
                gt_m[opacity < OPACITY_THRESH] = 0
                dropped_pct = (scored_before - int((gt_m != 0).sum())) / max(scored_before, 1) * 100

            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_m).long(),
                                                 torch.from_numpy(pred).long(), nc)
            sm = semantic_surface_metrics(GTSurfaceIndex(pts, gt_m, nc), pred)
            rec = {"tag": "opengaussian_frozen", "scene": scene, "class_set": cs,
                   "miou": float(miou) * 100, "macc": float(macc) * 100,
                   "dropped_pct": dropped_pct, "n_leaves_live": int((occu >= 2).sum())}
            rec.update({k: float(sm[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(sm.get(k), (int, float))})
            rows.append(rec)
            print(f"  og {scene} {cs:15s} mIoU={rec['miou']:5.2f} mAcc={rec['macc']:5.2f} "
                  f"scd={rec['scd']:.4f} bF1={rec['boundary_f1']:.3f} "
                  f"dropped={dropped_pct:.1f}% leaves={rec['n_leaves_live']}/320", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    for cs in a.class_sets.split(","):
        v = [r for r in rows if r["class_set"] == cs]
        if v:
            m = lambda k: float(np.mean([x[k] for x in v]))
            print(f"{cs:16s} n={len(v):>2}  mIoU {m('miou'):6.2f}  mAcc {m('macc'):6.2f}  "
                  f"SCD {m('scd'):.4f}  HD95 {m('hd95'):.4f}  BF1 {100*m('boundary_f1'):.2f}  "
                  f"del {m('dropped_pct'):.1f}%")


if __name__ == "__main__":
    main()
