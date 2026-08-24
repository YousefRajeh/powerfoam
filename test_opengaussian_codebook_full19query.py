"""Like-for-like with OpenGaussian's eval_scannet.py: the text query set is ALL 19 class
names (eval_scannet.py:107-112 builds target_names from target_id regardless of what is
present in the scene), and mIoU/mAcc average only over classes present in GT
(eval_scannet.py:83-87). My earlier run restricted the query set to the 7 present classes,
which is an easier problem. This re-scores per-cell vs 320-cluster codebook under the
harder, correct query set.

CPU-only. Single scene (scene0347_00) -> PROVISIONAL.
"""
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, embed_class_names,
    classify_primitives, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import spherical_kmeans
from run_cluster_classify_eval import two_level_position_aware

SCENE = "scene0347_00"
K = 320


def plain_argmax(feats, text_feats):
    """eval_scannet.py:155-159 verbatim in shape: F.normalize both, cosine, argmax over
    classes (dim=0 of a [n_cls, n_item] matrix). A zero row -> all-zero column -> argmax
    returns index 0, i.e. the FIRST class name in the query list."""
    return (text_feats @ F.normalize(feats, dim=-1).T).argmax(dim=0)


def main():
    enable_determinism()
    torch.set_num_threads(8)

    names = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    text_feats = embed_class_names(names, "cpu")
    print(f"[query set] {len(names)} names, first = {names[0]!r} "
          f"(this is what a zeroed cluster is forced to predict)")

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\train\{SCENE}", "segment20")
    # GT remap: label space is the 19 og names, in the SAME order as the query set.
    name_to_id = {n: i for i, n in enumerate(all_names)}
    target_ids = [name_to_id[n] for n in names]
    lut = {t: i + 1 for i, t in enumerate(target_ids)}
    gt = np.zeros_like(raw_labels)
    for t, newv in lut.items():
        gt[raw_labels == t] = newv
    gt_t = torch.from_numpy(gt).long()
    present = sorted(set(gt[gt > 0].tolist()))
    print(f"[gt] classes present: {[names[i-1] for i in present]}")

    ck = torch.load(rf"D:\Downloads\powerfoam\output\scannet_{SCENE}_nonfrozen\model.pt",
                    map_location="cpu", weights_only=False)
    centers = ck["points"].numpy().astype(np.float64)
    radii = F.softplus(ck["radii"], beta=100).numpy().astype(np.float64)
    solved = torch.load(rf"D:\Downloads\powerfoam\artifacts\scannet\{SCENE}\solved_geometric_median_nonfrozen_l3.pt",
                        map_location="cpu", weights_only=True)
    vm = solved["valid_mask"].numpy()
    vi = np.where(vm)[0]
    unit = F.normalize(solved["primitive_features"].float()[torch.from_numpy(vi)], dim=-1)
    positions = torch.from_numpy(centers[vi]).float()

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0

    lab, _ = spherical_kmeans(unit, K, seed=0)
    plab = two_level_position_aware(positions, unit, seed=0)

    def pooled_of(l):
        p = torch.zeros(K, unit.shape[1]).index_add_(0, l, unit)
        c = torch.bincount(l, minlength=K)
        return p / c.clamp_min(1).unsqueeze(1).float(), c

    fp, fc = pooled_of(lab)
    pp, pc = pooled_of(plab)

    def score(prim_cls_valid, tag):
        pc_all = np.zeros(centers.shape[0], dtype=np.int64)
        pc_all[vi] = prim_cls_valid.numpy()
        pred = np.zeros(gt_points.shape[0], dtype=np.int64)
        pred[owned] = pc_all[assigned[owned]] + 1
        _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(names) + 1)
        print(f"  {tag:<46} mIoU={miou*100:6.2f}  mAcc={macc*100:6.2f}  oAcc={acc*100:6.2f}")

    print("\n=== opengaussian19, FULL 19-name query set (OpenGaussian's own protocol) ===")
    score(plain_argmax(unit, text_feats), "per-cell, plain argmax (OG rule)")
    score(classify_primitives(unit, text_feats), "per-cell, hubness argmax (ours)")
    score(plain_argmax(fp, text_feats)[lab], "flat-kmeans320, plain argmax")
    score(classify_primitives(fp, text_feats)[lab], "flat-kmeans320, hubness argmax")
    score(plain_argmax(pp, text_feats)[plab], "pos-aware 64x5, plain argmax")
    score(classify_primitives(pp, text_feats)[plab], "pos-aware 64x5, hubness argmax")

    # occu<2 analogue + a deliberately aggressive version, to measure what the zeroing DOES.
    for frac in (0.0, 0.10, 0.25, 0.50):
        z = fp.clone()
        if frac > 0:
            thr = torch.quantile(fc.float(), frac)
            kill = fc.float() <= thr
        else:
            kill = fc < 2
        z[kill] = 0.0
        cellfrac = torch.isin(lab, torch.nonzero(kill).squeeze(1)).float().mean() * 100
        pred_cls = plain_argmax(z, text_feats)[lab]
        forced = (pred_cls == 0).float().mean() * 100
        print(f"  [zero {int(kill.sum()):3d}/320 clusters = {cellfrac:5.2f}% of cells; "
              f"{forced:5.2f}% of cells now predict {names[0]!r}]")
        score(pred_cls, f"flat-kmeans320 + zero (rule={'occu<2' if frac==0 else f'bottom-{int(frac*100)}%'})")


if __name__ == "__main__":
    main()
