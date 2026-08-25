"""Semantic-surface metrics for the ablation, with the GT side built once per scene.

WHAT THESE MEASURE, and why they sit beside mIoU. mIoU counts points; it cannot distinguish
"the predicted region is slightly the wrong size" from "the predicted region is on the far
side of the room". These do, by measuring DISTANCE between the predicted and true regions of
the same point cloud (per class c, over classes present in that scene's GT):

    mae_pred2gt(c)  mean over PRED_c of distance to nearest GT_c   -- false-positive depth
    mae_gt2pred(c)  mean over GT_c of distance to nearest PRED_c   -- false-negative depth
    scd(c)          (mae_pred2gt + mae_gt2pred) / 2                -- semantic Chamfer-L1
    hd95(c)         max of the two 95th percentiles                -- outlier-trimmed worst case
    boundary_f1(c)  F1 under "nearest same-class point within tau" (tau = 2 cm)

A low scd with a low mIoU is the informative case: errors are near the right surface (boundary
slop) rather than teleported across the scene.

NAMING follows the project rule: a mean distance is MAE in centimetres, never "accuracy";
precision / recall / F1 appear only with their criterion stated. Definitions are identical to
eval_semantic_surface.py -- verified elementwise in test_ablation_surface.py -- so ablation
numbers stay comparable with the ones already published from that script.

MISSED CLASSES ARE COUNTED, NOT DROPPED. If a class is never predicted its distances are
undefined; excluding it silently would reward a method for predicting nothing. Such classes
are excluded from the means and reported as `n_missed`, which must be read alongside them.

THE OPTIMISATION, and the part of it that was initially WRONG. The reference builds two
KD-trees per class per evaluation. In an ablation scoring one scene under many
(recon x solver x grouping) combinations the GT side is IDENTICAL every time, so the per-class
GT trees are built once per (scene, class-set) and reused. That alone is worth 1.10-1.17x.

Multi-threaded queries were then added on the assumption they would help. They do not, at
small scale: thread-spawn overhead dominates, and blanket workers=-1 made a 51k-point scene
1.7x SLOWER than the reference. Measured, per method cell:

    scene0062_00   51,610 pts, 6 classes    ref 0.63s   workers=1 0.57s (1.10x)   workers=-1 1.07s (0.59x)
    scene0140_00  372,941 pts, 8 classes    ref 11.19s  workers=1 9.54s (1.17x)   workers=-1 3.28s (3.41x)

So the worker count is chosen from the point count rather than fixed. The arithmetic is
untouched either way -- only tree rebuild count and query threading change, and the output is
verified elementwise identical to the reference in test_ablation_surface.py.
"""
import numpy as np
from scipy.spatial import cKDTree

TAU = 0.02          # 2 cm boundary criterion

# Below this many points, spawning query threads costs more than it saves (measured 0.59x on
# a 51k-point scene); above it, threading is the dominant win (3.41x on a 373k-point scene).
# The crossover was measured, not guessed, and sits well inside the gap between those two.
PARALLEL_MIN_POINTS = 150_000


class GTSurfaceIndex:
    """Per-class GT points and KD-trees for one (scene, class-set). Build once, reuse."""

    def __init__(self, points, gt, n_classes):
        self.points = points
        self.n_classes = n_classes
        self.gt_pts = {}
        self.gt_trees = {}
        self.n_gt = {}
        for c in range(1, n_classes):
            m = gt == c
            n = int(m.sum())
            if n == 0:
                continue                  # absent from this scene: not scored, per the protocol
            self.n_gt[c] = n
            self.gt_pts[c] = points[m]
            self.gt_trees[c] = cKDTree(self.gt_pts[c])

    def classes(self):
        return sorted(self.n_gt)


def semantic_surface_metrics(index: GTSurfaceIndex, pred, tau=TAU, workers=None):
    """Same definitions as eval_semantic_surface.semantic_surface_metrics, reusing GT trees.

    `pred` is 1-based class ids with 0 = no prediction, matching calculate_metrics.
    """
    if workers is None:
        workers = -1 if len(index.points) >= PARALLEL_MIN_POINTS else 1
    per_class = {}
    for c in index.classes():
        pm = pred == c
        n_pred = int(pm.sum())
        n_gt = index.n_gt[c]
        if n_pred == 0:
            per_class[c] = {"n_gt": n_gt, "n_pred": 0, "missed": True}
            continue
        ppts = index.points[pm]
        gpts = index.gt_pts[c]
        # predicted -> true region: query the PREBUILT GT tree (the reuse that pays)
        d_p2g, _ = index.gt_trees[c].query(ppts, k=1, workers=workers)
        # true -> predicted region: the prediction changes per method, so this tree is new
        d_g2p, _ = cKDTree(ppts).query(gpts, k=1, workers=workers)
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
        return {"n_classes_present": len(per_class), "n_missed": n_missed,
                "n_scored": 0, "tau": tau, "per_class": per_class}
    agg = {k: float(np.mean([m[k] for m in live])) for k in
           ("mae_pred2gt", "mae_gt2pred", "scd", "median_pred2gt", "hd95",
            "boundary_precision", "boundary_recall", "boundary_f1")}
    agg.update({"n_classes_present": len(per_class), "n_missed": n_missed,
                "n_scored": len(live), "tau": tau, "per_class": per_class})
    return agg
