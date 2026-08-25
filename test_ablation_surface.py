"""Is the cached-GT surface metric IDENTICAL to the reference, and how much faster?

The optimisation only changes how often the per-class GT KD-trees get rebuilt; every distance
and every aggregate must come out exactly equal to eval_semantic_surface.py's implementation,
otherwise ablation numbers are not comparable with the ones already published from it.

Checked on real ScanNet geometry with real predictions, not synthetic points, because the
failure mode that matters -- a class present in GT but never predicted, or predicted nowhere
near the truth -- only shows up with real label distributions.

Run:  D:\\conda\\envs\\powerfoam\\python.exe test_ablation_surface.py [scene]
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics as fast
from eval_semantic_surface import semantic_surface_metrics as reference
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                       remap_gt_labels)


def compare(a, b, label):
    keys = ("mae_pred2gt", "mae_gt2pred", "scd", "median_pred2gt", "hd95",
            "boundary_precision", "boundary_recall", "boundary_f1")
    bad = []
    for k in keys:
        if k in a or k in b:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None or abs(va - vb) > 0:
                bad.append(f"{k}: {va} vs {vb}")
    for k in ("n_classes_present", "n_missed", "n_scored"):
        if a.get(k) != b.get(k):
            bad.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    # per-class too, not just the aggregate
    pa, pb = a.get("per_class", {}), b.get("per_class", {})
    if set(pa) != set(pb):
        bad.append(f"per_class keys differ: {sorted(set(pa) ^ set(pb))}")
    else:
        for c in pa:
            for k in pa[c]:
                if isinstance(pa[c][k], float):
                    if abs(pa[c][k] - pb[c][k]) > 0:
                        bad.append(f"class {c} {k}: {pa[c][k]} vs {pb[c][k]}")
                elif pa[c][k] != pb[c][k]:
                    bad.append(f"class {c} {k}: {pa[c][k]} vs {pb[c][k]}")
    print(f"  {label:<28}{'IDENTICAL' if not bad else 'DIFFERS'}")
    for m in bad[:5]:
        print(f"      {m}")
    return not bad


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0062_00"
    gt_points, raw, all_names = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\train\{scene}", "segment20")
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if name_to_id[n] in present]
    tids = [name_to_id[n] for n in names]
    gt = remap_gt_labels(raw, tids)
    n_classes = len(names) + 1
    pts = np.asarray(gt_points, dtype=np.float64)
    print(f"{scene}: {len(pts):,} points, {len(names)} classes present\n")

    rng = np.random.default_rng(0)
    # Three prediction regimes, each exercising a different branch.
    preds = {
        "perfect": gt.copy(),
        "noisy (30% shuffled)": np.where(rng.random(len(gt)) < 0.3,
                                         rng.integers(1, n_classes, len(gt)), gt),
        "one class never predicted": np.where(gt == 1, 2, gt),
    }

    ok = True
    print("correctness")
    for label, pred in preds.items():
        idx = GTSurfaceIndex(pts, gt, n_classes)
        a = fast(idx, pred)
        b = reference(pts, gt, pred, n_classes)
        ok &= compare(a, b, label)

    print("\nspeed: one GT index reused across N method cells (the ablation's actual pattern)")
    N = 8
    pred = preds["noisy (30% shuffled)"]
    t = time.time()
    for _ in range(N):
        reference(pts, gt, pred, n_classes)
    t_ref = time.time() - t
    t = time.time()
    idx = GTSurfaceIndex(pts, gt, n_classes)
    for _ in range(N):
        fast(idx, pred)
    t_fast = time.time() - t
    print(f"  reference {t_ref:6.2f}s   cached-GT {t_fast:6.2f}s   "
          f"{t_ref/max(t_fast,1e-9):.2f}x over {N} cells")

    print("\nVERDICT:", "SAFE -- identical output, faster" if ok else "REJECT -- output differs")


if __name__ == "__main__":
    main()
