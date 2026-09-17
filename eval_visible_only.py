"""How much of the REAL pipeline's score is decided on points no view ever saw?

The oracle now scores only GT points that were visible in some view -- a point no ray reached
deposited evidence on nothing, so scoring it measures the capture's coverage rather than the lift.
The real evaluation (`evaluate_point_cloud_miou.evaluate_powerfoam`) does NOT do this: it scores
every labelled point, and so do the published baselines, because OpenGaussian's protocol does.

That is defensible for COMPARABILITY -- everyone scores the same points -- but it means the absolute
numbers include ~10.6% of points that are unanswerable from the images, and the fraction is uneven
across classes (shower curtain 31.3% invisible, picture 0.9%), so class-averaged mIoU is affected
unevenly. This measures the size of that effect on the real solved fields.

Reported per arm: mIoU over ALL labelled points (the published protocol) vs over the VISIBLE subset,
and the difference. Nothing here changes the headline tables; it quantifies what they include.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
from diagnose_holes import SCENES, GT_ROOT, geometry

FOAM = {"truefrozen", "nonfrozen"}


def one(scene, arm, solver, class_set, dev="cuda"):
    fp = f"artifacts/scannet/{scene}/solved_{solver}_{arm}_ogl3.pt"
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    d = torch.load(fp, map_location=dev, weights_only=True)
    X = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    T = embed_class_names(kept, dev)
    pred_prim = (F.normalize(X, dim=-1) @ T.T).argmax(1) + 1

    if arm in FOAM:
        centers, radii, _ = geometry(scene, arm)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    else:
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu",
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        centers = sp["means"].float().numpy()
        assigned = assign_points_to_nearest_center(gt_pts, centers, valid=valid)

    pred = np.zeros(gt_pts.shape[0], np.int64)
    owned = assigned >= 0
    pred[owned] = pred_prim.cpu().numpy()[assigned[owned]]

    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    lab = gt_lab > 0
    g_all = torch.from_numpy(gt_lab[lab]); p_all = torch.from_numpy(pred[lab])
    m = lab & vis
    g_vis = torch.from_numpy(gt_lab[m]); p_vis = torch.from_numpy(pred[m])
    _, mi_all, ac_all, _ = calculate_metrics(g_all, p_all, C + 1)
    _, mi_vis, ac_vis, _ = calculate_metrics(g_vis, p_vis, C + 1)
    # mIoU averages only over classes PRESENT in gt, so a class that vanishes when the
    # invisible points are dropped would silently change the denominator between the two
    # numbers. Record both counts; any row where they differ is not a like-for-like delta.
    k_all = int((torch.unique(g_all) != 0).sum())
    k_vis = int((torch.unique(g_vis) != 0).sum())
    return dict(scene=scene, arm=arm, solver=solver, C=C, k_all=k_all, k_vis=k_vis,
                n_all=int(lab.sum()), n_vis=int(m.sum()),
                miou_all=float(mi_all), acc_all=float(ac_all),
                miou_vis=float(mi_vis), acc_vis=float(ac_vis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--solver", default="weighted")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--out", default="artifacts/scannet/eval_visible_only.json")
    a = ap.parse_args()
    import json
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                rows.append(one(sc, arm, a.solver, a.class_set))
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            torch.cuda.empty_cache()
        json.dump(rows, open(a.out, "w"), indent=1)
    if not rows:
        return
    print(f"\nREAL pipeline (solver={a.solver}), all points vs VISIBLE-only\n")
    print(f"{'arm':<12}{'n':>4}{'pts all':>10}{'pts vis':>10}{'mIoU all':>10}"
          f"{'mIoU vis':>10}{'delta':>8}{'acc all':>9}{'acc vis':>9}")
    drop = [r for r in rows if r["k_vis"] != r["k_all"]]
    if drop:
        print("  NOT like-for-like -- class disappeared when invisible points dropped:")
        for r in drop:
            print(f"    {r['arm']}/{r['scene']}: {r['k_all']} -> {r['k_vis']} classes")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{arm:<12}{len(s):>4}{f('n_all'):>10,.0f}{f('n_vis'):>10,.0f}"
              f"{f('miou_all')*100:>10.2f}{f('miou_vis')*100:>10.2f}"
              f"{(f('miou_vis')-f('miou_all'))*100:>+8.2f}"
              f"{f('acc_all')*100:>9.2f}{f('acc_vis')*100:>9.2f}")


if __name__ == "__main__":
    main()
