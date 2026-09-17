"""Score any solved feature field in the reported point protocol, by tag.

One scorer for every solver so comparisons are like-for-like: same power-cell assignment, same
GT opacity mask, same per-primitive cosine argmax. Only the `solved_<tag>_<arm>_ogl3.pt` file
changes between arms.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics,
                                       apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT, geometry


def score(scene, arm, tag, class_set, opac_t, gt_mask, dev="cuda", cache={}):
    fp = f"artifacts/scannet/{scene}/solved_{tag}_{arm}_ogl3.pt"
    d = torch.load(fp, map_location="cpu", weights_only=True)
    X = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].numpy()

    key = (scene, arm, class_set, gt_mask)
    if key not in cache:
        centers, radii, density = geometry(scene, arm)
        cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
        gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        pres = set(np.unique(raw).tolist())
        kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
        if gt_mask:
            alpha = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
            gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, alpha, opac_t, scene)
        cache[key] = (gt_pts, gt_lab, assigned, kept, embed_class_names(kept, dev))
    gt_pts, gt_lab, assigned, kept, text = cache[key]

    pred = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    pl = np.zeros(len(gt_pts), np.int64)
    own = assigned >= 0
    pl[own] = pred[assigned[own]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pl).long(), len(kept) + 1)
    return float(miou), float(macc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="geometric_median,weighted,spheredeconv")
    ap.add_argument("--arms", default="truefrozen,nonfrozen")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/score_solved.json")
    a = ap.parse_args()
    res = {}
    for cs in a.class_sets.split(","):
        print(f"\n=== {cs} ===")
        print(f"{'arm':<12}{'tag':<18}{'mIoU':>8}{'mAcc':>8}{'n':>4}{'  vs base':>10}")
        for arm in a.arms.split(","):
            base = None
            for tag in a.tags.split(","):
                vals = []
                for sc in a.scenes.split(","):
                    try:
                        vals.append(score(sc, arm, tag, cs, a.opacity_threshold,
                                          not a.no_gt_opacity_mask))
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(f"   [{arm}/{tag}/{sc}] SKIP {type(e).__name__}: {e}")
                if not vals:
                    continue
                mi = float(np.mean([v[0] for v in vals])) * 100
                ma = float(np.mean([v[1] for v in vals])) * 100
                res[f"{cs}|{arm}|{tag}"] = dict(miou=mi, macc=ma, n=len(vals),
                                                per_scene=[v[0] * 100 for v in vals])
                if base is None:
                    base = mi
                print(f"{arm:<12}{tag:<18}{mi:>8.2f}{ma:>8.2f}{len(vals):>4}{mi-base:>+10.2f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
