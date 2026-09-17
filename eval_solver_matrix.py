"""What exactly is "our solver", and what is "foam"? Both solvers on both representations.

Three things are conflated easily and should not be:

  LIFT        who accumulates the per-primitive evidence. For the 3DGS arms this is Splat Feature
              Solver's own `distill.py` (the stats sidecar records
              "source": "splat-distiller/distill.py FFL_STATS_OUT path"), and our exported weights
              were verified equal to their `vis` to 3.6e-5. For the foam it is PowerFoam's
              rasteriser.
  SOLVER      how those accumulated sums become one feature per primitive.
                weighted          = sum_i w_i B_i / sum_i w_i   -- SFS Eq. 6/18, their closed form
                geometric_median  = streaming cosine geometric median -- OURS, in neither SFS nor
                                    NormLift
  READOUT     per-primitive cosine argmax against the text embeddings. Identical everywhere.

So "our solver" differs from theirs only in the SOLVER column, and only when geometric_median is
used. This prints the full 2x2 so a table row can be named correctly instead of described loosely.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics,
                                       load_gaussian_means_opacities, apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
from diagnose_holes import SCENES, GT_ROOT, geometry

ARMS = {
    "foam_truefrozen": dict(kind="foam", recon="truefrozen",
                            feat="solved_{solver}_truefrozen_ogl3"),
    "gs_froz": dict(kind="gs", recon="nonfrozen", feat="solved_{solver}_gs_froz_ogl3",
                    ckpt="recon_remote/gs_froz/{scene}/ckpt.pt"),
}
SOLVERS = ["weighted", "geometric_median"]


def one(scene, arm, solver, class_set, opac_t, gt_mask, dev="cuda"):
    cfg = ARMS[arm]
    fp = f"artifacts/scannet/{scene}/{cfg['feat'].format(solver=solver)}.pt"
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
    text = embed_class_names(kept, dev)

    if cfg["kind"] == "foam":
        centers, radii, density = geometry(scene, cfg["recon"])
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
        if gt_mask:
            alpha = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
            gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, alpha, opac_t, scene)
    else:
        means, opac = load_gaussian_means_opacities(cfg["ckpt"].format(scene=scene), dev)
        assigned = assign_points_to_nearest_center(
            gt_pts, means, valid=None if gt_mask else (opac >= opac_t))
        if gt_mask:
            gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, opac, opac_t, scene)

    pred = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    pl = np.zeros(len(gt_pts), np.int64)
    own = assigned >= 0
    pl[own] = pred[assigned[own]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pl).long(), C + 1)
    return float(miou), float(macc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/solver_matrix.json")
    a = ap.parse_args()
    res = {}
    for cs in a.class_sets.split(","):
        print(f"\n=== {cs} ===")
        print(f"{'arm':<18}{'solver':<18}{'mIoU':>8}{'mAcc':>8}{'n':>4}")
        for arm in ARMS:
            for sv in SOLVERS:
                vals = []
                for sc in a.scenes.split(","):
                    try:
                        vals.append(one(sc, arm, sv, cs, a.opacity_threshold,
                                        not a.no_gt_opacity_mask))
                    except Exception as e:
                        print(f"   [{arm}/{sv}/{sc}] SKIP {type(e).__name__}: {e}")
                if not vals:
                    continue
                mi = st.mean(v[0] for v in vals) * 100
                ma = st.mean(v[1] for v in vals) * 100
                res[f"{cs}|{arm}|{sv}"] = dict(miou=mi, macc=ma, n=len(vals))
                tag = "  <- SFS's own closed form" if sv == "weighted" else "  <- ours"
                print(f"{arm:<18}{sv:<18}{mi:>8.2f}{ma:>8.2f}{len(vals):>4}{tag}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
