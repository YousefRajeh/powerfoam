"""Which part of the error does Cut Pursuit actually fix?

The project's error decomposition says grouping should barely matter: 95.6% of error is upstream of
the solve, geometry costs only 1-4 mIoU, and 83% of wrong points are INTERIOR cells of coherent
regions rather than boundary cases. Yet Cut Pursuit is the only intervention that has produced a
reproducible gain (+0.0192 out-of-sample, 8/9, p=0.004). Those two facts have to be reconciled, and
the way to do it is to look at the points it flips rather than to argue from the decomposition.

THE MEASUREMENT. For every scored GT point, take the pre-CP prediction and the post-CP prediction
and classify the flip (fixed / broken / unchanged). Then attribute each flip two ways:

1. REGION PURITY BEFORE POOLING. For a point that CP fixed, what fraction of its own CP region was
   ALREADY correct before pooling? If the fixed points sit in regions that were near 50% correct,
   Cut Pursuit is averaging away per-cell NOISE -- a solve-side effect, the cells disagreed and the
   majority was right. If it fixes regions that were near 0% correct, it cannot be denoising: the
   region's evidence was coherently wrong and pooling must be importing signal from elsewhere.

2. BOUNDARY VS INTERIOR. A cell is a GT boundary cell when at least one facet neighbour carries a
   different GT label. The 83%/17% interior/boundary split caps any purely boundary-side mechanism
   at 17% of the error mass, so if CP's gain is concentrated on boundary cells its ceiling is low
   and known; if it is on interior cells it is reaching the dominant error mass.

The control question -- whether pooling merely re-labels whole regions to their own majority -- is
answered by also computing the MAJORITY-VOTE prediction per region. Pooling features and then
classifying is NOT the same operation as voting on labels, and the gap between them says how much
of CP's effect is explainable as plain majority voting.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "D:/Downloads/powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, apply_gt_opacity_mask,
                                       classify_primitives, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from point_cloud_query import assign_points_to_power_cells


def predictions(feats, text, cell, keep):
    pred_prim = classify_primitives(feats, text)
    if torch.is_tensor(pred_prim):
        pred_prim = pred_prim.cpu().numpy()
    # classify_primitives returns 0-based class indices; GT from remap_gt_labels is 1..K with
    # 0 reserved for unlabelled, so shift into the same space (as evaluate_powerfoam does).
    return np.asarray(pred_prim)[cell][keep] + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0000_00,scene0062_00,scene0070_00,scene0097_00,"
                                        "scene0140_00,scene0200_00,scene0347_00,scene0400_00,"
                                        "scene0590_00,scene0645_00")
    ap.add_argument("--recon", default="nonfrozen")
    ap.add_argument("--base", default="solved_geometric_median_nonfrozen_ogl3")
    ap.add_argument("--cp", default="solved_nf_cp0.03")
    ap.add_argument("--adjacency", default="adjacency_true_facet")
    ap.add_argument("--classes", default="opengaussian19")
    ap.add_argument("--output", default="D:/Downloads/claude_logs/cp_error_attribution.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for scene in a.scenes.split(","):
        A = "artifacts/scannet/%s" % scene
        fb, fc = "%s/%s.pt" % (A, a.base), "%s/%s.pt" % (A, a.cp)
        if not (os.path.exists(fb) and os.path.exists(fc)):
            print("[skip] %s" % scene)
            continue
        cand = [q for q in glob.glob("D:/Downloads/scannet_pointcept/*/%s" % scene) if os.path.isdir(q)]
        gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        pres = set(np.unique(raw).tolist())
        kept = [n for n in OPENGAUSSIAN_CLASS_SETS[a.classes] if n2i[n] in pres]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])

        ck = torch.load("output/scannet_%s_%s/model.pt" % (scene, a.recon),
                        map_location="cpu", weights_only=False)
        centers = ck["points"].float()
        radii = ck["radii"].float().reshape(-1)
        density = ck["density"].float().reshape(-1)
        cell = np.asarray(assign_points_to_power_cells(torch.as_tensor(gt_pts).float(),
                                                       centers, radii))
        alpha_p = 1.0 - np.exp(-density.numpy() * radii.numpy() * 2.0)
        gt_lab, _ = apply_gt_opacity_mask(gt_lab, cell, alpha_p, 0.1, scene)
        keep = gt_lab > 0
        y = gt_lab[keep]

        text = embed_class_names(kept, dev)
        sb = torch.load(fb, map_location="cpu", weights_only=True)
        sc = torch.load(fc, map_location="cpu", weights_only=True)
        pb = predictions(sb["primitive_features"].float().to(dev), text, cell, keep)
        pc = predictions(sc["primitive_features"].float().to(dev), text, cell, keep)
        lab = sc["labels"].numpy() if "labels" in sc else None
        if lab is None:
            print("[skip] %s: cp file has no labels" % scene)
            continue
        reg_of_pt = lab[cell][keep]

        ok_b, ok_c = pb == y, pc == y
        fixed = (~ok_b) & ok_c
        broken = ok_b & (~ok_c)

        # (1) how correct was each point's region BEFORE pooling?
        R = int(lab.max()) + 1
        cnt = np.bincount(reg_of_pt, minlength=R).astype(np.float64)
        cor = np.bincount(reg_of_pt, weights=ok_b.astype(np.float64), minlength=R)
        purity = np.divide(cor, np.maximum(cnt, 1))
        pur_fixed = purity[reg_of_pt[fixed]]
        pur_broken = purity[reg_of_pt[broken]]

        # (2) boundary vs interior, by GT label disagreement across facets
        ad = torch.load("%s/%s.pt" % (A, a.adjacency), map_location="cpu", weights_only=False)
        P = int(ad["num_primitives"])
        off = ad["offsets"].to(torch.int64).numpy()
        adj = ad["adjacent"].to(torch.int64).numpy()
        prim_lab = np.zeros(P, np.int64)
        v = np.zeros((P, int(gt_lab.max()) + 1), np.int32)
        np.add.at(v, (cell[keep], y), 1)
        prim_lab = v.argmax(1)
        has = v.sum(1) > 0
        src = np.repeat(np.arange(P), np.diff(off))
        diff = has[src] & has[adj] & (prim_lab[src] != prim_lab[adj])
        is_bnd = np.zeros(P, bool)
        np.logical_or.at(is_bnd, src[diff], True)
        bnd_pt = is_bnd[cell[keep]]

        # (3) what would plain label-majority voting have done?
        H = np.zeros((R, int(y.max()) + 1), np.int64)
        np.add.at(H, (reg_of_pt, pb), 1)
        maj = H.argmax(1)
        ok_maj = maj[reg_of_pt] == y

        row = dict(
            scene=scene, n_points=int(keep.sum()),
            acc_base=float(ok_b.mean()), acc_cp=float(ok_c.mean()), acc_major=float(ok_maj.mean()),
            fixed=int(fixed.sum()), broken=int(broken.sum()), net=int(fixed.sum() - broken.sum()),
            purity_fixed_mean=float(pur_fixed.mean()) if fixed.any() else None,
            purity_fixed_median=float(np.median(pur_fixed)) if fixed.any() else None,
            purity_broken_mean=float(pur_broken.mean()) if broken.any() else None,
            frac_points_boundary=float(bnd_pt.mean()),
            frac_fixed_boundary=float(bnd_pt[fixed].mean()) if fixed.any() else None,
            frac_broken_boundary=float(bnd_pt[broken].mean()) if broken.any() else None,
            err_base_boundary_share=float(bnd_pt[~ok_b].mean()),
        )
        rows.append(row)
        print("%-14s base=%.4f cp=%.4f major=%.4f | fixed=%d broken=%d net=%+d | "
              "purity(fixed)=%.2f | fixed on boundary=%.2f (points %.2f)"
              % (scene, row["acc_base"], row["acc_cp"], row["acc_major"], row["fixed"],
                 row["broken"], row["net"], row["purity_fixed_mean"] or -1,
                 row["frac_fixed_boundary"] or -1, row["frac_points_boundary"]), flush=True)

    json.dump(rows, open(a.output, "w"), indent=2)
    if rows:
        m = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
        print("\n=== AGGREGATE over %d scenes ===" % len(rows))
        print("  point accuracy: base %.4f -> CP %.4f   (label-majority would give %.4f)"
              % (m("acc_base"), m("acc_cp"), m("acc_major")))
        print("  fixed %d, broken %d, net %+d (summed)"
              % (sum(r["fixed"] for r in rows), sum(r["broken"] for r in rows),
                 sum(r["net"] for r in rows)))
        print("  pre-pooling purity of the regions CP FIXES:  %.3f" % m("purity_fixed_mean"))
        print("  pre-pooling purity of the regions CP BREAKS: %.3f" % m("purity_broken_mean"))
        print("  share of all points on a GT boundary:        %.3f" % m("frac_points_boundary"))
        print("  share of BASE ERRORS on a GT boundary:       %.3f" % m("err_base_boundary_share"))
        print("  share of CP's FIXES on a GT boundary:        %.3f" % m("frac_fixed_boundary"))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
