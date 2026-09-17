"""How much of the Gram's off-diagonal mass crosses a CLASS boundary? Label-free.

A25 established the point that makes o_i the wrong predictor of segmentation: blending two
primitives of the SAME class cannot move an argmax, so only class-crossing mixing can cost a
label. o_i is class-blind and counts both, which is why it tracks the residual but not mIoU
(A24, A26).

The class-aware version needs no ground truth. Partition the off-diagonal mass of G by whether
the two endpoints carry the same PREDICTED class:

    o_total = sum_{j != k} G_jk
    o_cross = sum_{j != k, pred_j != pred_k} G_jk

using pred from the closed-form field itself. Both are one pass over the cached gram edges -- no
rendering, no labels, no solve. `cross_share = o_cross / o_total` is then an operator-plus-readout
quantity that could plausibly predict segmentation where the operator-only o_i provably does not.

Also reported per primitive: the share of ITS OWN off-diagonal mass that crosses a class
boundary, so the quantity can be correlated against per-primitive correctness rather than only
compared between arms.
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
                                       remap_gt_labels)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT, geometry
from validate_bound import load_arm, sym_offdiag, ARMS

TAG = {"foam_truefrozen": "truefrozen", "foam_nonfrozen": "nonfrozen"}


def one(scene, arm, class_set, solver_tag, dev="cuda", chunk=1 << 24):
    stats_name, covis_name = ARMS[arm]
    D, Gd, AtB, cov = load_arm(scene, stats_name, covis_name)
    P = D.numel()
    r, c, v = sym_offdiag(cov, P, dev)

    d = torch.load(f"artifacts/scannet/{scene}/solved_{solver_tag}_{TAG[arm]}_ogl3.pt",
                   map_location=dev, weights_only=True)
    X = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {nm: i for i, nm in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [nm for nm in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[nm] in pres]
    C = len(kept)
    text = embed_class_names(kept, dev)
    pred = (F.normalize(X, dim=-1) @ text.T).argmax(1)          # predicted class, label-free

    tot = torch.zeros(P, device=dev, dtype=torch.float64)
    crs = torch.zeros(P, device=dev, dtype=torch.float64)
    for s0 in range(0, r.numel(), chunk):
        e = slice(s0, s0 + chunk)
        ri, ci, wi = r[e].long(), c[e].long(), v[e]
        tot.index_add_(0, ri, wi)
        crs.index_add_(0, ri, wi * (pred[ri] != pred[ci]).double())
    o_total, o_cross = float(tot.sum()), float(crs.sum())

    # per-primitive cross share, and its correlation with correctness
    live = (D.to(dev) > 0) & torch.from_numpy(valid).to(dev)
    share = (crs / tot.clamp_min(1e-30))

    gt_lab = remap_gt_labels(raw, [n2i[nm] for nm in kept])
    centers, radii, density = geometry(scene, TAG[arm])
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = torch.from_numpy(votes.argmax(1)).to(dev)
    prim_gt[torch.from_numpy(votes.max(1) == 0).to(dev)] = 0

    m = live & (prim_gt > 0)
    correct = (pred[m] + 1 == prim_gt[m]).float()
    sh = share[m].float()
    # RANK-based bins, not value quantiles. The per-primitive cross-share piles up at exactly 0
    # (a primitive all of whose co-visible neighbours share its predicted class), so value
    # quantiles collapse to identical edges and the low bins come out empty -- which is what
    # produced NaN for Q1-Q3 on the first run. Sorting and splitting by COUNT always fills.
    order = torch.argsort(sh)
    cs = correct[order]
    quint = [float(x.mean()) for x in torch.chunk(cs, 5)]
    # AUC of LOW cross-share predicting correctness; 0.5 is chance
    y = correct.cpu().numpy().astype(bool)
    sc = (-sh).cpu().numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 and n0:
        o2 = np.argsort(sc, kind="mergesort")
        rk = np.empty(len(sc), np.float64)
        rk[o2] = np.arange(1, len(sc) + 1)
        auc_cross = float((rk[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    else:
        auc_cross = float("nan")
    frac_zero = float((sh == 0).float().mean())
    acc_zero = float(correct[sh == 0].mean()) if int((sh == 0).sum()) else float("nan")
    acc_nonzero = float(correct[sh > 0].mean()) if int((sh > 0).sum()) else float("nan")
    return dict(scene=scene, arm=arm, P=int(P),
                o_total=o_total, o_cross=o_cross,
                cross_share=o_cross / max(o_total, 1e-30),
                mean_o=o_total / float(D.sum()),
                mean_o_cross=o_cross / float(D.sum()),
                acc=float(correct.mean()), n=int(m.sum()), quintiles=quint,
                auc_cross=auc_cross, frac_zero=frac_zero,
                acc_zero=acc_zero, acc_nonzero=acc_nonzero)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="foam_truefrozen,foam_nonfrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--solver-tag", default="weighted")
    ap.add_argument("--out", default="artifacts/scannet/cross_class_overlap.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, arm, a.class_set, a.solver_tag)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] mean_o {r['mean_o']:.4f}  cross {r['mean_o_cross']:.4f} "
                  f"({r['cross_share']:.1%})  acc {r['acc']:.4f}  AUC {r['auc_cross']:.4f}  "
                  f"zero-share {r['frac_zero']:.1%} acc {r['acc_zero']:.4f} vs "
                  f"{r['acc_nonzero']:.4f}", flush=True)
    if rows:
        print(f"\n{'arm':<18}{'mean o':>9}{'o_cross':>10}{'cross%':>9}{'acc':>8}{'n':>4}")
        for arm in a.arms.split(","):
            rs = [x for x in rows if x["arm"] == arm]
            if not rs:
                continue
            f = lambda k: float(np.mean([x[k] for x in rs]))
            print(f"{arm:<18}{f('mean_o'):>9.4f}{f('mean_o_cross'):>10.4f}"
                  f"{f('cross_share'):>9.1%}{f('acc'):>8.4f}{len(rs):>4}"
                  f"   AUC {f('auc_cross'):.4f}   zero-share {f('frac_zero'):.1%} "
                  f"acc {f('acc_zero'):.4f} vs {f('acc_nonzero'):.4f}")
            q = np.array([x["quintiles"] for x in rs], float)
            print("   accuracy by per-primitive cross-share quintile: " +
                  "  ".join(f"Q{i+1} {np.nanmean(q[:, i]):.3f}" for i in range(5)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
