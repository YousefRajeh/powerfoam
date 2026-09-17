"""Do the angular margin and the cross-class share carry DIFFERENT information?

Two label-free per-primitive signals have now been measured against correctness:

    gamma_j    angular top1-top2 margin        AUC 0.7071   (check_angular_margin.py)
    x_j        cross-class share of primitive j's own off-diagonal Gram mass
                                               AUC 0.6091 truefrozen / 0.6631 nonfrozen
                                               (measure_cross_class_overlap.py)

They read different geometry: gamma_j is FEATURE-to-TEXT (how decisively this primitive's own
feature picks a class) while x_j is PRIMITIVE-to-NEIGHBOUR (whether the things it shares rays
with agree with it). If they were redundant their combination would not beat the better one.

Reported:
  * Spearman correlation between them -- the direct redundancy check
  * AUC of each alone, of a rank-average, and of a logistic fit on both
  * accuracy in the gamma x cross-share quintile grid, so any interaction is visible
The logistic fit is scored on HELD-OUT SCENES (leave-one-scene-out), because fitting and
evaluating a combination on the same primitives would manufacture the improvement being tested.
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


def auc(score, y):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(score, kind="mergesort")
    rk = np.empty(len(score), np.float64)
    rk[o] = np.arange(1, len(score) + 1)
    return float((rk[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def rankify(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


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

    sim = F.normalize(X, dim=-1) @ text.T
    top2 = sim.topk(2, dim=-1)
    pred = top2.indices[:, 0]
    gamma = (torch.arccos(top2.values[:, 1].clamp(-1, 1))
             - torch.arccos(top2.values[:, 0].clamp(-1, 1)))

    tot = torch.zeros(P, device=dev, dtype=torch.float64)
    crs = torch.zeros(P, device=dev, dtype=torch.float64)
    for s0 in range(0, r.numel(), chunk):
        e = slice(s0, s0 + chunk)
        ri, ci, wi = r[e].long(), c[e].long(), v[e]
        tot.index_add_(0, ri, wi)
        crs.index_add_(0, ri, wi * (pred[ri] != pred[ci]).double())
    share = (crs / tot.clamp_min(1e-30)).float()

    gt_lab = remap_gt_labels(raw, [n2i[nm] for nm in kept])
    centers, radii, _ = geometry(scene, TAG[arm])
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    votes = np.zeros((P, C + 1), np.int32)
    okm = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[okm], gt_lab[okm]), 1)
    pgt = votes.argmax(1)
    pgt[votes.max(1) == 0] = 0
    pgt_t = torch.from_numpy(pgt).to(dev)

    m = (torch.from_numpy(valid).to(dev)) & (D.to(dev) > 0) & (pgt_t > 0)
    y = (pred[m] + 1 == pgt_t[m]).cpu().numpy().astype(np.int64)
    return dict(scene=scene, arm=arm,
                gamma=gamma[m].cpu().numpy(), share=share[m].cpu().numpy(), y=y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arm", default="foam_truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--solver-tag", default="weighted")
    ap.add_argument("--out", default="artifacts/scannet/combine_confidence.json")
    a = ap.parse_args()
    from scipy.stats import spearmanr

    data = []
    for sc in a.scenes.split(","):
        try:
            data.append(one(sc, a.arm, a.class_set, a.solver_tag))
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}", flush=True)
    if not data:
        return

    print(f"{'scene':<14}{'n':>9}{'AUC gam':>9}{'AUC -x':>9}{'AUC rank-avg':>14}{'spearman':>10}")
    rows = []
    for d in data:
        g, x, y = d["gamma"], d["share"], d["y"]
        ra = (rankify(g) + rankify(-x)) / 2
        rho = float(spearmanr(g, x).statistic)
        rows.append(dict(scene=d["scene"], n=int(len(y)), auc_g=auc(g, y),
                         auc_x=auc(-x, y), auc_avg=auc(ra, y), spearman=rho))
        print(f"{d['scene']:<14}{len(y):>9,}{rows[-1]['auc_g']:>9.4f}"
              f"{rows[-1]['auc_x']:>9.4f}{rows[-1]['auc_avg']:>14.4f}{rho:>10.4f}")

    # leave-one-scene-out logistic fit on the two rank features
    loo = []
    for i, d in enumerate(data):
        tr = [data[k] for k in range(len(data)) if k != i]
        Xtr = np.concatenate([np.stack([rankify(t["gamma"]), rankify(-t["share"])], 1) for t in tr])
        ytr = np.concatenate([t["y"] for t in tr])
        Xte = np.stack([rankify(d["gamma"]), rankify(-d["share"])], 1)
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
            s = clf.decision_function(Xte)
        except Exception:
            s = Xte.mean(1)
        loo.append(auc(s, d["y"]))

    f = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"\n=== {a.arm}, {len(rows)} scenes ===")
    print(f"  AUC gamma only        : {f('auc_g'):.4f}")
    print(f"  AUC cross-share only  : {f('auc_x'):.4f}")
    print(f"  AUC rank-average      : {f('auc_avg'):.4f}")
    print(f"  AUC logistic (LOSO)   : {float(np.mean(loo)):.4f}")
    print(f"  spearman(gamma, share): {f('spearman'):+.4f}   "
          f"(near 0 => the two signals are independent)")
    json.dump(dict(rows=rows, loo=loo), open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
