"""Does the ANGULAR MARGIN predict per-primitive correctness? The gate for every margin bound.

CLIP features live on the sphere and are read out by cosine, so the quantity that decides a label
is not a Euclidean distance but an angle. For primitive j the label survives a rotation theta_j
whenever theta_j < gamma_j / 2, where gamma_j is the ANGULAR GAP between the best and second-best
class. gamma_j is computable from the lifted features and the text embeddings alone -- no labels --
which is what would make a margin bound usable.

Every margin-style bound rests on gamma_j actually tracking correctness. If its AUC against
per-primitive correctness is near chance, no sphere-consistent objective rescues the family and
the remaining ideas should not be built. This is deliberately the cheapest possible test: it needs
only an existing solved feature field, so it runs before any solver work.

Reported per scene and pooled:
  AUC(gamma)      angular top1-top2 gap, in radians
  AUC(cos gap)    the same gap in cosine units, since the readout compares cosines directly
  AUC(top1)       the raw best-class cosine, as a baseline -- if top1 alone does as well, the
                  "margin" adds nothing and is just a confidence score in disguise
  accuracy by gamma quintile, so the shape is visible rather than summarised to one number
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


def auc(score, y):
    """Rank-based AUC; ties handled by average rank."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = np.sort(score)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def one(scene, arm, tag, class_set, dev="cuda"):
    d = torch.load(f"artifacts/scannet/{scene}/solved_{tag}_{arm}_ogl3.pt",
                   map_location=dev, weights_only=True)
    X = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()

    centers, radii, density = geometry(scene, arm)
    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    text = embed_class_names(kept, dev)

    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    P = centers.shape[0]
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = votes.argmax(1)
    prim_gt[votes.max(1) == 0] = 0

    sim = F.normalize(X, dim=-1) @ text.T                      # (P, C) cosines
    top2 = sim.topk(2, dim=-1)
    pred = top2.indices[:, 0].cpu().numpy() + 1
    cos_gap = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
    # angular gap in radians: the quantity theta_j must stay below half of
    ang_gap = (torch.arccos(top2.values[:, 1].clamp(-1, 1))
               - torch.arccos(top2.values[:, 0].clamp(-1, 1))).cpu().numpy()
    top1 = top2.values[:, 0].cpu().numpy()

    m = (prim_gt > 0) & valid
    y = (pred[m] == prim_gt[m]).astype(np.int64)
    g, cg, t1 = ang_gap[m], cos_gap[m], top1[m]
    qs = np.quantile(g, np.linspace(0, 1, 6))
    quint = []
    for b in range(5):
        sel = (g >= qs[b]) & (g <= qs[b + 1] if b == 4 else g < qs[b + 1])
        quint.append(float(y[sel].mean()) if sel.sum() else float("nan"))
    return dict(scene=scene, arm=arm, n=int(m.sum()), acc=float(y.mean()),
                auc_ang=auc(g, y), auc_cos=auc(cg, y), auc_top1=auc(t1, y),
                gap_p50=float(np.median(cg)), quintiles=quint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen")
    ap.add_argument("--tag", default="weighted")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--out", default="artifacts/scannet/angular_margin.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, arm, a.tag, a.class_set)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}")
                continue
            rows.append(r)
            print(f"[{arm}/{sc}] n {r['n']:>7,} acc {r['acc']:.4f} | AUC ang {r['auc_ang']:.4f} "
                  f"cos {r['auc_cos']:.4f} top1 {r['auc_top1']:.4f} | cos-gap p50 "
                  f"{r['gap_p50']:.4f}", flush=True)
    if rows:
        f = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\n=== {len(rows)} scenes ===")
        print(f"  AUC angular gap : {f('auc_ang'):.4f}")
        print(f"  AUC cosine gap  : {f('auc_cos'):.4f}")
        print(f"  AUC top1 cosine : {f('auc_top1'):.4f}   <- if this matches, the margin adds nothing")
        q = np.array([r["quintiles"] for r in rows], float)
        print("  accuracy by angular-gap quintile: " +
              "  ".join(f"Q{i+1} {np.nanmean(q[:, i]):.3f}" for i in range(5)))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
