"""Can the geometry loss be BOUNDED by a quantity that needs no ground-truth labels?

`geom_acc = 1 - purity` is an identity, but purity is measured FROM the labels, so it explains the
loss without predicting it. A bound has to be computable from the representation alone.

THE STRUCTURE. A cell can only be impure if a class boundary passes through it. If cells have extent
delta, every impure point must lie within ~delta of a boundary, so

    1 - purity   <=   Pr[ point lies within delta of a class boundary ]

The right-hand side factorises into one term per side of the problem:
    delta              the representation's cell scale -- median nearest-neighbour spacing between
                       primitive centres. No labels involved.
    boundary density   the scene's own geometry -- how much of the surface lies near a class change.
                       Identical for every arm on a scene, so it cannot flatter one representation.

This measures both and checks (a) that the inequality HOLDS, and (b) whether it is tight enough to
be worth stating. A bound that holds by a factor of 50 is true and useless.

`d_bnd(p)` = distance from a scored point to the nearest scored point carrying a DIFFERENT class.
Points with `d_bnd <= delta` are the ones a cell of that scale can confuse.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT, geometry

FOAM = {"truefrozen", "nonfrozen"}


def centers_of(scene, arm):
    if arm in FOAM:
        c, _, _ = geometry(scene, arm)
        return c.astype(np.float32)
    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    return sp["means"].float().numpy()


def boundary_distance(P, g):
    """d_bnd(p): distance to the nearest scored point of a DIFFERENT class.

    Done per class: for class c, query the tree built on all points NOT in c. That is exact and
    avoids a k-NN scan whose k would have to grow with the size of a same-class blob.
    """
    d = np.full(P.shape[0], np.inf, np.float32)
    for c in np.unique(g):
        m = g == c
        if m.all():
            continue
        t = cKDTree(P[~m])
        d[m] = t.query(P[m], k=1, workers=-1)[0]
    return d


def one(scene, arm, stats):
    z = np.load(os.path.join(stats, f"{arm}_{scene}.npz"))
    live = z["live"].astype(bool)
    cent = centers_of(scene, arm)
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    m = (gl > 0) & vis
    P, g = pts[m].astype(np.float32), gl[m]

    # delta: the representation's cell scale, from centres alone (k=2 -> nearest OTHER centre)
    C = cent[live]
    nn = cKDTree(C).query(C, k=2, workers=-1)[0][:, 1]
    delta = float(np.median(nn))

    # the scene's own boundary structure -- identical for every arm
    db = boundary_distance(P, g)

    # ACTUAL impurity under the live partition: a point is lost iff its class differs from its
    # owner's majority class (this is exactly 1 - purity)
    d_own, own = cKDTree(C).query(P, k=1, workers=-1)
    d_own = d_own.astype(np.float32)
    K = int(g.max())
    occ = np.zeros((C.shape[0], K + 1), np.int64)
    np.add.at(occ, (own, g), 1)
    maj = occ[:, 1:].argmax(1) + 1
    actual = float((maj[own] != g).mean())

    return {
        "scene": scene, "arm": arm, "live": int(live.sum()), "n_scored": int(m.sum()),
        "delta": delta,
        "actual_impurity": actual,
        # the bound, at delta and at delta/2 (a point needs a boundary INSIDE its cell, and the
        # centre sits roughly mid-cell, so delta/2 is the tighter geometric reading)
        "bound_delta": float((db <= delta).mean()),
        "bound_half_delta": float((db <= delta / 2).mean()),
        # PER-POINT bound. The delta version assumed every point sits within the median
        # centre spacing of its owner -- false when centres are uneven (3DGS-unfrozen has
        # 1.9M centres incl. floaters, median spacing 0.0022, violated 10/10). Use each
        # point OWN distance to its owner: two points share a cell only if both are within
        # reach of the same centre, so a point is safe when every differently labelled
        # point is farther than 2*d_own. Centres only, still no labels.
        "d_own_median": float(np.median(d_own)),
        "bound_2down": float((db <= 2 * d_own).mean()),
        "bound_1down": float((db <= d_own).mean()),
        "dbnd_median": float(np.median(db)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/scannet/geom_bound.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                r = one(sc, arm, a.stats)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            ok = "OK" if r["bound_2down"] >= r["actual_impurity"] else "*** VIOLATED ***"
            print("[%s/%s] d_own %.4f actual %.4f bound(2*down) %.4f  %s" % (arm, sc, r["d_own_median"], r["actual_impurity"], r["bound_2down"], ok), flush=True)
    if not rows:
        return
    print("%-12s%9s%10s%14s%12s%9s%12s" % ("arm","d_own","actual","bound 2*down","bound down","slack x","violations"))
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        v = sum(1 for r in s if r.get("bound_2down", 0) < r["actual_impurity"])
        sl = f("bound_2down") / max(f("actual_impurity"), 1e-9)
        print("%-12s%9.4f%10.4f%14.4f%12.4f%9.1f%7d/%d" % (arm, f("d_own_median"), f("actual_impurity"), f("bound_2down"), f("bound_1down"), sl, v, len(s)))
    print("\nbound holds if bound >= actual. slack = how loose it is; a bound true by 50x is useless.")


if __name__ == "__main__":
    main()
