"""Split the geometry loss into SURFACE error and STRADDLE error.

A point can be lost under a perfect solver for two unrelated reasons:

  A. SURFACE / containment.  No primitive's extent actually covers the point. It is still assigned to
     one -- the partition is exhaustive, every point has a nearest cell -- but that assignment is a
     default, not a representation. The root cause is that the reconstructed surface sits away from
     the true surface (depth error), or that the covering primitive is dead. The assignment rule
     (nearest centre / power cell / Mahalanobis) changes which primitive catches the point, not
     whether it was represented.

  B. STRADDLE.  The point IS covered, and by the right primitive, but that primitive's own extent
     spans a class boundary, so one label cannot serve all of its points.

These call for different fixes -- A wants better geometry or more coverage, B wants finer or
boundary-aligned partitioning -- so reporting them as one number ("geometry: 1.87") hides which.

CONTAINMENT TEST.  `d_own(p) <= extent(owner)`:
  foam   extent = the power radius r_j, the primitive's own scale parameter.
  3DGS   extent = the largest Gaussian axis, i.e. max(exp(scale)); a Gaussian has no compact
         support, so this is the conventional 1-sigma reach and is reported as such rather than
         pretended to be a cell boundary.

Every point is then in exactly one of:
    covered + pure owner        correct under the ceiling
    covered + straddling owner  cause B
    not covered                 cause A
and the three partition the scored set exactly, which is asserted.
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


def arm_geom(scene, arm):
    """-> centres (P,3), extent (P,) in world units."""
    if arm in FOAM:
        c, r, _ = geometry(scene, arm)
        return c.astype(np.float32), r.astype(np.float32)
    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    return (sp["means"].float().numpy(),
            torch.exp(sp["scales"].float()).max(dim=-1).values.numpy())


def one(scene, arm, stats):
    z = np.load(os.path.join(stats, f"{arm}_{scene}.npz"))
    live = z["live"].astype(bool)
    cent, ext = arm_geom(scene, arm)
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    m = (gl > 0) & vis
    P, g = pts[m].astype(np.float32), gl[m]

    idx = np.nonzero(live)[0]
    C, E = cent[live], ext[live]
    d_own, own_l = cKDTree(C).query(P, k=1, workers=-1)
    d_own = d_own.astype(np.float32)

    # ceiling prediction and which points it loses
    K = int(g.max())
    occ = np.zeros((C.shape[0], K + 1), np.int64)
    np.add.at(occ, (own_l, g), 1)
    maj = occ[:, 1:].argmax(1) + 1
    wrong = maj[own_l] != g

    covered = d_own <= E[own_l]            # A: is the point inside its owner's own extent?
    # distance to the nearest primitive of ANY kind, live or not -- pure surface fidelity,
    # independent of which primitives happened to receive evidence
    d_any = cKDTree(cent).query(P, k=1, workers=-1)[0].astype(np.float32)

    n = P.shape[0]
    a = int((wrong & ~covered).sum())       # lost, not covered  -> surface / coverage
    b = int((wrong & covered).sum())        # lost, covered      -> straddle
    assert a + b == int(wrong.sum())
    return {
        "scene": scene, "arm": arm, "n_scored": n,
        "loss_total": float(wrong.mean()),
        "loss_surface": a / n, "loss_straddle": b / n,
        "frac_covered": float(covered.mean()),
        "median_d_own": float(np.median(d_own)),
        "median_d_any": float(np.median(d_any)),
        "median_extent": float(np.median(E)),
        # surface fidelity: how far a GT point sits from the nearest primitive at all
        "d_any_p90": float(np.quantile(d_any, 0.9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/scannet/geom_causes.json")
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
            print(f"[{arm}/{sc}] loss {r['loss_total']:.4f} = surface {r['loss_surface']:.4f} + "
                  f"straddle {r['loss_straddle']:.4f}   covered {r['frac_covered']:.1%}", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'loss':>9}{'SURFACE':>10}{'STRADDLE':>10}{'covered':>10}"
          f"{'d_any med':>11}{'extent med':>12}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{arm:<12}{f('loss_total'):>9.4f}{f('loss_surface'):>10.4f}{f('loss_straddle'):>10.4f}"
              f"{f('frac_covered'):>9.1%}{f('median_d_any'):>11.4f}{f('median_extent'):>12.4f}")
    print("\nSURFACE  = point not inside any owning primitive's extent (bad geometry / dead cover)")
    print("STRADDLE = point covered, but its primitive's extent spans a class boundary")


if __name__ == "__main__":
    main()
