"""Split the geometry term: is the ceiling limited by the PARTITION or by COVERAGE?

`ceiling` assigns every scored point to its nearest LIVE primitive and labels it with that
primitive's own majority GT class. A primitive is live only if some ray carrying a labelled pixel
deposited on it -- and 22-40% of primitives are not. So two different things are folded into the
geometry loss:

  partition impurity  the owning cell genuinely straddles a class boundary. Irreducible for this
                      partition: no solver, and no amount of extra evidence, recovers it.
  coverage failure    the point's truly-nearest primitive is DEAD, so the point falls to a more
                      distant live one that may carry a different class. This is NOT a property of
                      the partition -- it is a property of how much of the scene the rays reached,
                      and it moves with view count, labelled-pixel coverage and the transmittance
                      floor.

Measuring the ceiling twice -- once over live primitives only, once over ALL primitives -- separates
them. The all-primitive ceiling is what the partition could support with perfect coverage, so

    coverage cost = ceiling_all - ceiling_live

and whatever remains below 100 is the partition's own limit. If the residual is mostly coverage,
"100 mIoU with everything else fixed" is reachable by observing more, not by changing the geometry.

Ownership is Euclidean-nearest-centre in both cases, matching the metric the paper reports.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import torch

from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels, calculate_metrics
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


def ceiling(cent, pts, gt, C, dev="cuda"):
    """mIoU and accuracy when every point takes its nearest primitive's own majority class."""
    P = cent.shape[0]
    # KD-tree, not cdist: the unfrozen 3DGS arm has ~1.9M primitives, so a dense
    # (points x primitives) distance block is ~1 TB even chunked over points only. A tree is
    # O(n log n), runs on CPU, and gives the identical nearest-centre assignment.
    from scipy.spatial import cKDTree
    _, nn = cKDTree(cent).query(pts.astype(np.float32), k=1, workers=-1)
    own = torch.from_numpy(np.asarray(nn, dtype=np.int64)).to(dev)
    g = torch.from_numpy(gt).to(dev)
    occ = torch.zeros(P, C + 1, device=dev)
    occ.index_put_((own, g), torch.ones(g.numel(), device=dev), accumulate=True)
    maj = occ[:, 1:].argmax(1) + 1
    pred = maj[own]
    _, mi, acc, _ = calculate_metrics(g.cpu(), pred.cpu(), C + 1)
    return float(mi) * 100, float(acc) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,gs_froz")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/scannet/ceiling_split.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            try:
                z = np.load(os.path.join(a.stats, f"{arm}_{sc}.npz"))
                live = z["live"].astype(bool)
                cent = centers_of(sc, arm)
                assert cent.shape[0] == live.shape[0], f"{cent.shape[0]} centers vs {live.shape[0]} live"
                d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
                pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
                n2i = {n: i for i, n in enumerate(names)}
                pres = set(np.unique(raw).tolist())
                kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
                C = len(kept)
                gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
                vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
                m = (gl > 0) & vis
                pts_s, gt_s = pts[m], gl[m]
                mi_l, ac_l = ceiling(cent[live], pts_s, gt_s, C)
                mi_a, ac_a = ceiling(cent, pts_s, gt_s, C)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True); continue
            r = dict(arm=arm, scene=sc, P=int(live.size), live=int(live.sum()),
                     n_scored=int(m.sum()), C=C,
                     ceil_live_miou=mi_l, ceil_all_miou=mi_a,
                     ceil_live_acc=ac_l, ceil_all_acc=ac_a)
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] live {r['live']:,}/{r['P']:,}  ceiling mIoU live {mi_l:6.2f} "
                  f"-> all {mi_a:6.2f}  (coverage cost {mi_a - mi_l:+5.2f})", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'ceil live':>11}{'ceil ALL':>10}{'coverage':>10}{'partition':>11}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        l = float(np.mean([r["ceil_live_miou"] for r in s]))
        al = float(np.mean([r["ceil_all_miou"] for r in s]))
        print(f"{arm:<12}{l:>11.2f}{al:>10.2f}{al - l:>10.2f}{100 - al:>11.2f}   n={len(s)}")
    print("\ncoverage = recoverable by observing more; partition = irreducible for this geometry")


if __name__ == "__main__":
    main()
