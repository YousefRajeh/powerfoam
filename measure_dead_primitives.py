"""Why are dead primitives dead -- unobserved, or unobservABLE?

27-40% of primitives receive no evidence, and that costs 1.1-1.9 mIoU of "coverage" (the ceiling
computed over live primitives sits below the ceiling computed over all of them). But `dead` conflates
two things with opposite implications:

  unobserved      a surface cell that rays happened not to reach. Recoverable: more views, better
                  masks, a lower transmittance floor.
  unobservABLE    a cell with ~zero density, or one buried inside a volume where no ray can ever
                  terminate. No view count fixes it, so the loss it causes is irreducible and must
                  NOT be reported as recoverable headroom.

The separator used here is deliberately weak, because a strong one would beg the question:
  * density / opacity -- a primitive the renderer cannot express cannot be observed, whatever the
    camera does. Read directly from the checkpoint, no rendering involved.
  * distance to the nearest labelled GT point -- GT points lie ON the surface, so a cell far from
    every one of them is interior (or a floater) and owns nothing that is scored.

The decisive number is not the size of the dead set: it is how much of the COVERAGE LOSS it causes.
A dead primitive only costs anything if some scored point's nearest primitive is that primitive. So
this reports the density and surface-distance of the primitives that ACTUALLY OWN the moved points,
not of the dead set at large.
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
from ablation_opacity import primitive_alpha

FOAM = {"truefrozen", "nonfrozen"}


def load_arm(scene, arm):
    if arm in FOAM:
        c, r, dens = geometry(scene, arm)
        return c.astype(np.float32), dens.astype(np.float32)
    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    return (sp["means"].float().numpy(),
            torch.sigmoid(sp["opacities"].float().reshape(-1)).numpy())


def one(scene, arm, stats):
    z = np.load(os.path.join(stats, f"{arm}_{scene}.npz"))
    live = z["live"].astype(bool)
    cent, dens = load_arm(scene, arm)
    assert cent.shape[0] == live.shape[0] == dens.shape[0]

    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    m = (gl > 0) & vis
    P = pts[m].astype(np.float32)

    # distance from every primitive to the nearest SCORED point: cells far from the surface own
    # nothing that is measured, so their deadness cannot cost anything
    dist_to_gt = cKDTree(P).query(cent, k=1, workers=-1)[0]
    # which primitive each scored point would pick with perfect coverage, and with real coverage
    own_all = cKDTree(cent).query(P, k=1, workers=-1)[1]
    idx_live = np.nonzero(live)[0]
    own_live = idx_live[cKDTree(cent[live]).query(P, k=1, workers=-1)[1]]
    moved = own_all != own_live                       # points whose true owner is dead

    dead = ~live
    med_sp = float(np.median(dist_to_gt[live])) if live.any() else float("nan")
    # the primitives that actually cause the coverage loss
    culprit = np.unique(own_all[moved]) if moved.any() else np.array([], dtype=np.int64)
    return {
        "scene": scene, "arm": arm, "P": int(live.size), "live": int(live.sum()),
        "dead_frac": float(dead.mean()),
        "n_scored": int(m.sum()),
        "moved_frac": float(moved.mean()),
        # of the DEAD set at large
        "dead_zero_density": float((dens[dead] < 0.01).mean()) if dead.any() else float("nan"),
        "dead_median_dist_to_gt": float(np.median(dist_to_gt[dead])) if dead.any() else float("nan"),
        "live_median_dist_to_gt": med_sp,
        # of the primitives that actually cost us something
        "n_culprits": int(culprit.size),
        "culprit_zero_density": float((dens[culprit] < 0.01).mean()) if culprit.size else float("nan"),
        "culprit_median_density": float(np.median(dens[culprit])) if culprit.size else float("nan"),
        "culprit_median_dist_to_gt": float(np.median(dist_to_gt[culprit])) if culprit.size else float("nan"),
        "culprit_frac_of_dead": float(culprit.size / max(int(dead.sum()), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,gs_froz")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/scannet/dead_primitives.json")
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
            print(f"[{arm}/{sc}] dead {r['dead_frac']:.1%}  moved pts {r['moved_frac']:.2%}  "
                  f"culprits {r['n_culprits']:,} ({r['culprit_frac_of_dead']:.1%} of dead)  "
                  f"culprit zero-density {r['culprit_zero_density']:.1%}", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'dead':>8}{'moved pts':>11}{'culprits/dead':>15}"
          f"{'culprit zero-dens':>19}{'culprit dist':>14}{'live dist':>11}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        g = lambda k: float(np.nanmean([r[k] for r in s]))
        print(f"{arm:<12}{g('dead_frac'):>7.1%}{g('moved_frac'):>11.2%}{g('culprit_frac_of_dead'):>15.1%}"
              f"{g('culprit_zero_density'):>19.1%}{g('culprit_median_dist_to_gt'):>14.4f}"
              f"{g('live_median_dist_to_gt'):>11.4f}")
    print("\nculprits = dead primitives that some scored point would have chosen. Only these cost "
          "anything.\nIf they are zero-density or far from the surface, their loss is IRREDUCIBLE, "
          "not recoverable headroom.")


if __name__ == "__main__":
    main()
