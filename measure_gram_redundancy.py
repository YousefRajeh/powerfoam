"""How much of G's off-diagonal is BLUR (deconvolvable) and how much is REDUNDANCY (not)?

A partition basis and an overlapping basis put different things in the off-diagonal of G, and the
difference decides what any solver can achieve:

  BLUR        voxels, and power cells: two primitives share a ray only by lying at different
              depths along it. The mixing is geometric and depth-ordered, the system stays
              identifiable, and iteration can undo it. This is the regime SIRT was built for.
  REDUNDANCY  overlapping primitives explain the SAME point in space, so they are indistinguishable
              from every view at once. That is null space, not blur: the information was never
              measured and no iteration recovers it.

The normalised co-visibility correlation separates them without an eigendecomposition:

    c_jk = G_jk / sqrt(G_jj G_kk)   in [0, 1] by Cauchy-Schwarz

c_jk -> 1 means primitives j and k receive their ray weight in almost exactly the same proportions
from every ray -- they are near-duplicates of one another as columns of A, i.e. a near-null
direction (u_j - u_k). Reporting max_k c_jk per primitive gives the per-primitive redundancy, and
its upper tail is the near-null mass. One pass over the cached gram edges; no eigensolver, and
exact rather than estimated.

Also reported: rho_j = (sum_{k != j} G_jk)/G_jj, the per-primitive overlap the solver's step size
is built from, so the two can be read side by side -- a primitive can have large rho (much blur)
and small c (no redundancy), which is the deconvolvable case.
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from diagnose_holes import SCENES
from solve_sphere_deconv import ARMS, load


def one(scene, arm, dev="cuda", chunk=1 << 24):
    D, Gd, AtB, r, c, w, P = load(scene, arm, dev)
    live = D > 0
    gd = Gd.clamp_min(1e-30)

    cmax = torch.zeros(P, device=dev, dtype=torch.float64)
    offsum = torch.zeros(P, device=dev, dtype=torch.float64)
    for s0 in range(0, r.numel(), chunk):
        e = slice(s0, s0 + chunk)
        ri, ci, wi = r[e].long(), c[e].long(), w[e]
        cc = wi / (gd[ri] * gd[ci]).sqrt()          # normalised co-visibility correlation
        cmax.index_reduce_(0, ri, cc, "amax", include_self=True)
        offsum.index_add_(0, ri, wi)
    rho = offsum / gd

    m = live
    cm = cmax[m]
    rh = rho[m]
    q = lambda t, x: float(torch.quantile(x.float(), t))
    return dict(scene=scene, arm=arm, P=int(P), live=int(m.sum()),
                c_p50=q(.5, cm), c_p90=q(.9, cm), c_p99=q(.99, cm), c_max=float(cm.max()),
                frac_c_gt_90=float((cm > 0.90).float().mean()),
                frac_c_gt_99=float((cm > 0.99).float().mean()),
                rho_p50=q(.5, rh), rho_p90=q(.9, rh), rho_mean=float(rh.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen")
    ap.add_argument("--out", default="artifacts/scannet/gram_redundancy.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                rows.append(one(sc, arm))
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            r = rows[-1]
            print(f"[{arm}/{sc}] c_jk p50 {r['c_p50']:.3f} p90 {r['c_p90']:.3f} "
                  f"p99 {r['c_p99']:.3f}  >0.9 {r['frac_c_gt_90']:.2%}  >0.99 "
                  f"{r['frac_c_gt_99']:.3%}  | rho mean {r['rho_mean']:.3f}", flush=True)
    if rows:
        print(f"\n{'arm':<14}{'c p50':>8}{'c p90':>8}{'c p99':>8}{'>0.9':>9}{'>0.99':>9}"
              f"{'rho mean':>10}{'n':>4}")
        for arm in a.arms.split(","):
            rs = [x for x in rows if x["arm"] == arm]
            if not rs:
                continue
            f = lambda k: float(np.mean([x[k] for x in rs]))
            print(f"{arm:<14}{f('c_p50'):>8.3f}{f('c_p90'):>8.3f}{f('c_p99'):>8.3f}"
                  f"{f('frac_c_gt_90'):>9.2%}{f('frac_c_gt_99'):>9.3%}"
                  f"{f('rho_mean'):>10.3f}{len(rs):>4}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
