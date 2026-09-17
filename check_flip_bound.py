"""Is theta_j < gamma_j/2 an actual BOUND on label flips, or only a correlate?

A22 reported ||X' - Xhat||/||Xhat|| = 1.0, which is meaningless: the readout is invariant to any
per-primitive positive scalar, so a Euclidean distance dominated by a norm difference measures
something no label can see. The quantity that governs a label is the ANGLE

    theta_j = angle(x'_j, xhat_j)

and the label survives it whenever theta_j is smaller than half the angular top1-top2 gap:

    angle(u, t1) + theta < angle(u, t2) - theta   =>   argmax unchanged
    i.e.   theta_j < gamma_j / 2 ,   gamma_j = angle(u, t2) - angle(u, t1)

gamma_j is label-free (AUC 0.707 against per-primitive correctness, `check_angular_margin.py`).

THIS TESTS THE IMPLICATION, NOT THE CORRELATION. A bound earns its name only if
`theta_j < gamma_j/2` implies no flip with ZERO violations. Violations are counted explicitly;
a single one falsifies it. Also reported: how tight it is (what fraction of the certified-safe
set exists at all), because a bound that certifies nothing is vacuous even when true.

"Flip" here means argmax(x'_j) != argmax(xhat_j): the label change caused by the closed form
relative to the least-squares optimum -- which is exactly the error the theory is about.

ERROR ATTRIBUTION. Xhat is the best any solver working from this evidence can do, so comparing
both fields against ground truth splits every mistake into two disjoint causes:

    lift-caused   X' wrong AND Xhat right   -- a better solver would fix it
    upstream      X' wrong AND Xhat wrong   -- the evidence itself is wrong; no solver helps

That converts the certificate from "these labels cannot move" into a budget: how much of the
reported accuracy is actually reachable by improving the lift. A20 put the upstream share at
~45% of scored errors by a residual argument; this measures it directly, per primitive, against
the optimum rather than by elimination.
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
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT
from validate_bound import load_arm, sym_offdiag, cg, ARMS


def one(scene, arm, class_set, iters, dev="cuda"):
    stats_name, covis_name = ARMS[arm]
    D, Gd, AtB, cov = load_arm(scene, stats_name, covis_name)
    P = D.numel()
    r, c, v = sym_offdiag(cov, P, dev)
    Dt = D.to(dev)
    live = Dt > 0
    idx = torch.nonzero(live, as_tuple=True)[0]
    remap = torch.full((P,), -1, dtype=torch.long, device=dev)
    remap[idx] = torch.arange(idx.numel(), device=dev)
    keep = live[r] & live[c]
    rr = remap[r[keep]].to(torch.int32)
    cc = remap[c[keep]].to(torch.int32)
    vv = v[keep]
    del r, c, v, keep
    torch.cuda.empty_cache()

    n = idx.numel()
    dg = Gd.to(dev)[idx]
    dd = Dt[idx]
    B = AtB.to(dev).double()[idx]
    Fdim = B.shape[1]
    eb = max(1, int(5.0e8 // (8 * Fdim)))

    def Gmv(X):
        y = dg.unsqueeze(-1) * X
        for s0 in range(0, rr.numel(), eb):
            e = slice(s0, s0 + eb)
            y.index_add_(0, rr[e].long(), vv[e].unsqueeze(-1) * X[cc[e].long()])
        return y

    Xp = B / dd.unsqueeze(-1)
    Xh, res = cg(Gmv, B, dg.clamp_min(1e-12).unsqueeze(-1), iters, 1e-8)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {nm: i for i, nm in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [nm for nm in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[nm] in pres]
    C = len(kept)
    T = embed_class_names(kept, dev).double()

    # per-primitive ground truth, on the same live subset the solve used
    from evaluate_point_cloud_miou import remap_gt_labels as _rgl
    from point_cloud_query import assign_points_to_power_cells as _apc
    from diagnose_holes import geometry as _geom
    arm_tag = {"foam_truefrozen": "truefrozen", "foam_nonfrozen": "nonfrozen"}[arm]
    centers, radii, _dens = _geom(scene, arm_tag)
    gt_lab = _rgl(raw, [n2i[nm] for nm in kept])
    lv = live.cpu().numpy()
    assigned = _apc(gt_pts, centers, radii, valid=lv, k=64)
    votes = np.zeros((P, C + 1), np.int32)
    okm = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[okm], gt_lab[okm]), 1)
    pgt_full = votes.argmax(1)
    pgt_full[votes.max(1) == 0] = 0
    pgt = torch.from_numpy(pgt_full).to(dev)[idx]

    up = F.normalize(Xp, dim=-1)
    uh = F.normalize(Xh, dim=-1)
    theta = torch.arccos((up * uh).sum(-1).clamp(-1, 1))         # angle between the two fields

    sim = up @ T.T
    top2 = sim.topk(2, dim=-1)
    gamma = (torch.arccos(top2.values[:, 1].clamp(-1, 1))
             - torch.arccos(top2.values[:, 0].clamp(-1, 1)))
    flip = top2.indices[:, 0] != (uh @ T.T).argmax(1)

    certified = theta < gamma / 2
    violations = int((certified & flip).sum())

    # --- error attribution against the optimum ---
    pp = top2.indices[:, 0] + 1
    ph = (uh @ T.T).argmax(1) + 1
    sc_m = pgt > 0
    okp = (pp == pgt) & sc_m
    okh = (ph == pgt) & sc_m
    n_sc = int(sc_m.sum())
    err = sc_m & ~okp
    lift_caused = int((err & okh).sum())
    upstream = int((err & ~okh).sum())
    acc_cert = float(okp[sc_m & certified].float().mean()) if int((sc_m & certified).sum()) else float("nan")
    acc_unc = float(okp[sc_m & ~certified].float().mean()) if int((sc_m & ~certified).sum()) else float("nan")

    return dict(scene=scene, arm=arm, live=int(n), cg_residual=res,
                n_scored=n_sc,
                acc_closed=float(okp[sc_m].float().mean()) if n_sc else float("nan"),
                acc_optimum=float(okh[sc_m].float().mean()) if n_sc else float("nan"),
                err_lift_caused=lift_caused / max(n_sc, 1),
                err_upstream=upstream / max(n_sc, 1),
                lift_share_of_errors=lift_caused / max(lift_caused + upstream, 1),
                acc_certified=acc_cert, acc_uncertified=acc_unc,
                theta_p50=float(theta.median()), theta_p90=float(torch.quantile(theta.float(), .9)),
                gamma_p50=float(gamma.median()),
                frac_certified=float(certified.float().mean()),
                frac_flip=float(flip.float().mean()),
                violations=violations,
                flip_rate_certified=float(flip[certified].float().mean()) if int(certified.sum()) else float("nan"),
                flip_rate_uncertified=float(flip[~certified].float().mean()) if int((~certified).sum()) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="foam_truefrozen",
                    help="keys of validate_bound.ARMS: foam_truefrozen / foam_nonfrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="artifacts/scannet/flip_bound.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, arm, a.class_set, a.iters)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] cert {r['frac_certified']:.1%} viol {r['violations']} | "
                  f"acc X' {r['acc_closed']:.4f} vs Xhat {r['acc_optimum']:.4f} | "
                  f"err: lift {r['err_lift_caused']:.2%} upstream {r['err_upstream']:.2%} "
                  f"(lift = {r['lift_share_of_errors']:.1%} of errors) | "
                  f"acc cert {r['acc_certified']:.4f} uncert {r['acc_uncertified']:.4f}",
                  flush=True)
    if rows:
        f = lambda k: float(np.mean([x[k] for x in rows]))
        tot_v = sum(x["violations"] for x in rows)
        print(f"\n=== {len(rows)} scenes ===")
        print(f"  theta p50 {f('theta_p50'):.4f} rad, gamma p50 {f('gamma_p50'):.4f} rad")
        print(f"  certified safe   : {f('frac_certified'):.2%} of primitives")
        print(f"  actual flip rate : {f('frac_flip'):.2%}")
        print(f"  flip rate | certified {f('flip_rate_certified'):.4f}  |  uncertified "
              f"{f('flip_rate_uncertified'):.4f}")
        print(f"  BOUND VIOLATIONS : {tot_v:,}   -> {'HOLDS' if tot_v == 0 else 'FALSIFIED'}")
        print(f"\n--- error attribution (X' vs the optimum Xhat) ---")
        print(f"  accuracy, closed form : {f('acc_closed'):.4f}")
        print(f"  accuracy, optimum     : {f('acc_optimum'):.4f}   <- ceiling for ANY solver "
              f"on this evidence")
        print(f"  errors lift-caused    : {f('err_lift_caused'):.2%}  "
              f"({f('lift_share_of_errors'):.1%} of all errors)")
        print(f"  errors upstream       : {f('err_upstream'):.2%}")
        print(f"  accuracy | certified {f('acc_certified'):.4f}  uncertified "
              f"{f('acc_uncertified'):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
