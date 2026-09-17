"""Does the L2 -> mIoU chain survive, or is it vacuous? Measure before investing in the minimax step.

FINDINGS3 supplied both halves of a route from an L2 bound to a decision bound:

  Eq.(4)   ||Y_k - Y_true||  <=  ||H^k Y_true|| + sqrt(k) * delta        (deterministic, any error)
  Eq.(13)  1{yhat != y}  <=  1{ ||s_k,j - s_true,j||_inf >= Delta_j / 2 }

and A52 supplied the missing `delta`, label-free. This assembles them and measures the slack at every
link, because a valid bound is not necessarily a useful one.

THE ONE PIECE THAT IS FREE. `X_true_j` is a UNIT CLASS EMBEDDING `T_c`, so the true score vector is
`s_true,j = T_c T^T` and the true margin is

    Delta_j = 1 - max_{c' != c} <T_c, T_c'>,

which depends only on the class. Hence the worst case over classes

    Delta_min = 1 - max_{c != c'} <T_c, T_c'>

is computable FROM THE TEXT HEAD ALONE -- no labels. (This is why the margin is available here and
not in general; FINDINGS3's warning that "a teacher margin is not a true margin" does not bite,
because the reference field is literally built from text embeddings.)

THE CHAIN, in the D-metric so no 1/min_j D_j conversion constant appears:

  misclassified at j  =>  ||s_k,j - s_true,j||_inf >= Delta_j/2 >= Delta_min/2
  ||s_k,j - s_true,j||_inf = max_c |<e_j, T_c>| <= ||e_j||        (unit T_c, Cauchy-Schwarz)
  => misclassified at j  =>  ||e_j|| >= Delta_min/2
  => Markov on the D-weighted mass:

      sum_j D_j 1{misclassified}  <=  (4 / Delta_min^2) * ||Y_k - Y_true||^2_F        (*)

and `||Y_k - Y_true||` is bounded by Eq.(4) with the A52 delta, plus `||H^k Y_true|| <= ||Y_true||`
and `||Y_true||^2 = sum_j D_j` (unit embeddings) -- also label-free.

WHAT IS REPORTED. Every link, so the vacuity is visible rather than assumed:
  ||Y_1 - Y_true||  actual   vs   the Eq.(4) bound using delta_upper
  D-weighted misclassified mass  actual   vs   the Markov bound (*)
  the bound as a FRACTION OF TOTAL MASS -- if that exceeds 1 the chain is vacuous.

--selftest verifies the two inequalities on dense synthetic problems, and that (*) holds exactly as
stated including the Markov step.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(300):
        P, C, d = 40, 6, 9
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        off = T @ T.T - np.eye(C)
        dmin = 1.0 - float(off.max())
        cls = rng.integers(0, C, P)
        Xt = T[cls]
        Xk = Xt + 0.35 * rng.normal(size=(P, d))
        Dj = rng.uniform(0.1, 3.0, P)

        # link A: the true margin is a per-class quantity, and Delta_min lower-bounds it
        for j in range(P):
            s = T[cls[j]] @ T.T
            m = s[cls[j]] - np.max(np.delete(s, cls[j]))
            assert m >= dmin - 1e-9, (m, dmin)

        # link B: score error is dominated by feature error
        e = Xk - Xt
        assert (np.abs(e @ T.T).max(1) <= np.linalg.norm(e, axis=1) + 1e-9).all()

        # link C: misclassification implies a feature error of at least Delta_min/2
        pred = (Xk @ T.T).argmax(1)
        bad = pred != cls
        assert (np.linalg.norm(e[bad], axis=1) >= dmin / 2 - 1e-9).all()

        # link D: the Markov bound (*)
        lhs = float(Dj[bad].sum())
        rhs = (4.0 / dmin ** 2) * float((Dj * (e ** 2).sum(1)).sum())
        assert lhs <= rhs + 1e-9, (lhs, rhs)
    print("  selftest OK: Delta_min lower-bounds every true class margin; score error <= feature "
          "error; misclassification implies ||e_j|| >= Delta_min/2; the D-weighted Markov bound holds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--out", default="artifacts/scannet/miou_chain.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    from determinism import enable_determinism
    enable_determinism()
    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           remap_gt_labels)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev)
            nnz = val.numel()
            o = torch.argsort(row); row, col, val = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
            colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
            live = colsum > 0
            Dinv = 1.0 / colsum.clamp_min(torch.finfo(val.dtype).eps)

            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            _st = starts.cpu().tolist()
            BUD = max(1, int(3e8 // max(Treg.shape[1], 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            Cc = T.shape[0]
            offd = (T @ T.T) - torch.eye(Cc, device=dev)
            dmin = 1.0 - float(offd.max())                 # LABEL-FREE worst-case true margin

            d = Treg.shape[1]
            rhs = torch.zeros((P, d), device=dev)
            for s0 in range(0, nnz, 8_000_000):
                e0 = min(s0 + 8_000_000, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Treg[gid[row[s0:e0]]])
            Xk = rhs * Dinv[:, None]                       # X' = k=1 iterate

            # label-free delta upper bound (A52)
            rvec = torch.zeros(R, device=dev).index_add_(0, row, val)
            hit = torch.zeros(R, dtype=torch.bool, device=dev); hit[row] = True
            idx = hit.nonzero(as_tuple=True)[0]
            gi = gid[idx]; ri = rvec[idx]
            cosmin = (Treg @ T.T).min(1).values
            bn2 = (Treg ** 2).sum(1)
            delta_up = float((bn2[gi] + ri ** 2 - 2.0 * ri * cosmin[gi]).sum()) ** 0.5

            # oracle field
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            livenp = live.cpu().numpy(); recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=livenp, k=64)
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu", weights_only=False)
                spp = ck["splats"] if "splats" in ck else ck
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(), valid=livenp)
            gtv = gl[m]; okm = own >= 0
            ow = torch.from_numpy(own[okm]).to(dev); gv = torch.from_numpy(gtv[okm]).to(dev)
            cnt = torch.zeros((P, Cc + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            maj = cnt.argmax(1); has = (cnt.sum(1) > 0) & (maj > 0)
            Xt = torch.zeros((P, d), device=dev)
            Xt[has] = T[(maj[has] - 1).clamp_min(0)]

            Dj = colsum
            Ytrue_n2 = float((Dj[has] * (Xt[has] ** 2).sum(1)).sum())       # = sum_j D_j (unit)
            e = Xk - Xt
            Yerr_n2 = float((Dj[has] * (e[has] ** 2).sum(1)).sum())
            # actual D-weighted misclassified mass
            pred = (Xk[has] @ T.T).argmax(1) + 1
            badm = pred != maj[has]
            mass_bad = float(Dj[has][badm].sum())
            mass_tot = float(Dj[has].sum())
            # the chain's bound
            bound_Yerr = (Ytrue_n2 ** 0.5) + delta_up            # ||H Y|| <= ||Y||, k=1
            bound_mass = (4.0 / dmin ** 2) * bound_Yerr ** 2
            # and with the ACTUAL Yerr, to separate the two links' slack
            bound_mass_tight = (4.0 / dmin ** 2) * Yerr_n2

            r_ = {"arm": arm, "scene": sc, "P": int(P), "C": Cc, "delta_min_margin": dmin,
                  "Ytrue_norm": Ytrue_n2 ** 0.5, "Yerr_norm_actual": Yerr_n2 ** 0.5,
                  "Yerr_norm_bound": bound_Yerr, "delta_upper": delta_up,
                  "mass_total": mass_tot, "mass_misclassified": mass_bad,
                  "bound_mass_full_chain": bound_mass, "bound_mass_with_true_Yerr": bound_mass_tight,
                  "frac_actual": mass_bad / max(mass_tot, 1e-30),
                  "frac_bound": bound_mass / max(mass_tot, 1e-30),
                  "frac_bound_tight": bound_mass_tight / max(mass_tot, 1e-30),
                  "wall_s": round(time.time() - t0, 1)}
            out.append(r_); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] Delta_min {dmin:.4f} | ||Y_err|| actual {Yerr_n2**0.5:.1f} vs bound "
                  f"{bound_Yerr:.1f} ({bound_Yerr/max(Yerr_n2**0.5,1e-30):.1f}x) | misclassified mass "
                  f"{100*r_['frac_actual']:.1f}% | CHAIN BOUND {100*r_['frac_bound']:.0f}% of mass "
                  f"({'VACUOUS' if r_['frac_bound']>1 else 'useful'}) | with true Yerr "
                  f"{100*r_['frac_bound_tight']:.0f}%  {r_['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, Xk, Xt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
