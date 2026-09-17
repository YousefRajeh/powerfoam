"""An UPPER bound on achievable accuracy from the data alone: what the solve cannot fix.

A52 measured that 99.24% of delta lives in range(A) -- the semantic error is one the operator can
represent, so a good fit says nothing about being right. This turns that observation into a bound.

THE QUESTION. Given only `A` and `B`, and knowing the true field satisfies ||B - A X*|| <= delta,
how many class assignments for primitive j fit the data equally well? If several do, no estimator can
choose between them, and the achievable accuracy is capped regardless of how good the solver is.

THE CLOSED FORM. Let `X^(j->c)` be X* with primitive j's class changed to c, and R = B - A X*.
Only rays touching j change, and

    ||B - A X^(j->c)||^2 - ||B - A X*||^2
        = -2 <[A^T R]_j, T_c - T_c*(j)>  +  G_jj ||T_c - T_c*(j)||^2   =: Delta_j(c)

with G_jj = sum_i A_ij^2. So a single pass giving `A^T R` (P x d) and `G_jj` (P) yields Delta_j(c)
for EVERY primitive and EVERY class at once -- no per-flip re-evaluation.

THE COUNT.  N_j(delta) = #{ c : ||B - A X*||^2 + Delta_j(c) <= delta^2 }.
c*(j) is always counted (Delta_j(c*) = 0 and ||B - AX*|| <= delta), so N_j >= 1.

THE BOUND -- CORRECTED. A first version of this file claimed "worst-case accuracy <= mean_j 1/N_j"
from the count of classes fitting WITHIN delta. The self-test refuted it immediately (plug-in
accuracy 0.857 against a claimed bound of 0.200). Two errors: fitting *within* delta is not
indistinguishability -- an alternative that fits worse is still distinguishable, and the data prefers
the truth -- and 1/N_j is an average-case quantity, not a worst-case one.

The defensible statement uses classes that fit AT LEAST AS WELL:

    MISLED_j  :=  [ min_c Delta_j(c) < 0 ]        i.e. some wrong class fits STRICTLY better,
                                                  with every other primitive held at its true value

If MISLED_j, then X* is not a coordinate-wise minimum of ||B - A X||: flipping j alone strictly
reduces the residual. Hence **any algorithm returning a coordinate-wise residual minimum disagrees
with the truth at j**, no matter how well it optimises. So

    accuracy of any coordinate-wise residual minimiser  <=  1 - frac(MISLED)      (*)

This is a bound on residual-minimising estimators -- which is exactly the class we care about, since
X', Xhat, X_ball, CG and sphere-deconvolution are all in it. It is NOT a bound over all conceivable
estimators, and is not claimed to be. The oracle reference makes it the sharpest such statement:
even given every other primitive exactly, the data points at the wrong class.

delta is supplied by the A52 label-free bound, so the whole thing needs no labels EXCEPT for the
reference X* used to define the flips -- reported here as a diagnostic. With delta from A52 and any
admissible reference the bound is valid; using the oracle reference makes it as tight as we can make
it, hence the strongest statement of the form "even knowing the truth, the data cannot confirm it".

--selftest verifies:
  1. the closed form Delta_j(c) against explicit re-evaluation of ||B - A X^(j->c)||^2;
  2. Delta_j(c*) = 0 exactly;
  3. N_j >= 1 always, and N_j = C when delta is huge;
  4. MISLED_j holds exactly when coordinate descent on the residual, started AT the truth, moves
     primitive j away from it -- the operational meaning of the bound.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def delta_flip(AtR, Gjj, T, cstar):
    """Delta_j(c) for every primitive and class at once. Works for numpy OR torch.

    Cost: one (P,d)x(d,C) matmul plus O(P C). With C ~ 14 the whole table is ~1M entries, so no
    per-flip re-evaluation is ever needed -- that is the entire optimisation, and it is exact.
    AtR (P,d), Gjj (P,), T (C,d), cstar (P,) in [0,C).
    """
    Tc = T[cstar]                                   # (P, d) current class embedding
    diff_ip = AtR @ T.T - (AtR * Tc).sum(1, keepdims=True)          # <AtR_j, T_c - T_c*>
    sq = (T ** 2).sum(1)[None, :] - 2.0 * (Tc @ T.T) + (Tc ** 2).sum(1)[:, None]   # ||T_c - T_c*||^2
    return -2.0 * diff_ip + Gjj[:, None] * sq


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R, P, C, d = 40, 7, 5, 6
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.4, 1.0, (R, 1))
        cstar = rng.integers(0, C, P)
        Xs = T[cstar]
        B = A @ Xs + 0.3 * rng.normal(size=(R, d))
        Rr = B - A @ Xs
        AtR = A.T @ Rr; Gjj = (A ** 2).sum(0)
        D = delta_flip(AtR, Gjj, T, cstar)
        base = float((Rr ** 2).sum())
        # 1 & 2: closed form vs explicit re-evaluation
        for j in range(P):
            for c in range(C):
                X2 = Xs.copy(); X2[j] = T[c]
                exact = float(((B - A @ X2) ** 2).sum()) - base
                assert abs(exact - D[j, c]) < 1e-8 * max(1.0, abs(exact)), (j, c, exact, D[j, c])
            assert abs(D[j, cstar[j]]) < 1e-10
        # 3: N_j >= 1, and = C for huge delta
        N = (base + D <= base + 1e-12).sum(1)
        assert (N >= 1).all()
        Nbig = (base + D <= 1e18).sum(1)
        assert (Nbig == C).all()
        # 4: MISLED_j <=> coordinate descent started AT the truth moves j away from it
        misled = D.min(1) < -1e-12
        for j in range(P):
            best = int(np.argmin(D[j]))
            moved = best != cstar[j]
            assert moved == bool(misled[j]), (j, best, cstar[j], D[j].min())
    print("  selftest OK: closed form matches explicit re-evaluation; Delta_j(c*) = 0; N_j >= 1 and "
          "saturates at C; MISLED_j is exactly 'coordinate descent from the truth moves j'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--out", default="artifacts/scannet/identifiability.json")
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
            Gjj = torch.zeros(P, device=dev).index_add_(0, col, val * val)
            rvec = torch.zeros(R, device=dev).index_add_(0, row, val)

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            Cc = T.shape[0]
            # whitened class space: cheap, and a partial isometry so delta only contracts
            Mm = (T @ T.T).double(); ev, Vv = torch.linalg.eigh(Mm)
            Tw = ((Vv @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ Vv.T) @ T.double()).float()
            Treg = Treg @ Tw.T; T = T @ Tw.T
            d = Treg.shape[1]

            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            _st = starts.cpu().tolist()
            BUD = max(1, int(3e8 // max(d, 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            # oracle reference field
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
            cstar = (maj - 1).clamp_min(0)
            Xs = torch.zeros((P, d), device=dev); Xs[has] = T[cstar[has]]

            # residual and A^T R, row-blocked
            base = 0.0
            AtR = torch.zeros((P, d), device=dev)
            for r0, r1, s_, e_ in blocks:
                lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                t = torch.zeros((r1 - r0, d), device=dev)
                t.index_add_(0, lr, vw * Xs[cs])
                hit = torch.zeros(r1 - r0, dtype=torch.bool, device=dev); hit[lr] = True
                idx = hit.nonzero(as_tuple=True)[0]
                res = torch.zeros((r1 - r0, d), device=dev)
                res[idx] = Treg[gid[idx + r0]] - t[idx]
                base += float((res[idx] ** 2).sum())
                AtR.index_add_(0, cs, vw * res[lr])
                del t, hit, idx, res

            # label-free delta (A52)
            hit = torch.zeros(R, dtype=torch.bool, device=dev); hit[row] = True
            idx = hit.nonzero(as_tuple=True)[0]
            gi = gid[idx]; ri = rvec[idx]
            cosmin = (Treg @ T.T).min(1).values
            delta2 = float(((Treg[gi] ** 2).sum(1) + ri ** 2 - 2.0 * ri * cosmin[gi]).sum())

            # Delta_j(c) for all j,c -- the SAME function the self-test validates, not a copy.
            Dfl = delta_flip(AtR, Gjj, T, cstar)
            # decisive in-run check: flipping to the current class must change nothing, exactly.
            chk = float(Dfl.gather(1, cstar[:, None]).abs().max())
            scale = float(Dfl.abs().max())
            assert chk <= 1e-4 * max(scale, 1.0), f"Delta_j(c*) != 0 on real data: {chk:.3e}"
            # NOTE: rays the operator never touches contribute the same constant to every X, so they
            # cancel in Delta and are correctly excluded from `base` above.

            sel = has & live
            N = ((base + Dfl) <= delta2).sum(1).clamp_min(1)[sel].float()   # within delta (context)
            best = Dfl.argmin(1)
            misled = (best != cstar)[sel]
            frac_misled = float(misled.float().mean())
            bound_acc = 1.0 - frac_misled                    # the corrected bound (*)
            n_strictly_better = ((Dfl < -1e-9).sum(1))[sel].float()
            r_ = {"arm": arm, "scene": sc, "P": int(P), "C": Cc, "n_scored": int(sel.sum()),
                  "base_resid2": base, "delta2_labelfree": delta2,
                  "N_mean": float(N.mean()), "N_median": float(N.median()),
                  "frac_fully_ambiguous": float((N >= Cc).float().mean()),
                  "frac_unique": float((N <= 1).float().mean()),
                  "frac_misled": frac_misled,
                  "n_strictly_better_mean": float(n_strictly_better.mean()),
                  "bound_accuracy": bound_acc, "wall_s": round(time.time() - t0, 1)}
            out.append(r_); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] C={Cc} | MISLED {100*frac_misled:.1f}% of primitives "
                  f"(mean {r_['n_strictly_better_mean']:.2f} strictly-better wrong classes) | "
                  f"ACCURACY BOUND for residual minimisers {100*bound_acc:.1f}% | N_j within delta "
                  f"mean {r_['N_mean']:.2f}  {r_['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, AtR, Xs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
