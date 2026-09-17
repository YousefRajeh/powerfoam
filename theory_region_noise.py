"""Corrected solve theory: the noise is REGION-level, not ray-level.

Why the iid version fails. Every ray hitting a SAM region receives the IDENTICAL feature vector
(`B_i = Treg[gid_i]` exactly), so the per-ray noise is perfectly correlated within a region. Measured:
~15,000,000 rays against M = 78-213 distinct regions, i.e. **~78,000 rays per region**. Cov(vec N) is
not near-isotropic, it is RANK-DEFICIENT -- N lives in an (M d)-dimensional subspace of an
(R d)-dimensional one. The iid model therefore believes it has 15M independent measurements averaging
the noise away, which is why it predicts "keep iterating" (argmin k = 12, capped) while the empirical
optimum is k = 2, and why sigma^2 = RSS/(n_rays d) comes out ~100x too small.

The corrected model. Let S be the one-hot ray->region matrix and F = F_true + E the region features,
E rows iid with per-channel covariance sigma_reg^2 I_M. Then with W = A^T S and V = D^-1/2 W:

    g = D^-1/2 A^T S F = V F,      n = V E,      Cov(n) = sigma_reg^2 V V^T  per channel

Assuming the noiseless data is consistent (`V F_true = C Y_true`, Y = D^1/2 X), the iterates
Y_k = phi_k(C) C^+ g give

    e_k = -(I-C)^k Y_true  +  phi_k(C) C^+ n

    E||e_k||^2 = || (I-C)^k Y_true ||_F^2            [bias, decreasing in k]
               + d sigma_reg^2 || psi_k(C) V ||_F^2  [variance, increasing in k]

    psi_k(lam) = phi_k(lam)/lam = sum_{j<k} (1-lam)^j

**psi_k is a POLYNOMIAL, so C^+ disappears entirely.** Both terms are exact, need no CG, no
Hutchinson probes and no difference of large numbers -- the three things that broke the iid
estimator. Each is accumulated at ONE matvec per k:

    Q_k   = (I-C)^k V           one matvec from Q_{k-1}
    Psi_k = sum_{j<k} Q_j       running sum
    var_k = ||Psi_k||_F^2

V is P x M with M ~ 200, so the whole curve costs one operator pass over a P x 200 block.

sigma_reg^2 is NOT estimated from the ray residual (that is the quantity the iid model got wrong).
Instead the curve is reported as a function of sigma_reg, and we solve for the critical value at
which argmin_k moves to each k. That makes the claim falsifiable: the theory predicts k = k_emp iff
sigma_reg lies in a stated interval, which can be checked against region-feature variability.

--selftest verifies on dense synthetic data:
  1. psi_k(C) V computed incrementally == phi_k(C) C^+ V computed by dense pinv;
  2. the risk identity matches Monte-Carlo with genuinely REGION-correlated noise;
  3. the iid formula is WRONG under region noise (it under-predicts the variance term), which is the
     failure mode being corrected;
  4. argmin_k is monotone non-increasing in sigma_reg, so a critical sigma per k is well defined.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def curves_dense(C, V, Ytrue, K):
    """(bias_k, var_k) for k=1..K. var_k = ||psi_k(C) V||_F^2."""
    P = C.shape[0]
    bias, var = [], []
    Yk = Ytrue.copy()
    Q = V.copy()
    Psi = np.zeros_like(V)
    for _ in range(K):
        Yk = Yk - C @ Yk
        bias.append(float((Yk ** 2).sum()))
        Psi = Psi + Q
        Q = Q - C @ Q
        var.append(float((Psi ** 2).sum()))
    return np.array(bias), np.array(var)


def sigma_reg_hat(W, F):
    """Mass-weighted WITHIN-PRIMITIVE spread of the region features.

    A primitive hit by several regions gives several noisy observations of (ideally) the same
    underlying feature, so their weighted spread estimates sigma_reg^2 with no labels at all.
    For weights w and mean Fbar, E[sum_m w_m ||F_m - Fbar||^2] = sigma^2 d (sum w - sum w^2 / sum w),
    which fixes the degrees of freedom.

    UPPER BOUND, not an equality: a primitive straddling two objects has genuine signal spread as
    well as noise, which inflates it. That direction is the useful one -- an upper bound on sigma_reg
    that still fails to reach the k=1 threshold rules k=1 out.

    W is (P, M) mass, F is (M, d).
    """
    tot = W.sum(1)
    keep = tot > 0
    W = W[keep]; tot = tot[keep]
    Fbar = (W @ F) / tot[:, None]
    sq = (F ** 2).sum(1)
    num = float((W @ sq).sum() - (tot * (Fbar ** 2).sum(1)).sum())
    dof = float((tot - (W ** 2).sum(1) / tot).sum())
    return float(np.sqrt(max(num, 0.0) / max(dof * F.shape[1], 1e-30)))


def selftest():
    rng = np.random.default_rng(0)
    for trial in range(6):
        R, P, M, d, K = 200, 12, 6, 3, 10
        S = np.zeros((R, M)); S[np.arange(R), rng.integers(0, M, R)] = 1.0
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.6, 1.0, (R, 1))
        D = A.sum(0); G = A.T @ A
        Dh = np.diag(1 / np.sqrt(D)); Dhi = np.diag(np.sqrt(D))
        C = Dh @ G @ Dh
        W = A.T @ S; V = Dh @ W
        Cp = np.linalg.pinv(C)

        # 1. psi_k(C) V == phi_k(C) C^+ V
        Q = V.copy(); Psi = np.zeros_like(V)
        for k in range(1, K + 1):
            Psi = Psi + Q; Q = Q - C @ Q
            IC = np.linalg.matrix_power(np.eye(P) - C, k)
            phi_Cp = (np.eye(P) - IC) @ Cp
            assert np.abs(Psi - phi_Cp @ V).max() < 1e-6, (trial, k, np.abs(Psi - phi_Cp @ V).max())

        # 2. risk identity vs Monte-Carlo with REGION-correlated noise
        Ftrue = rng.normal(size=(M, d))
        Xt = np.linalg.pinv(A) @ (S @ Ftrue)          # consistent noiseless data
        Ytrue = Dhi @ Xt
        sig = 0.3
        bias, var = curves_dense(C, V, Ytrue, K)
        ex = bias + d * sig ** 2 * var
        NT = 3000
        acc = np.zeros(K)
        for _ in range(NT):
            F = Ftrue + sig * rng.normal(size=(M, d))
            B = S @ F
            Y = np.zeros((P, d))
            for k in range(K):
                Y = Y + (Dh @ (A.T @ B) - C @ Y)
                acc[k] += ((Y - Ytrue) ** 2).sum()
        mc = acc / NT
        rel = np.abs(mc - ex) / np.maximum(np.abs(ex), 1e-30)
        assert rel.max() < 0.08, (trial, rel.max(), mc[:3], ex[:3])

        # 3. the iid formula UNDER-predicts the variance here (the failure being corrected)
        lam = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
        var_iid = np.array([float(((1 - (1 - lam) ** k) ** 2 / lam).sum()) for k in range(1, K + 1)])
        assert var_iid[-1] < var[-1], (trial, var_iid[-1], var[-1])

        # 3b. the within-primitive estimator recovers sigma_reg when primitives are single-source
        Fn = Ftrue + sig * rng.normal(size=(M, d))
        est = sigma_reg_hat(W.copy(), Fn)            # W is already P x M
        # with genuine signal spread across regions this is an UPPER bound; with a constant true
        # field it must be tight
        Fc = np.tile(Ftrue[:1], (M, 1)) + sig * rng.normal(size=(M, d))
        tight = sigma_reg_hat(W.copy(), Fc)
        assert tight < sig * 1.5 and tight > sig * 0.5, (tight, sig)
        assert est >= tight - 1e-9

        # 4. argmin_k is monotone non-increasing in sigma
        prev = None
        for sg in np.geomspace(1e-3, 1e3, 40):
            am = int(np.argmin(bias + d * sg ** 2 * var)) + 1
            if prev is not None:
                assert am <= prev, (trial, sg, am, prev)
            prev = am
    print("  selftest OK: psi_k(C)V == phi_k(C)C^+V; risk matches region-correlated Monte-Carlo "
          "to <8%; iid formula under-predicts the variance; argmin monotone in sigma_reg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--out", default="artifacts/scannet/region_noise.json")
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
    from estimate_sigma_crit import _class_head

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
            sq = (1.0 / colsum.clamp_min(torch.finfo(val.dtype).eps)).sqrt()

            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            _st = starts.cpu().tolist()
            Mreg = int(Treg.shape[0])
            BUD = max(1, int(3e8 // max(Mreg, 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            def Cmul(X):
                out_ = torch.zeros_like(X)
                for r0, r1, s_, e_ in blocks:
                    lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                    t = torch.zeros((r1 - r0, X.shape[1]), device=dev)
                    Xs = sq[:, None] * X
                    t.index_add_(0, lr, vw * Xs[cs])
                    out_.index_add_(0, cs, vw * t[lr])
                    del t, Xs
                return sq[:, None] * out_

            # W = A^T S  (P x M): mass of each primitive on each region.  V = D^-1/2 W
            W = torch.zeros((P, Mreg), device=dev)
            for s0 in range(0, nnz, 8_000_000):
                e0 = min(s0 + 8_000_000, nnz)
                W.index_put_((col[s0:e0], gid[row[s0:e0]]), val[s0:e0], accumulate=True)
            V = sq[:, None] * W
            W_cpu = W.detach().cpu().numpy().astype(np.float64)
            del W

            # target: primitive's true class embedding, whitened class space
            Tcls = _class_head(sc, dev)
            Mm = (Tcls @ Tcls.T).double()
            ev, Vv = torch.linalg.eigh(Mm)
            Tw = ((Vv @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ Vv.T) @ Tcls.double()).float()
            d = Tw.shape[0]
            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            Tt = embed_class_names(kept, dev); Tt = Tt / Tt.norm(dim=-1, keepdim=True)
            Tt = Tt @ Tw.T
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
            cnt = torch.zeros((P, len(kept) + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            maj = cnt.argmax(1); has = (cnt.sum(1) > 0) & (maj > 0)
            Xtrue = torch.zeros((P, d), device=dev)
            Xtrue[has] = Tt[(maj[has] - 1).clamp_min(0)]
            Ytrue = Xtrue / sq.clamp_min(1e-30)[:, None]

            # ---- the two curves, exact, one matvec each per k
            bias, var = [], []
            Yk = Ytrue.clone(); Q = V.clone(); Psi = torch.zeros_like(V)
            for _ in range(a.kmax):
                Yk = Yk - Cmul(Yk); bias.append(float((Yk * Yk).sum()))
                Psi = Psi + Q; Q = Q - Cmul(Q); var.append(float((Psi * Psi).sum()))
            bias = np.array(bias); var = np.array(var)

            # ---- empirical curve, same metric
            Bt = Treg @ Tw.T
            rhs = torch.zeros((P, d), device=dev)
            for s0 in range(0, nnz, 8_000_000):
                e0 = min(s0 + 8_000_000, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Bt[gid[row[s0:e0]]])
            g = sq[:, None] * rhs
            Yk = torch.zeros_like(g); emp = []
            for _ in range(a.kmax):
                Yk = Yk + (g - Cmul(Yk)); emp.append(float(((Yk - Ytrue) ** 2).sum()))
            k_emp = int(np.argmin(emp)) + 1

            # ---- critical sigma_reg for each k, and the sigma that reproduces k_emp
            sig_grid = np.geomspace(1e-4, 1e3, 400)
            argmins = np.array([int(np.argmin(bias + d * s2 ** 2 * var)) + 1 for s2 in sig_grid])
            hit = np.where(argmins == k_emp)[0]
            k_reachable = sorted(set(int(x) for x in argmins))
            sig_lo = float(sig_grid[hit[0]]) if hit.size else float("nan")
            sig_hi = float(sig_grid[hit[-1]]) if hit.size else float("nan")
            hit1 = np.where(argmins == 1)[0]
            sig_k1 = float(sig_grid[hit1[0]]) if hit1.size else float("nan")

            sig_hat = sigma_reg_hat(W_cpu, Bt.detach().cpu().numpy().astype(np.float64))
            r = {"arm": arm, "scene": sc, "P": int(P), "M": Mreg, "d": int(d), "rays": R,
                 "sigma_reg_hat": sig_hat,
                 "k_predicted": int(np.argmin(bias + d * sig_hat ** 2 * var)) + 1,
                 "bias": bias.tolist(), "var": var.tolist(), "emp": emp,
                 "k_emp": k_emp, "sigma_reg_lo": sig_lo, "sigma_reg_hi": sig_hi,
                 "sigma_reg_k1": sig_k1, "k_reachable": k_reachable,
                 "k_emp_reachable": bool(hit.size), "wall_s": round(time.time() - t0, 1)}
            out.append(r); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] M={Mreg} | emp k={k_emp} | sigma_reg_hat={sig_hat:.4f} "
                  f"-> PREDICTED k={r['k_predicted']} | k={k_emp} needs sigma in "
                  f"[{sig_lo:.3g},{sig_hi:.3g}] | k=1 needs >={sig_k1:.3g}  {r['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, V, g, Ytrue, Xtrue
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
