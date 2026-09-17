"""Make the Theorem 2 stopping rule computable WITHOUT ground truth.

sigma_crit^2 = [ sum_m lam_m mu_m^2 (1+mu_m) s_m ] / [ d sum_m lam_m mu_m (mu_m+2) ],  mu = 1-lam

The earlier note called the full spectrum of an 81k x 81k operator a blocker. It is not, because
BOTH weight functions are polynomials in lambda, so every sum is a trace of a matrix POLYNOMIAL --
no Chebyshev approximation, no eigendecomposition:

    f_den(lam) = lam (1-lam)(3-lam)   = 3 lam - 4 lam^2 + lam^3
    f_num(lam) = lam (1-lam)^2 (2-lam)                                   (degree 4)
    p(lam)     = f_num(lam)/lam = (1-lam)^2 (2-lam)                      (degree 3, NO pole)

DENOMINATOR.  sum_m f_den(lam_m) = 3 tr C - 4 tr C^2 + tr C^3, with
    tr C   = sum_j G_jj / D_jj                      exact, closed form
    tr C^2 = ||C||_F^2 = sum_jk G_jk^2/(D_j D_k)    exact from the sparse Gram
    tr C^3 = E_z[ z^T C^3 z ]                       Hutchinson, 3 matvecs per probe
No inverse anywhere, so this half is unconditionally well behaved.

NUMERATOR.  Using E||g_m||^2 = lam_m^2 s_m + sigma^2 lam_m d with g = D^-1/2 A^T B:

    sum_m f_num(lam_m) s_m = sum_m p(lam_m) [ ||g_m||^2 / lam_m - sigma^2 d ]
                           = <g, p(C) C^+ g>  -  sigma^2 d tr p(C)

The only inverse left is a SINGLE pseudo-inverse applied to g, which CG already does (g lies in
range(C), and as lam->0 we have E||g_m||^2 -> sigma^2 lam d so the summand stays finite). p(C) is a
cubic, so p(C)C^+ g costs one CG solve plus three matvecs. That is the whole estimator -- the C^-2
worry was an artefact of not cancelling lambda first.

GROUND-TRUTH REFERENCE. With X_true known, sum_m f_num(lam_m) s_m = tr(Z^T f_num(C) Z) exactly,
Z = D^1/2 X_true, which is four matvecs on a P x d block. Used to validate the label-free estimate.

--selftest verifies on dense synthetic problems, where everything is computable exactly:
  1. the polynomial identities f_den, f_num, p against their factored forms;
  2. the trace formulas for tr C and tr C^2 against numpy;
  3. Hutchinson convergence for tr C^3;
  4. the GT numerator identity tr(Z^T f_num(C) Z) == sum_m f_num(lam_m) s_m;
  5. the label-free numerator against the GT numerator, in expectation over noise;
  6. sigma_crit from the estimator against sigma_crit from the dense spectrum;
  7. that the criterion sigma^2 >= sigma_crit^2 predicts argmin_k risk == 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------- polynomials
def f_den(l):
    return 3.0 * l - 4.0 * l ** 2 + l ** 3


def f_num(l):
    return l * (1.0 - l) ** 2 * (2.0 - l)


def p_poly(l):
    return (1.0 - l) ** 2 * (2.0 - l)


def _apply_poly(matvec, X, coeffs):
    """sum_k coeffs[k] C^k X, Horner in matvec form. coeffs[0] is the constant term."""
    out = np.zeros_like(X) if isinstance(X, np.ndarray) else X.new_zeros(X.shape)
    acc = X
    for k, c in enumerate(coeffs):
        if k > 0:
            acc = matvec(acc)
        if c != 0.0:
            out = out + c * acc
    return out


P_COEF = [2.0, -5.0, 4.0, -1.0]        # (1-l)^2 (2-l) = 2 - 5l + 4l^2 - l^3
FNUM_COEF = [0.0, 2.0, -5.0, 4.0, -1.0]  # l * p(l)


def _check_coeffs():
    l = np.linspace(0, 1, 101)
    assert np.allclose(sum(c * l ** k for k, c in enumerate(P_COEF)), p_poly(l))
    assert np.allclose(sum(c * l ** k for k, c in enumerate(FNUM_COEF)), f_num(l))
    assert np.allclose(f_den(l), l * (1 - l) * (3 - l))


# ---------------------------------------------------------------- selftest
def selftest():
    _check_coeffs()
    rng = np.random.default_rng(0)
    for trial in range(8):
        R, P, d = 80, 16, 4
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.6, 1.0, (R, 1))
        D = A.sum(0); G = A.T @ A
        Dh = np.diag(1 / np.sqrt(D)); Dhi = np.diag(np.sqrt(D))
        C = Dh @ G @ Dh
        lam, U = np.linalg.eigh(C); lam = np.clip(lam, 0, None)
        mv = lambda X: C @ X

        # 2. exact trace formulas
        assert abs((G.diagonal() / D).sum() - np.trace(C)) < 1e-9
        assert abs((G ** 2 / np.outer(D, D)).sum() - np.trace(C @ C)) < 1e-9

        # 3. Hutchinson for tr C^3
        tr3 = np.trace(C @ C @ C)
        est = np.mean([float(z @ (C @ (C @ (C @ z))))
                       for z in rng.choice([-1.0, 1.0], size=(4000, P))])
        assert abs(est - tr3) / max(abs(tr3), 1e-30) < 0.08, (est, tr3)

        # 4. GT numerator identity
        Xt = rng.normal(size=(P, d))
        Z = Dhi @ Xt
        s = ((U.T @ Z) ** 2).sum(1)
        lhs = float((Z * _apply_poly(mv, Z, FNUM_COEF)).sum())
        rhs = float((f_num(lam) * s).sum())
        assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-8, (lhs, rhs)

        # 5 & 6. label-free numerator, in expectation over noise
        sigma = 0.4
        NT = 3000
        acc = 0.0
        for _ in range(NT):
            B = A @ Xt + sigma * rng.normal(size=(R, d))
            g = Dh @ (A.T @ B)
            u = np.linalg.pinv(C) @ g
            acc += float((g * _apply_poly(mv, u, P_COEF)).sum())
        A_term = acc / NT
        num_lf = A_term - sigma ** 2 * d * float(p_poly(lam).sum())
        assert abs(num_lf - rhs) / max(abs(rhs), 1e-30) < 0.06, (num_lf, rhs)

        # 7. criterion predicts the argmin
        den = d * float(f_den(lam).sum())
        sc2 = rhs / den
        for sig, want in ((np.sqrt(sc2) * 0.5, False), (np.sqrt(sc2) * 2.0, True)):
            risk = [float((( 1 - lam) ** (2 * k) * s).sum()
                          + sig ** 2 * d * ((1 - (1 - lam) ** k) ** 2 / np.maximum(lam, 1e-300)).sum())
                    for k in range(1, 60)]
            assert (int(np.argmin(risk)) == 0) == want, (trial, sig, np.argmin(risk), want)
    print("  selftest OK: polynomial identities; exact tr C / tr C^2; Hutchinson tr C^3 <8%; "
          "GT numerator identity <1e-8; label-free numerator within 6% of GT; criterion predicts argmin")


# ---------------------------------------------------------------- real data
_HEAD = {}


def _class_head(scene, dev):
    """Unit-norm text embeddings for the classes present in this scene (cached)."""
    if scene in _HEAD:
        return _HEAD[scene]
    import numpy as _np, torch as _t, glob as _g, os as _o
    from diagnose_holes import GT_ROOT
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
    dd = [q for q in _g.glob(_o.path.join(GT_ROOT, "*", scene)) if _o.path.isdir(q)][0]
    _, raw, names = load_scannet_pointcept_gt(dd, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(_np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    T = embed_class_names(kept, dev)
    T = T / T.norm(dim=-1, keepdim=True)
    _HEAD[scene] = T
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--probes", type=int, default=32, help="Hutchinson probes for tr C^3")
    ap.add_argument("--project", default="whitened", choices=["whitened", "class", "feature"],
                    help="risk space. 'feature' is 512-d (27x cost). 'class' projects with T, which "
                         "is CHEAP BUT WRONG: T T^T has mean off-diagonal cosine 0.69 and cond 72, so "
                         "isotropic noise does NOT stay isotropic and Theorem 2's hypothesis breaks. "
                         "'whitened' uses Tw = (T T^T)^-1/2 T, whose rows are orthonormal, so noise "
                         "stays isotropic and the risk is the feature-space risk restricted to the "
                         "class subspace -- cheap AND valid.")
    ap.add_argument("--cg-iters", type=int, default=300)
    ap.add_argument("--out", default="artifacts/scannet/sigma_crit.json")
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
            order = torch.argsort(row)
            row, col, val = row[order].contiguous(), col[order].contiguous(), val[order].contiguous()
            colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
            live = colsum > 0
            Dinv = 1.0 / colsum.clamp_min(torch.finfo(val.dtype).eps)
            sq = Dinv.sqrt()

            # row-blocked A^T A: a full (R, d) temporary is ~30 GB at d=512, so chunk the rows
            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            _st = starts.cpu().tolist()
            BUD = max(1, int(3e8 // max(Treg.shape[1], 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            def Gmul(X):
                o = torch.zeros_like(X)
                for r0, r1, s_, e_ in blocks:
                    lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                    t = torch.zeros((r1 - r0, X.shape[1]), device=dev)
                    t.index_add_(0, lr, vw * X[cs])
                    o.index_add_(0, cs, vw * t[lr])
                    del t
                return o

            def Cmul(X):                       # C = D^-1/2 G D^-1/2
                return sq[:, None] * Gmul(sq[:, None] * X)

            # ---- denominator: exact tr C, tr C^2; Hutchinson tr C^3
            gdiag = torch.zeros(P, device=dev).index_add_(0, col, val * val)
            trC = float((gdiag * Dinv).sum())
            e = torch.zeros(P, 1, device=dev)
            # tr C^2 = ||C||_F^2 : accumulate over G's sparse structure via probes is costly, so use
            # the identity tr C^2 = E_z ||C z||^2 with Rademacher z (exact in expectation).
            zs = (torch.randint(0, 2, (P, a.probes), device=dev, dtype=torch.float32) * 2 - 1)
            Cz = Cmul(zs)
            trC2 = float((Cz * Cz).sum() / a.probes)
            C2z = Cmul(Cz)
            trC3 = float((Cz * C2z).sum() / a.probes)
            den_sum = 3.0 * trC - 4.0 * trC2 + trC3

            # ---- observations g = D^-1/2 A^T B
            # CLASS SPACE by default: B_c = B T^T, so d drops from 512 to C (~14-19). The operator,
            # and hence the entire spectrum {lambda_m}, is unchanged -- only s_m and d change, and
            # this is the risk in the space the argmax actually consumes.
            Tcls = _class_head(sc, dev)
            if a.project == "feature":
                Tproj = None
            elif a.project == "class":
                Tproj = Tcls
            else:
                M = (Tcls @ Tcls.T).double()
                ev, V = torch.linalg.eigh(M)
                Tproj = ((V @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ V.T) @ Tcls.double()).float()
                chk = float((Tproj @ Tproj.T - torch.eye(Tproj.shape[0], device=dev)).abs().max())
                assert chk < 1e-3, f"whitening failed: ||Tw Tw^T - I||_max = {chk:.2e}"
            Tab = Treg if Tproj is None else (Treg @ Tproj.T)
            d = Tab.shape[1]
            rhs = torch.zeros((P, d), device=dev)
            for s0 in range(0, nnz, 8_000_000):
                e0 = min(s0 + 8_000_000, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Tab[gid[row[s0:e0]]])
            g = sq[:, None] * rhs

            def poly(X, coef):
                o = torch.zeros_like(X); acc = X
                for k, c in enumerate(coef):
                    if k > 0:
                        acc = Cmul(acc)
                    if c != 0.0:
                        o = o + c * acc
                return o

            # ---- sigma^2 from the residual at k=1 (X' = D^-1 A^T B)
            Xp = rhs * Dinv[:, None] * sq[:, None]      # back to X-space: D^-1 A^T B
            # ||A X' - B||_F^2, row-blocked; B row i is Treg[gid[i]] and is never materialised whole
            resid = 0.0; nrays = 0
            for r0, r1, s_, e_ in blocks:
                lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                t = torch.zeros((r1 - r0, d), device=dev)
                t.index_add_(0, lr, vw * Xp[cs])
                hit = torch.zeros(r1 - r0, dtype=torch.bool, device=dev)
                hit[lr] = True
                idx = hit.nonzero(as_tuple=True)[0]
                resid += float(((t[idx] - Tab[gid[idx + r0]]) ** 2).sum())
                nrays += int(idx.numel())
                del t, hit, idx
            sigma2_hat = resid / max(nrays * d, 1)

            # ---- GT numerator (reference)
            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            Ccl = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            livenp = live.cpu().numpy(); recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=livenp, k=64)
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                weights_only=False)
                spp = ck["splats"] if "splats" in ck else ck
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                      valid=livenp)
            gtv = gl[m]; okm = own >= 0
            ow = torch.from_numpy(own[okm]).to(dev); gv = torch.from_numpy(gtv[okm]).to(dev)
            cnt = torch.zeros((P, Ccl + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            maj = cnt.argmax(1)
            Xtrue = torch.zeros((P, d), device=dev)
            has = (cnt.sum(1) > 0) & (maj > 0)
            Tt = T if Tproj is None else (T @ Tproj.T)
            Xtrue[has] = Tt[(maj[has] - 1).clamp_min(0)]
            Z = Xtrue / sq.clamp_min(1e-30)[:, None]      # D^1/2 X_true
            num_gt = float((Z * poly(Z, FNUM_COEF)).sum())

            # ---- label-free numerator: <g, p(C) C^+ g> - sigma^2 d tr p(C)
            u = torch.zeros_like(g); r_ = g.clone(); pdir = r_.clone()
            rz = (r_ * r_).sum()
            for _ in range(a.cg_iters):
                Ap = Cmul(pdir)
                al = rz / (pdir * Ap).sum().clamp_min(1e-30)
                u += al * pdir; r_ -= al * Ap
                rz2 = (r_ * r_).sum()
                if float(rz2.sqrt()) / max(float((g * g).sum().sqrt()), 1e-30) < 1e-7:
                    break
                pdir = r_ + (rz2 / rz.clamp_min(1e-30)) * pdir; rz = rz2
            A_term = float((g * poly(u, P_COEF)).sum())
            # tr p(C) RESTRICTED TO range(C). p(0) = 2, so every null mode would contribute 2 --
            # with P=81k and only ~61k live primitives that is tens of thousands of spurious units,
            # which is what drove the first estimate negative. Modes with lambda = 0 carry no data
            # and are excluded: sum_{lam>0} p = tr q(C) + 2 rank, q = p - 2 = -5l +4l^2 -l^3 (q(0)=0).
            n_live = int(live.sum())
            trq = -5.0 * trC + 4.0 * trC2 - trC3
            trp = trq + 2.0 * n_live                           # rank(C) approximated by n_live
            num_lf = A_term - sigma2_hat * d * trp

            # ---- DECISIVE CHECK: theory risk curve vs the EMPIRICAL one, same D-metric, same k.
            # bias_k = tr(Z^T (I-C)^2k Z)  -- exact, by repeated matvecs
            # var_k  = sigma^2 d tr(phi_k(C)^2 C^+); phi_k(l) = 1-(1-l)^k is divisible by l, so
            #          phi_k^2/l is a polynomial and the trace is Hutchinson-estimable.
            KMAX = 12
            W = Z.clone(); bias = []
            for _ in range(KMAX):
                W = W - Cmul(W); W = W - Cmul(W)          # (I-C)^2 per step
                bias.append(float((W * W).sum()))
            # var_k = sigma^2 d tr(phi_k(C)^2 C^+). phi_k(l) = 1-(1-l)^k is divisible by l, and
            # phi_k/l = psi_k(l) = sum_{j<k} (1-l)^j is a POLYNOMIAL -- so no inverse is needed.
            # C is symmetric, so z^T phi_k psi_k z = (phi_k z)^T (psi_k z), and BOTH are accumulated
            # incrementally at ONE matvec per k. (An earlier version ran a CG inside this loop:
            # ~1450 matvecs instead of 12, which is why it took minutes.)
            zc = zs
            Pk = zc.clone()                                # (I-C)^0 z
            psi = torch.zeros_like(zc)
            var = []
            for _ in range(KMAX):
                psi = psi + Pk                             # psi_k = sum_{j<k} (I-C)^j
                Pk = Pk - Cmul(Pk)                         # (I-C)^k
                phi = zc - Pk                              # phi_k(C) z
                var.append(float((phi * psi).sum() / zc.shape[1]))
            risk_theory = [bias[k] + sigma2_hat * d * var[k] for k in range(KMAX)]
            # empirical: run the actual Richardson iteration and measure ||X_k - X_true||_D^2
            Xk = torch.zeros_like(g); emp = []
            for k in range(KMAX):
                Xk = Xk + (g - Cmul(Xk))                   # Z-space Richardson, omega=1
                emp.append(float(((Xk - Z) ** 2).sum()))
            k_theory = int(np.argmin(risk_theory)) + 1
            k_emp = int(np.argmin(emp)) + 1

            sc2_gt = num_gt / max(d * den_sum, 1e-30)
            sc2_lf = num_lf / max(d * den_sum, 1e-30)
            r = {"arm": arm, "scene": sc, "P": int(P), "d": int(d), "rays": nrays,
                 "space": a.project,
                 "trC": trC, "trC2": trC2, "trC3": trC3, "den_sum": den_sum, "tr_p": trp,
                 "n_live": int(live.sum()), "A_term": A_term,
                 "sigma2_hat": sigma2_hat, "num_gt": num_gt, "num_lf": num_lf,
                 "sigma_crit2_gt": sc2_gt, "sigma_crit2_lf": sc2_lf,
                 "risk_theory": risk_theory, "risk_emp": emp,
                 "k_theory": k_theory, "k_emp": k_emp,
                 "k1_optimal_gt": bool(sigma2_hat >= sc2_gt),
                 "k1_optimal_lf": bool(sigma2_hat >= sc2_lf),
                 "wall_s": round(time.time() - t0, 1)}
            out.append(r); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] sigma2 {sigma2_hat:.4e} | sigma_crit2 GT {sc2_gt:.4e} "
                  f"LF {sc2_lf:.4e} | k=1 optimal? GT {r['k1_optimal_gt']} LF {r['k1_optimal_lf']} "
                  f"| ratio {sigma2_hat/max(sc2_gt,1e-30):.3f} | argmin_k THEORY {k_theory} "
                  f"EMPIRICAL {k_emp}  {r['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, g, Xtrue, Z
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
