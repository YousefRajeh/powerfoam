"""Semiconvergence: is the closed form X' the BOTTOM of the U, or just on its right side?

The lifting problem is a classical ill-posed inverse problem -- limited views, noisy observations
(region-CLIP is right 47.9% of the time), overlapping primitives -> `G = A^T A` ill-conditioned with
||xhat|| reaching 1e7 in near-null directions. In that regime the least-squares optimum is NOT the
estimator you want and truncated iteration IS the regulariser (Hansen; Engl-Hanke-Neubauer).

X' = D^-1 A^T B is exactly ONE preconditioned Richardson/SIRT step from zero:

    X_k = X_{k-1} + omega * D^-1 A^T (B - A X_{k-1}),   X_0 = 0
    X_1 = omega * D^-1 A^T B = omega * X'                  (so omega=1, k=1 IS X')

Every measurement we have walks X' -> Xhat, i.e. the RIGHT branch (k > 1), where things get worse.
Nobody walked the left branch. This does the whole curve and reports:

  * ||X_k - X_true||    the classical semiconvergence object. X_true is each primitive's TRUE class
                        embedding (majority GT class of the points it owns) -- the oracle field.
  * cosine error        scale-invariant version, matching what the readout actually consumes.
  * mIoU(X_k)           the decision-space quantity we care about.

A prediction worth stating before running: on the left branch a single damped step gives exactly
`omega * X'`, a GLOBAL positive scalar multiple, and argmax is invariant to that. So mIoU must be
FLAT for k=1 at any omega in (0,1]. If it is, then X' is the discrete minimum by default and the only
open question is whether k=2 beats k=1.

--selftest proves, on hand-computable inputs, before any GPU time:
  1. k=1, omega=1 from X_0=0 reproduces X' = D^-1 A^T B to machine precision.
  2. a single damped step is exactly omega*X' (hence argmax-invariant).
  3. iterating with a stable omega converges to Xhat = G^-1 A^T B (residual decreasing to ~0).
  4. the stability threshold omega < 2/lambda_max(D^-1 G) is real: just above it, divergence.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import torch


def _richardson(A, B, omega, K, record):
    """Reference (dense) implementation of the iteration the GPU path mirrors."""
    D = A.sum(0)                                     # diag(A^T 1)
    X = np.zeros((A.shape[1], B.shape[1]))
    out = {}
    for k in range(1, K + 1):
        X = X + omega * (A.T @ (B - A @ X)) / D[:, None]
        if k in record:
            out[k] = X.copy()
    return out


def selftest():
    rng = np.random.default_rng(0)
    m, n, d = 60, 12, 4
    A = np.abs(rng.normal(size=(m, n))) * (rng.random((m, n)) < 0.4)   # nonneg, sparse, overlapping
    A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(m, (A.sum(0) == 0).sum())))
    B = rng.normal(size=(m, d))
    D = A.sum(0); G = A.T @ A
    Xp = (A.T @ B) / D[:, None]                                        # the closed form

    # 1. k=1, omega=1 IS X'
    got = _richardson(A, B, 1.0, 1, {1})[1]
    assert np.abs(got - Xp).max() < 1e-12, np.abs(got - Xp).max()

    # 2. a single damped step is exactly omega * X'  -> a global scalar -> argmax invariant
    for om in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
        g = _richardson(A, B, om, 1, {1})[1]
        assert np.abs(g - om * Xp).max() < 1e-12
    T = rng.normal(size=(d, 5)); T /= np.linalg.norm(T, axis=0, keepdims=True)
    lab0 = (Xp @ T).argmax(1)
    for om in (0.1, 0.5, 2.0, 37.0):
        assert ((om * Xp) @ T).argmax(1).tolist() == lab0.tolist(), "scaling changed argmax"

    # 3. with a stable omega the iteration converges to Xhat = G^-1 A^T B
    Dh = np.diag(1 / np.sqrt(D))
    lmax = np.linalg.eigvalsh(Dh @ G @ Dh).max()
    om = 1.0 / lmax
    Xhat = np.linalg.solve(G, A.T @ B)
    ks = [1, 10, 100, 1000, 5000]
    it = _richardson(A, B, om, max(ks), set(ks))
    errs = [np.linalg.norm(it[k] - Xhat) for k in ks]
    # strict decrease only while above machine precision -- once converged the last digits wobble
    for i in range(len(errs) - 1):
        if errs[i] > 1e-12 * max(np.linalg.norm(Xhat), 1.0):
            assert errs[i + 1] < errs[i], (i, errs)
    assert errs[-1] / max(errs[0], 1e-30) < 1e-3, errs

    # 4. the stability threshold is real
    div = _richardson(A, B, 2.05 / lmax, 400, {400})[400]
    assert not np.isfinite(div).all() or np.linalg.norm(div) > 1e3 * np.linalg.norm(Xhat), \
        "expected divergence above 2/lambda_max"

    # 5. semiconvergence itself is reproducible in miniature: with a noisy B around a clean target,
    #    error to the TRUTH is U-shaped in k even though the residual falls monotonically.
    Xtrue = rng.normal(size=(n, d))
    Bc = A @ Xtrue; Bn = Bc + 0.5 * rng.normal(size=Bc.shape) * np.abs(Bc).mean()
    ks2 = list(range(1, 4000, 25))
    it2 = _richardson(A, Bn, om, max(ks2), set(ks2))
    e = np.array([np.linalg.norm(it2[k] - Xtrue) for k in ks2])
    r = np.array([np.linalg.norm(A @ it2[k] - Bn) for k in ks2])
    assert all(r[i + 1] <= r[i] + 1e-9 for i in range(len(r) - 1)), "residual must fall monotonically"
    assert e.argmin() not in (0, len(e) - 1), f"expected an interior minimum, got idx {e.argmin()}"
    print(f"selftest OK  (miniature semiconvergence: residual monotone, error to truth "
          f"minimised at k={ks2[int(e.argmin())]} of {ks2[-1]}, not at either end)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--ks", default="1,2,3,4,5,7,10,15,20,30,50,75,100,150,200")
    ap.add_argument("--omegas-left", default="0.25,0.5,0.75", help="single damped steps, k=1")
    ap.add_argument("--out", default="artifacts/scannet/semiconv.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    from determinism import enable_determinism
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           remap_gt_labels, calculate_metrics)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    KS = sorted({int(x) for x in a.ks.split(",") if x})
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True)
                continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, args = XB.build(sc, arm, a.views, a.cap, dev)
            nnz = val.numel(); M = Treg.shape[0]
            colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
            live = colsum > 0
            if not bool((row[1:] >= row[:-1]).all()):
                o = torch.argsort(row)
                row, col, val = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
                del o
            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            BUD = max(1, int(3e8 // M))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            _st = starts.cpu().tolist()
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            Gram = (Treg @ Treg.T).double()
            jit = 1e-6 * float(torch.diagonal(Gram).mean())
            Lf = torch.linalg.cholesky(
                Gram + jit * torch.eye(M, device=dev, dtype=torch.float64)).float()

            def AtA(x):
                o = torch.zeros((P, x.shape[1]), device=dev)
                for r0, r1, s, e in blocks:
                    lr = row[s:e] - r0; cs = col[s:e]; vw = val[s:e, None]
                    ap_ = torch.zeros((r1 - r0, x.shape[1]), device=dev)
                    ap_.index_add_(0, lr, vw * x[cs])
                    o.index_add_(0, cs, vw * ap_[lr])
                    del ap_
                return o

            rhs = torch.zeros((P, M), device=dev)
            for s in range(0, nnz, 8_000_000):
                e = min(s + 8_000_000, nnz)
                rhs.index_add_(0, col[s:e], val[s:e, None] * Lf[gid[row[s:e]]])
            Dinv = 1.0 / colsum.clamp_min(torch.finfo(rhs.dtype).eps)
            Xp_z = rhs * Dinv[:, None]                       # X' in Z-space

            # lambda_max of D^-1 G via the symmetric similar matrix D^-1/2 G D^-1/2
            sq = Dinv.clamp_min(0).sqrt()
            v = torch.randn(P, 1, device=dev); v /= v.norm()
            for _ in range(30):
                v = sq[:, None] * AtA(sq[:, None] * v)
                v = v / v.norm().clamp_min(1e-30)
            lmax = float((v * (sq[:, None] * AtA(sq[:, None] * v))).sum())
            omega = min(1.0, 1.0 / max(lmax, 1e-30))

            # ---- targets: each primitive's TRUE class = majority GT class of the points it owns
            d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            C = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            livenp = live.cpu().numpy(); recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=livenp, k=64)
            else:
                ckp = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                 weights_only=False)
                spp = ckp["splats"] if "splats" in ckp else ckp
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                      valid=livenp)
            gtv = gl[m]; okm = own >= 0
            ow = torch.from_numpy(own[okm]).to(dev)
            gv = torch.from_numpy(gtv[okm]).to(dev)
            cnt = torch.zeros((P, C + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            has = cnt.sum(1) > 0
            maj = cnt.argmax(1)                              # 1..C  (0 = unlabelled)
            tgt = has & live & (maj > 0)
            Xtrue = T[(maj[tgt] - 1).clamp_min(0)]           # unit-norm target per primitive

            def feats(Z_):
                return torch.linalg.solve_triangular(Lf.T, Z_.T, upper=True).T @ Treg

            def score_feat(Xf):
                lab = torch.zeros(P, dtype=torch.long, device=dev)
                lab[live] = (torch.nn.functional.normalize(Xf[live], dim=-1) @ T.T).argmax(1) + 1
                pr = np.zeros(gtv.shape[0], np.int64)
                pr[okm] = lab.cpu().numpy()[own[okm]]
                _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
                return float(mi) * 100, float(ac) * 100

            def errs(Xf):
                Xs = Xf[tgt]
                l2 = float((Xs - Xtrue).norm(dim=-1).mean())
                cos = float(1.0 - (torch.nn.functional.normalize(Xs, dim=-1) * Xtrue).sum(-1).mean())
                return l2, cos

            def resid(Z_):
                tot = torch.zeros((), device=dev)
                for r0, r1, s, e in blocks:
                    az = torch.zeros((r1 - r0, M), device=dev)
                    az.index_add_(0, row[s:e] - r0, val[s:e, None] * Z_[col[s:e]])
                    tot += az.pow(2).sum()
                    del az
                return float(tot - 2.0 * (Z_ * rhs).sum())

            # ---- IN-RUN MATH CHECKS against the real operator
            Z1 = omega * (rhs * Dinv[:, None])
            assert float((Z1 - omega * Xp_z).abs().max()) < 1e-4, "k=1 step is not omega * X'"
            mi_p, ac_p = score_feat(feats(Xp_z))
            for om in (0.3, 0.7, 2.0):
                mi_o, _ = score_feat(feats(om * Xp_z))
                assert abs(mi_o - mi_p) < 1e-6, f"argmax not scale-invariant at {om}"

            l2p, cosp = errs(feats(Xp_z))
            traj = [{"k": 1, "omega": 1.0, "miou": mi_p, "acc": ac_p, "l2": l2p, "cos": cosp,
                     "resid": resid(Xp_z), "is_xprime": True}]
            Zk = torch.zeros_like(rhs)
            for k in range(1, max(KS) + 1):
                Zk = Zk + omega * ((rhs - AtA(Zk)) * Dinv[:, None])
                if k in KS:
                    Xf = feats(Zk)
                    mi, ac = score_feat(Xf); l2, cs = errs(Xf)
                    traj.append({"k": k, "omega": omega, "miou": mi, "acc": ac, "l2": l2,
                                 "cos": cs, "resid": resid(Zk),
                                 "is_xprime": bool(k == 1 and omega == 1.0)})
                    del Xf
            r = {"arm": arm, "scene": sc, "P": int(P), "M": int(M), "C": C,
                 "lmax_Dinv_G": lmax, "omega": omega, "n_targets": int(tgt.sum()),
                 "miou_xprime": mi_p, "l2_xprime": l2p, "cos_xprime": cosp,
                 "traj": traj, "wall_s": round(time.time() - t0, 1)}
            out.append(r); json.dump(out, open(a.out, "w"), indent=1)
            bl = min(traj, key=lambda z: z["l2"]); bm = max(traj, key=lambda z: z["miou"])
            print(f"[{arm}/{sc}] lmax {lmax:.3f} omega {omega:.4f} | X' mIoU {mi_p:.2f} "
                  f"l2 {l2p:.4f} || best l2 @k={bl['k']} ({bl['l2']:.4f}) | "
                  f"best mIoU @k={bm['k']} ({bm['miou']:.2f})  {r['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, Zk, Xp_z
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
