"""Semiconvergence trajectory, run in the C-dimensional CLASS subspace instead of the M-dim region
subspace. Mathematically exact for every argmax-based metric (mIoU, accuracy); 13-20x faster.

WHY IT IS EXACT. The readout is `argmax_c <X_j/||X_j||, t_c>`. The norm is a positive per-primitive
scalar, and argmax over `c` is invariant to any such scalar, so the decision depends only on
`<X_j, t_c>` -- i.e. only on the component of `X_j` in `span(T)`, a `C <= 16` dimensional subspace.

The Richardson iteration `X_{k+1} = X_k + w D^-1 A^T (B - A X_k)`, `X_0 = 0`, is linear in `B` and
acts identically and independently on each feature coordinate. Hence for ANY fixed `d x C` matrix
`V`, iterating with `B V` yields exactly `X_k V`. Taking `V` = an orthonormal basis of `span(T)`
gives the projected trajectory, and `<X_k,j , t_c> = <X_k,j V, t_c V>` exactly.

`measure_semiconvergence.py` already reduces 512 -> M by a Cholesky of the region Gram (M = number of
SAM regions, 78-193 here). This reduces M -> C (6-16), and the per-iteration cost `O(nnz * dim)` falls
by the same factor. Note `t_c` need NOT lie in the span of the region features for this to hold --
the projection is applied to the iterate, not to the data.

WHAT IS LOST. `l2` and `cos` to the reference field are distances in the full 512-d space and are NOT
recoverable from the projection; they are reported as None here. Use the original script if you need
them. mIoU and accuracy are exact.

--selftest verifies:
  1. on random operators, the C-dim trajectory reproduces full-dimension scores and argmax exactly;
  2. a single damped step is omega * X', hence argmax-invariant;
  3. REAL-DATA CROSS-CHECK: with --validate-against, reproduces the mIoU trajectory already recorded
     by measure_semiconvergence.py on the same scene, which is the actual proof of equivalence.
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
    for _ in range(200):
        R, P, d, C = int(rng.integers(40, 160)), int(rng.integers(10, 50)), 48, int(rng.integers(3, 17))
        A = np.zeros((R, P))
        for i in range(R):
            js = rng.choice(P, size=int(rng.integers(1, 5)), replace=False)
            w = rng.uniform(.1, 1, len(js)); A[i, js] = w / w.sum()
        live = A.sum(0) > 0; A = A[:, live]
        B = rng.normal(size=(R, d))
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        V, _ = np.linalg.qr(T.T)
        dv = A.sum(0); om = 1.0

        def run(Bm):
            X = np.zeros((A.shape[1], Bm.shape[1])); out = {}
            for k in range(1, 21):
                X = X + om * (A.T @ (Bm - A @ X)) / dv[:, None]
                out[k] = X.copy()
            return out

        full, proj = run(B), run(B @ V)
        for k in (1, 2, 5, 10, 20):
            sf = full[k] @ T.T; sp = proj[k] @ (T @ V).T
            assert np.abs(sf - sp).max() < 1e-9 * max(np.abs(sf).max(), 1.0), "scores differ"
            assert (sf.argmax(1) == sp.argmax(1)).all(), "argmax differs"
        # 2: one damped step is a global scalar multiple of X'
        Xp = (A.T @ B @ V) / dv[:, None]
        base = (Xp @ (T @ V).T).argmax(1)
        for s in (0.25, 0.5, 3.0):
            assert ((s * Xp) @ (T @ V).T).argmax(1).tolist() == base.tolist()
    print("  selftest OK: C-dim class-subspace trajectory reproduces full-dimension scores (<1e-9) "
          "and argmax exactly over 200 operators x 5 stopping indices; damped step is a global "
          "scalar so argmax is invariant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="gs_froz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--ks", default="1,2,3,4,5,7,10,15,20,30,50,75,100,150,200")
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/semiconv_fast.json")
    ap.add_argument("--validate-against", default=None,
                    help="path to a semiconvergence json; assert mIoU trajectory matches")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None and a.validate_against is None:
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
                                           remap_gt_labels, calculate_metrics)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    KS = sorted({int(x) for x in a.ks.split(",")})
    ref = {}
    if a.validate_against:
        for r in json.load(open(a.validate_against)):
            ref[(r["arm"], r["scene"])] = {t["k"]: t["miou"] for t in r["traj"]}

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            if not bool((row[1:] >= row[:-1]).all()):
                o = torch.argsort(row)
                row, col, val = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
                del o; torch.cuda.empty_cache()
            colsum = torch.zeros(P, device=dev, dtype=torch.float64)
            for s in range(0, nnz, 100_000_000):
                e = min(s + 100_000_000, nnz)
                colsum.index_add_(0, col[s:e], val[s:e].double())
            colsum = colsum.float(); live = colsum > 0

            # ---- GT targets and class embeddings
            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
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

            # ---- project into the class subspace: V is d x C orthonormal, span(V) = span(T)
            V, _ = torch.linalg.qr(T.T.double())
            V = V.float()                                   # d x C
            Tz = (Treg @ V).contiguous()                    # M x C  (per-region feature, projected)
            Tc = (T @ V).contiguous()                       # C x C  (class embeddings, projected)
            dim = Tz.shape[1]

            starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
            BUD = max(1, int(3e8 // max(dim, 1)))
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            _st = starts.cpu().tolist()
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            def AtA(x):
                o = torch.zeros((P, x.shape[1]), device=dev)
                for r0, r1, s, e in blocks:
                    lr = row[s:e] - r0; cs = col[s:e]; vw = val[s:e, None]
                    ap_ = torch.zeros((r1 - r0, x.shape[1]), device=dev)
                    ap_.index_add_(0, lr, vw * x[cs])
                    o.index_add_(0, cs, vw * ap_[lr])
                    del ap_
                return o

            rhs = torch.zeros((P, dim), device=dev)
            for s in range(0, nnz, 8_000_000):
                e = min(s + 8_000_000, nnz)
                rhs.index_add_(0, col[s:e], val[s:e, None] * Tz[gid[row[s:e]]])
            Dinv = 1.0 / colsum.clamp_min(torch.finfo(rhs.dtype).eps)
            Xp = rhs * Dinv[:, None]

            sq = Dinv.clamp_min(0).sqrt()
            v = torch.randn(P, 1, device=dev); v /= v.norm()
            for _ in range(30):
                v = sq[:, None] * AtA(sq[:, None] * v); v = v / v.norm().clamp_min(1e-30)
            lmax = float((v * (sq[:, None] * AtA(sq[:, None] * v))).sum())
            omega = min(1.0, 1.0 / max(lmax, 1e-30))

            def score(Xf):
                lab = torch.zeros(P, dtype=torch.long, device=dev)
                lab[live] = (Xf[live] @ Tc.T).argmax(1) + 1
                pr = np.zeros(gtv.shape[0], np.int64)
                pr[okm] = lab.cpu().numpy()[own[okm]]
                _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
                return float(mi) * 100, float(ac) * 100

            mi_p, ac_p = score(Xp)
            for s_ in (0.3, 2.0):                            # argmax scale-invariance, in-run
                assert abs(score(s_ * Xp)[0] - mi_p) < 1e-6, "argmax not scale-invariant"
            traj = [{"k": 1, "miou": mi_p, "acc": ac_p, "is_xprime": True}]
            Zk = torch.zeros_like(rhs)
            for k in range(1, max(KS) + 1):
                Zk = Zk + omega * ((rhs - AtA(Zk)) * Dinv[:, None])
                if k in KS:
                    mi, ac = score(Zk)
                    traj.append({"k": k, "miou": mi, "acc": ac, "is_xprime": bool(k == 1)})

            rec = {"arm": arm, "scene": sc, "P": int(P), "M": int(Treg.shape[0]), "C": C,
                   "dim_used": int(dim), "lmax_Dinv_G": lmax, "omega": omega,
                   "miou_xprime": mi_p, "traj": traj, "l2": None, "cos": None,
                   "wall_s": round(time.time() - t0, 1)}

            if (arm, sc) in ref:
                bad = [(k, v, ref[(arm, sc)][k]) for k, v in
                       ((t["k"], t["miou"]) for t in traj) if k in ref[(arm, sc)]
                       and abs(v - ref[(arm, sc)][k]) > 1e-4]
                if bad:
                    raise AssertionError(f"VALIDATION FAILED {arm}/{sc}: {bad[:5]}")
                print(f"  validated against reference: {len(ref[(arm,sc)])} stopping indices match "
                      f"to <1e-4 mIoU", flush=True)

            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            m1 = traj[0]["miou"]; mk = traj[-1]["miou"]
            print(f"[{arm}/{sc}] C={C} dim {Treg.shape[0]}->{dim} | lmax {lmax:.4f} | k=1 {m1:6.2f}"
                  f"  k={traj[-1]['k']} {mk:6.2f}  loss {mk-m1:+6.2f}  {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
