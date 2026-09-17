"""Can delta be bounded label-free? Yes, rigorously -- the question is whether it is USEFUL.

FINDINGS3 (§C6) established that the framework we wanted already exists: the discrepancy principle
assumes only ||B - K Y_true|| <= delta with ARBITRARY error direction, so a deterministic semantic
bias is admissible. What we lack is a credible delta. It also warned that an uninformed bound
(||B_i - (A X_true)_i|| <= 1 + r_i ~ 2) can be satisfied at k = 0 and is therefore useless.

THE BOUND. X_true rows are unit class embeddings T_c, and row i of A is non-negative with
r_i = sum_j A_ij <= 1. So (A X_true)_i is a SUB-CONVEX combination of unit vectors:

    (A X_true)_i  in  r_i * conv{T_c}

The distance from a fixed point to a convex hull is maximised at a vertex, hence

    ||B_i - (A X_true)_i||  <=  max_c ||B_i - r_i T_c||

    delta^2  <=  sum_i [ ||B_i||^2 + r_i^2 - 2 r_i min_c cos(B_i, T_c) ]     (unit B_i)

Every quantity is observable: B is the data, r comes from the operator, T is the text head. No labels.
This is tight exactly when some class embedding sits maximally far from the observed feature in the
direction the ray actually mixes -- i.e. loose when the true class is NOT the worst-case class, which
is the normal situation.

THREE QUANTITIES, so the gap is visible rather than assumed:
  delta_lower  ||(I - A A^+) B||_F   the part of B no field can explain. Label-free. A valid LOWER
               bound on delta for every X, computed here via CG on the normal equations.
  delta_true   ||A X_true - B||_F    with the oracle field. Diagnostic only.
  delta_upper  the bound above.      Label-free, rigorous.

THE POINT OF delta_lower. delta^2 = ||Z_A B||^2 + ||A(Xhat - X_true)||^2 exactly, with Z_A = I-AA^+
and Xhat = A^+ B. The semantic bias lives in RANGE(A) and so contributes nothing to the first term.
That is precisely why a residual-based sigma was ~100x too small (A50): the residual cannot see a
bias that the operator can represent. Measuring lower/true/upper together shows how much of delta is
invisible to the residual.

--selftest checks, before any GPU time:
  1. the hull bound holds for random sub-convex combinations (never violated);
  2. it is attained when the true class IS the worst-case class (so the bound is not vacuous-by-slack);
  3. ||Z_A B|| <= ||B - A X|| for every X (lower-bound property);
  4. the exact split delta^2 = ||Z_A B||^2 + ||A(Xhat - X_true)||^2.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def delta_upper_rows(B, r, T):
    """sum_i max_c ||B_i - r_i T_c||^2, vectorised. B (R,d) unit rows, r (R,), T (C,d) unit rows."""
    cos = B @ T.T                                   # (R, C)
    return (B ** 2).sum(1) + r ** 2 - 2.0 * r * cos.min(1)


def selftest():
    rng = np.random.default_rng(0)
    worst_slack = np.inf
    for _ in range(300):
        R, P, C, d = 40, 6, 5, 7
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        r = A.sum(1)
        cls = rng.integers(0, C, P)
        Xt = T[cls]                                  # unit class embeddings per primitive
        B = rng.normal(size=(R, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)

        # 1. the hull bound is never violated
        actual = ((B - A @ Xt) ** 2).sum(1)
        ub = delta_upper_rows(B, r, T)
        assert (actual <= ub + 1e-9).all(), (actual - ub).max()
        worst_slack = min(worst_slack, float((ub - actual).min()))

        # 3. lower-bound property of the range-orthogonal part
        Ap = np.linalg.pinv(A)
        Z = np.eye(R) - A @ Ap
        lo = ((Z @ B) ** 2).sum()
        for _ in range(5):
            X = rng.normal(size=(P, d))
            assert lo <= ((B - A @ X) ** 2).sum() + 1e-9
        # 4. the exact split
        Xhat = Ap @ B
        assert abs(((B - A @ Xt) ** 2).sum()
                   - (lo + ((A @ (Xhat - Xt)) ** 2).sum())) < 1e-6

    # 5. THE WHITENING OPTIMISATION must not break the bound. Tw = (T T^T)^-1/2 T has orthonormal
    #    rows, so the projection is a partial isometry: it maps the sub-convex hull to a sub-convex
    #    hull and keeps B in the unit ball. The bound must survive, with ||B_i||^2 no longer 1.
    for _ in range(200):
        R, P, C, d = 30, 5, 6, 12
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        ev, V = np.linalg.eigh(T @ T.T)
        Tw = (V @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-8))) @ V.T) @ T
        assert np.abs(Tw @ Tw.T - np.eye(C)).max() < 1e-8, "whitening failed"
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        r = A.sum(1)
        Xt = T[rng.integers(0, C, P)]
        B = rng.normal(size=(R, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)
        Bp, Tp, Xtp = B @ Tw.T, T @ Tw.T, Xt @ Tw.T
        assert (np.linalg.norm(Bp, axis=1) <= 1.0 + 1e-9).all(), "projection left the unit ball"
        # projection commutes with the linear map: (A Xt) Tw^T == A (Xt Tw^T)
        assert np.abs((A @ Xt) @ Tw.T - A @ Xtp).max() < 1e-9
        # the bound still holds in the projected space
        act_p = ((Bp - A @ Xtp) ** 2).sum(1)
        ub_p = delta_upper_rows(Bp, r, Tp)
        assert (act_p <= ub_p + 1e-9).all(), (act_p - ub_p).max()
        # and the projected delta is <= the full-space delta (partial isometry contracts)
        assert act_p.sum() <= ((B - A @ Xt) ** 2).sum() + 1e-9

    # 2. attained when the true class is the worst-case class (single primitive, r=1)
    T = np.eye(3); A = np.ones((1, 1)); r = np.array([1.0])
    B = np.array([[1.0, 0.0, 0.0]])
    worst = int(np.argmin(B @ T.T))
    Xt = T[[worst]]
    assert abs(float(((B - A @ Xt) ** 2).sum()) - float(delta_upper_rows(B, r, T)[0])) < 1e-12
    print(f"  selftest OK: hull bound never violated over 300 operators (min slack "
          f"{worst_slack:.3e}); attained when the true class is worst-case; ||Z_A B|| is a valid "
          f"lower bound; exact split verified; WHITENING preserves the bound, commutes with A, "
          f"and contracts delta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cg-iters", type=int, default=120)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--tau", type=float, default=1.01)
    ap.add_argument("--out", default="artifacts/scannet/delta_bound.json")
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
            BUD = max(1, int(3e8 // max(Treg.shape[1], 1)))   # conservative: pre-projection width
            tg = torch.arange(0, nnz + BUD, BUD, device=dev)
            bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                          torch.tensor([R], device=dev)]))
            blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                      for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

            def AtA(X):
                out_ = torch.zeros_like(X)
                for r0, r1, s_, e_ in blocks:
                    lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                    t = torch.zeros((r1 - r0, X.shape[1]), device=dev)
                    t.index_add_(0, lr, vw * X[cs]); out_.index_add_(0, cs, vw * t[lr]); del t
                return out_

            def resid2(X):
                """||A X - B||_F^2, row-blocked; B row i = Treg[gid[i]], never materialised."""
                tot = 0.0
                for r0, r1, s_, e_ in blocks:
                    lr = row[s_:e_] - r0; cs = col[s_:e_]; vw = val[s_:e_, None]
                    t = torch.zeros((r1 - r0, d), device=dev)
                    t.index_add_(0, lr, vw * X[cs])
                    hit = torch.zeros(r1 - r0, dtype=torch.bool, device=dev); hit[lr] = True
                    idx = hit.nonzero(as_tuple=True)[0]
                    tot += float(((t[idx] - Treg[gid[idx + r0]]) ** 2).sum())
                    del t, hit, idx
                return tot

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            # WHITENED CLASS SPACE (d: 512 -> C). Tw has orthonormal rows, so the projection is a
            # partial isometry: B stays in the unit ball and the class embeddings still span a
            # sub-convex hull, so the bound below is unchanged in form. 27x cheaper.
            Mm = (T @ T.T).double(); ev, Vv = torch.linalg.eigh(Mm)
            Tw = ((Vv @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ Vv.T) @ T.double()).float()
            Treg = Treg @ Tw.T
            T = T @ Tw.T
            d = Treg.shape[1]
            rhs = torch.zeros((P, d), device=dev)
            for s0 in range(0, nnz, 8_000_000):
                e0 = min(s0 + 8_000_000, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Treg[gid[row[s0:e0]]])
            rvec = torch.zeros(R, device=dev).index_add_(0, row, val)
            cosmin = (Treg @ T.T).min(1).values                 # per REGION, then gathered per ray
            bn2 = (Treg ** 2).sum(1)
            hit = torch.zeros(R, dtype=torch.bool, device=dev); hit[row] = True
            idx = hit.nonzero(as_tuple=True)[0]
            gi = gid[idx]                          # gid is indexed by RAY, not by nnz position
            ri = rvec[idx]
            delta_up2 = float((bn2[gi] + ri ** 2 - 2.0 * ri * cosmin[gi]).sum())

            # lower bound ||Z_A B||^2 = ||B - A Xhat||^2, Xhat by CG on the normal equations
            X = torch.zeros((P, d), device=dev); rr = rhs.clone(); pp = rr.clone(); rz = (rr * rr).sum()
            for _ in range(a.cg_iters):
                Ap = AtA(pp); al = rz / (pp * Ap).sum().clamp_min(1e-30)
                X += al * pp; rr -= al * Ap; rz2 = (rr * rr).sum()
                if float(rz2.sqrt()) / max(float((rhs * rhs).sum().sqrt()), 1e-30) < 1e-7: break
                pp = rr + (rz2 / rz.clamp_min(1e-30)) * pp; rz = rz2
            delta_lo2 = resid2(X)

            # true delta with the oracle field
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
            Xtrue[has] = T[(maj[has] - 1).clamp_min(0)]     # already in whitened class space
            delta_tr2 = resid2(Xtrue)

            # ONE Richardson trajectory; every stopping index is then a lookup into this curve.
            # (An earlier version re-ran the whole trajectory per delta -- 4x the work for nothing.)
            Xk = torch.zeros((P, d), device=dev); rcurve = []
            for _ in range(a.kmax):
                Xk = Xk + (rhs - AtA(Xk)) * Dinv[:, None]
                rcurve.append(resid2(Xk))
            def stop_index(delta2):
                t2 = (a.tau ** 2) * delta2
                for k, rv in enumerate(rcurve, 1):
                    if rv <= t2:
                        return k
                return -1
            k_up = stop_index(delta_up2); k_tr = stop_index(delta_tr2)
            # DECISIVE CHECK ON REAL DATA: the three quantities must be ordered, or an optimisation
            # has broken the math. lower <= true by construction; true <= upper is the bound itself.
            assert delta_lo2 <= delta_tr2 * (1 + 1e-6) + 1e-9,                 f"lower bound violated: {delta_lo2:.6e} > {delta_tr2:.6e}"
            assert delta_tr2 <= delta_up2 * (1 + 1e-6) + 1e-9,                 f"HULL BOUND VIOLATED on real data: {delta_tr2:.6e} > {delta_up2:.6e}"
            # the residual curve must be non-increasing (Richardson on a consistent system)
            assert all(rcurve[i + 1] <= rcurve[i] * (1 + 1e-6) for i in range(len(rcurve) - 1)),                 "residual not monotone -- iteration or blocking is wrong"

            r_ = {"arm": arm, "scene": sc, "P": int(P), "rays_hit": int(idx.numel()), "d": int(d),
                  "delta_lower": delta_lo2 ** 0.5, "delta_true": delta_tr2 ** 0.5,
                  "delta_upper": delta_up2 ** 0.5,
                  "tightness_upper_over_true": (delta_up2 / max(delta_tr2, 1e-30)) ** 0.5,
                  "invisible_frac": 1.0 - delta_lo2 / max(delta_tr2, 1e-30),
                  "k_stop_upper": k_up, "k_stop_true": k_tr, "resid_curve": rcurve,
                  "wall_s": round(time.time() - t0, 1)}
            out.append(r_); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] delta lower {r_['delta_lower']:.1f} < true {r_['delta_true']:.1f} "
                  f"< upper {r_['delta_upper']:.1f} | upper/true {r_['tightness_upper_over_true']:.2f}x "
                  f"| residual-invisible {100*r_['invisible_frac']:.1f}% | k_stop upper={k_up} "
                  f"true={k_tr}  {r_['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, X, Xtrue
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
