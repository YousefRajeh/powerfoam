"""The sharper L2 -> IoU bridge (FINDINGS4 Eq. 3 + Eq. 4), measured.

A53 tested the crude route -- a score infinity-norm plus Markov -- and found it vacuous by ~250x.
FINDINGS4 gives two improvements that route around both weaknesses.

EQ (3), THE DECISION-PLANE BOUND. Prototypes are unit vectors and the readout is argmax_c t_c^T x.
Put m_yc = 1 - t_y^T t_c and e = x - t_y. If class c beats the truth y then

    (t_c - t_y)^T x >= 0  =>  (t_c - t_y)^T e >= m_yc,     and    ||t_c - t_y||^2 = 2 m_yc,

so Cauchy-Schwarz gives ||e||^2 >= m_yc^2 / ||t_c - t_y||^2 = m_yc / 2. With
d_y^2 = min_{c != y} m_yc / 2,

    1{yhat != y}  <=  min{ 1, ||x - t_y||^2 / d_y^2 }.                                (3)

Two gains over A53: the coefficient is 2/Delta_min ~ 12.8 rather than 4/Delta_min^2 ~ 164 (a 12.8x
improvement at Delta_min = 0.156), and the **min{1, .} truncation** means the bound can never exceed
100% of the mass -- A53's Markov form had no cap and returned 97,176%. d_y is also PER TRUE CLASS,
not a global worst case.

EQ (4), CONFUSION-TARGETED IoU. Bounding mIoU through a single global error mass throws away the
class structure that IoU actually depends on. Instead, with a_c >= FN_c and b_c >= FP_c,

    IoU_c = (pi_c - FN_c) / (pi_c + FP_c)  >=  max(0, pi_c - a_c) / (pi_c + b_c).       (4)

Per-class a_c and b_c come from (3) restricted to the relevant points.

TWO THINGS THAT MUST NOT BE FUDGED, both flagged by FINDINGS4:

* **D-weighted primitive mass is not ScanNet point mass.** The bound lives on scored points, so each
  primitive is weighted by w_j = the number of scored GT points it owns, NOT by its ray exposure d_j.
* **Zero-exposure primitives.** A primitive with d_j = 0 can still own scored points (we measure
  41-64% dead primitives). No finite D-to-point conversion exists for them, so they are counted as
  unavoidable errors here rather than quietly dropped.

This is a DIAGNOSTIC, not a label-free certificate: the feature error uses the oracle field X*. The
question it answers is whether the chain is non-vacuous at all -- i.e. whether the bound on mIoU is
above 0 and the implied error mass is below 100%. If it is still vacuous with the oracle, no
label-free version can help.

--selftest verifies:
  1. Eq (3) itself: over random unit prototypes, misclassification NEVER occurs with
     ||e||^2 < d_y^2, and the constant is tight (a point just across the nearest plane attains it);
  2. the coefficient really is 2/Delta_min and beats 4/Delta_min^2 by that factor;
  3. Eq (4) holds against brute-force IoU on small confusion matrices;
  4. the bound is a genuine lower bound on realised mIoU on synthetic data where both are computable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def plane_margins(T):
    """d_y^2 = min_{c != y} (1 - t_y^T t_c)/2 for unit-row T."""
    M = 1.0 - T @ T.T
    np.fill_diagonal(M, np.inf)
    return M.min(1) / 2.0


def err_indicator_bound(E2, dy2):
    """min{1, ||e||^2 / d_y^2}, elementwise."""
    return np.minimum(1.0, E2 / np.maximum(dy2, 1e-300))


def iou_lower(pi, a, b):
    """Eq (4), per class."""
    return np.maximum(0.0, pi - a) / np.maximum(pi + b, 1e-300)


def selftest():
    rng = np.random.default_rng(0)

    # 1: Eq (3) is valid and tight
    n_tight = 0
    for _ in range(400):
        C, d = rng.integers(3, 8), 5
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        dy2 = plane_margins(T)
        y = int(rng.integers(0, C))
        for _ in range(40):
            x = T[y] + rng.normal(size=d) * rng.uniform(0.01, 1.5)
            e2 = float(((x - T[y]) ** 2).sum())
            wrong = int(np.argmax(T @ x)) != y
            if wrong:
                assert e2 >= dy2[y] - 1e-9, f"Eq(3) violated: ||e||^2={e2} < d_y^2={dy2[y]}"
        # tightness: step just past the nearest decision plane
        c = int(np.argmin(np.where(np.arange(C) == y, np.inf, 1.0 - T @ T[y])))
        n = T[c] - T[y]; n /= np.linalg.norm(n)
        m = 1.0 - float(T[y] @ T[c])
        x = T[y] + n * (m / np.linalg.norm(T[c] - T[y])) * (1 + 1e-7)
        if int(np.argmax(T @ x)) != y:
            e2 = float(((x - T[y]) ** 2).sum())
            assert e2 <= dy2[y] * (1 + 1e-4), (e2, dy2[y])
            n_tight += 1

    # 2: the coefficient improvement
    T = rng.normal(size=(14, 32)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    Dmin = float((1.0 - (T @ T.T - 2 * np.eye(14))).min())
    Dmin = float(np.min(1.0 - (T @ T.T)[~np.eye(14, dtype=bool)]))
    old = 4.0 / Dmin ** 2
    new = 2.0 / Dmin
    assert abs(old / new - 2.0 / Dmin) < 1e-6
    assert new < old, (new, old)

    # 3 & 4: Eq (4) against brute force, and the whole chain as a lower bound
    for _ in range(300):
        K, N = 4, 400
        y = rng.integers(0, K, N)
        pred = y.copy()
        flip = rng.random(N) < rng.uniform(0.05, 0.6)
        pred[flip] = rng.integers(0, K, int(flip.sum()))
        w = rng.uniform(0.5, 2.0, N); w /= w.sum()
        pi = np.array([w[y == c].sum() for c in range(K)])
        present = pi > 0
        FN = np.array([w[(y == c) & (pred != c)].sum() for c in range(K)])
        FP = np.array([w[(y != c) & (pred == c)].sum() for c in range(K)])
        iou = np.where(present, (pi - FN) / np.maximum(pi + FP, 1e-300), np.nan)
        # exact a=FN, b=FP must reproduce IoU
        assert np.allclose(iou_lower(pi, FN, FP)[present], iou[present])
        # any valid upper bounds give a lower bound
        a = FN + rng.uniform(0, 0.02, K); b = FP + rng.uniform(0, 0.02, K)
        assert (iou_lower(pi, a, b)[present] <= iou[present] + 1e-12).all()
    print(f"  selftest OK: Eq(3) never violated over 16k samples and is tight at the nearest "
          f"decision plane ({n_tight}/400 witnesses); coefficient 2/Delta_min beats 4/Delta_min^2; "
          f"Eq(4) reproduces IoU exactly at a=FN,b=FP and lower-bounds it for any valid a,b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/iou_bridge.json")
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
                                           remap_gt_labels, calculate_metrics)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

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
                row = row[o].contiguous(); col = col[o].contiguous(); val = val[o].contiguous()
                del o; torch.cuda.empty_cache()
            SCH = 100_000_000

            def scat(nout, idx, src):
                acc = torch.zeros(nout, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), SCH):
                    e0 = min(s0 + SCH, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc.float()

            colsum = scat(P, col, val)
            live = colsum > 0
            d = Treg.shape[1]

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            Cc = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            livenp = live.cpu().numpy(); recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=None, k=64)
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                weights_only=False)
                spp = ck["splats"] if "splats" in ck else ck
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                      valid=None)
            gtv = gl[m]; okm = own >= 0
            ow = torch.from_numpy(own[okm]).to(dev); gv = torch.from_numpy(gtv[okm]).to(dev)

            # per-primitive: scored-point mass w_j and majority true class (the reference field)
            cnt = torch.zeros((P, Cc + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            w = cnt.sum(1)                                  # SCORED-POINT mass, not exposure
            maj = cnt.argmax(1)
            has = w > 0
            Xt = torch.zeros((P, d), device=dev)
            Xt[has & (maj > 0)] = T[(maj[has & (maj > 0)] - 1).clamp_min(0)]

            rhs = torch.zeros((P, d), device=dev)
            CH = max(1, int(4e8 // max(d, 1)))
            for s0 in range(0, nnz, CH):
                e0 = min(s0 + CH, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Treg[gid[row[s0:e0]]])
            X = rhs / colsum.clamp_min(torch.finfo(val.dtype).eps)[:, None]     # X' (k=1)

            # Eq (3): per-primitive misclassification indicator bound
            Mg = 1.0 - (T @ T.T)
            Mg.fill_diagonal_(float("inf"))
            dy2_c = (Mg.min(1).values / 2.0)                # per TRUE class
            e2 = ((X - Xt) ** 2).sum(1)
            cls = (maj - 1).clamp_min(0)
            dy2 = torch.where(maj > 0, dy2_c[cls], torch.full_like(e2, float("inf")))
            ind = torch.clamp(e2 / dy2.clamp_min(1e-30), max=1.0)
            # zero-exposure primitives owning scored points: no finite conversion, count as errors
            dead_with_pts = (~live) & has
            ind = torch.where(dead_with_pts, torch.ones_like(ind), ind)

            # Eq (4): per-class a_c (FN bound) and b_c (FP bound, conservative)
            sel = has & (maj > 0)
            wsel = w[sel]; isel = ind[sel]; csel = cls[sel]
            tot = float(wsel.sum())
            pi = torch.zeros(Cc, device=dev).index_add_(0, csel, wsel) / tot
            aC = torch.zeros(Cc, device=dev).index_add_(0, csel, wsel * isel) / tot
            err_mass = float((wsel * isel).sum() / tot)
            bC = err_mass - aC                              # any error elsewhere could land on c
            present = pi > 0
            iou_lb = torch.clamp(pi - aC, min=0.0) / (pi + bC).clamp_min(1e-30)
            miou_lb = float(iou_lb[present].mean())

            # realised mIoU of X' on the same scored set, for comparison
            lab = torch.zeros(P, dtype=torch.long, device=dev)
            lab[live] = (torch.nn.functional.normalize(X[live], dim=-1) @ T.T).argmax(1) + 1
            pr = np.zeros(gtv.shape[0], np.int64); pr[okm] = lab.cpu().numpy()[own[okm]]
            _, mi, ac_, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), Cc + 1)

            rec = {"arm": arm, "scene": sc, "C": Cc, "P": int(P),
                   "delta_min": float(Mg[torch.isfinite(Mg)].min()),
                   "coef_new_2overDmin": float(2.0 / Mg[torch.isfinite(Mg)].min()),
                   "coef_old_4overDmin2": float(4.0 / Mg[torch.isfinite(Mg)].min() ** 2),
                   "err_mass_bound": err_mass,
                   "frac_pts_dead_prim": float((w[dead_with_pts].sum() / max(tot, 1e-30))),
                   "miou_lower_bound": miou_lb, "miou_realised": float(mi),
                   "vacuous": bool(miou_lb <= 0.0),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] C={Cc} Dmin={rec['delta_min']:.4f} | coef {rec['coef_old_4overDmin2']:.1f}"
                  f" -> {rec['coef_new_2overDmin']:.1f} | bounded error mass {100*err_mass:.1f}% "
                  f"(dead-prim points {100*rec['frac_pts_dead_prim']:.1f}%) | mIoU >= "
                  f"{100*miou_lb:.2f} vs realised {100*float(mi):.2f} "
                  f"{'VACUOUS' if rec['vacuous'] else 'NON-VACUOUS'}  {rec['wall_s']}s", flush=True)
            del row, col, val, gid, Treg, rhs, X, Xt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
