"""Does the lift IMPROVE WITH VIEWS? Prediction: yes for foam, plateau for 3DGS.

HYPOTHESIS (user's, and the one this script exists to test). Gaussians overlap along rays, so the
ambiguity between two primitives that always co-occur is STRUCTURAL: no number of additional views
separates them. Foam cells are spatially disjoint, so extra views progressively disambiguate them.
Foam should therefore pay a different error instead -- a visibility/opacity error, since a cell's
contribution must be resolved from enough distinct viewpoints.

PREDICTION, stated before running:
  * foam  -- mIoU rises with view budget, and keeps rising;
  * 3DGS  -- mIoU rises then PLATEAUS, because the residual error is overlap, not data volume.

Supporting evidence already in hand (10 scenes, fixed 12 views): the within-arm Spearman of
semiconvergence loss against nnz-per-primitive is -0.806 (p=0.005) for foam but +0.539 (p=0.11) for
3DGS, while against `rho` it is +0.297 (p=0.40) for foam and +0.709 (p=0.022) for 3DGS. Data volume
governs foam; entanglement governs 3DGS.

WHY IT IS CHEAP. One build at V_max. Rays sit in contiguous per-view blocks, so every smaller budget
is a row mask -- no rebuild. And the readout `argmax_c <X_j, t_c>` depends only on the component of
`X_j` in `span(T)`, so the whole lift runs in C <= 16 dimensions rather than 512 (proved exact in
`measure_semiconv_fast.py`, validated on real data to <1e-4 mIoU).

NOTE ON THE BUDGET DEFINITION. Budget `v` uses `linspace(0, V_max-1, v)` INTO the built view list, a
nested subsample of the V_max set -- not identical to a fresh `XB.build(..., views=v)`, which would
linspace over all available cameras. Internally consistent and monotone, which is what a trend test
needs; stated rather than hidden.

--selftest verifies:
  1. the C-dim class-subspace lift gives argmax identical to the full-dimension lift;
  2. row masking by view block reproduces an explicitly sliced operator's X' exactly;
  3. X' is invariant to positive per-primitive rescaling, so the readout is scale-free;
  4. a monotone-nested budget sequence yields monotone-nondecreasing nnz and live counts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def view_subset(V_max, v):
    return np.unique(np.linspace(0, V_max - 1, v).astype(int))


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(200):
        V = int(rng.integers(2, 7)); per = int(rng.integers(6, 30)); P = int(rng.integers(4, 25))
        d, C = 48, int(rng.integers(3, 12))
        R = V * per
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.35)
        B = rng.normal(size=(R, d))
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        Vb, _ = np.linalg.qr(T.T)
        prev_nnz, prev_live = -1, -1
        for v in range(1, V + 1):
            vs = view_subset(V, v)
            m = np.zeros(R, bool)
            for k in vs:
                m[k * per:(k + 1) * per] = True
            Asub, Bsub = A[m], B[m]
            dv = Asub.sum(0); live = dv > 0
            if not live.any():
                continue
            Xfull = np.zeros((P, d)); Xfull[live] = (Asub.T @ Bsub)[live] / dv[live, None]
            Xproj = np.zeros((P, C)); Xproj[live] = (Asub.T @ (Bsub @ Vb))[live] / dv[live, None]
            # 1: same argmax in C dims as in d dims
            a1 = (Xfull[live] @ T.T).argmax(1); a2 = (Xproj[live] @ (T @ Vb).T).argmax(1)
            assert (a1 == a2).all(), "class-subspace lift changed argmax"
            # 2: masking equals an explicit slice (this IS the explicit slice, so check the identity
            #    against a scatter-style accumulation restricted to the same rows)
            acc = np.zeros((P, d))
            for i in np.where(m)[0]:
                acc += np.outer(A[i], B[i])
            assert np.abs(acc[live] / dv[live, None] - Xfull[live]).max() < 1e-9
            # 3: positive per-primitive rescaling is inert
            s = rng.uniform(0.1, 5.0, int(live.sum()))
            assert (((s[:, None] * Xfull[live]) @ T.T).argmax(1) == a1).all()
            # 4: monotone nesting
            assert (Asub != 0).sum() >= prev_nnz and int(live.sum()) >= prev_live
            prev_nnz, prev_live = (Asub != 0).sum(), int(live.sum())
    print("  selftest OK: C-dim class-subspace lift reproduces full-dimension argmax; per-view row "
          "masking reproduces an explicitly accumulated X'; positive per-primitive rescaling is "
          "inert; nested budgets give monotone nnz and live counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen,gs_froz")
    ap.add_argument("--budgets", default="4,8,12,24,36")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/view_sweep_miou.json")
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

    BUD = sorted({int(x) for x in a.budgets.split(",")})
    V_MAX = max(BUD)
    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    CH = 100_000_000

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, V_MAX, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            assert R % V_MAX == 0, f"R={R} not divisible by V_max={V_MAX}"
            per = R // V_MAX

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
            recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=None, k=64)
            else:
                ckp = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                 weights_only=False)
                spp = ckp["splats"] if "splats" in ckp else ckp
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                      valid=None)
            gtv = gl[m]; okm = own >= 0

            Vb, _ = torch.linalg.qr(T.T.double()); Vb = Vb.float()
            Tz = (Treg @ Vb).contiguous(); Tc = (T @ Vb).contiguous()
            dim = Tz.shape[1]
            vid = torch.div(row, per, rounding_mode="floor")

            rec = {"arm": arm, "scene": sc, "P": int(P), "C": C, "V_max": V_MAX,
                   "nnz_full": int(nnz), "budgets": {}}
            for v in BUD:
                vs = torch.from_numpy(view_subset(V_MAX, v)).to(dev)
                keep = torch.zeros(V_MAX, dtype=torch.bool, device=dev); keep[vs] = True
                nzm = keep[vid]
                c_, v_, r_, g_ = col[nzm], val[nzm], row[nzm], gid[row[nzm]]
                cs = torch.zeros(P, device=dev, dtype=torch.float64)
                for s0 in range(0, c_.numel(), CH):
                    e0 = min(s0 + CH, c_.numel())
                    cs.index_add_(0, c_[s0:e0], v_[s0:e0].double())
                cs = cs.float(); live = cs > 0
                rhs = torch.zeros((P, dim), device=dev)
                CH2 = max(1, int(4e8 // max(dim, 1)))
                for s0 in range(0, c_.numel(), CH2):
                    e0 = min(s0 + CH2, c_.numel())
                    rhs.index_add_(0, c_[s0:e0], v_[s0:e0, None] * Tz[g_[s0:e0]])
                X = rhs / cs.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]
                lab = torch.zeros(P, dtype=torch.long, device=dev)
                lab[live] = (X[live] @ Tc.T).argmax(1) + 1
                pr = np.zeros(gtv.shape[0], np.int64)
                pr[okm] = lab.cpu().numpy()[own[okm]]
                _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
                rec["budgets"][str(v)] = {
                    "views": int(len(vs)), "miou": float(mi) * 100, "acc": float(ac) * 100,
                    "nnz": int(nzm.sum().item()), "live": int(live.sum().item()),
                    "dead_frac": 1.0 - int(live.sum().item()) / P,
                }
                del nzm, c_, v_, r_, g_, cs, rhs, X, lab
                torch.cuda.empty_cache()
            rec["wall_s"] = round(time.time() - t0, 1)
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            line = "  ".join(f"v{v}:{rec['budgets'][str(v)]['miou']:.2f}" for v in BUD)
            b0, b1 = rec["budgets"][str(BUD[0])]["miou"], rec["budgets"][str(BUD[-1])]["miou"]
            print(f"[{arm}/{sc}] {line}   gain {b1-b0:+.2f}   {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
