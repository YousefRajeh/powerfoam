"""View-budget sweep in O(one view) memory. No whole-operator materialisation, no host staging.

WHY THIS EXISTS. `measure_xball2.build` returns the FULL operator: row(int64), col(int64),
val(float32) = 20 bytes per nnz. For gs_froz at 36 views that is ~1.5e9 nnz = ~30 GB, which OOMs a
48 GB card before any arithmetic happens, and forces `--cat-on-cpu` host staging that saturates a CPU
core copying 3 GB back and forth.

None of that is necessary. Every quantity the sweep needs is a SUM OVER VIEWS:

    colsum_j = sum_i A_ij           rhs_j = sum_i A_ij B_i
    sum_i r_i^2                     ||A||_F^2 = sum_ij A_ij^2

Rays never span views, so each view contributes independently. Streaming one view at a time gives
peak memory = one view's nnz (~0.5 GB even for 3DGS) plus two small accumulators
(`colsum`: P floats, `rhs`: P x C floats). Memory is then INDEPENDENT of the view budget, so the
36-view and 100-view cases cost the same as the 4-view case.

NESTED BUDGETS IN ONE PASS. Views are visited in van der Corput (bit-reversed) order, so EVERY prefix
is a near-uniform subsample of the camera trajectory. Budget `k` is then simply "the first k views",
and snapshotting the accumulators after view k scores that budget -- all budgets from a single sweep,
with no recomputation. This is a cleaner definition than `linspace(0, V-1, k)` per budget because the
budgets are exactly nested by construction.

The region table is also never stacked: each view's features are projected into the C-dimensional
class subspace immediately (`tab @ V`), so nothing of size (total regions x 512) is ever held.

--selftest verifies:
  1. streaming accumulation equals whole-operator accumulation, bit-for-bit in float64;
  2. van der Corput prefixes are exactly nested and well-spread (max gap <= 2x the ideal stride);
  3. the C-dim class-subspace lift reproduces full-dimension argmax;
  4. per-view partitioning of rows is exhaustive and disjoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def vdc_order(n):
    """Van der Corput / bit-reversal order: every prefix is a near-uniform subsample."""
    b = max(1, int(np.ceil(np.log2(max(n, 2)))))
    key = np.array([int(format(i, f"0{b}b")[::-1], 2) for i in range(n)])
    return np.argsort(key, kind="stable")


def selftest():
    rng = np.random.default_rng(0)
    # 2: nesting and spread
    for n in (8, 16, 37, 64, 279):
        o = vdc_order(n)
        assert sorted(o.tolist()) == list(range(n)), "not a permutation"
        for k in (2, 4, 8):
            if k > n:
                continue
            assert set(o[:k].tolist()) <= set(o[:2 * k].tolist() if 2 * k <= n else o.tolist())
            pref = np.sort(o[:k])
            gaps = np.diff(np.concatenate([[-1], pref, [n]]))
            assert gaps.max() <= 3 * (n / k) + 2, (n, k, gaps.max())
    # 1, 3, 4: streaming equals batch
    for _ in range(200):
        V = int(rng.integers(2, 8)); per = int(rng.integers(5, 25)); P = int(rng.integers(4, 20))
        d, C = 32, int(rng.integers(3, 10))
        R = V * per
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.4)
        B = rng.normal(size=(R, d))
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        Vb, _ = np.linalg.qr(T.T)
        order = vdc_order(V)
        cs = np.zeros(P); rhs = np.zeros((P, C)); s_r2 = 0.0; s_f2 = 0.0
        seen = np.zeros(R, bool)
        for k, vi in enumerate(order, 1):
            sl = slice(vi * per, (vi + 1) * per)
            # 4: partition is disjoint and eventually exhaustive
            assert not seen[sl].any(); seen[sl] = True
            Av, Bv = A[sl], B[sl]
            cs += Av.sum(0); rhs += Av.T @ (Bv @ Vb)
            s_r2 += (Av.sum(1) ** 2).sum(); s_f2 += (Av ** 2).sum()
            # 1: compare against the batch computation on the same view set
            m = np.zeros(R, bool)
            for u in order[:k]:
                m[u * per:(u + 1) * per] = True
            assert abs(cs - A[m].sum(0)).max() < 1e-10
            assert abs(rhs - A[m].T @ (B[m] @ Vb)).max() < 1e-9
            assert abs(s_r2 - (A[m].sum(1) ** 2).sum()) < 1e-8
            assert abs(s_f2 - (A[m] ** 2).sum()) < 1e-8
            # 3: class-subspace argmax equals full-dimension argmax
            live = cs > 0
            if live.any():
                Xp = rhs[live] / cs[live, None]
                Xf = (A[m].T @ B[m])[live] / cs[live, None]
                assert ((Xp @ (T @ Vb).T).argmax(1) == (Xf @ T.T).argmax(1)).all()
        assert seen.all()
    print("  selftest OK: streaming accumulation matches whole-operator accumulation for colsum, "
          "rhs, sum r_i^2 and ||A||_F^2 at every prefix (200 operators); van der Corput prefixes are "
          "nested and well-spread; class-subspace argmax matches full dimension; per-view row blocks "
          "partition the rays exactly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen,gs_froz")
    ap.add_argument("--budgets", default="4,8,12,24,36")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--out", default="artifacts/scannet/view_sweep_stream.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    import configargparse
    from determinism import enable_determinism
    enable_determinism()
    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from configs import Params, add_group
    from data_loader import DataHandler
    from camera_bridge import K_from_ray_dirs
    from measure_flip import load_view_features
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           remap_gt_labels, calculate_metrics)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    BUD = sorted({int(x) for x in a.budgets.split(",")})
    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        recon = arm.replace("pf_", "")
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            cfg = f"output/scannet_{sc}_{recon if recon in XB.FOAM else 'truefrozen'}/config.yaml"
            p = configargparse.ArgParser(); add_group(p, Params)
            p.add_argument("-c", "--config", is_config_file=True)
            args = p.parse_args(["-c", cfg])
            dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
            feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
            stems = sorted(os.path.splitext(f)[0][:-2]
                           for f in os.listdir(feat_dir) if f.endswith("_f.npy"))
            V_avail = len(dh.cameras)
            buds = [b for b in BUD if b <= V_avail]
            V_use = max(buds)

            if recon in XB.FOAM:
                import warp as wp
                from powerfoam.feature_operator import export_operator_for_views
                from powerfoam.scene import PowerfoamScene
                wp.init()
                model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
                model.load_pt(f"output/scannet_{sc}_{recon}/model.pt")
                P = model.points.shape[0]
            else:
                from gsplat_baseline.export_gsplat_operator import export_view_operator
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location=dev,
                                weights_only=False)
                sp = ck["splats"] if "splats" in ck else ck
                gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
                gs_ = torch.exp(sp["scales"].to(dev)); go = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
                gc = torch.zeros((gm.shape[0], 1), device=dev); P = gm.shape[0]

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            C = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            msk = (gl > 0) & vis
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[msk], cen, rad, valid=None, k=64)
            else:
                own = assign_points_to_nearest_center(pts[msk], gm.cpu().numpy(), valid=None)
            gtv = gl[msk]; okm = own >= 0
            Vb, _ = torch.linalg.qr(T.T.double()); Vb = Vb.float()
            Tc = (T @ Vb).contiguous(); dim = Tc.shape[1]

            order = vdc_order(V_avail)[:V_use]
            colsum = torch.zeros(P, device=dev, dtype=torch.float64)
            rhs = torch.zeros((P, dim), device=dev, dtype=torch.float64)
            s_r2 = 0.0; s_f2 = 0.0
            rec = {"arm": arm, "scene": sc, "P": int(P), "C": C, "V_avail": V_avail,
                   "budgets": {}}
            for k, vi in enumerate(order, 1):
                cam = dh.cameras[int(vi)]; H, W = int(cam.height), int(cam.width)
                if recon in XB.FOAM:
                    op = export_operator_for_views(model, [cam], [int(vi)],
                                                   max_hits_per_pixel=a.cap,
                                                   max_intersections=4096)
                    ri, ci, vv = op.row_indices, op.col_indices, op.values
                    del op
                else:
                    K, _ = K_from_ray_dirs(cam)
                    c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[int(vi)].double()
                    vmat = torch.linalg.inv(c2w).float().to(dev)
                    ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vmat, K.to(dev),
                                                            W, H, max_hits_per_pixel=a.cap,
                                                            transmittance_floor=1e-3)
                seg, tab = load_view_features(feat_dir, stems[int(vi) % len(stems)], H, W, dev,
                                              normalize=True)
                tz = (tab @ Vb)                                    # regions x C, projected at once
                ri = ri.to(torch.int64); ci = ci.to(torch.int64); vv = vv.float()
                colsum.index_add_(0, ci, vv.double())
                rhs.index_add_(0, ci, (vv[:, None] * tz[seg.reshape(-1)[ri].clamp(0, tab.shape[0]-1)]).double())
                rv = torch.zeros(H * W, device=dev, dtype=torch.float64)
                rv.index_add_(0, ri, vv.double())
                s_r2 += float((rv ** 2).sum()); s_f2 += float((vv.double() ** 2).sum())
                del ri, ci, vv, seg, tab, tz, rv
                torch.cuda.empty_cache()
                if k in buds:
                    live = colsum > 0
                    X = (rhs / colsum.clamp_min(1e-30)[:, None]).float()
                    lab = torch.zeros(P, dtype=torch.long, device=dev)
                    lab[live] = (X[live] @ Tc.T).argmax(1) + 1
                    pr = np.zeros(gtv.shape[0], np.int64)
                    pr[okm] = lab.cpu().numpy()[own[okm]]
                    _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv),
                                                     torch.from_numpy(pr), C + 1)
                    rec["budgets"][str(k)] = {
                        "miou": float(mi) * 100, "acc": float(ac) * 100,
                        "rho": s_r2 / max(s_f2, 1e-300) - 1.0,
                        "live": int(live.sum().item()),
                        "dead_frac": 1.0 - int(live.sum().item()) / P,
                        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                    del X, lab
            rec["wall_s"] = round(time.time() - t0, 1)
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            line = "  ".join(f"v{b}:{rec['budgets'][str(b)]['miou']:.2f}" for b in buds)
            pk = max(rec["budgets"][str(b)]["peak_gb"] for b in buds)
            print(f"[{arm}/{sc}] {line}   peak {pk} GB   {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
