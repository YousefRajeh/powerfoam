"""X_ball, converged: region-space change of variables + warm start + FISTA.

The first attempt did plain projected gradient in 512-d from X' and moved the loss by 0.04% in 60
iterations while still descending -- the "+0.10 mIoU" it reported described a point barely displaced
from X', not the constrained optimum. Three changes make it actually converge:

1. REGION SPACE, EXACTLY. Observations are per-SAM-region lookups, so `B = S T_reg` with `S` one-hot
   (R x M) and `T_reg` the stacked region features (M x 512), M = 78..247 rather than 512. Any
   iterate stays in `span(T_reg)`, so write `X = Y T_reg`. Then with `Gram = T_reg T_reg^T = L L^T`
   and `Z = Y L`:

       ||X_j||_2 = ||Y_j||_Gram = ||Z_j||_2            the ball constraint stays a BALL
       ||A X - B||_F^2 = ||A Z - S L||_F^2             the objective is unchanged

   so the whole problem is M-dimensional and the projection is still a simple rescale. `S L` is a
   row lookup because `S` is one-hot -- it is never materialised.

2. WARM START from the projected unconstrained optimum. `Xhat` is cheap by CG; `Xhat / max(1,||Xhat||)`
   is feasible and near the constrained optimum, so FISTA refines instead of crawling from X'.

3. FISTA rather than plain PGD: O(1/k^2) instead of O(1/k), one momentum term, no tuning.

Convergence is CHECKED, not assumed: the run reports the relative loss change over the last
checkpoint interval and refuses to call itself converged if the objective is still moving.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from camera_bridge import K_from_ray_dirs
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT, geometry
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
from measure_flip import load_view_features

FOAM = {"truefrozen", "nonfrozen"}


def build(scene, arm, views, cap, dev, feat_dirname="openclip_features_sam_l3",
          normalize_features=True):
    """Operator plus the factorised observations: seg ids per ray and the stacked region table."""
    recon = arm.replace("pf_", "")
    cfg = f"output/scannet_{scene}_{recon if recon in FOAM else 'truefrozen'}/config.yaml"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, feat_dirname)
    stems = sorted(os.path.splitext(f)[0][:-2] for f in os.listdir(feat_dir) if f.endswith("_f.npy"))

    if recon in FOAM:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
        P = model.points.shape[0]
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=dev, weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gs_ = torch.exp(sp["scales"].to(dev))
        go = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)
        P = gm.shape[0]

    rows, cols, vals, gid_all, tabs, offs, moff = [], [], [], [], [], 0, 0
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        if recon in FOAM:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=1e-3)
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W, dev,
                                      normalize=normalize_features)
        rows.append(ri.to(torch.int64).to(dev) + offs)
        cols.append(ci.to(torch.int64).to(dev))
        vals.append(vv.float().to(dev))
        gid_all.append(seg.clamp(0, tab.shape[0] - 1) + moff)     # global region id per ray
        tabs.append(tab)
        offs += H * W; moff += tab.shape[0]
        del ri, ci, vv
    return (torch.cat(rows), torch.cat(cols), torch.cat(vals),
            torch.cat(gid_all), torch.cat(tabs, 0), P, offs, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--cg-iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-5, help="relative loss change to call converged")
    ap.add_argument("--out", default="artifacts/scannet/xball2.json")
    a = ap.parse_args()
    from determinism import enable_determinism
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    dev = "cuda"
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            t0 = time.time()
            try:
                row, col, val, gid, Treg, P, R, args = build(sc, arm, a.views, a.cap, dev)
                nnz = val.numel(); M = Treg.shape[0]
                colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
                live = colsum > 0
                if not bool((row[1:] >= row[:-1]).all()):
                    o = torch.argsort(row)
                    row, col, val, = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
                    del o
                starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
                BUD = max(1, int(3e8 // M))
                tg = torch.arange(0, nnz + BUD, BUD, device=dev)
                bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                              torch.tensor([R], device=dev)]))
                _st = starts.cpu().tolist()
                blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                          for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

                # Gram = Treg Treg^T = L L^T; Z = Y L makes the ellipsoid a unit ball
                Gram = (Treg @ Treg.T).double()
                jit = 1e-6 * float(torch.diagonal(Gram).mean())
                L = torch.linalg.cholesky(Gram + jit * torch.eye(M, device=dev, dtype=torch.float64))
                Lf = L.float()

                def AtA(x):
                    o = torch.zeros((P, x.shape[1]), device=dev)
                    for r0, r1, s, e in blocks:
                        lr = row[s:e] - r0; cs = col[s:e]; vw = val[s:e, None]
                        ap_ = torch.zeros((r1 - r0, x.shape[1]), device=dev)
                        ap_.index_add_(0, lr, vw * x[cs])
                        o.index_add_(0, cs, vw * ap_[lr])
                        del ap_
                    return o

                # rhs in Z-space: A^T (S L) -- S is one-hot so S L is a row lookup
                SL = Lf                                            # (M, M): row m of S L is L[m]
                rhs = torch.zeros((P, M), device=dev)
                CH = 8_000_000
                for s in range(0, nnz, CH):
                    e = min(s + CH, nnz)
                    rhs.index_add_(0, col[s:e], val[s:e, None] * SL[gid[row[s:e]]])

                # unconstrained optimum by CG, then project -> warm start
                diag = torch.zeros(P, device=dev).index_add_(0, col, val * val)
                lam = 1e-6 * float(diag.mean()); Mp = (diag + lam).clamp_min(1e-30)
                Z = torch.zeros((P, M), device=dev)
                rr = rhs.clone(); zz = rr / Mp[:, None]; pd = zz.clone(); rz = (rr * zz).sum()
                r0n = float(rr.norm())
                for _ in range(a.cg_iters):
                    Ap = AtA(pd) + lam * pd
                    al = rz / (pd * Ap).sum().clamp_min(1e-30)
                    Z += al * pd; rr -= al * Ap
                    if float(rr.norm()) / max(r0n, 1e-30) < 1e-7:
                        break
                    zz = rr / Mp[:, None]; rz2 = (rr * zz).sum()
                    pd = zz + (rz2 / rz.clamp_min(1e-30)) * pd; rz = rz2

                Xp_z = rhs / colsum.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]   # X' in Z-space
                # CHECK the change of variables: ||Z_j|| must equal ||X_j|| for X' (<=1 by construction)
                Xp_feat = torch.linalg.solve_triangular(Lf.T, Xp_z.T, upper=True).T @ Treg
                dev_norm = float((Xp_z.norm(dim=-1) - Xp_feat.norm(dim=-1)).abs().max())
                assert dev_norm < 1e-2, f"change of variables wrong: norm dev {dev_norm:.3e}"

                v = torch.randn(P, 1, device=dev); v /= v.norm()
                for _ in range(12):
                    v = AtA(v); v = v / v.norm().clamp_min(1e-30)
                lmax = float((v * AtA(v)).sum()); eta = 1.0 / max(lmax, 1e-30)

                def proj(x):
                    return x * torch.clamp(1.0 / x.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)

                def loss(Z_):
                    tot = torch.zeros((), device=dev)
                    for r0, r1, s, e in blocks:
                        az = torch.zeros((r1 - r0, M), device=dev)
                        az.index_add_(0, row[s:e] - r0, val[s:e, None] * Z_[col[s:e]])
                        tot += az.pow(2).sum()
                        del az
                    return float(tot - 2.0 * (Z_ * rhs).sum())

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
                if recon in FOAM:
                    cen, rad, _ = geometry(sc, recon)
                    own = assign_points_to_power_cells(pts[m], cen, rad, valid=livenp, k=64)
                else:
                    ckp = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                     weights_only=False)
                    spp = ckp["splats"] if "splats" in ckp else ckp
                    own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                          valid=livenp)
                gtv = gl[m]; okm = own >= 0
                TT_z = torch.linalg.solve_triangular(Lf.T, (Treg @ T.T), upper=True)  # Z-space readout

                def score(Z_):
                    Xf = torch.linalg.solve_triangular(Lf.T, Z_.T, upper=True).T @ Treg
                    lab = torch.zeros(P, dtype=torch.long, device=dev)
                    lab[live] = (torch.nn.functional.normalize(Xf[live], dim=-1) @ T.T).argmax(1) + 1
                    pr = np.zeros(gtv.shape[0], np.int64); pr[okm] = lab.cpu().numpy()[own[okm]]
                    _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
                    return float(mi) * 100, float(ac) * 100

                mi_p, ac_p = score(Xp_z); l_p = loss(Xp_z)
                Zh = proj(Z); mi_h, ac_h = score(Z); mi_hp, ac_hp = score(Zh)
                # FISTA from the projected unconstrained optimum
                Zk = Zh.clone(); Yk = Zk.clone(); tk = 1.0
                traj = []; prev = loss(Zk); conv = False
                for k in range(1, a.iters + 1):
                    Zn = proj(Yk - eta * (AtA(Yk) - rhs))
                    tn = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5)
                    Yk = Zn + ((tk - 1.0) / tn) * (Zn - Zk)
                    Zk = Zn; tk = tn
                    if k % 50 == 0 or k == a.iters:
                        lk = loss(Zk); rel = abs(lk - prev) / max(abs(prev), 1e-30)
                        mi, ac = score(Zk); traj.append((k, mi, ac, lk, rel))
                        if rel < a.tol:
                            conv = True; prev = lk; break
                        prev = lk
                r = {"arm": arm, "scene": sc, "P": int(P), "M": int(M), "C": C,
                     "miou_xprime": mi_p, "miou_xhat": mi_h, "miou_xhat_proj": mi_hp,
                     "miou_xball": traj[-1][1] if traj else mi_hp,
                     "loss_xprime": l_p, "loss_xball": traj[-1][3] if traj else float("nan"),
                     "converged": bool(conv), "last_rel": traj[-1][4] if traj else float("nan"),
                     "iters_run": traj[-1][0] if traj else 0, "traj": traj,
                     "wall_s": round(time.time() - t0, 1)}
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            out.append(r); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] M={r['M']} mIoU X' {r['miou_xprime']:.2f} | Xhat {r['miou_xhat']:.2f} "
                  f"| Xhat_proj {r['miou_xhat_proj']:.2f} | X_ball {r['miou_xball']:.2f} "
                  f"({r['miou_xball'] - r['miou_xprime']:+.2f})  "
                  f"{'CONVERGED' if r['converged'] else 'NOT CONVERGED'} rel {r['last_rel']:.2e} "
                  f"@{r['iters_run']}it  {r['wall_s']}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
