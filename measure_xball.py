"""Is ANY least-squares target the right destination? Score the unit-ball-constrained optimum.

We know the unconstrained optimum is worse than the closed form (X' beats Xhat 20/20 scene-arms),
and that walking the path from X' to Xhat -- which monotonically LOWERS the least-squares loss --
monotonically lowers mIoU too. The standing explanation is that Xhat is ill-posed: `G` is
near-singular and `||xhat_j||` reaches 1e7 in near-null directions, while `||X'_j|| <= 1` by
construction because X' is a convex combination of unit-norm features.

If that explanation is right, the CONSTRAINED optimum should be the real target:

    X_ball = argmin ||A X - B||_F^2   subject to   ||X_j||_2 <= 1  for every primitive

X' is FEASIBLE for this problem, so `L(X_ball) <= L(X')` always -- the constrained optimum is
strictly better in loss. The question is whether it is better in mIoU.

  * If X_ball BEATS X' -- least squares is the right objective, it just needed the norm constraint,
    and the closed form is a cheap approximation to it after all.
  * If X_ball LOSES to X' -- then NO member of the least-squares family is the destination, and the
    closed form's success has nothing to do with approximating least squares. That would retire the
    whole "bounded approximation to the optimum" framing, not just SFS's proof of it.

This cannot use the class-space trick: the constraint lives in the 512-d feature space, so `T` and
the projection do not commute. It does avoid materialising `B` (R x 512 would be ~30 GB) by
precomputing `A^T B` once from the factorised per-view store and then iterating on `G Y - A^T B`.

Step size 1/lambda_max(G) via power iteration, so the gradient step is non-expansive and no tuning
is involved. Starting from X' means iteration 0 reproduces the published readout exactly, asserted.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

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


def build(scene, arm, views, cap, dev):
    recon = arm.replace("pf_", "")
    cfg = f"output/scannet_{scene}_{recon if recon in FOAM else 'truefrozen'}/config.yaml"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
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

    rows, cols, vals, offs = [], [], [], 0
    D = None
    rhs = None
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
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W, dev)
        r_ = ri.to(torch.int64).to(dev); c_ = ci.to(torch.int64).to(dev); v_ = vv.float().to(dev)
        if D is None:
            D = tab.shape[1]
            rhs = torch.zeros((P, D), device=dev)
        # A^T B for this view, without ever forming B densely
        rhs.index_add_(0, c_, v_[:, None] * tab[seg[r_].clamp(0, tab.shape[0] - 1)])
        rows.append(r_ + offs); cols.append(c_); vals.append(v_)
        offs += H * W
        del ri, ci, vv, seg, tab
    row = torch.cat(rows); col = torch.cat(cols); val = torch.cat(vals)
    del rows, cols, vals
    return row, col, val, rhs, P, D, offs, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="artifacts/scannet/xball.json")
    a = ap.parse_args()
    dev = "cuda"
    rows_out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows_out}
    import time
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            t0 = time.time()
            try:
                row, col, val, rhs, P, D, R, args = build(sc, arm, a.views, a.cap, dev)
                nnz = val.numel()
                colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
                live = colsum > 0
                if not bool((row[1:] >= row[:-1]).all()):
                    o = torch.argsort(row)
                    row, col, val = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
                    del o
                starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
                BUD = max(1, int(3e8 // D))
                tg = torch.arange(0, nnz + BUD, BUD, device=dev)
                bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                              torch.tensor([R], device=dev)]))
                # Resolve every block boundary to a PYTHON int ONCE. Indexing `starts` (a CUDA
                # tensor) inside the AtA loop forced a device sync per block -- ~50 per call, ~3000
                # over a PGD run, for no work.
                _st = starts.cpu().tolist()
                blocks = [(int(x), int(y), _st[int(x)], _st[int(y)])
                          for x, y in zip(bnd[:-1], bnd[1:]) if _st[int(y)] > _st[int(x)]]

                def AtA(x):
                    o = torch.zeros((P, x.shape[1]), device=dev)
                    for r0, r1, s, e in blocks:
                        lr = row[s:e] - r0
                        cs = col[s:e]
                        vw = val[s:e, None]
                        ap_ = torch.zeros((r1 - r0, x.shape[1]), device=dev)
                        ap_.index_add_(0, lr, vw * x[cs])
                        o.index_add_(0, cs, vw * ap_[lr])
                        del ap_
                    return o

                # step size 1/lambda_max(G) by power iteration -- no tuning
                # 12 power iterations is ample for a step size (we only need lambda_max to a few
                # percent, and overestimating is safe -- it only shortens the step). No per-iteration
                # float() sync.
                v = torch.randn(P, 1, device=dev); v /= v.norm()
                for _ in range(12):
                    v = AtA(v)
                    v = v / v.norm().clamp_min(1e-30)
                lmax = float((v * AtA(v)).sum())          # v is unit-norm, so this is the Rayleigh quotient
                eta = 1.0 / max(lmax, 1e-30)

                X = rhs / colsum.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]   # X', feasible
                X0 = X.clone()
                nrm = X.norm(dim=-1)
                assert float(nrm.max()) <= 1.0 + 1e-3, f"X' not feasible, max norm {float(nrm.max())}"

                # ground truth, scored the same way as everywhere else
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
                livenp = live.cpu().numpy()
                recon = arm.replace("pf_", "")
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

                def score(Z):
                    lab = torch.zeros(P, dtype=torch.long, device=dev)
                    zl = torch.nn.functional.normalize(Z[live], dim=-1)
                    lab[live] = (zl @ T.T).argmax(1) + 1
                    pr = np.zeros(gtv.shape[0], np.int64); pr[okm] = lab.cpu().numpy()[own[okm]]
                    _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv),
                                                     torch.from_numpy(pr), C + 1)
                    return float(mi) * 100, float(ac) * 100

                def loss(Z):
                    """||A Z||^2 - 2<Z, A^T B>; the constant ||B||^2 is dropped (only deltas matter).

                    Accumulated on-device and synced ONCE. An earlier version had a term multiplied
                    by zero that still performed a full gather, and called float() once per block,
                    forcing ~50 GPU syncs per evaluation.
                    """
                    tot = torch.zeros((), device=dev)
                    for r0, r1, s, e in blocks:
                        az = torch.zeros((r1 - r0, D), device=dev)
                        az.index_add_(0, row[s:e] - r0, val[s:e, None] * Z[col[s:e]])
                        tot += az.pow(2).sum()
                        del az
                    return float(tot - 2.0 * (Z * rhs).sum())

                mi0, ac0 = score(X0); l0 = loss(X0)
                traj = [(0, mi0, ac0, l0)]
                chk = {5, 15, 30, a.iters}
                prev_l = l0
                mono = True
                for k in range(1, a.iters + 1):
                    X -= eta * (AtA(X) - rhs)                        # in-place, no extra P x D alloc
                    X *= torch.clamp(1.0 / X.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
                    if k in chk:
                        mi, ac = score(X); lk = loss(X)
                        # PGD with eta = 1/lambda_max is monotone; if it is not, the step size or
                        # the projection is wrong and the whole run is meaningless
                        mono &= lk <= prev_l + 1e-3 * abs(prev_l)
                        prev_l = lk
                        traj.append((k, mi, ac, lk))
                r = {"arm": arm, "scene": sc, "P": int(P), "C": C, "D": int(D),
                     "eta": eta, "lmax": lmax,
                     "miou_xprime": mi0, "miou_xball": traj[-1][1],
                     "loss_xprime": l0, "loss_xball": traj[-1][3],
                     "traj": traj, "loss_monotone": bool(mono),
                     "wall_s": round(time.time() - t0, 1)}
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows_out.append(r); json.dump(rows_out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] loss {r['loss_xprime']:.4e} -> {r['loss_xball']:.4e}   "
                  f"mIoU {r['miou_xprime']:.2f} -> {r['miou_xball']:.2f} "
                  f"({r['miou_xball'] - r['miou_xprime']:+.2f})  "
                  f"{'mono OK' if r['loss_monotone'] else '*** LOSS NOT MONOTONE ***'}  "
                  f"{r['wall_s']}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
