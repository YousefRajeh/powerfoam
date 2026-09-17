"""Does the feature-space excess predict an ARGMAX FLIP? Solved in CLASS space, exactly.

THE QUESTION. `L(X') - L(Xhat) <= E_H(Xhat)` is exact in feature space, but mIoU only moves when the
READOUT changes: a large feature error that leaves argmax alone is free, a tiny one that crosses a
decision boundary costs a point. Every feature-space quantity we tried (E_H, gamma, gap/rays,
mean_o) predicts the scene-level mIoU solve gap at r ~ +0.27, i.e. not at all. The missing link is
whether it predicts the FLIP.

THE OPTIMISATION, AND WHY IT IS EXACT. The flip depends only on `argmax(x T^T)`. The text projection
acts on the RIGHT while `G^-1` acts on the LEFT, so they commute:

    Xhat T^T = (G^-1 A^T B) T^T = G^-1 A^T (B T^T)
    X'   T^T = (D^-1 A^T B) T^T = D^-1 A^T (B T^T)

so solving with `B_c = B T^T` gives the class-space solutions EXACTLY, with C = 6..19 right-hand
sides instead of 512. That is ~13x beyond the region-space reduction and ~50x beyond the original.
`compute_beta.py` cannot use it because beta and gamma need the full 512-d residual; the flip does
not.

It is also better aligned with the target: the per-primitive gap it produces is measured in the
space where the decision is actually made, rather than in a 512-d space whose norm mixes directions
the readout never looks at.

AUC rather than a correlation: `flip` is binary and the gap is heavy-tailed. AUC answers "is a
flipped primitive's gap larger than a non-flipped one's?" with no distributional assumption.
0.5 = no information.
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
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT

FOAM = {"truefrozen", "nonfrozen"}


def load_view_features(feat_dir, stem, H, W, dev, normalize=True):
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy")).astype(np.float32)
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[-1] if s.shape[0] <= 4 else s[..., -1]
    if s.shape != (H, W):
        from PIL import Image
        s = np.array(Image.fromarray(s.astype(np.int32)).resize((W, H), Image.NEAREST))
    t = torch.from_numpy(np.ascontiguousarray(f)).float().to(dev)
    # NOTE: region features are L2-normalised here, for EVERY caller. That is what makes
    # ||x'_j|| <= 1 (a convex combination of unit vectors) and is load-bearing for the unit-ball
    # bound. `normalize=False` is only for measuring what that normalisation is worth; it requires
    # a feature dir that actually stores un-normalised vectors (openclip_features_sam_l3_nonorm).
    if normalize:
        t = torch.nn.functional.normalize(t, dim=-1)
    return torch.from_numpy(np.ascontiguousarray(s)).long().reshape(-1).to(dev), t


def run(scene, arm, views, cap, cg_iters, dev="cuda"):
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    _, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    C = len(kept)
    T = embed_class_names(kept, dev)
    T = T / T.norm(dim=-1, keepdim=True)

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

    rows, cols, vals, segs, tabs, offs = [], [], [], [], [], 0
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
        rows.append(ri.to(torch.int64).to(dev) + offs); cols.append(ci.to(torch.int64).to(dev))
        vals.append(vv.float().to(dev))
        # project THIS view's region features to class scores once: (n_regions, C)
        segs.append(seg); tabs.append(tab @ T.T)
        offs += H * W
        del ri, ci, vv
    row = torch.cat(rows); col = torch.cat(cols); val = torch.cat(vals)
    del rows, cols, vals
    R, nnz = offs, val.numel()

    # B_c: per-ray class scores, gathered from the factorised per-view store
    Bc = torch.zeros((R, C), device=dev)
    base = 0
    for seg, tb in zip(segs, tabs):
        n = seg.numel()
        Bc[base:base + n] = tb[seg.clamp(0, tb.shape[0] - 1)]
        base += n
    del segs, tabs

    rhs = torch.zeros((P, C), device=dev)
    CH = 8_000_000
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        rhs.index_add_(0, col[s:e], val[s:e, None] * Bc[row[s:e]])
    colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
    diag = torch.zeros(P, device=dev).index_add_(0, col, val * val)
    live = colsum > 0

    if not bool((row[1:] >= row[:-1]).all()):
        o = torch.argsort(row); row, col, val = row[o].contiguous(), col[o].contiguous(), val[o].contiguous()
        del o
    starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
    BUD = max(1, int(3e8 // max(C, 1)))
    tg = torch.arange(0, nnz + BUD, BUD, device=dev)
    bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tg).clamp(0, R),
                                  torch.tensor([R], device=dev)]))
    blocks = [(int(x), int(y)) for x, y in zip(bnd[:-1], bnd[1:]) if int(y) > int(x)]

    def AtA(x):
        o = torch.zeros((P, x.shape[1]), device=dev)
        for r0, r1 in blocks:
            s, e = int(starts[r0]), int(starts[r1])
            if e <= s:
                continue
            lr = row[s:e] - r0
            ap = torch.zeros((r1 - r0, x.shape[1]), device=dev)
            ap.index_add_(0, lr, val[s:e, None] * x[col[s:e]])
            o.index_add_(0, col[s:e], val[s:e, None] * ap[lr])
            del ap
        return o

    lam = 1e-6 * float(diag.mean())
    # CG on (G + lam I) y = rhs, with C right-hand sides
    y = torch.zeros((P, C), device=dev)
    r_ = rhs.clone(); M = (diag + lam).clamp_min(1e-30)
    z = r_ / M[:, None]; pdir = z.clone(); rz = (r_ * z).sum()
    r0n = float(r_.norm())
    for _ in range(cg_iters):
        Ap = AtA(pdir) + lam * pdir
        al = rz / (pdir * Ap).sum().clamp_min(1e-30)
        y += al * pdir; r_ -= al * Ap
        if float(r_.norm()) / max(r0n, 1e-30) < 1e-6:
            break
        z = r_ / M[:, None]; rz2 = (r_ * z).sum()
        pdir = z + (rz2 / rz.clamp_min(1e-30)) * pdir; rz = rz2
    resid = float(r_.norm()) / max(r0n, 1e-30)

    xp = rhs / colsum.clamp_min(1e-30)[:, None]              # X' T^T, exactly
    dc = xp - y
    gap = torch.zeros(P, device=dev)
    for r0, r1 in blocks:
        s, e = int(starts[r0]), int(starts[r1])
        if e <= s:
            continue
        gap.index_add_(0, col[s:e], (val[s:e, None] * dc[col[s:e]]).pow(2).sum(-1))

    # ---- what does a flip COST? -------------------------------------------------------------
    # A flip is a changed DECISION; it only moves mIoU if the primitive owns scored points, and it
    # moves mIoU a lot only if those points belong to a rare class (class-averaging). So score the
    # GT points twice -- once under X' and once under Xhat -- and read the difference directly.
    # This is the REAL pipeline (features are the actual SAM+CLIP ones), so the delta is the mIoU
    # actually available from solving exactly instead of in closed form.
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
    from diagnose_holes import geometry as _geom
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    ptsxyz, _, _ = load_scannet_pointcept_gt(d, "segment20")
    msk = (gl > 0) & vis
    livenp = live.cpu().numpy()
    if recon in FOAM:
        cen, rad, _ = _geom(scene, recon)
        own = assign_points_to_power_cells(ptsxyz[msk], cen, rad, valid=livenp, k=64)
    else:
        cen = sp["means"].float().cpu().numpy()
        own = assign_points_to_nearest_center(ptsxyz[msk], cen, valid=livenp)
    gtv = gl[msk]

    lab_p = torch.zeros(P, dtype=torch.long, device=dev)
    lab_h = torch.zeros(P, dtype=torch.long, device=dev)
    lab_p[live] = xp[live].argmax(1) + 1
    lab_h[live] = y[live].argmax(1) + 1
    lpn, lhn = lab_p.cpu().numpy(), lab_h.cpu().numpy()
    okm = own >= 0
    pr_p = np.zeros(gtv.shape[0], np.int64); pr_p[okm] = lpn[own[okm]]
    pr_h = np.zeros(gtv.shape[0], np.int64); pr_h[okm] = lhn[own[okm]]
    from evaluate_point_cloud_miou import calculate_metrics as _cm
    _, mi_p, ac_p, _ = _cm(torch.from_numpy(gtv), torch.from_numpy(pr_p), C + 1)
    _, mi_h, ac_h, _ = _cm(torch.from_numpy(gtv), torch.from_numpy(pr_h), C + 1)
    # per-point outcome of the flips
    ch = pr_p != pr_h
    fixed = int(((pr_h == gtv) & (pr_p != gtv) & ch).sum())     # exact solve rescues the point
    broke = int(((pr_p == gtv) & (pr_h != gtv) & ch).sum())     # exact solve breaks it
    neutral = int((ch & (pr_p != gtv) & (pr_h != gtv)).sum())   # both wrong, different wrong
    cost = {"miou_xprime": float(mi_p) * 100, "miou_xhat": float(mi_h) * 100,
            "acc_xprime": float(ac_p) * 100, "acc_xhat": float(ac_h) * 100,
            "pts_scored": int(gtv.shape[0]), "pts_changed": int(ch.sum()),
            "pts_fixed": fixed, "pts_broken": broke, "pts_neutral": neutral}

    # ---- WALK THE LEAST-SQUARES PATH ---------------------------------------------------------
    # L((1-t)X' + t Xhat) is convex in t with its minimum at t=1, so moving along this path
    # MONOTONICALLY DECREASES the least-squares loss. If mIoU falls as we walk it, the objective
    # everyone optimises is pointing away from the task -- not merely a loose proxy for it.
    # Done in class space, which is exact for the readout (T and G^-1 commute).
    path = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        xt = (1.0 - t) * xp + t * y
        lt = torch.zeros(P, dtype=torch.long, device=dev)
        lt[live] = xt[live].argmax(1) + 1
        pr = np.zeros(gtv.shape[0], np.int64); pr[okm] = lt.cpu().numpy()[own[okm]]
        _, mi_t, ac_t, _ = _cm(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
        path.append((t, float(mi_t) * 100, float(ac_t) * 100))
    cost["ls_path"] = path

    lp, lh = xp[live].argmax(1), y[live].argmax(1)
    flip = lp != lh
    g = gap[live]
    nf, nn_ = int(flip.sum()), int((~flip).sum())
    auc = float("nan"); ratio = float("nan")
    if nf and nn_:
        rk = torch.argsort(torch.argsort(g.double())).double() + 1.0
        auc = float((rk[flip].sum() - nf * (nf + 1) / 2) / (nf * nn_))
        ratio = float(g[flip].mean() / g[~flip].mean().clamp_min(1e-30))
    return {"scene": scene, "arm": arm, "P": int(P), "live": int(live.sum()), "C": C,
            "rays": int(R), "nnz": int(nnz), "cg_resid": resid,
            "flip_frac": float(flip.float().mean()), "flip_n": nf,
            "flip_auc": auc, "flip_gap_ratio": ratio,
            "gap_total": float(gap.sum()), "gap_per_live": float(gap.sum() / max(int(live.sum()), 1)),
            **cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="pf_truefrozen,gs_froz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cg-iters", type=int, default=300)
    ap.add_argument("--out", default="artifacts/scannet/flip.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    import time
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            t0 = time.time()
            try:
                r = run(sc, arm, a.views, a.cap, a.cg_iters)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] C={r['C']} nnz {r['nnz']:,} resid {r['cg_resid']:.1e}  "
                  f"flip {r['flip_frac']:.2%}  AUC {r['flip_auc']:.3f}  "
                  f"ratio {r['flip_gap_ratio']:.2f} | mIoU X' {r['miou_xprime']:.2f} -> Xhat "
                  f"{r['miou_xhat']:.2f} ({r['miou_xhat'] - r['miou_xprime']:+.2f})  "
                  f"fixed {r['pts_fixed']:,} broke {r['pts_broken']:,}  {r['wall_s']}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
