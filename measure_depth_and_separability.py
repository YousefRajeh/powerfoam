"""Two things the current bound is missing: DEPTH ORDER, and an IRREDUCIBLE separability floor.

1. DEPTH-SPLIT CONTAMINATION. `G_jk = sum_i A_ij A_ik` is symmetric, so it cannot distinguish
   "primitive j was contaminated by an occluder IN FRONT of it" from "by something behind". But
   A_ij = alpha_j * T_j, and T_j is set entirely by what precedes j along the ray, so the physics
   is ordered: a low-opacity mug in front deposits mug features onto the table behind it, not the
   reverse. Splitting the off-diagonal mass by traversal order,

       G^front_j = sum_i sum_{s < t, col(t)=j} A_is A_it     (j contaminated from in front)
       G^back_j  = sum_i sum_{s < t, col(s)=j} A_is A_it     (j contaminated from behind)
       G^front_j + G^back_j = sum_{k != j} G_jk

   ties the bound to OPACITY, which is a controllable design parameter, rather than to a purely
   descriptive overlap count.

   ORDER IS VERIFIED, NOT ASSUMED. If slot order really is front-to-back then
   alpha_k = A_k / (1 - sum_{s<k} A_s) must lie in [0,1] for every hit; a scrambled order produces
   alpha > 1. This matters because the operator's first slot agrees with the rasteriser's
   `front_prim_idx` only 85-90% of the time on foam and 26-38% on the nonfrozen arm, so "slot 0 is
   the visible surface" is definitely false and "slots are depth-ordered" needs its own evidence.

2. SEPARABILITY FLOOR. Two primitives occupying the same place are indistinguishable from every
   view at once: that is null space, not blur, and no solver, no upstream fix and no amount of data
   recovers it. The current bound has no such term -- it says the excess shrinks with overlap, full
   stop. The normalised co-visibility correlation

       c_jk = G_jk / sqrt(G_jj G_kk)  in [0,1]   (Cauchy-Schwarz)

   is 1 exactly when columns j and k of A are proportional. Foam measures median 0.097 with 0.000%
   above 0.99 (`measure_gram_redundancy.py`); this runs the same measurement on 3DGS, which is the
   half that makes the partition-vs-overlap claim quantitative.

RAY SUBSAMPLING. 3DGS averages ~33 hits per ray, so pair enumeration is ~528 per ray and ~8e9 per
scene -- infeasible. Rays are subsampled uniformly at `--ray-frac`. c_jk is a RATIO in which
numerator and denominator scale together, so its distribution survives subsampling; the absolute
G mass does not, and is reported per sampled ray rather than as a total.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
from configs import Params, add_group
from data_loader import DataHandler
from diagnose_holes import SCENES


def verify_depth_order(row, col, val, dev, sample=200000):
    """alpha_k = A_k / (1 - sum_{s<k} A_s) must be in [0,1] if slots are front-to-back."""
    n = row.numel()
    if n == 0:
        return float("nan"), float("nan")
    order = torch.argsort(row, stable=True)
    r, v = row[order], val[order]
    cnt = torch.bincount(r, minlength=int(r.max()) + 1)
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, 0)[:-1]
    # within-ray EXCLUSIVE cumulative sum. The previous form indexed csum[starts[1:]-1], which
    # misaligns as soon as any ray has zero hits (starts repeats), and reported alpha up to 1.2e6.
    # Subtracting the running sum at each entry's own row start is index-safe.
    csum = torch.cumsum(v, 0)
    rs = starts[r]
    prev = csum - v - (csum[rs] - v[rs])
    T = 1.0 - prev
    # A SATURATED ray carries no information about ordering. Once the front primitives have
    # absorbed essentially all transmittance, T -> 0 and alpha = A/T is dominated by float
    # residue: clamping T at 1e-8 made a v~0.08 entry report alpha ~ 8e6, and 17% of entries
    # "violated" a bound that was never about depth. Those entries are EXCLUDED and counted,
    # rather than being allowed to masquerade as an ordering failure. The test is only
    # meaningful where there is transmittance left to divide by.
    ok = T > 1e-3
    excl = float((~ok).float().mean())
    if not bool(ok.any()):
        return float("nan"), float("nan"), excl
    a = v[ok] / T[ok]
    return float(a.max()), float((a > 1.0 + 1e-3).float().mean()), excl


def per_view(row, col, val, P, dev, pair_budget=1 << 22):
    """Depth-split off-diagonal mass, and (key, weight) pairs for the Gram."""
    order = torch.argsort(row, stable=True)
    row, col, val = row[order], col[order], val[order]
    R = int(row.max()) + 1
    cnt = torch.bincount(row, minlength=R)
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, 0)[:-1]
    gf = torch.zeros(P, device=dev, dtype=torch.float64)
    gb = torch.zeros(P, device=dev, dtype=torch.float64)
    keys, wts, cts = [], [], []
    for k in torch.unique(cnt):
        ki = int(k)
        if ki < 2:
            continue
        rows = (cnt == k).nonzero(as_tuple=True)[0]
        ii, jj = torch.triu_indices(ki, ki, offset=1, device=dev)   # ii < jj => ii is IN FRONT
        npair = ii.numel()
        step = max(1, pair_budget // max(npair, 1))
        ar = torch.arange(ki, device=dev)
        for s0 in range(0, rows.numel(), step):
            rr = rows[s0:s0 + step]
            idx = starts[rr][:, None] + ar[None, :]
            c = col[idx]
            v = val[idx]
            a, b = c[:, ii].reshape(-1), c[:, jj].reshape(-1)        # a in front of b
            w = (v[:, ii] * v[:, jj]).reshape(-1).double()
            gb.index_add_(0, a, w)      # a is contaminated from BEHIND by b
            gf.index_add_(0, b, w)      # b is contaminated from the FRONT by a
            lo = torch.minimum(a, b).long()
            hi = torch.maximum(a, b).long()
            keys.append(lo * P + hi)
            wts.append(w)
            cts.append(torch.ones_like(w))
            del idx, c, v, a, b, w, lo, hi
    if keys:
        kk = torch.cat(keys); ww = torch.cat(wts); cc = torch.cat(cts)
        uk, inv = torch.unique(kk, return_inverse=True)
        uw = torch.zeros(uk.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, ww)
        uc = torch.zeros(uk.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, cc)
        return gf, gb, uk, uw, uc
    z = torch.zeros(0, dtype=torch.long, device=dev)
    zf = torch.zeros(0, device=dev, dtype=torch.float64)
    return gf, gb, z, zf, zf


def run(scene, arm, n_views, cap, ray_frac, seed, min_pair_rays=5, dev="cuda"):
    is_gs = arm.startswith("gs")
    recon = "nonfrozen" if is_gs else arm
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    if is_gs:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        from camera_bridge import K_from_ray_dirs
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gsc = torch.exp(sp["scales"].to(dev))
        gop = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
        gcol = torch.zeros((gm.shape[0], 1), device=dev)
        P = gm.shape[0]
        model = None
    else:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
        P = model.points.shape[0]

    sel = np.linspace(0, len(dh.cameras) - 1, min(n_views, len(dh.cameras))).astype(int).tolist()
    GF = torch.zeros(P, device=dev, dtype=torch.float64)
    GB = torch.zeros(P, device=dev, dtype=torch.float64)
    GD = torch.zeros(P, device=dev, dtype=torch.float64)
    ak, aw, ac = None, None, None
    amax, abad, aexcl, nray_used, nnz_tot = 0.0, 0.0, 0.0, 0, 0
    g = torch.Generator(device=dev).manual_seed(seed)
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if is_gs:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gsc, gop, gcol, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=1e-3)
        else:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        ri, ci, vv = ri.to(torch.int64), ci.to(torch.int64), vv.float()
        if ray_frac < 1.0:
            keep_ray = torch.rand(H * W, device=dev, generator=g) < ray_frac
            m = keep_ray[ri]
            ri, ci, vv = ri[m], ci[m], vv[m]
            nray_used += int(keep_ray.sum())
        else:
            nray_used += H * W
        if ri.numel() == 0:
            continue
        nnz_tot += int(ri.numel())
        am, ab, aex = verify_depth_order(ri, ci, vv, dev)
        amax, abad = max(amax, am), max(abad, ab)
        aexcl = max(aexcl, aex)
        GD.index_add_(0, ci, (vv * vv).double())
        gf, gb, uk, uw, uc = per_view(ri, ci, vv, P, dev)
        GF += gf; GB += gb
        if uk.numel():
            if ak is None:
                ak, aw, ac = uk, uw, uc
            else:
                ck_ = torch.cat([ak, uk]); cw = torch.cat([aw, uw]); cn = torch.cat([ac, uc])
                ak, inv = torch.unique(ck_, return_inverse=True)
                aw = torch.zeros(ak.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, cw)
                ac = torch.zeros(ak.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, cn)
        del ri, ci, vv
    live = GD > 0
    off = GF + GB
    rho = (off / GD.clamp_min(1e-30))[live]
    fshare = (GF / off.clamp_min(1e-30))[live & (off > 0)]
    if ak is not None and ak.numel():
        # MINIMUM SUPPORT. A pair co-occurring on a single ray has c_jk = 1 by construction, so at
        # a low ray fraction most pairs are singletons and the c distribution is manufactured, not
        # measured -- a first run reported 1.72% of foam primitives above c = 0.99 where the full
        # cached gram says 0.000%. Pairs seen by fewer than `min_pair_rays` sampled rays are
        # dropped, and the dropped fraction is reported so the filter cannot hide the problem.
        keep_p = ac >= min_pair_rays
        frac_singleton = float((~keep_p).float().mean())
        ak, aw = ak[keep_p], aw[keep_p]
        lo, hi = torch.div(ak, P, rounding_mode="floor"), ak % P
        c = aw / (GD[lo] * GD[hi]).clamp_min(1e-30).sqrt()
        cmax = torch.zeros(P, device=dev, dtype=torch.float64)
        cmax.index_reduce_(0, lo, c, "amax", include_self=True)
        cmax.index_reduce_(0, hi, c, "amax", include_self=True)
        cm = cmax[live]
    else:
        c = cm = torch.zeros(0, device=dev, dtype=torch.float64)
        frac_singleton = float("nan")
    q = lambda t, x: float(torch.quantile(x.float(), t)) if x.numel() else float("nan")
    return dict(scene=scene, arm=arm, P=int(P), live=int(live.sum()), views=len(sel),
                rays_used=int(nray_used), nnz=int(nnz_tot),
                alpha_max=amax, alpha_violations=abad, alpha_excluded_saturated=aexcl,
                front_share=float(fshare.mean()) if fshare.numel() else float("nan"),
                rho_p50=q(.5, rho), rho_mean=float(rho.mean()) if rho.numel() else float("nan"),
                c_p50=q(.5, cm), c_p90=q(.9, cm), c_p99=q(.99, cm),
                frac_c_gt_90=float((cm > .90).float().mean()) if cm.numel() else float("nan"),
                frac_c_gt_99=float((cm > .99).float().mean()) if cm.numel() else float("nan"),
                n_pairs=int(ak.numel()) if ak is not None else 0,
                frac_pairs_dropped=frac_singleton, min_pair_rays=min_pair_rays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00")
    ap.add_argument("--arms", default="truefrozen,gs_froz")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--ray-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pair-rays", type=int, default=5,
                    help="drop pairs seen by fewer sampled rays; c_jk = 1 is trivial for a "
                         "singleton pair")
    ap.add_argument("--out", default="artifacts/scannet/depth_separability.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = run(sc, arm, a.views, a.cap, a.ray_frac, a.seed, a.min_pair_rays)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] depth-order alpha_max {r['alpha_max']:.3f} "
                  f"viol {r['alpha_violations']:.2%} (sat-excl {r['alpha_excluded_saturated']:.1%}) | front-share {r['front_share']:.3f} | "
                  f"rho p50 {r['rho_p50']:.3f} | c p50 {r['c_p50']:.3f} p99 {r['c_p99']:.3f} "
                  f">0.9 {r['frac_c_gt_90']:.2%} >0.99 {r['frac_c_gt_99']:.3%} "
                  f"({r['n_pairs']:,} pairs, {r['frac_pairs_dropped']:.1%} singletons dropped)",
                  flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
