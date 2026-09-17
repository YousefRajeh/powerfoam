"""Self-consistent oracle at FULL view budget: SFS Eq.6 / Eq.18 vs the streaming geometric median.

WHY THIS EXISTS. `test_consistent_oracle.py` concatenates every view's operator into one COO
before it solves, because the sphere-deconvolution arm needs A on every iteration. That caps the
view budget: foam at 38 views is already 90M nnz, and 3DGS averages far more hits per ray than
foam, so a 279-view splat scene is billions of nonzeros and cannot be held. The 10-scene oracle
was therefore run at 4 views -- and measured foam at 80.48 mIoU where 38 views gives 94.79. The
view budget, not the representation, was most of that gap.

Eq. 6, Eq. 18 and the streaming geometric median are all ONE-PASS: each folds a view in and never
looks at it again. Dropping the deconvolution arm buys the full view budget on both arms at
bounded memory -- one view's operator resident at a time, never the concatenation.

THE ORACLE. B = A Z T with Z the one-hot emission of each primitive's nearest labelled GT point,
so the exact solve is Z itself and scores 1.0 by construction (reported as the control). Any
shortfall is attributable to the lifting geometry alone, with the upstream held perfect.

GEOMETRIC MEDIAN. Reproduces `AccumulatedFeatureStats.accumulate_view`'s streaming Riemannian
update (VALA, arXiv:2509.05515) in class space: per view, each primitive's direction is
f = y/||y|| with y = sum_i A_ij M_i, and z <- Norm(z + eta (f - <f,z> z)), eta = w_new/(w_prev+w_new).
Because everything lives in span(T), norms and inner products carry the metric TT^T rather than
being Euclidean in C -- getting that wrong would silently score a different statistic.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT


def one(scene, recon, n_views, class_set, cap, squeeze=None, weiszfeld=False, dev="cuda"):
    is_gs = recon.startswith("gs_")
    cfg_recon = "nonfrozen" if is_gs else recon
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{cfg_recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    if is_gs:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        from camera_bridge import K_from_ray_dirs
        ckpt = torch.load(f"recon_remote/{recon}/{scene}/ckpt.pt", map_location=dev,
                          weights_only=False)
        sp = ckpt["splats"] if "splats" in ckpt else ckpt
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gsc = torch.exp(sp["scales"].to(dev))
        _ol = sp["opacities"].to(dev).reshape(-1)
        gop = torch.sigmoid(_ol * squeeze) if squeeze else torch.sigmoid(_ol)
        gcol = torch.zeros((gm.shape[0], 1), device=dev)
        centers = gm.detach().cpu().numpy()
        model = None
    else:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        ck = f"output/scannet_{scene}_{recon}"
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"{ck}/model.pt")
        centers = model.points.detach().cpu().numpy()
        radii = model.get_radii().detach().cpu().numpy()
    P = centers.shape[0]

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    if is_gs:
        from point_cloud_query import assign_points_to_nearest_center
        assigned = assign_points_to_nearest_center(gt_pts, centers, valid=None)
    else:
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=None, k=64)
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = votes.argmax(1).astype(np.int64)
    prim_gt[votes.max(1) == 0] = 0
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)

    from scipy.spatial import cKDTree
    lab_pts, lab_cls = gt_pts[gt_lab > 0], gt_lab[gt_lab > 0]
    _, nn = cKDTree(lab_pts).query(centers, k=1, workers=-1)
    emit = torch.from_numpy(lab_cls[nn].astype(np.int64)).to(dev)
    T = embed_class_names(kept, dev)
    TT = T @ T.T                                    # metric on span(T), and the readout

    ncam = len(dh.cameras)
    sel = (list(range(ncam)) if n_views <= 0 else
           np.linspace(0, ncam - 1, min(n_views, ncam)).astype(int).tolist())

    # ---- STREAMING ACCUMULATORS: one view resident at a time ----
    GZ = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
    GZ2 = torch.zeros(P, C, device=dev); D2 = torch.zeros(P, device=dev)
    gz = torch.zeros(P, C, device=dev); gw = torch.zeros(P, device=dev)
    s_o = s_ocross = s_den = 0.0
    R = nnz = 0

    def sph_norm(Y):
        return ((Y @ TT) * Y).sum(-1).clamp_min(0).sqrt()

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
        r_ = ri.to(torch.int64).to(dev); c_ = ci.to(torch.int64).to(dev); v_ = vv.float().to(dev)
        del ri, ci, vv
        nr = H * W; R += nr; nnz += v_.numel()

        M = torch.zeros(nr, C, device=dev)
        M.index_put_((r_, emit[c_] - 1), v_, accumulate=True)
        v2 = v_ * v_
        GZ.index_add_(0, c_, v_.unsqueeze(-1) * M[r_]);  D.index_add_(0, c_, v_)
        GZ2.index_add_(0, c_, v2.unsqueeze(-1) * M[r_]); D2.index_add_(0, c_, v2)

        rs = torch.zeros(nr, device=dev).index_add_(0, r_, v_)
        rq = torch.zeros(nr, device=dev).index_add_(0, r_, v2)
        s_o += float((rs * rs - rq).clamp_min(0).sum())
        s_ocross += float((rs * rs - (M * M).sum(1)).clamp_min(0).sum())
        s_den += float(rs.sum())

        # --- streaming Riemannian median: this view's direction per primitive ---
        yv = torch.zeros(P, C, device=dev).index_add_(0, c_, v_.unsqueeze(-1) * M[r_])
        wv = torch.zeros(P, device=dev).index_add_(0, c_, v_)
        seen = (wv > 0) & (sph_norm(yv) > 1e-20)
        fv = torch.zeros_like(yv)
        fv[seen] = yv[seen] / sph_norm(yv[seen]).unsqueeze(-1)
        init = seen & (gw <= 0); upd = seen & (gw > 0)
        gz[init] = fv[init]; gw[init] = wv[init]
        if upd.any():
            zp, wp_, wn = gz[upd], gw[upd], wv[upd]
            eta = (wn / (wp_ + wn)).clamp_max(1.0)
            cos = ((fv[upd] @ TT) * zp).sum(-1, keepdim=True)
            if weiszfeld:
                chord = (2.0 - 2.0 * cos).clamp_min(1e-8).sqrt()
                eta = (eta * (chord.median().clamp_min(1e-8) / chord).squeeze(-1)).clamp_max(1.0)
            zn = zp + eta[:, None] * (fv[upd] - cos * zp)
            n = sph_norm(zn).clamp_min(1e-30)
            gz[upd] = zn / n.unsqueeze(-1); gw[upd] = wp_ + wn
        del M, r_, c_, v_, v2, rs, rq, yv, wv, fv
        torch.cuda.empty_cache()

    live = D > 0
    Wp = torch.zeros(P, C, device=dev); Wp[live] = GZ[live] / D[live].unsqueeze(-1)
    Wt = torch.zeros(P, C, device=dev); Wt[live] = GZ2[live] / D2[live].clamp_min(1e-30).unsqueeze(-1)
    Z = torch.zeros(P, C, device=dev); Z[torch.arange(P, device=dev), emit - 1] = 1.0

    # POINT-LEVEL SCORING SET. Per-primitive mIoU is averaged over primitives, so an arm with more
    # of them is scored on a finer partition and gets an easier task for reasons that have nothing
    # to do with lifting quality: gs_unfroz carries 1,952,900 primitives against truefrozen's
    # 156,998, a 12.4x difference. Scoring instead at the GT POINTS puts every arm on the same
    # fixed sample with the same rule -- each labelled GT point takes the prediction of its nearest
    # LIVE primitive centre -- so primitive count no longer sets the denominator.
    from scipy.spatial import cKDTree as _KD
    live_np = live.cpu().numpy()
    live_idx = np.nonzero(live_np)[0]
    pt_nn = None
    if live_idx.size and lab_pts.shape[0]:
        _, _loc = _KD(centers[live_idx]).query(lab_pts, k=1, workers=-1)
        pt_nn = torch.from_numpy(live_idx[_loc].astype(np.int64)).to(dev)
        pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)

    def score(Wmat, sel_=None):
        s = live if sel_ is None else sel_
        pred = torch.zeros(P, dtype=torch.long, device=dev)
        pred[s] = (Wmat[s] @ TT).argmax(1) + 1
        acc = float((pred[s] == emit[s]).float().mean())
        _, miou, _, macc = calculate_metrics(emit[s].cpu(), pred[s].cpu(), C + 1)
        if pt_nn is None:
            return acc, float(miou), float(macc), float("nan"), float("nan")
        pp = pred[pt_nn]
        pacc = float((pp == pt_gt).float().mean())
        _, pmiou, _, _ = calculate_metrics(pt_gt.cpu(), pp.cpu(), C + 1)
        return acc, float(miou), float(macc), float(pmiou), pacc

    a_e, mi_e, ma_e, pmi_e, pa_e = score(Z)
    a_p, mi_p, ma_p, pmi_p, pa_p = score(Wp)
    a_t, mi_t, ma_t, pmi_t, pa_t = score(Wt)
    a_g, mi_g, ma_g, pmi_g, pa_g = score(gz)
    labelled = prim_gt_t > 0
    return dict(scene=scene, recon=recon, views=len(sel), P=int(P), rays=int(R), nnz=int(nnz),
                labelled_frac=float(labelled.float().mean()), live=int(live.sum()), C=C,
                n_gt_pts=int(lab_pts.shape[0]),
                mean_o=s_o / max(s_den, 1e-30), mean_o_cross=s_ocross / max(s_den, 1e-30),
                cross_share=s_ocross / max(s_o, 1e-30),
                acc_closed=a_p, miou_closed=mi_p, macc_closed=ma_p,
                acc_tikhonov=a_t, miou_tikhonov=mi_t, macc_tikhonov=ma_t,
                acc_geomed=a_g, miou_geomed=mi_g, macc_geomed=ma_g,
                acc_exact=a_e, miou_exact=mi_e, macc_exact=ma_e,
                # point-level: same GT sample and same rule for every arm
                pt_miou_closed=pmi_p, pt_acc_closed=pa_p,
                pt_miou_tikhonov=pmi_t, pt_acc_tikhonov=pa_t,
                pt_miou_geomed=pmi_g, pt_acc_geomed=pa_g,
                pt_miou_exact=pmi_e, pt_acc_exact=pa_e, squeeze=(squeeze or 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen,gs_froz")
    ap.add_argument("--views", type=int, default=-1, help="-1 = every camera in the scene")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--squeeze", type=float, default=None)
    ap.add_argument("--weiszfeld", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/oracle_stream.json")
    a = ap.parse_args()
    rows = []
    if os.path.exists(a.out):
        try:
            rows = json.load(open(a.out))
        except Exception:
            rows = []
    done = {(r["recon"], r["scene"]) for r in rows}
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            if (rec, sc) in done:
                print(f"[{rec}/{sc}] cached", flush=True)
                continue
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.squeeze, a.weiszfeld)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] v{r['views']} P {r['P']:,} nnz {r['nnz']/1e6:.0f}M "
                  f"o {r['mean_o']:.4f} || per-prim Eq18 {r['miou_tikhonov']*100:6.2f} "
                  f"(ctl {r['miou_exact']*100:6.2f}) || PT Eq6 {r['pt_miou_closed']*100:6.2f} "
                  f"Eq18 {r['pt_miou_tikhonov']*100:6.2f} GM {r['pt_miou_geomed']*100:6.2f} "
                  f"ctl {r['pt_miou_exact']*100:6.2f}", flush=True)
            torch.cuda.empty_cache()
    if rows:
        print("\n=== self-consistent oracle, full view budget ===")
        print(f"{'arm':<12}{'P':>10}{'mean o':>9}|{'ppEq18':>9}{'ppCtl':>8}|"
              f"{'ptEq6':>8}{'ptEq18':>8}{'ptGM':>8}{'ptCtl':>8}{'n':>4}")
        for rec in a.recons.split(","):
            rs = [r for r in rows if r["recon"] == rec]
            if not rs:
                continue
            f = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"{rec:<12}{f('P'):>10,.0f}{f('mean_o'):>9.4f}|"
                  f"{f('miou_tikhonov')*100:>9.2f}{f('miou_exact')*100:>8.2f}|"
                  f"{f('pt_miou_closed')*100:>8.2f}{f('pt_miou_tikhonov')*100:>8.2f}"
                  f"{f('pt_miou_geomed')*100:>8.2f}{f('pt_miou_exact')*100:>8.2f}{len(rs):>4}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
