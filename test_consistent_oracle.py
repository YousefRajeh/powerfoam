"""The theorem's prediction, isolated exactly: a SELF-CONSISTENT oracle upstream.

An earlier version built the oracle from the FRONT primitive of each ray -- "a perfect 2D model
labels a pixel with the visible surface". That is wrong for this purpose. The rendering model
composites everything the ray passes through, so the target consistent with the operator is the
alpha-composited mixture, not the first hit. Using the front surface introduced an error that had
nothing to do with the solve and everything to do with a heuristic.

Let Z be the (P, C) one-hot of per-primitive ground-truth classes and T the (C, F) text
embeddings, so the true feature field is Xtrue = Z T. Let each primitive emit its own class
embedding; then the rendered per-ray evidence is exactly

    B = A Z T          (the compositing the rasteriser already performs)

and therefore

    A^T B = A^T A Z T = G Z T   =>   Xhat = G^-1 A^T B = Z T = Xtrue    EXACTLY.

Three things follow, and they are what make this the right experiment:

  * the least-squares solve is perfect BY CONSTRUCTION -- no CG, no convergence residual, no
    rank-deficiency, and no ill-conditioned Xhat to contaminate the comparison;
  * the closed form's deviation is exactly the object the theorem bounds,
        X' - Xhat = D^-1 A^T B - Xhat = (D^-1 G - I) Z T = -D^-1 L Z T,
    which vanishes iff G = D, i.e. iff rays are disjoint;
  * every label error in X' is therefore attributable to OVERLAP ALONE.

EVERY PRIMITIVE EMITS, AND EVERY PRIMITIVE IS SCORED. Two earlier versions got this wrong in
turn. The first let only primitives CONTAINING a ground-truth point emit -- 49.9% of them on
truefrozen but 8.5% on nonfrozen -- so most of the nonfrozen operator's row mass vanished and its
measured overlap came out BELOW truefrozen's (0.083 vs 0.112), inverting the true ordering. The
second fixed emission but still SCORED only GT-containing primitives, which is worse: the
primitives a high-overlap representation adds are precisely the ones without their own
ground-truth point, so restricting the metric to the other 8.5% hides the cost that overlap is
supposed to impose.

Z is the SIGNAL, and the question is whether the lift recovers it. Each primitive is assigned the
class of its nearest ground-truth point, that field is rendered through the operator, and the
lift is scored against it on every primitive the renderer touches (D_jj > 0). The exact solve
then returns Z exactly, so the control is 100.00 by construction and every point of deficit in
the closed form is overlap.

SPLAT FEATURE SOLVER'S OWN SOLVER IS INCLUDED, not just the plain weighted mean. Their
Tikhonov Guidance is two separate things (see `splat-distiller`, and fp/lifting.py):

    Eq. 6    x_j = sum_i A_ij B_i / sum_i A_ij          the weighted mean  ("Eq6")
    Eq. 18   x_j = sum_i A_ij^2 B_i / sum_i A_ij^2      SQUARED weights    ("Eq18")
    Eq. 17   alpha~ = sigmoid(lambda * theta), lam=1.2  opacity squeeze    ("--squeeze")

Squaring the weights replaces the denominator D_jj = sum_i A_ij with G_jj = sum_i A_ij^2, so it
concentrates each primitive on its dominant contributor -- it is, in this paper's language, an
OVERLAP-REDUCTION heuristic, and the oracle setting measures exactly how much of the overlap
cost it recovers. The squeeze sharpens the opacities before rendering and so changes A itself;
it applies to the splat arms, where opacity is a stored logit.

Everything lives in C ~ 7-19 dimensions: with M = A Z (rays x C) the closed form scores are
    S' = (A^T M / D) (T T^T),      pred_j = argmax_c S'_{jc}
and the row normalisation of x'_j drops out of the argmax as a positive per-primitive scalar.
No 512-dimensional array is ever formed, and no linear system is ever solved.
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


def one(scene, recon, n_views, class_set, cap, squeeze=None, deconv_iters=12,
        deconv_damp=0.5, theta=1e9, step_mode="gershgorin", dev="cuda"):
    """`recon` is a PowerFoam arm (truefrozen / nonfrozen) or a splat arm (gs_froz / gs_unfroz).

    Both paths produce the same object: A, the renderer's compositing weights. For the foam that
    is the rasteriser's per-cell weights; for the splats it is alpha*T from the splat rasteriser
    -- the very quantity Splat Feature Solver's own closed form consumes. Nothing downstream
    knows or cares which renderer produced it.
    """
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
        # SFS Eq. 17: the squeeze acts on the raw stored logit, before activation
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
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)          # 0 = no GT point inside: SCORING set

    # EMISSION set: nearest labelled ground-truth point, defined for every primitive
    from scipy.spatial import cKDTree
    lab_pts = gt_pts[gt_lab > 0]
    lab_cls = gt_lab[gt_lab > 0]
    _, nn = cKDTree(lab_pts).query(centers, k=1, workers=-1)
    emit = torch.from_numpy(lab_cls[nn].astype(np.int64)).to(dev)
    T = embed_class_names(kept, dev)
    TT = T @ T.T

    # ---- operator A over an evenly spaced view subset ----
    sel = np.linspace(0, len(dh.cameras) - 1, n_views).astype(int).tolist()
    rows, cols, vals, offs = [], [], [], 0
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if is_gs:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gsc, gop, gcol, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=1e-3)
        else:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        rows.append(ri.to(torch.int64) + offs)
        cols.append(ci.to(torch.int64))
        vals.append(vv.float())
        offs += H * W
        del ri, ci, vv
    row = torch.cat(rows).to(dev); col = torch.cat(cols).to(dev); val = torch.cat(vals).to(dev)
    del rows, cols, vals
    R, nnz = offs, val.numel()

    labelled = prim_gt_t > 0                   # reported only, as a coverage statistic
    r_, c_, v_ = row, col, val                 # EVERY primitive emits, so A is used in full

    # M = A Z  (rays x C): the composited class mass each output element actually receives
    M = torch.zeros(R, C, device=dev)
    M.index_put_((r_, emit[c_] - 1), v_, accumulate=True)
    # A^T M = G Z  (P x C)
    GZ = torch.zeros(P, C, device=dev)
    GZ.index_add_(0, c_, v_.unsqueeze(-1) * M[r_])
    D = torch.zeros(P, device=dev).index_add_(0, c_, v_)

    # SFS Eq. 18, Tikhonov Guidance: the SOLVER weights by A_ij^2 instead of A_ij, which
    # replaces the denominator D_jj = sum_i A_ij with G_jj = sum_i A_ij^2. The evidence itself
    # is still rendered with the true weights -- only the solve changes.
    v2 = v_ * v_
    GZ2 = torch.zeros(P, C, device=dev)
    GZ2.index_add_(0, c_, v2.unsqueeze(-1) * M[r_])
    D2 = torch.zeros(P, device=dev).index_add_(0, c_, v2)

    # scored on EVERYTHING the renderer touches, against the signal that was injected
    live = D > 0
    Wp = torch.zeros(P, C, device=dev)
    Wp[live] = GZ[live] / D[live].unsqueeze(-1)            # X' in class space
    Wt = torch.zeros(P, C, device=dev)
    Wt[live] = GZ2[live] / D2[live].clamp_min(1e-30).unsqueeze(-1)   # Eq. 18, squared weights

    # overlap on the same operator, and its class-crossing part
    rs = torch.zeros(R, device=dev).index_add_(0, r_, v_)
    rq = torch.zeros(R, device=dev).index_add_(0, r_, v_ * v_)
    o_ray = (rs * rs - rq).clamp_min(0)
    o_cross = (rs * rs - (M * M).sum(1)).clamp_min(0)
    denom = float(rs.sum().clamp_min(1e-30))
    mean_o = float(o_ray.sum()) / denom
    mean_o_cross = float(o_cross.sum()) / denom
    cross_share = float(o_cross.sum()) / max(float(o_ray.sum()), 1e-30)

    def score(Wmat, sel=None):
        sel = live if sel is None else sel
        pred = torch.zeros(P, dtype=torch.long, device=dev)
        pred[sel] = (Wmat[sel] @ TT).argmax(1) + 1
        acc = float((pred[sel] == emit[sel]).float().mean())
        _, miou, _, macc = calculate_metrics(emit[sel].cpu(), pred[sel].cpu(), C + 1)
        return acc, float(miou), float(macc)

    # the exact solve is Z itself; scoring it is the CONTROL and must come out ~1.0
    Z = torch.zeros(P, C, device=dev)
    Z[torch.arange(P, device=dev), emit - 1] = 1.0
    a_e, mi_e, ma_e = score(Z)
    a_p, mi_p, ma_p = score(Wp)
    a_t, mi_t, ma_t = score(Wt)

    # ---- SPHERE-DECONVOLVED SOLVER (ours) -------------------------------------------------
    # NormLift Eq. 4 maximises sum_i A_ij <u, B_i> over the unit sphere. That objective is LINEAR
    # in u, so by linearity of the inner product it decouples across primitives no matter how much
    # overlap there is -- Cauchy-Schwarz gives u_j = f_j/||f_j||, the weighted mean renormalised.
    # It is therefore structurally blind to overlap, and under argmax the renormalisation is a
    # per-primitive positive scalar, so it predicts exactly what Eq. 6 predicts.
    #
    # The sphere-constrained LEAST SQUARES objective is what actually sees overlap:
    #     min_{||u_j||=1} sum_i || sum_j A_ij u_j - B_i ||^2
    #        = tr(U^T G U) - 2 <U, A^T B> + const
    # The linear term is theirs; the quadratic term carries the Gram. When G = D it equals
    # sum_j D_jj ||u_j||^2 = sum_j D_jj, CONSTANT on the sphere, so it drops out and the problem
    # collapses exactly to Eq. 4. Their solver is optimal in the disjoint limit and only there.
    #
    # Stationarity off that limit gives the update
    #     u_j  <-  normalise( (A^T B)_j - sum_{k != j} G_jk u_k )
    # i.e. subtract what the co-visible neighbours already explain. Everything stays in span(T),
    # so it runs in C dimensions with the metric T T^T; ||u_j||^2 = y_j^T (T T^T) y_j.
    # SAFEGUARDED, per primitive. The plain iteration is Jacobi-type: row j contracts only if the
    # off-diagonal mass it subtracts is small against its own diagonal. Define
    #     rho_j = (sum_{k != j} G_jk) / G_jj
    # -- the per-primitive version of the overlap mass, and under (P2) just (D_jj - G_jj)/G_jj,
    # so it needs `support` and `support2` and no labels. Scaling the correction by
    #     alpha_j = min(1, theta / rho_j)
    # enforces the contraction row by row. alpha_j = 0 leaves (A^T B)_j untouched, which IS the
    # back-projection direction, so the argmax degrades to Eq. 6 EXACTLY rather than diverging:
    # the method cannot do worse than the estimator it replaces, and improves wherever rho_j is
    # small. One theta is used for every arm -- no per-representation tuning.
    Gs = torch.zeros(P, device=dev)                       # sum_k G_jk = A^T (A 1)
    _s = torch.zeros(R, device=dev).index_add_(0, r_, v_)
    Gs.index_add_(0, c_, v_ * _s[r_])
    rho = (Gs - D2).clamp_min(0) / D2.clamp_min(1e-30)
    alpha = torch.clamp(theta / rho.clamp_min(1e-30), max=1.0)
    alpha[~live] = 0.0

    # THE SAFEGUARD IS THE OBJECTIVE, NOT THE STEP. Throttling the correction with a small
    # alpha_j moves the FIXED POINT and so converges to a biased answer (measured: alpha_mean
    # 0.063 on the high-overlap arm cost 19 mIoU against the same solver at alpha = 1 with a
    # smaller step). Damping changes only the PATH. So run the unbiased iteration and guard it by
    # tracking the actual least-squares objective
    #     J(U) = || A U - B ||^2 ,   U = Y T ,  so  J = sum_i (AY - AZ)_i (T T^T) (AY - AZ)_i^T
    # keeping the best iterate and STARTING from the back-projection. The returned solution is
    # then never worse than Eq. 6 on the quantity the bound is about -- that is the floor, and it
    # is a guarantee rather than a hope -- while still converging to the deconvolved optimum
    # wherever the iteration is contractive.
    Y = Wp.clone()

    def sph_norm(Yk):
        q = ((Yk @ TT) * Yk).sum(-1).clamp_min(1e-30).sqrt()
        return Yk / q.unsqueeze(-1)

    AZc = torch.zeros(R, C, device=dev)
    AZc.index_add_(0, r_, v_.unsqueeze(-1) * Z[c_])

    def objective(Yk):
        E = torch.zeros(R, C, device=dev)
        E.index_add_(0, r_, v_.unsqueeze(-1) * Yk[c_])
        E -= AZc
        return float(((E @ TT) * E).sum())

    Y[live] = sph_norm(Y[live])
    Y_best, j_best = Y.clone(), objective(Y)
    j_init = j_best
    for _ in range(deconv_iters):
        AY = torch.zeros(R, C, device=dev)
        AY.index_add_(0, r_, v_.unsqueeze(-1) * Y[c_])
        GY = torch.zeros(P, C, device=dev)
        GY.index_add_(0, c_, v_.unsqueeze(-1) * AY[r_])
        if step_mode == "gershgorin":
            # Projected gradient on J(U) = tr(U^T G U) - 2<U, A^T B>, grad = 2(GU - A^T B).
            # Stable for eta < 2/lambda_max(G), and Gershgorin bounds lambda_max(G) by
            # max_j sum_k G_jk for free because G is entrywise non-negative. Applied PER ROW as
            # eta_j = 1/sum_k G_jk, a safe diagonal preconditioner: no hand-set damping, and the
            # fixed point is untouched, so this stays exactly the deconvolved solution.
            Yn = Y - (GY - GZ) / Gs.clamp_min(1e-30).unsqueeze(-1)
            Y = torch.zeros_like(Y)
            Y[live] = sph_norm(Yn[live])
        else:
            off = (GY - D2.unsqueeze(-1) * Y) * alpha.unsqueeze(-1)
            Yn = torch.zeros_like(Y)
            Yn[live] = sph_norm((GZ - off)[live])
            Y = (1.0 - deconv_damp) * Y + deconv_damp * Yn
            Y[live] = sph_norm(Y[live])
        j_now = objective(Y)
        if j_now < j_best:
            j_best, Y_best = j_now, Y.clone()
    a_d, mi_d, ma_d = score(Y_best)


    # how far the closed form moved each primitive off its own class, in class space
    own = Wp[live].gather(1, (emit[live] - 1).unsqueeze(1)).squeeze(1)
    # secondary: the same metric restricted to primitives that own a GT point, for comparison
    sub = live & labelled
    a_s, mi_s, _ = score(Wp, sub)
    return dict(scene=scene, recon=recon, views=n_views, P=int(P), rays=int(R), nnz=int(nnz),
                labelled_frac=float(labelled.float().mean()),
                emit_agree=float((emit[labelled] == prim_gt_t[labelled]).float().mean()),
                live=int(live.sum()), C=C,
                mean_o=mean_o, mean_o_cross=mean_o_cross, cross_share=cross_share,
                own_mass_mean=float(own.mean()), own_mass_p50=float(own.median()),
                acc_closed=a_p, miou_closed=mi_p, macc_closed=ma_p,
                acc_exact=a_e, miou_exact=mi_e, macc_exact=ma_e,
                acc_closed_gtonly=a_s, miou_closed_gtonly=mi_s,
                acc_tikhonov=a_t, miou_tikhonov=mi_t, macc_tikhonov=ma_t,
                acc_deconv=a_d, miou_deconv=mi_d, macc_deconv=ma_d,
                obj_init=j_init, obj_best=j_best,
                obj_drop=float(1.0 - j_best / max(j_init, 1e-30)),
                alpha_mean=float(alpha[live].mean()),
                alpha_frac_capped=float((alpha[live] >= 1.0).float().mean()),
                squeeze=(squeeze or 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen,nonfrozen")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--deconv-iters", type=int, default=12)
    ap.add_argument("--step-mode", default="gershgorin",
                    choices=["gershgorin", "jacobi"],
                    help="gershgorin: per-row eta_j = 1/sum_k G_jk, no tuning")
    ap.add_argument("--theta", type=float, default=1e9,
                    help="per-row contraction safety factor; alpha_j = min(1, theta/rho_j)")
    ap.add_argument("--deconv-damp", type=float, default=0.5)
    ap.add_argument("--squeeze", type=float, default=None,
                    help="SFS Eq. 17 opacity squeeze lambda (their reported optimum 1.2); "
                         "splat arms only, since it acts on a stored opacity logit")
    ap.add_argument("--out", default="artifacts/scannet/consistent_oracle.json")
    a = ap.parse_args()
    rows = []
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.squeeze,
                        a.deconv_iters, a.deconv_damp, a.theta, a.step_mode)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] o {r['mean_o']:.4f}  o_cross {r['mean_o_cross']:.4f} "
                  f"({r['cross_share']:.1%})  own_mass {r['own_mass_mean']:.3f}  || "
                  f"CLOSED mIoU {r['miou_closed']*100:6.2f} acc {r['acc_closed']*100:6.2f}  "
                  f"| EXACT(control) mIoU {r['miou_exact']*100:6.2f}  "
                  f"| Eq18 {r['miou_tikhonov']*100:6.2f}  | DECONV {r['miou_deconv']*100:6.2f}  "
                  f"(scored {r['live']:,}, GT-owning {r['labelled_frac']:.1%})", flush=True)
    if rows:
        print(f"\n=== self-consistent oracle, {a.views} views ===")
        print(f"{'arm':<12}{'mean o':>9}{'o_cross':>9}{'cross%':>8}{'own_mass':>10}"
              f"{'Eq6 mIoU':>11}{'Eq18 mIoU':>11}{'DECONV':>10}{'Eq6 acc':>10}"
              f"{'DECONV acc':>12}{'exact(ctl)':>12}{'n':>4}")
        for rec in a.recons.split(","):
            rs = [r for r in rows if r["recon"] == rec]
            if not rs:
                continue
            f = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"{rec:<12}{f('mean_o'):>9.4f}{f('mean_o_cross'):>9.4f}{f('cross_share'):>8.1%}"
                  f"{f('own_mass_mean'):>10.3f}{f('miou_closed')*100:>11.2f}"
                  f"{f('miou_tikhonov')*100:>11.2f}{f('miou_deconv')*100:>10.2f}"
                  f"{f('acc_closed')*100:>10.2f}{f('acc_deconv')*100:>12.2f}"
                  f"{f('miou_exact')*100:>12.2f}{len(rs):>4}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
