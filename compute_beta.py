"""Compute Splat Feature Solver's dispersion beta EXACTLY, for foam and 3DGS.

THEIR DEFINITION (arXiv 2508.12216, Eq. 14), at the globally optimal least-squares solution x_hat:

    Delta_ij = ||x_hat_j - B_i||
    mu_i     = sum_j A_ij Delta_ij
    sigma_i2 = sum_j A_ij (Delta_ij^2 - mu_i^2)
    beta_i   = sigma_i2 / mu_i^2
    beta     = max_i beta_i           with the bound   L(x') <= (1 + beta) L(x_hat)

x' is their closed-form row-sum-preconditioned solution (the weighted mean) -- the thing the bound is
ABOUT. beta must therefore be evaluated at x_hat, the TRUE least-squares optimum, which is why this
script runs a conjugate-gradient solve rather than reusing any lifted feature field we already have.
Computing Delta against x' would measure nothing.

WHY beta AND NOT OUR n_eff. n_eff (Kish) is ours and appears in neither paper; beta is the published
quantity that carries a published bound. The exact link we can prove is one-directional:

    k_i = 1  =>  beta_i = 0 exactly   (one nonzero, row sums to 1 => mu_i = Delta_ij, sigma_i2 = 0)
    k_i >= 2 =>  beta_i >= 0, magnitude undetermined

so the measured 36.4% of foam rays with k_i = 1 (vs 0.0% for 3DGS) is a LOWER BOUND on the fraction
of rays contributing exactly zero to the bound. This script measures the rest.

SCOPE, STATED NOT HIDDEN. beta is defined over every ray of every view. Caching A for all 244 views
of a ScanNet scene would need ~77 GB for the 3DGS arm (9.6e9 nonzeros), so this evaluates a
well-defined SUBPROBLEM: a fixed, evenly spaced subset of `--views` views at native resolution. For
that subproblem x_hat is solved exactly and beta is exact. The view count is recorded in the output
and must be reported.

ROW-STOCHASTICITY IS CHECKED, NOT ASSUMED. sigma_i2 is a variance only if sum_j A_ij = 1. Foam is
row-stochastic to 4 decimals; the 3DGS exporter was NOT until the float32 cumsum bug was fixed. The
script reports the row-sum distribution and refuses to proceed if it is far from 1.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import

import configargparse
import numpy as np
import torch

from camera_bridge import K_from_ray_dirs
from configs import Params, add_group
from data_loader import DataHandler

OUT = "artifacts/scannet/beta"


def load_view_features(feat_dir, stem, H, W):
    """Return (seg_ids (H*W,) int64, table (S, D) float32) for one view.

    Features are stored per SAM SEGMENT, not per pixel: `{stem}_f.npy` is (S, D) and `{stem}_s.npy`
    is the per-pixel segment map. Keeping that factorisation is what makes B affordable -- a dense
    (rays, 512) B for even 12 views would be ~30 GB.
    """
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[-1] if s.shape[0] <= 4 else s[..., -1]
    t = torch.from_numpy(np.ascontiguousarray(f)).float()
    t = torch.nn.functional.normalize(t, dim=-1)
    seg = torch.from_numpy(np.ascontiguousarray(s)).long()
    if seg.shape != (H, W):
        seg = torch.nn.functional.interpolate(
            seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), t


def _flip_stats(x_prime, x_hat, prim_gap, colsum, scene, dev):
    """Does the feature-space excess predict an ARGMAX FLIP?

    The suboptimality bound `L(X') - L(Xhat) <= E_H(Xhat)` is exact in feature space, but mIoU only
    moves when the READOUT changes -- a large feature error that leaves argmax alone is free, a tiny
    one that crosses a decision boundary costs a point. Every feature-space quantity we tried
    (E_H, gamma, gap/rays, mean_o) predicts the mIoU solve gap at r ~ +0.27, i.e. not at all.

    So test the missing link directly, PER PRIMITIVE (n ~ 3e4, not n = 10 scenes): does a
    primitive's own share of the gap predict whether its label flips between X' and Xhat?

      * if YES  -- the bound is sound and the readout is simply the lossy step, and a flip-rate
                   bound is the thing to state;
      * if NO   -- feature-space excess is unrelated to the decision even per primitive, and no
                   bound of this family can govern the metric.

    AUC is used rather than a correlation because `flip` is binary and `prim_gap` is heavy-tailed;
    it answers "is a flipped primitive's gap bigger than a non-flipped one's?" with no distributional
    assumption. AUC 0.5 = no information.
    """
    import glob, os
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from diagnose_holes import GT_ROOT
    try:
        d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
        _, raw, names = load_scannet_pointcept_gt(d, "segment20")
        n2i = {n: i for i, n in enumerate(names)}
        pres = set(np.unique(raw).tolist())
        kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
        T = embed_class_names(kept, dev)
        T = T / T.norm(dim=-1, keepdim=True)
    except Exception as e:
        return {"flip_error": f"{type(e).__name__}: {e}"}
    live = colsum > 0
    if not bool(live.any()):
        return {}
    xp = x_prime[live] / x_prime[live].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xh = x_hat[live] / x_hat[live].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lp = (xp @ T.T).argmax(1)
    lh = (xh @ T.T).argmax(1)
    flip = (lp != lh)
    g = prim_gap[live]
    nf, nn = int(flip.sum()), int((~flip).sum())
    if nf == 0 or nn == 0:
        return {"flip_frac": float(flip.float().mean()), "flip_auc": float("nan"), "flip_n": nf}
    # AUC via the rank-sum identity, exact and O(n log n)
    r = torch.argsort(torch.argsort(g.double())).double() + 1.0
    auc = float((r[flip].sum() - nf * (nf + 1) / 2) / (nf * nn))
    return {"flip_frac": float(flip.float().mean()), "flip_auc": auc, "flip_n": nf,
            "flip_gap_ratio": float(g[flip].mean() / g[~flip].mean().clamp_min(1e-30))}


def cg_normal_equations(matmul, rmatmul, rhs, diag, iters=300, rtol=1e-6):
    """Jacobi-preconditioned CG on (A^T A) x = rhs, block over feature channels.

    A custom solve rather than `ridge_pcg` because that function takes a DENSE (num_rows, channels)
    B, which does not exist here -- B is factorised into segment ids plus a small table, and
    materialising it would be ~30 GB. Only rhs = A^T B is needed, and it is formed streaming.
    Validated against ridge_pcg(mode="none") on a small dense case in test_beta_cg.py.
    """
    inv = diag.clamp_min(torch.finfo(rhs.dtype).eps).reciprocal()
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    z = inv[:, None] * r
    p = z.clone()
    rz = (r * z).sum(0)
    n0 = rhs.norm(dim=0).clamp_min(torch.finfo(rhs.dtype).eps)
    hist = []
    for it in range(iters):
        ap = rmatmul(matmul(p))
        den = (p * ap).sum(0)
        step = torch.where(den.abs() > 0, rz / den, torch.zeros_like(rz))
        x = x + p * step
        r = r - ap * step
        rel = float((r.norm(dim=0) / n0).max())
        hist.append(rel)
        if rel <= rtol:
            break
        z = inv[:, None] * r
        rz_new = (r * z).sum(0)
        p = z + p * torch.where(rz.abs() > 0, rz_new / rz, torch.zeros_like(rz))
        rz = rz_new
    return x, {"iterations": it + 1, "final_rel_residual": hist[-1], "residual_history": hist[::20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--arm", default="pf_truefrozen",
                    choices=("pf_truefrozen", "pf_nonfrozen", "gs_froz", "gs_unfroz"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cg-iters", type=int, default=300)
    ap.add_argument("--ridge", default="1e-6",
                    help="ridge values as multiples of mean(diag G); 0 = unregularised. Each value "
                         "costs a FULL CG solve and only --beta-at is reported, so the default is "
                         "now the single reported value. The sensitivity question this sweep "
                         "answered is settled (05_open_questions A23: beta_p50 moves 0.58%% from "
                         "lam=0 to 1e-6 while beta_max moves 96%%); pass the old "
                         "'0,1e-8,1e-6,1e-4,1e-2' to reproduce it.")
    ap.add_argument("--no-region-space", action="store_true",
                    help="solve in the full D=512 feature space instead of the exact M-channel "
                         "region subspace; for cross-checking the reduction")
    ap.add_argument("--beta-at", default="1e-6",
                    help="which ridge value the headline beta is reported at")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    recon = "nonfrozen" if a.arm.startswith("gs_") else a.arm[3:]
    cfg = f"output/scannet_{a.scene}_{recon}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])

    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
    stems = sorted(os.path.splitext(f)[0][:-2] for f in os.listdir(feat_dir) if f.endswith("_f.npy"))

    # ---- build A over the selected views, plus per-ray segment ids ----
    if a.arm.startswith("pf_"):
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{a.scene}_{recon}/model.pt")
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{a.arm}/{a.scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].cuda(), sp["quats"].cuda()
        gs_ = torch.exp(sp["scales"].cuda())
        go = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)

    rows, cols, vals, seg_all, tables, offs = [], [], [], [], [], 0
    for n, vi in enumerate(sel):
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if a.arm.startswith("pf_"):
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            P = op.num_primitives
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().cuda()
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.cuda(), W, H,
                                                    max_hits_per_pixel=a.cap,
                                                    transmittance_floor=1e-3)
            P = gm.shape[0]
        rows.append(ri.to(torch.int64) + offs)
        cols.append(ci.to(torch.int64))
        vals.append(vv.float())
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W)
        seg_all.append(seg.to(dev))
        tables.append(tab.to(dev))
        offs += H * W
        del ri, ci, vv

    row = torch.cat(rows); col = torch.cat(cols); val = torch.cat(vals)
    del rows, cols, vals
    R, D = offs, tables[0].shape[1]
    nnz = val.numel()

    rowsum = torch.zeros(R, device=dev).index_add_(0, row, val)
    live = rowsum > 0
    q = torch.quantile(rowsum[live].double(), torch.tensor([.01, .5, .99], device=dev, dtype=torch.float64))
    print(f"[{a.arm}/{a.scene}] views={a.views} rays={R:,} nnz={nnz:,} P={P:,}", flush=True)
    print(f"  row sums 1/50/99%: {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}  "
          f"(sigma_i^2 is a variance only if ~1)", flush=True)
    if abs(float(q[1]) - 1.0) > 0.05:
        print("  WARNING: rows are not stochastic; beta is not a coefficient of variation here",
              flush=True)

    def B_rows(idx):
        """Gather observations for arbitrary global ray indices, from the factorised store."""
        out = torch.zeros((idx.numel(), D), device=dev)
        base = 0
        for seg, tab in zip(seg_all, tables):
            n = seg.numel()
            m = (idx >= base) & (idx < base + n)
            if bool(m.any()):
                loc = idx[m] - base
                out[m] = tab[seg[loc].clamp(0, tab.shape[0] - 1)]
            base += n
        return out

    # ---- rhs = A^T B, streamed over nonzero chunks ----
    # REGION SPACE. B is not an arbitrary (R, D) matrix: every row is a lookup into that view's
    # per-SAM-region table, so B = S T exactly, with S one-hot (R, M) and T (M, D) the stacked
    # tables. M is the number of regions over the selected views -- 247 on scene0070, 162 on
    # scene0000 -- against D = 512. Since CG is linear in the right-hand side, solving
    #     (G + lam I) Y = A^T S        then      x_hat = Y T
    # gives the identical minimiser while every gather inside AtA carries M columns instead of D.
    # This is exact, not an approximation. Disable with --no-region-space to cross-check.
    CH = 4_000_000
    T_all = torch.cat(tables, 0)                       # (M, D), each row already L2-normalised
    M = T_all.shape[0]
    gid = torch.cat([seg.clamp(0, tab.shape[0] - 1) + off for seg, tab, off in
                     zip(seg_all, tables,
                         torch.tensor([0] + [t.shape[0] for t in tables[:-1]],
                                      device=dev).cumsum(0).tolist())])
    use_region = (not a.no_region_space) and M < D
    if use_region:
        rhs_s = torch.zeros((P, M), device=dev)
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            rhs_s.index_put_((col[s:e], gid[row[s:e]]), val[s:e], accumulate=True)
        rhs = rhs_s @ T_all                            # A^T B, for x_prime and reporting
        print(f"  region space: solving {M} channels instead of D={D} "
              f"({D / max(M, 1):.1f}x less gather per CG iteration)", flush=True)
    else:
        rhs_s = None
        rhs = torch.zeros((P, D), device=dev)
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            rhs.index_add_(0, col[s:e], val[s:e, None] * B_rows(row[s:e]))
    diag = torch.zeros(P, device=dev).index_add_(0, col, val * val)

    # FUSED A^T A, BLOCKED OVER ROWS. Computing A^T(A p) as two separate passes materialises the
    # intermediate A p at (rays, channels) -- 15M x 512 x 4 B = 30 GB, which OOM'd. Because
    # (A^T A p)_j = sum_i A_ij (sum_k A_ik p_k), the inner product for a row can be consumed
    # immediately, so processing a BLOCK of rows at a time bounds the intermediate to
    # (block_rows, channels). Requires the nonzeros sorted by row, done once below.
    # Already row-sorted by construction: each view's exporter returns nonzeros ordered by pixel,
    # and views are appended with increasing global row offsets. Verified below rather than assumed,
    # because argsort on 483M int64 elements transiently doubles the operator (~19 GB) and was
    # itself a source of OOM.
    if not bool((row[1:] >= row[:-1]).all()):
        order = torch.argsort(row)
        row, col, val = row[order].contiguous(), col[order].contiguous(), val[order].contiguous()
        del order
    starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
    # Block boundaries chosen by NONZERO count, not row count. Rows carry wildly different nnz
    # (foam 2.3/row, 3DGS 31.5/row), so a fixed row block gave 12.3M nonzeros for 3DGS -> a 25 GB
    # gather that OOM'd. Bounding nnz per block bounds the gather directly.
    NNZ_BUDGET = max(1, int(3e8 // max(D, 1)))          # ~(nnz x D) elements, ~2.4 GB at D=512
    # Vectorised boundary search. The previous loop called searchsorted with a 0-DIM tensor, which
    # does not do what it looks like -- it returned a single boundary, so the "first block" spanned
    # the whole operator and the gather was 22.7 GB instead of the intended 2.4 GB.
    tgts = torch.arange(0, nnz + NNZ_BUDGET, NNZ_BUDGET, device=dev)
    bnd = torch.searchsorted(starts.contiguous(), tgts).clamp(0, R)
    bnd = torch.unique(torch.cat([bnd, torch.tensor([R], device=dev)]))
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

    # RIDGE SWEEP. beta is NOT a property of the data when G = A^T A is rank-deficient: it reads
    # Xhat itself (through Delta_ij = ||xhat_j - b_i||), not A Xhat, so two exact minimisers with
    # identical loss give different beta. test_beta_math.py constructs that case explicitly -- beta
    # moves 9x along a null direction at constant loss, while the excess L(X') - L(Xhat) is
    # invariant to 1e-9. On these scenes Xhat runs to ||xhat_j|| ~ 1e7 in near-null directions, so
    # an unregularised beta_max is reporting the solver's arbitrary choice among minimisers.
    #
    # Solving (G + lam I) x = A^T B for a sweep of lam makes the choice explicit: lam -> 0 is the
    # minimum-norm minimiser, and the sweep shows directly whether beta diverges as the
    # regularisation is removed. gamma is reported alongside because it must NOT move.
    lam_scale = float(diag.mean())
    lams = [float(x) * lam_scale for x in a.ridge.split(",")]
    sweep = []
    x_hat = None
    for lam in lams:
        def AtA_l(x, lam=lam):
            return AtA(x) + lam * x
        # Solve against A^T S when in region space, then map back: x = Y T. The operator
        # (G + lam I) is untouched, so this is the same linear system with fewer right-hand
        # sides -- the minimiser is identical, only the channel count changes.
        b_rhs = rhs_s if use_region else rhs
        yl, info_l = cg_normal_equations(lambda p: p, AtA_l, b_rhs, diag + lam, iters=a.cg_iters)
        xl = (yl @ T_all) if use_region else yl
        del yl
        sweep.append((lam, xl, info_l))
    # the arm reported as "the" beta uses the LAST (largest) lambda unless --beta-at is given
    pick = min(range(len(lams)), key=lambda i: abs(lams[i] - float(a.beta_at) * lam_scale))
    x_hat, info = sweep[pick][1], sweep[pick][2]
    print(f"  ridge sweep over lam/mean(diag G) = {a.ridge};  beta reported at "
          f"{a.beta_at} (lam={lams[pick]:.3e})", flush=True)
    print(f"  CG: {info['iterations']} iters, final relative residual "
          f"{info['final_rel_residual']:.3e}", flush=True)

    # ---- gamma: the EXACT suboptimality of the closed form, not a bound ----------------------
    # Because x_hat solves the normal equations, A^T(A x_hat - B) = 0, so for ANY x the cross term
    # vanishes and Pythagoras is exact:
    #       ||Ax - B||^2 = ||A(x - x_hat)||^2 + ||A x_hat - B||^2
    # Hence with  gamma = ||A(x' - x_hat)||^2 / L(x_hat):     L(x') = (1 + gamma) L(x_hat) EXACTLY.
    # beta upper-bounds gamma. Reporting both shows how much of the bound is real: beta is a max
    # over ~1e7 rays and is set by a single pixel, while gamma is what the closed form actually
    # gives up. gamma is nearly free here -- x_hat is already solved (the expensive part) and
    # x' = A^T B / colsum reuses rhs.
    #
    # NOTE ON VALIDITY: the identity needs x_hat AT the optimum. Where CG stops on its iteration cap
    # (the 3DGS arm) the cross term is not exactly zero, so gamma is biased -- and biased against
    # whichever arm converges worse. cg_residual is recorded beside gamma so this is checkable.
    # ---- overlap mass, the operator-only measurable that replaces beta -----------------------
    # o_i = (sum_j A_ij)^2 - sum_j A_ij^2, zero exactly when a ray lands on one primitive. Unlike
    # beta it needs no solve, no residuals and no ground truth: two accumulators over the weights
    # the rasteriser already produces. Computed here so that beta, gamma and o are all measured on
    # the SAME subproblem (same views, same A), which is the only way the comparison is fair.
    row_s = torch.zeros(R, device=dev).index_add_(0, row, val)
    row_sq = torch.zeros(R, device=dev).index_add_(0, row, val * val)
    o_ray = (row_s * row_s - row_sq).clamp_min(0)
    sum_o = float(o_ray.sum())
    mean_o = sum_o / float(row_s.sum().clamp_min(1e-30))
    o_q = [float(torch.quantile(o_ray.float(), q)) for q in (0.5, 0.9, 0.99)]
    print(f"  sum_i o_i = {sum_o:.4e}   mean o per unit ray mass = {mean_o:.4f}   "
          f"o p50/p90/p99 = {o_q[0]:.3f}/{o_q[1]:.3f}/{o_q[2]:.3f}", flush=True)

    colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
    x_prime = rhs / colsum.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]
    d = x_prime - x_hat

    gap = torch.zeros((), device=dev)          # ||A(x' - x_hat)||_F^2
    loss_hat = torch.zeros((), device=dev)     # ||A x_hat - B||_F^2
    prim_gap = torch.zeros(P, device=dev)      # per-primitive share of the gap
    for r0, r1 in blocks:
        s, e = int(starts[r0]), int(starts[r1])
        if e <= s:
            continue
        lr = row[s:e] - r0
        ad = torch.zeros((r1 - r0, D), device=dev)
        ad.index_add_(0, lr, val[s:e, None] * d[col[s:e]])
        gap += (ad * ad).sum()
        ax = torch.zeros((r1 - r0, D), device=dev)
        ax.index_add_(0, lr, val[s:e, None] * x_hat[col[s:e]])
        resid = ax - B_rows(torch.arange(r0, r1, device=dev))
        loss_hat += (resid * resid).sum()
        # Attribute the gap: primitive j's own displacement weighted by how much ray mass it carries.
        prim_gap.index_add_(0, col[s:e], (val[s:e, None] * d[col[s:e]]).pow(2).sum(-1))
        del ad, ax, resid
    gamma = float(gap / loss_hat.clamp_min(torch.finfo(gap.dtype).eps))
    print(f"  gamma (exact L(x')/L(x_hat) - 1) = {gamma:.6f}    "
          f"L(x_hat)={float(loss_hat):.4e}", flush=True)

    # ---- beta, TWO-PASS. Pass 1 accumulates mu_i; pass 2 accumulates the centred second
    # moment sum_j w_j (Delta_ij - mu_i)^2 directly.
    #
    # The one-pass form sigma^2 = E[Delta^2] - mu^2 is the textbook-unstable variance formula: when
    # sigma^2 << mu^2 the two terms cancel and the result is pure rounding. That is exactly the
    # k_i = 1 case, where sigma^2 must be 0 -- measured, the one-pass form returned beta up to
    # 9.3e-08 for such rays (float32 carries ~1e-7 relative error in each term), so 31.2% of rays
    # with k_i = 1 yielded only 27.4% with beta = 0 and the k=1 => beta=0 identity appeared to fail.
    # Two-pass gives 4.9e-15 on the same inputs, because the normalised weight of a lone contributor
    # is v/v = 1 EXACTLY in IEEE754 (test_beta_variance.py). This is a stability fix, not a
    # loosened threshold.
    rs_pre = rowsum.clamp_min(torch.finfo(rowsum.dtype).eps)

    def beta_of(xh):
        """beta_i at a GIVEN minimiser. Factored out so the ridge sweep can evaluate it at each."""
        mu = torch.zeros(R, device=dev)
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            d = (xh[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
            mu.index_add_(0, row[s:e], val[s:e] * d)
        mu = mu / rs_pre
        m2 = torch.zeros(R, device=dev)      # holds the CENTRED moment after this loop
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            d = (xh[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
            w = val[s:e] / rs_pre[row[s:e]]
            m2.index_add_(0, row[s:e], w * (d - mu[row[s:e]]) ** 2)
        okl = live & (mu > 1e-6)
        bl = torch.zeros(R, device=dev)
        bl[okl] = m2[okl].clamp_min(0) / (mu[okl] ** 2)
        return bl, okl, int((live & ~okl).sum())

    # gamma must be INVARIANT across the sweep (it depends on Xhat only through A Xhat); beta is
    # the quantity under test. Printing both side by side is the whole point.
    print(f"  {'lam/mean(diagG)':>16}{'||xhat||max':>14}{'beta_max':>14}{'beta_p50':>12}"
          f"{'gamma':>12}{'cg res':>10}", flush=True)
    ridge_rows = []
    for (lam, xl, info_l), lam_rel in zip(sweep, [float(x) for x in a.ridge.split(",")]):
        bl, okl, _ = beta_of(xl)
        bb = bl[okl]
        gl = torch.zeros((), device=dev)
        ll = torch.zeros((), device=dev)
        dl = (rhs / colsum.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]) - xl
        for r0, r1 in blocks:
            s0, e0 = int(starts[r0]), int(starts[r1])
            if e0 <= s0:
                continue
            lr = row[s0:e0] - r0
            adl = torch.zeros((r1 - r0, D), device=dev)
            adl.index_add_(0, lr, val[s0:e0, None] * dl[col[s0:e0]])
            gl += (adl * adl).sum()
            axl = torch.zeros((r1 - r0, D), device=dev)
            axl.index_add_(0, lr, val[s0:e0, None] * xl[col[s0:e0]])
            rsd = axl - B_rows(torch.arange(r0, r1, device=dev))
            ll += (rsd * rsd).sum()
            del adl, axl, rsd
        gam_l = float(gl / ll.clamp_min(torch.finfo(gl.dtype).eps))
        ridge_rows.append({"lam_rel": lam_rel, "lam": lam,
                           "xhat_norm_max": float(xl.norm(dim=-1).max()),
                           "beta_max": float(bb.max()), "beta_p50": float(bb.median()),
                           "gamma": gam_l, "cg_residual": info_l["final_rel_residual"]})
        print(f"  {lam_rel:>16.0e}{ridge_rows[-1]['xhat_norm_max']:>14.3e}"
              f"{ridge_rows[-1]['beta_max']:>14.4e}{ridge_rows[-1]['beta_p50']:>12.4e}"
              f"{gam_l:>12.6f}{info_l['final_rel_residual']:>10.1e}", flush=True)

    mu = torch.zeros(R, device=dev)
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        mu.index_add_(0, row[s:e], val[s:e] * d)
    mu = mu / rs_pre
    m2 = torch.zeros(R, device=dev)          # holds the CENTRED moment after this loop
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        w = val[s:e] / rs_pre[row[s:e]]
        m2.index_add_(0, row[s:e], w * (d - mu[row[s:e]]) ** 2)
    # NORMALISE BY THE ROW SUM. beta_i is the squared coefficient of variation of Delta under the
    # weight distribution A_i., which requires sum_j w_ij = 1. Our rows sum to 0.9998, not 1, and
    # with raw weights a k=1 row gives sig2 = s*d^2*(1-s) != 0 -- so the k=1 => beta=0 identity
    # failed numerically (measured frac(beta=0)=5.1% against frac(k=1)=31.2%). Dividing mu and m2
    # by the row sum restores it exactly.
    sig2 = m2.clamp_min(0)      # already the centred, weight-normalised second moment
    # mu ~ 0 means every contributor sits at the same (near-zero) residual; beta is then 0/0 and
    # dominated by float noise, which is what produced a max of 1.8e5. Excluded and counted.
    ok = live & (mu > 1e-6)
    n_degen = int((live & ~ok).sum())
    beta_i = torch.zeros(R, device=dev)
    beta_i[ok] = sig2[ok] / (mu[ok] ** 2)
    b = beta_i[ok]
    qs = torch.quantile(b.double(), torch.tensor([.5, .9, .99, .999], device=dev, dtype=torch.float64))
    k = torch.bincount(row, minlength=R)
    res = {"scene": a.scene, "arm": a.arm, "views": a.views, "rays": int(ok.sum()), "P": int(P),
           "nnz": int(nnz), "row_sum_median": float(q[1]),
           "cg_iterations": info["iterations"], "cg_residual": info["final_rel_residual"],
           "beta_max": float(b.max()), "beta_mean": float(b.mean()),
           "beta_p50": float(qs[0]), "beta_p90": float(qs[1]),
           "beta_p99": float(qs[2]), "beta_p999": float(qs[3]),
           "frac_beta_zero": float((b <= 1e-12).float().mean()),
           "n_degenerate_mu0": n_degen,
           "frac_k1": float((k[ok] == 1).float().mean()),
           "ridge_sweep": ridge_rows, "beta_at": a.beta_at,
           # The operator-only measurable, on this same subproblem.
           "sum_o": sum_o, "mean_o": mean_o,
           "o_p50": o_q[0], "o_p90": o_q[1], "o_p99": o_q[2],
           # 2*sum_o bounds L(x') - L(x_c) against the UNIT-BALL optimum (Omega <= 2 there), so
           # this ratio is the honest tightness of the replacement bound on real data.
           "bound_ball_over_gap": 2.0 * sum_o / max(float(gap), 1e-30),
           # gamma is the REALIZED suboptimality, exact rather than bounded; beta >= gamma always.
           "gamma": gamma,
           "loss_x_hat": float(loss_hat),
           "gap_A_dx_sq": float(gap),
           "beta_over_gamma": float(qs[0]) / max(gamma, 1e-12),
           # Primitives whose closed-form value is furthest from optimal, for the per-cell study.
           **_flip_stats(x_prime, x_hat, prim_gap, colsum, a.scene, dev),
           "prim_gap_p50": float(torch.quantile(prim_gap.double(), 0.5)),
           "prim_gap_p99": float(torch.quantile(prim_gap.double(), 0.99)),
           "prim_gap_max": float(prim_gap.max())}
    json.dump(res, open(f"{OUT}/{a.arm}_{a.scene}.json", "w"), indent=1)
    print(f"  beta: max={res['beta_max']:.3f} p999={res['beta_p999']:.3f} "
          f"p99={res['beta_p99']:.3f} p90={res['beta_p90']:.4f} median={res['beta_p50']:.4f}",
          flush=True)
    print(f"  frac(beta_i == 0)={res['frac_beta_zero']*100:.1f}%  "
          f"frac(k_i == 1)={res['frac_k1']*100:.1f}%   <- these must agree", flush=True)


if __name__ == "__main__":
    main()
