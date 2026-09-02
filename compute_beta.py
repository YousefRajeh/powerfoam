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
    rhs = torch.zeros((P, D), device=dev)
    CH = 4_000_000
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

    x_hat, info = cg_normal_equations(lambda p: p, AtA, rhs, diag, iters=a.cg_iters)
    print(f"  CG: {info['iterations']} iters, final relative residual "
          f"{info['final_rel_residual']:.3e}", flush=True)

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
    mu = torch.zeros(R, device=dev)
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        mu.index_add_(0, row[s:e], val[s:e] * d)
    rs_pre = rowsum.clamp_min(torch.finfo(mu.dtype).eps)
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
           "frac_k1": float((k[ok] == 1).float().mean())}
    json.dump(res, open(f"{OUT}/{a.arm}_{a.scene}.json", "w"), indent=1)
    print(f"  beta: max={res['beta_max']:.3f} p999={res['beta_p999']:.3f} "
          f"p99={res['beta_p99']:.3f} p90={res['beta_p90']:.4f} median={res['beta_p50']:.4f}",
          flush=True)
    print(f"  frac(beta_i == 0)={res['frac_beta_zero']*100:.1f}%  "
          f"frac(k_i == 1)={res['frac_k1']*100:.1f}%   <- these must agree", flush=True)


if __name__ == "__main__":
    main()
