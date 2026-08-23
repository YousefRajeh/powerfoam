"""Cone-constrained solve entirely in coefficient space, using the precomputed block Hessian.

Change 1 of a sequence. This changes only WHERE the arithmetic happens, not the algorithm: same
FISTA, same feasible set, same starting point. It exists so the later algorithmic changes
(restart, preconditioning) can be tested in minutes instead of an hour each.

The objective, with f_j = U_j^T a_j (a_j >= 0 over cell j's observed CLIP embeddings):

    L(a) = 1/2 ||A U a - b||^2 = 1/2 a^T H a - a^T c + const
    H    = block-sparse, B_{jl} = S_{jl} (U_j U_l^T),   S = A^T A     [gram_blocks, validated 6.2e-07]
    c_j  = U_j (A^T b)_j                                             [A^T b is the streaming numerator]
    grad = H a - c

Neither b (~138 TB) nor the 512-dimensional features appear anywhere in the loop.

Reported metrics changed deliberately. The previous runs tracked a relative residual that
saturates near the noise floor of an overdetermined system and hides optimization progress; the
standard measure for a bound-constrained problem is the projected-gradient (KKT) norm
||min(a, grad)||_inf, which is zero exactly at the constrained optimum. Both that and the
objective are printed, and the objective is the one that must decrease monotonically for a
correct descent method.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP
from gram_blocks import accumulate_view_pairs, merge, build_blocks

D = 512


def kkt(a, grad):
    """||min(a, grad)||_inf -- zero exactly at a KKT point of min L(a) s.t. a >= 0."""
    return float(torch.minimum(a, grad).abs().max())


def build(a_args, device="cuda"):
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", a_args.config])
    ckpt = a_args.config.replace("/config.yaml", "").replace("\\config.yaml", "")

    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    stems = sorted(q.stem for q in (Path(args.data_path) / args.scene / "images").iterdir())
    assert len(stems) == len(cameras)
    n_views = len(cameras) if a_args.max_views is None else min(a_args.max_views, len(cameras))

    P, K = model.points.shape[0], a_args.topk
    feat_dir = Path(a_args.feature_folder)
    Atb = torch.zeros(P, D, device=device)
    support = torch.zeros(P, device=device)
    top_w = torch.zeros(P, K, device=device)
    U = torch.zeros(P, K, D, device=device)
    keys, svals = [], []
    t0 = time.time()

    for vi in range(n_views):
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H_, W_,
                                                   sam_level=a_args.sam_level)
        if float(fmap.abs().max()) == 0.0:
            continue
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = H_ * W_
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.repeat_interleave(torch.arange(npix, device=device), slots_used.long())
        f_pix = fmap.reshape(-1, D)

        w_cell = torch.zeros(P, device=device).index_add_(0, cols, vals)
        f_cell = torch.zeros(P, D, device=device)
        CH = 1_000_000
        for s in range(0, cols.numel(), CH):
            e = min(s + CH, cols.numel())
            f_cell.index_add_(0, cols[s:e], vals[s:e, None] * f_pix[rows[s:e]])
        Atb += f_cell
        support += w_cell

        seen = w_cell > 0
        f_view = torch.zeros_like(f_cell)
        f_view[seen] = F.normalize(f_cell[seen], dim=-1)
        worst = top_w.argmin(dim=1)
        wv = top_w.gather(1, worst[:, None]).squeeze(1)
        better = seen & (w_cell > wv)
        if bool(better.any()):
            idx = torch.where(better)[0]
            top_w[idx, worst[idx]] = w_cell[idx]
            U[idx, worst[idx]] = f_view[idx]

        # S is accumulated here, and the ray triples are then DISCARDED -- unlike the
        # matrix-free solver, nothing needs them again
        accumulate_view_pairs(cols, vals, slots_used, P, keys, svals)
        if len(keys) > 24:
            k, v = merge(keys, svals)
            keys, svals = [k], [v]
        del out_col, out_val, fmap, f_pix, rows, cols, vals, f_cell, f_view
        torch.cuda.empty_cache()

    k, v = merge(keys, svals)
    print(f"[build] {n_views} views, {k.numel():,} edges, {time.time()-t0:.0f}s", flush=True)

    valid = support > 0
    x_diag = torch.zeros(P, D, device=device)
    x_diag[valid] = Atb[valid] / support[valid][:, None]

    # augment the cone with the incumbent so the feasible set provably contains it
    Kt = K + 1
    U_aug = torch.zeros(P, Kt, D, device=device)
    U_aug[:, :K] = U
    dn = x_diag.norm(dim=-1)
    hasd = dn > 0
    U_aug[hasd, K] = x_diag[hasd] / dn[hasd, None]
    have = torch.zeros(P, Kt, dtype=torch.bool, device=device)
    have[:, :K] = top_w > 0
    have[:, K] = hasd
    del U
    U = U_aug
    torch.cuda.empty_cache()

    t1 = time.time()
    Hb = build_blocks(k, v, U, P, Kt)
    print(f"[build] blocks {Hb.bytes()/2**30:.2f} GiB in {time.time()-t1:.0f}s", flush=True)
    c = torch.einsum("pkd,pd->pk", U, Atb)
    return Hb, c, U, have, dn, valid, x_diag, P, Kt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--restart", type=int, default=1,
                   help="1 = adaptive gradient restart (default), 0 = plain FISTA, for A/B")
    p.add_argument("--precond", type=int, default=1,
                   help="1 = per-cell diagonal preconditioning (default), 0 = single global step")
    a = p.parse_args()
    device = "cuda"

    Hb, c, U, have, dn, valid, x_diag, P, K = build(a, device)
    hf = have.float()

    # start exactly at the incumbent: all mass on the augmented direction
    coef = torch.zeros(P, K, device=device)
    coef[:, K - 1] = dn

    def obj(x):
        return float(0.5 * (x * Hb.matvec(x)).sum() - (x * c).sum())

    def grad(x):
        return (Hb.matvec(x) - c) * hf

    g0 = grad(coef)
    print(f"[start] objective {obj(coef):.6e}   KKT {kkt(coef, g0):.6e}", flush=True)

    # ---- change 3: per-cell diagonal preconditioning ---------------------------------
    # A single global step 1/L makes every cell move at the pace of the worst-conditioned one.
    # S_jj = sum_r A[r,j]^2 spans orders of magnitude across cells (a wall seen head-on in 40
    # views versus a sliver glimpsed edge-on in two), so the global step is drastically too
    # small almost everywhere. Because the true diagonal of H is S_jj*||u_jk||^2 = S_jj for
    # every k (unit basis vectors), the right preconditioner is a per-CELL scalar -- which
    # leaves the non-negativity projection a plain clamp, with no per-cell NNLS required.
    # Scaled gradient projection in a fixed diagonal metric (Bonettini, Zanella & Zanni,
    # Inverse Problems 25:015002, 2009).
    gen = torch.Generator(device=device).manual_seed(0)
    v = torch.randn(P, K, device=device, generator=gen) * hf
    v = v / v.norm().clamp_min(1e-12)
    L = 1.0
    for _ in range(20):
        Hv = Hb.matvec(v) * hf
        L = float(Hv.norm())
        v = Hv / max(L, 1e-12)
    if a.precond:
        rs = Hb.row_block_norms()
        # A cell with no edge in S was never crossed by any ray. Clamping its L_j to a tiny
        # epsilon would hand it a step of ~1e12; it is inert today only because `have` masks it,
        # which is too fragile to rely on. Give it exactly zero step instead.
        alive = rs > 0
        eta = torch.zeros(P, device=device)
        eta[alive] = 1.0 / rs[alive]
        eta = eta[:, None]
        q = torch.quantile(rs[alive].float(), torch.tensor([0.01, 0.5, 0.99], device=device))
        print(f"[precond] per-cell L_j percentiles 1/50/99: {q[0]:.3e} {q[1]:.3e} {q[2]:.3e}"
              f"   spread {float(q[2]/q[0]):.1f}x   (global L={L:.4e}, {int(alive.sum())}/{P} cells alive)", flush=True)
    else:
        eta = 1.0 / max(L, 1e-12)
        print(f"[fista] global L={L:.4e}  eta={eta:.4e}", flush=True)

    # ---- change 2: adaptive gradient restart (O'Donoghue & Candes, FoCM 2015) --------
    # Plain FISTA drives the momentum coefficient toward 1 regardless of the true strong-
    # convexity modulus, so on an ill-conditioned problem the iterates become UNDER-DAMPED and
    # ripple with period ~sqrt(L/mu) instead of converging. Observed here exactly: the KKT
    # residual fell to 1.82e1 by iteration 275 and then jumped back to 6.11e1 by 299.
    #
    # The generalized-gradient restart test for a projected/proximal scheme is
    #       (y_k - x_{k+1})^T (x_{k+1} - x_k) > 0   ->   reset momentum,
    # which uses only quantities the iteration already computed: no extra Hessian application,
    # no extra gradient, and no knowledge of the strong-convexity modulus. Restarting recovers
    # the optimal linear rate O(sqrt(L/mu) log 1/eps) in place of FISTA's O(1/k^2).
    y = coef.clone()
    t = 1.0
    n_restarts = 0
    t0 = time.time()
    for it in range(a.iters):
        g = grad(y)
        new = (y - eta * g).clamp_min(0.0) * hf
        restart_stat = float(((y - new) * (new - coef)).sum())
        if a.restart and restart_stat > 0.0:
            t_new = 1.0
            y = new.clone()
            n_restarts += 1
        else:
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = (new + ((t - 1.0) / t_new) * (new - coef)).clamp_min(0.0) * hf
        coef, t = new, t_new
        if it % 25 == 0 or it == a.iters - 1:
            gg = grad(coef)
            print(f"[fista] iter {it:4d}  obj {obj(coef):.6e}  KKT {kkt(coef, gg):.6e}  "
                  f"restarts {n_restarts}  stat {restart_stat:+.3e}  ({time.time()-t0:.1f}s)", flush=True)

    f = torch.einsum("pk,pkd->pd", coef, U)
    nz = valid & (f.norm(dim=-1) > 0)
    cos = F.cosine_similarity(F.normalize(f[nz], dim=-1), F.normalize(x_diag[nz], dim=-1), dim=-1)
    print(f"[compare] cosine(cone, diagonal): median {float(cos.median()):.4f}  "
          f"frac<0.99 {float((cos<0.99).float().mean())*100:.2f}%", flush=True)
    print(f"[cone] active basis vectors per cell (median): "
          f"{float((coef > 1e-8).sum(1)[nz].float().median()):.1f} of {K}", flush=True)
    torch.save({"primitive_features": f.cpu(), "valid_mask": nz.cpu()}, a.output)
    print(f"[solve_cone_fast] {int(nz.sum())}/{P} valid -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
