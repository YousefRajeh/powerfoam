"""Deconvolution physics WITHOUT leaving the CLIP cone.

WHAT THE COUPLED SOLVE GOT RIGHT AND WRONG

`solve_coupled_ridge.py` established that the exact linear inverse is solvable at ScanNet scale
and that it is much WORSE than the diagonal approximation everyone uses (27.15 vs 36.12 mIoU on
scene0347_00), with a ridge sweep confirming the mechanism is an interpolation: as lam grows the
solution slides monotonically back toward the diagonal answer (cosine 0.9204 -> 0.9996 -> 1.0000
for lam 1e-2 -> 1e1 -> 1e2).

The diagnosis: deconvolution SUBTRACTS what neighbouring cells contributed to a shared pixel, so
the recovered coefficients go negative and the solution stops being a CONVEX combination of
observed CLIP embeddings. Cosine-against-text is calibrated only on the cone spanned by real
embeddings. The diagonal restriction keeps every solution inside that cone by construction --
it is an implicit manifold constraint, which is exactly what it is buying.

So the defect was never that we solve the coupling. It is that we solve it UNCONSTRAINED.

THE FORMULATION

Constrain each cell's feature to the non-negative cone of the embeddings ACTUALLY OBSERVED for
that cell. Rather than projecting after each step, parameterize the constraint away: with
u_{j,1..K} the (unit, on-manifold) per-view features observed for cell j,

    f_j  =  sum_k  a_{j,k} u_{j,k} ,        a >= 0

Any a >= 0 gives an f inside the cone by construction, so the constraint can never be violated.
This also shrinks the unknowns from (P, 512) to (P, K) with K ~ 6 -- the solve happens in the
span of what was really seen, not in all of R^512, which is itself the regularization the
unconstrained solve lacked.

Objective and gradient, using only machinery that already exists:

    L(a) = || A (U a) - b ||^2
    dL/da_{j,k} = < u_{j,k},  (A^T A f - A^T b)_j >

`A^T b` is the streaming `numerator` and `A^T A` is the cached view-by-view matvec from
solve_coupled_ridge -- the dense observation matrix b (~138 TB/scene) is never needed. Steps use
exact line search, valid because the objective is quadratic in a, then clamp at zero.

WHY THIS IS A POWER-DIAGRAM MOVE

Two cells couple in A^T A only if some ray crosses BOTH. With DISJOINT cells at ~12 per ray that
coupling is sparse and geometrically meaningful (adjacency, or occluder/occludee). A Gaussian
method has 50+ overlapping primitives per ray and no disjoint ownership, so its coupling is far
denser and its "observed features per primitive" are themselves blends rather than single mask
embeddings -- there is no clean cone to constrain to. The constraint is only well-posed because
the partition makes per-(cell, view) observations on-manifold in the first place, which the
mask-purity measurement independently confirmed (median purity exactly 1.0000).

KILL CRITERION: if this does not beat the diagonal baseline (36.12 on scene0347_00), the whole
"solve the coupling" direction is closed -- unconstrained loses, constrained loses, and the
diagonal stands as the right answer rather than a convenient one.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP
from solve_coupled_ridge import CachedRayOperator, nnz_chunks, D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--topk", type=int, default=6, help="observed views kept per cell (cone basis)")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--max-views", type=int, default=None)
    a = p.parse_args()

    device = "cuda"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", a.config])
    ckpt = a.config.replace("/config.yaml", "").replace("\\config.yaml", "")

    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    stems = sorted(q.stem for q in (Path(args.data_path) / args.scene / "images").iterdir())
    assert len(stems) == len(cameras)
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))

    P, K = model.points.shape[0], a.topk
    feat_dir = Path(a.feature_folder)

    op = CachedRayOperator(device, chunk=a.chunk)
    Atb = torch.zeros(P, D, device=device)
    support = torch.zeros(P, device=device)
    top_w = torch.zeros(P, K, device=device)
    U = torch.zeros(P, K, D, device=device)      # the cone basis: observed, on-manifold
    t0 = time.time()

    for vi in range(n_views):
        cam = cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H, W, sam_level=a.sam_level)
        if float(fmap.abs().max()) == 0.0:
            continue
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = H * W
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.repeat_interleave(torch.arange(npix, device=device), slots_used.long())
        f_pix = fmap.reshape(-1, D)

        w_cell = torch.zeros(P, device=device).index_add_(0, cols, vals)
        f_cell = torch.zeros(P, D, device=device)
        for s, e in nnz_chunks(cols.numel(), a.chunk):
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

        op.add_view(cols, vals, slots_used, npix)
        del out_col, out_val, fmap, f_pix, rows, cols, vals, f_cell, f_view
        torch.cuda.empty_cache()
        if (vi + 1) % 20 == 0:
            print(f"  cached {vi+1}/{n_views} ({op.cached_bytes()/2**30:.1f} GiB, "
                  f"{time.time()-t0:.0f}s)", flush=True)

    print(f"[cache] {op.cached_bytes()/2**30:.2f} GiB triples, {time.time()-t0:.0f}s", flush=True)
    valid = support > 0
    zero_lam = torch.zeros(P, device=device)

    def f_of_init(c, Ub):
        return torch.einsum("pk,pkd->pd", c, Ub)

    # ---- the incumbent diagonal solution, and its projection onto the cone ----------
    x_diag = torch.zeros(P, D, device=device)
    x_diag[valid] = Atb[valid] / support[valid][:, None]
    # ---- initialize at the LEAST-SQUARES projection of the diagonal answer onto the cone ----
    # Not the naive inner products <u_k, x>: the basis vectors u_jk are observations of the same
    # cell from different views, so they are strongly correlated (measured view consistency
    # ~0.78-0.82 cosine) and nowhere near orthonormal. Using inner products as coefficients
    # therefore counts the shared direction once per view and inflates ||f|| by roughly a factor
    # of K. Measured consequence when it was done that way: projected gradient started at
    # relative residual 1.94 versus 0.050 for the diagonal answer itself, and after 20 iterations
    # had only reached 0.285 -- the optimizer was spending its whole budget undoing the
    # initialization rather than improving on the incumbent.
    #
    # The correct projection solves the small normal equations per cell,
    #     G_j a_j = U_j x_j,   G_j = U_j U_j^T   (K x K, K ~ 6)
    # batched over all cells, then clamps at zero. Unobserved slots are pinned to a_k = 0 by
    # setting their diagonal to 1 with a zero right-hand side, which keeps G invertible without
    # perturbing the observed block.
    have = top_w > 0
    G = torch.einsum("pkd,pld->pkl", U, U)
    rhs = torch.einsum("pkd,pd->pk", U, x_diag)
    eye = torch.eye(K, device=device).expand(P, K, K)
    missing = ~have
    G = torch.where(missing[:, :, None] | missing[:, None, :], eye, G)
    G = G + 1e-4 * eye                      # ridge for the correlated-basis case
    rhs = rhs * have.float()
    coef = torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)
    coef = coef.clamp_min(0.0) * have.float()
    with torch.no_grad():
        r0 = float((op.AtA(f_of_init(coef, U), zero_lam) - Atb).norm() / Atb.norm())
        rd = float((op.AtA(x_diag, zero_lam) - Atb).norm() / Atb.norm())
    print(f"[init] relative residual: cone-projected diagonal {r0:.6f}  vs  diagonal {rd:.6f}",
          flush=True)

    def f_of(c):
        return torch.einsum("pk,pkd->pd", c, U)

    def grad_a(c):
        g_f = op.AtA(f_of(c), zero_lam) - Atb
        return torch.einsum("pkd,pd->pk", U, g_f), g_f

    # ---- FISTA (accelerated projected gradient) ------------------------------------
    # Plain projected gradient is first-order with a linear rate, and on this problem the basis
    # vectors within a cell are strongly correlated (view consistency ~0.78-0.82), so the
    # reduced Hessian is ill-conditioned and steepest descent crawls: 20 iterations only took
    # the residual from 1.94 to 0.285. FISTA costs one extra stored iterate and gets an O(1/k^2)
    # rate, which matters here because a slow optimizer would produce a FALSE NEGATIVE -- we
    # would conclude the constrained solve does not beat the diagonal when it simply had not
    # converged.
    #
    # FISTA needs a fixed step 1/L rather than a line search, so L (the top eigenvalue of the
    # reduced Hessian a -> U^T A^T A U a) is estimated by power iteration first.
    gen = torch.Generator(device=device).manual_seed(0)
    v = torch.randn(P, K, device=device, generator=gen) * have.float()
    v = v / v.norm().clamp_min(1e-12)
    L = 1.0
    for _ in range(8):
        Hv = torch.einsum("pkd,pd->pk", U, op.AtA(f_of(v), zero_lam)) * have.float()
        L = float(Hv.norm())
        v = Hv / max(L, 1e-12)
    eta = 1.0 / max(L, 1e-12)
    print(f"[fista] Lipschitz estimate L={L:.4e}, step eta={eta:.4e}", flush=True)

    y = coef.clone()
    t = 1.0
    best = (r0, coef.clone())
    for it in range(a.iters):
        g, _ = grad_a(y)
        g = g * have.float()
        coef_new = (y - eta * g).clamp_min(0.0) * have.float()
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = coef_new + ((t - 1.0) / t_new) * (coef_new - coef)
        y = y.clamp_min(0.0) * have.float()      # keep the extrapolated point feasible too
        coef, t = coef_new, t_new
        if it % 10 == 0 or it == a.iters - 1:
            res = float((op.AtA(f_of(coef), zero_lam) - Atb).norm() / Atb.norm())
            if res < best[0]:
                best = (res, coef.clone())
            print(f"[fista] iter {it:3d}  |grad|={float(g.norm()):.4e}  "
                  f"relative residual={res:.6f}", flush=True)
    # FISTA is not monotone, so keep the best iterate actually seen rather than the last
    if best[0] < float((op.AtA(f_of(coef), zero_lam) - Atb).norm() / Atb.norm()):
        print(f"[fista] keeping best iterate (residual {best[0]:.6f})", flush=True)
        coef = best[1]

    x = f_of(coef)
    nz = valid & (x.norm(dim=-1) > 0)
    cos = F.cosine_similarity(F.normalize(x[nz], dim=-1), F.normalize(x_diag[nz], dim=-1), dim=-1)
    print(f"[compare] cosine(cone-constrained, diagonal): median {float(cos.median()):.4f}  "
          f"mean {float(cos.mean()):.4f}  frac<0.99 {float((cos<0.99).float().mean())*100:.2f}%",
          flush=True)
    print(f"[cone] active basis vectors per cell (median): "
          f"{float((coef > 1e-8).sum(1)[nz].float().median()):.1f} of {K}", flush=True)

    torch.save({"primitive_features": x.cpu(), "valid_mask": nz.cpu()}, a.output)
    print(f"[solve_cone_constrained] {int(nz.sum())}/{P} valid -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
