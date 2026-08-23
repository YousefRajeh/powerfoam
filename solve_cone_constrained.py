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

    # ---- the incumbent diagonal solution, and its projection onto the cone ----------
    x_diag = torch.zeros(P, D, device=device)
    x_diag[valid] = Atb[valid] / support[valid][:, None]
    # initialize a with the cone coefficients of the diagonal answer (clamped >= 0)
    coef = torch.einsum("pkd,pd->pk", U, x_diag).clamp_min(0.0)
    have = top_w > 0
    coef = coef * have.float()

    def f_of(c):
        return torch.einsum("pk,pkd->pd", c, U)

    def grad_a(c):
        g_f = op.AtA(f_of(c), zero_lam) - Atb
        return torch.einsum("pkd,pd->pk", U, g_f), g_f

    print("[pg] projected gradient with exact line search (objective is quadratic in a)",
          flush=True)
    for it in range(a.iters):
        g, _ = grad_a(coef)
        g = g * have.float()
        # only descend on free coordinates: at a=0 a positive gradient would push negative
        active = (coef > 0) | (g < 0)
        g = g * active.float()
        gn = float(g.norm())
        if gn == 0.0:
            print(f"[pg] iter {it}: zero gradient, stopping", flush=True)
            break
        d_f = f_of(g)
        Hd = op.AtA(d_f, zero_lam)
        denom = float((d_f * Hd).sum())
        if denom <= 0:
            print(f"[pg] iter {it}: non-positive curvature, stopping", flush=True)
            break
        eta = float((g * g).sum()) / denom
        coef = (coef - eta * g).clamp_min(0.0) * have.float()
        if it % 10 == 0:
            fj = f_of(coef)
            res = float((op.AtA(fj, zero_lam) - Atb).norm() / Atb.norm())
            print(f"[pg] iter {it:3d}  |grad|={gn:.4e}  relative residual={res:.6f}", flush=True)

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
