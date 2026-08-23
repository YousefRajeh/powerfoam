"""The full coupled deconvolution solve, which no baseline in this comparison performs.

WHAT EVERY METHOD HERE ACTUALLY SOLVES

Each pixel contributes ONE blended observation of the feature field:

    b_r  =  sum_j  A[r, j] * f_j          A[r, j] = alpha_j * T_j   (the render weight)

so recovering the per-primitive features from the observations is a linear inverse problem,
`min_f ||A f - b||^2 + lam ||f||^2`, whose normal equations are

    (A^T A + lam I) f  =  A^T b .

Every training-free lifter in this comparison -- ours included, and NormLift, SFS, Occam's,
LUDVIG -- approximates `A^T A` by its DIAGONAL. Back-projection with weight normalization IS
`f_j = (A^T b)_j / (A^T A)_jj`. That is exact only if distinct primitives never share a pixel.
They always do: on scene0347_00 the median pixel deposits into 12 cells, and the median cell's
per-pixel weight entropy is 1.65. The off-diagonal mass is precisely the cross-talk between
primitives that co-occur along rays, which is exactly the contamination the whole lifting
stage suffers from -- and the diagonal approximation throws it away by construction.

WHY THIS HAS NEVER BEEN RUN AT SCANNET SCALE

`ridge_pcg` in feature_foam_lifting.operator does the real solve, but it takes a materialized
`SparseFeatureOperator` plus the dense observation matrix `b`. For one ScanNet scene `b` is
(n_views * H * W, 512) = 54 * 1.25M * 512 floats ~ 138 TB. That is why the coupled solve was
documented as "batch/view-limited" and never actually run on a full scene.

THE OBSERVATION THAT MAKES IT FEASIBLE

Conjugate gradients on the normal equations never needs `b`. It needs
  (1) the right-hand side  A^T b, and
  (2) a matrix-vector product  x -> A^T A x.
And `A^T b` is EXACTLY the `numerator` accumulator the streaming pipeline already computes --
the same tensor `solve_weighted_from_stats` divides by `support`. So the 138 TB object is never
required, only a quantity we have been computing all along for free.

`A^T A x` is applied view by view without ever forming `A^T A` (which would be a P x P matrix
with the sparsity of "shares a pixel with"):

    t_r   = sum_j A[r, j] x_j        (scatter nonzeros into per-pixel sums)
    out_j = sum_r A[r, j] t_r        (gather back to primitives)

Both passes are chunked over nonzeros, so peak memory is bounded by the chunk size and never
by total nnz -- the same trap that OOM'd the facet-edge gather (37GB) and the lifting gather
(30GB) earlier in this project.

The ray triples are cached once so CG iterations cost arithmetic rather than re-rendering.
Rows are NOT cached: they are reconstructible exactly from each view's slot counter via
repeat_interleave, which cuts cache memory by a third.

Preconditioner is Jacobi on diag(A^T A) + lam, i.e. exactly the operator the diagonal
approximation inverts -- so CG starts from the incumbent solution's scaling and every
iteration measures what the COUPLING adds on top of it. If the coupled solve is no better,
that is a real and publishable negative: it would say the off-diagonal mass is not what limits
open-vocabulary lifting, which no paper in this space has checked.
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

D = 512


def nnz_chunks(n, ch):
    for s in range(0, n, ch):
        yield s, min(s + ch, n)


class CachedRayOperator:
    """A stored view-by-view, as compacted (cols, vals) plus each view's slot counter."""

    def __init__(self, device, chunk=1_000_000):
        self.views = []          # (cols int32, vals float32, slots int32, npix)
        self.device = device
        self.chunk = chunk
        self.P = None

    def add_view(self, cols, vals, slots, npix):
        self.views.append((cols.to(torch.int32), vals.to(torch.float32),
                           slots.to(torch.int32), npix))

    def _rows(self, slots, npix):
        return torch.repeat_interleave(
            torch.arange(npix, device=slots.device), slots.long())

    def cached_bytes(self):
        return sum(c.numel() * 4 + v.numel() * 4 + s.numel() * 4 for c, v, s, _ in self.views)

    def AtA(self, x, lam_diag):
        """x: (P, D) -> (A^T A + diag(lam)) x, chunked so peak memory is bounded."""
        out = x * lam_diag[:, None]
        for cols, vals, slots, npix in self.views:
            cols_l = cols.long()
            rows = self._rows(slots, npix)
            t = torch.zeros(npix, D, device=x.device, dtype=x.dtype)
            for s, e in nnz_chunks(cols.numel(), self.chunk):
                t.index_add_(0, rows[s:e], vals[s:e, None] * x[cols_l[s:e]])
            for s, e in nnz_chunks(cols.numel(), self.chunk):
                out.index_add_(0, cols_l[s:e], vals[s:e, None] * t[rows[s:e]])
            del t, rows
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="lam as a fraction of mean(diag(A^T A))")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--rtol", type=float, default=1e-4)
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
    images_dir = Path(args.data_path) / args.scene / "images"
    stems = sorted(q.stem for q in images_dir.iterdir())
    assert len(stems) == len(cameras), f"{len(stems)} images vs {len(cameras)} cameras"
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))

    P = model.points.shape[0]
    feat_dir = Path(a.feature_folder)

    # ---- pass 1: cache the ray operator and accumulate A^T b and diag(A^T A) -------
    op = CachedRayOperator(device, chunk=a.chunk)
    Atb = torch.zeros(P, D, device=device)
    diag = torch.zeros(P, device=device)
    support = torch.zeros(P, device=device)
    t0 = time.time()
    used = 0
    for vi in range(n_views):
        cam = cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            print(f"  [skip] no feature for {stems[vi]}", flush=True)
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H, W,
                                                   sam_level=a.sam_level)
        if float(fmap.abs().max()) == 0.0:
            print(f"  [skip] all-zero feature map for {stems[vi]}", flush=True)
            continue
        used += 1
        out_col, out_val, slots, overflow, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = H * W
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.repeat_interleave(torch.arange(npix, device=device), slots_used.long())
        f_pix = fmap.reshape(-1, D)

        for s, e in nnz_chunks(cols.numel(), a.chunk):
            Atb.index_add_(0, cols[s:e], vals[s:e, None] * f_pix[rows[s:e]])
        diag.index_add_(0, cols, vals * vals)
        support.index_add_(0, cols, vals)
        op.add_view(cols, vals, slots_used, npix)
        del out_col, out_val, fmap, f_pix, rows, cols, vals
        torch.cuda.empty_cache()
        if (vi + 1) % 20 == 0:
            print(f"  cached view {vi+1}/{n_views}  "
                  f"({op.cached_bytes()/2**30:.1f} GiB, {time.time()-t0:.0f}s)", flush=True)

    print(f"[cache] {used} views, {op.cached_bytes()/2**30:.2f} GiB of triples, "
          f"{time.time()-t0:.0f}s", flush=True)

    valid = support > 0
    lam = a.ridge * float(diag[valid].mean())
    lam_diag = torch.full((P,), lam, device=device)
    print(f"[ridge] lam = {lam:.6g}  ({a.ridge} x mean diag {float(diag[valid].mean()):.6g})",
          flush=True)

    # ---- the incumbent: the diagonal approximation everyone else uses ---------------
    x_diag = torch.zeros(P, D, device=device)
    x_diag[valid] = Atb[valid] / support[valid][:, None]

    # ---- preconditioned CG on (A^T A + lam I) x = A^T b ----------------------------
    M = (diag + lam).clamp_min(1e-20)
    x = x_diag.clone()                      # start from the incumbent
    r = Atb - op.AtA(x, lam_diag)
    z = r / M[:, None]
    pdir = z.clone()
    rz = (r * z).sum()
    b_norm = Atb.norm()
    print(f"[cg] initial residual {float(r.norm()/b_norm):.6f} (relative)", flush=True)
    for it in range(a.iters):
        Ap = op.AtA(pdir, lam_diag)
        denom = (pdir * Ap).sum()
        if float(denom) <= 0:
            print(f"[cg] non-positive curvature at iter {it}, stopping", flush=True)
            break
        alpha = rz / denom
        x = x + alpha * pdir
        r = r - alpha * Ap
        rel = float(r.norm() / b_norm)
        if it % 5 == 0 or rel < a.rtol:
            print(f"[cg] iter {it:3d}  relative residual {rel:.6f}", flush=True)
        if rel < a.rtol:
            break
        z = r / M[:, None]
        rz_new = (r * z).sum()
        pdir = z + (rz_new / rz) * pdir
        rz = rz_new

    # how far did the coupled solve actually move from the diagonal answer?
    cos = F.cosine_similarity(F.normalize(x[valid], dim=-1),
                              F.normalize(x_diag[valid], dim=-1), dim=-1)
    print(f"[compare] cosine(coupled, diagonal) over valid cells: "
          f"median {float(cos.median()):.4f}  mean {float(cos.mean()):.4f}  "
          f"frac below 0.99: {float((cos < 0.99).float().mean())*100:.2f}%", flush=True)

    torch.save({"primitive_features": x.cpu(), "valid_mask": valid.cpu()}, a.output)
    print(f"[solve_coupled_ridge] {int(valid.sum())}/{P} valid -> {a.output}", flush=True)
    diag_out = a.output.replace(".pt", "_diagbaseline.pt")
    torch.save({"primitive_features": x_diag.cpu(), "valid_mask": valid.cpu()}, diag_out)
    print(f"[solve_coupled_ridge] diagonal baseline -> {diag_out}", flush=True)


if __name__ == "__main__":
    main()
