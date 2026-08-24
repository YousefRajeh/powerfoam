"""Prove SparseGram.matvec equals the matrix-free A^T A, before any number it produces is used.

Same discipline as validate_gram_blocks.py, and for the same reason: the block-sparse Hessian was
90x faster and WRONG (relative error 4.5e-02 -- off-diagonal edges counted four times) until it was
checked against the slow path. solve_spmm_exact.py reuses exactly that upper-triangle convention,
so it inherits exactly that failure mode.

    fast:  (S X)_j = sum_l S_{jl} X_l          SparseGram.matvec, one scalar per edge
    slow:  (A^T A X)_j                         CachedRayOperator.AtA, streams the ray triples

These are the same quantity by construction (S = A^T A), so any disagreement beyond float
accumulation order is a bug. Unlike the block version there is no basis in between, so a mismatch
here can only be the pair expansion or the symmetric upper-triangle handling.

Also times both operators on the SAME views, which is the honest way to measure the claimed
50-150x on the EXACT operator (no cone, no basis, no restriction).
"""
import argparse
import time

import torch
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from solve_coupled_ridge import CachedRayOperator, D
from gram_blocks import accumulate_view_pairs, merge, maybe_merge
from solve_spmm_exact import SparseGram


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--max-views", type=int, default=3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--chunk", type=int, default=2_000_000, help="edge chunk in SparseGram.matvec")
    p.add_argument("--merge-limit", type=int, default=20_000_000)
    p.add_argument("--skip-slow", action="store_true",
                   help="timing only for the fast path (when the cached triples will not fit)")
    a = p.parse_args()

    device = "cuda"
    wp.init()
    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    P = model.points.shape[0]
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    print(f"{a.scene}/{a.variant}: {P:,} cells, {n_views}/{len(cameras)} views", flush=True)

    op = CachedRayOperator(device)
    keys, vals_s = [], []
    t0 = time.time()
    for vi in range(n_views):
        cam = cameras[vi]
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = int(cam.height) * int(cam.width)
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vv = out_val.reshape(-1)[keep]
        if not a.skip_slow:
            op.add_view(cols, vv, slots_used, npix)
        del out_col, out_val
        torch.cuda.empty_cache()
        accumulate_view_pairs(cols, vv, slots_used, P, keys, vals_s)
        del cols, vv
        torch.cuda.empty_cache()
        maybe_merge(keys, vals_s, limit=a.merge_limit)
        if vi % 10 == 0:
            print(f"  view {vi}: alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
                  f"{time.time()-t0:.0f}s", flush=True)

    maybe_merge(keys, vals_s, force=True)
    k, v = keys[0], vals_s[0]
    print(f"{n_views} views: {k.numel():,} unique edges, cached triples "
          f"{op.cached_bytes()/2**30:.2f} GiB, build {time.time()-t0:.0f}s", flush=True)

    S = SparseGram(k, v, P)
    zero = torch.zeros(P, device=device)
    g = torch.Generator(device=device).manual_seed(0)

    for trial in range(a.trials):
        # a full 512-channel dense block, exactly the object the solver applies S to
        X = torch.randn(P, D, device=device, generator=g)

        torch.cuda.synchronize(); t0 = time.time()
        fast = S.matvec(X, chunk=a.chunk)
        torch.cuda.synchronize(); t_fast = time.time() - t0

        if a.skip_slow:
            print(f"  trial {trial}: fast={t_fast*1000:.1f} ms  (slow reference skipped)",
                  flush=True)
            del X, fast
            torch.cuda.empty_cache()
            continue

        torch.cuda.synchronize(); t0 = time.time()
        slow = op.AtA(X, zero)
        torch.cuda.synchronize(); t_slow = time.time() - t0

        num = float((fast - slow).abs().max())
        den = float(slow.abs().max())
        rel = float((fast - slow).norm() / slow.norm())
        print(f"  trial {trial}: max|diff|={num:.4e}  scale={den:.4e}  relative={rel:.3e}   "
              f"fast={t_fast*1000:.1f} ms  slow={t_slow*1000:.1f} ms  "
              f"({t_slow/max(t_fast,1e-9):.1f}x)", flush=True)
        del X, fast, slow
        torch.cuda.empty_cache()

    print("\n  relative error ~1e-6 or below is float accumulation order, not a bug;")
    print("  anything ~1e-5 or larger means the pair expansion or the symmetric handling is wrong.")


if __name__ == "__main__":
    main()
