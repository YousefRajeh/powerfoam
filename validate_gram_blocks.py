"""Prove the block-sparse coefficient-space Hessian equals the matrix-free one.

A ~400x faster operator that computes something slightly different is worse than no speedup at
all, because every downstream conclusion would be quietly wrong. So before the fast path is used
for any result, it is checked against the slow path it replaces:

    fast:  (H a)_j = sum_l S_{jl} (U_j U_l^T) a_l        block-sparse, K x K blocks
    slow:  (H a)_j = U_j^T ( A^T A (U a) )_j             streams the ray triples in 512-d

These are the same quantity by construction, so any disagreement beyond float tolerance is a bug
in the pair expansion (most likely the range-expansion indexing or the symmetric upper-triangle
handling, where a missing transpose or a double-counted diagonal would show up immediately).

Run on few views so the slow reference is affordable.
"""
import argparse
import time

import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from solve_coupled_ridge import CachedRayOperator, D
from gram_blocks import accumulate_view_pairs, merge, build_blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--max-views", type=int, default=3)
    p.add_argument("--K", type=int, default=7)
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
    P, K = model.points.shape[0], a.K
    n_views = min(a.max_views, len(cameras))

    # a random but FIXED per-cell basis; correctness of the identity does not depend on U being
    # the real observed embeddings, and a random U is a stricter test (no accidental structure)
    g = torch.Generator(device=device).manual_seed(0)
    U = F.normalize(torch.randn(P, K, D, device=device, generator=g), dim=-1)

    op = CachedRayOperator(device)
    keys, vals_s = [], []
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
        op.add_view(cols, vv, slots_used, npix)
        accumulate_view_pairs(cols, vv, slots_used, P, keys, vals_s)
        del out_col, out_val
        torch.cuda.empty_cache()

    k, v = merge(keys, vals_s)
    print(f"{n_views} views: {k.numel():,} unique edges", flush=True)
    H = build_blocks(k, v, U, P, K)
    print(f"blocks built: {H.bytes()/2**30:.2f} GiB", flush=True)

    zero = torch.zeros(P, device=device)
    torch.manual_seed(0)
    for trial in range(3):
        av = torch.randn(P, K, device=device, generator=g)

        torch.cuda.synchronize(); t0 = time.time()
        fast = H.matvec(av)
        torch.cuda.synchronize(); t_fast = time.time() - t0

        torch.cuda.synchronize(); t0 = time.time()
        f = torch.einsum("pk,pkd->pd", av, U)
        slow = torch.einsum("pkd,pd->pk", U, op.AtA(f, zero))
        torch.cuda.synchronize(); t_slow = time.time() - t0

        num = float((fast - slow).abs().max())
        den = float(slow.abs().max())
        rel = float((fast - slow).norm() / slow.norm())
        print(f"  trial {trial}: max|diff|={num:.4e}  scale={den:.4e}  relative={rel:.3e}   "
              f"fast={t_fast*1000:.1f} ms  slow={t_slow*1000:.1f} ms  "
              f"({t_slow/max(t_fast,1e-9):.0f}x)", flush=True)

    print("\n  a relative error ~1e-6 or below is float accumulation order, not a bug;")
    print("  anything ~1e-2 or larger means the pair expansion or the symmetric handling is wrong.")


if __name__ == "__main__":
    main()
