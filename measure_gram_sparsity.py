"""How sparse is A^T A actually? This decides whether the coefficient-space reformulation is viable.

The block-sparse plan stores one K x K matrix per nonzero EDGE of S = A^T A. At K=7 that is
49 floats = 196 bytes per edge, so 50M edges is ~10 GiB and 250M edges is ~49 GiB and the plan
is dead. My estimate of "tens of millions" was a guess from ~12 cells per ray; this measures it
on the real exported operator instead of trusting the guess.

Also reports what the payoff would be: the per-iteration cost of a block-sparse matvec versus the
measured 33 s of the 512-dimensional matrix-free application.
"""
import argparse
import time
from pathlib import Path

import torch
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from gram_blocks import accumulate_view_pairs, merge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--max-views", type=int, default=None)
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
    P = model.points.shape[0]
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))

    keys, vals_s = [], []
    total_pairs = 0
    total_nnz = 0
    t0 = time.time()
    for vi in range(n_views):
        cam = cameras[vi]
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vv = out_val.reshape(-1)[keep]
        total_nnz += cols.numel()
        total_pairs += accumulate_view_pairs(cols, vv, slots_used, P, keys, vals_s)
        del out_col, out_val, cols, vv
        # periodic merge keeps the key list from growing without bound
        if (vi + 1) % 8 == 0 or vi == n_views - 1:
            k, v = merge(keys, vals_s)
            keys, vals_s = [k], [v]
            print(f"  view {vi+1}/{n_views}: running unique edges = {k.numel():,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()

    k, v = merge(keys, vals_s)
    E = k.numel()
    K = a.K
    print(f"\n=== {a.scene}: A^T A sparsity ===")
    print(f"  cells P                      {P:,}")
    print(f"  ray nonzeros (nnz of A)      {total_nnz:,}")
    print(f"  pair-instances expanded      {total_pairs:,}")
    print(f"  UNIQUE EDGES of A^T A        {E:,}   (upper triangle incl. diagonal)")
    print(f"  mean coupled cells per cell  {2.0*E/P:.1f}")
    print(f"  build time                   {time.time()-t0:.0f}s")

    blk = E * K * K * 4 / 2**30
    idx = E * 16 / 2**30
    print(f"\n=== cost of the coefficient-space plan (K={K}) ===")
    print(f"  K x K blocks                 {blk:.2f} GiB")
    print(f"  edge indices                 {idx:.2f} GiB")
    print(f"  TOTAL resident               {blk+idx:.2f} GiB   (A6000 has 48 GiB)")
    traffic = E * (K * K + 2 * K) * 4 * 2 / 2**30
    print(f"  traffic per matvec           {traffic:.2f} GiB  -> ~{traffic/700*1000:.1f} ms "
          f"at 700 GB/s")
    print(f"  versus the 512-dim matrix-free application: 3.02 TiB, ~33 s")
    print(f"  predicted speedup            ~{3.02*1024/max(traffic,1e-9):.0f}x in traffic")
    verdict = "VIABLE" if blk + idx < 30 else ("TIGHT" if blk + idx < 45 else "NOT VIABLE")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
