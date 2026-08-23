"""How much does pruning weak couplings actually change the operator?

Pruning is an APPROXIMATION, unlike everything else in the fast path, so it needs its own error
measurement rather than an assurance. The question is not "does it save memory" (obviously) but
"at what budget does the operator stop matching the exact one".

Measured on scene0347_00, whose 16.2M edges fit exactly, so the unpruned operator is available
as ground truth. Budgets below 16.2M then emulate what the large scenes are forced to do.

The relevant error is on H a for random a, not on the matrix entries: a pruned entry matters only
insofar as it moves the gradient.
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
from gram_blocks import accumulate_view_pairs, merge, maybe_merge, build_blocks, prune_edges

D = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--K", type=int, default=7)
    p.add_argument("--budgets", default="16000000,8000000,4000000,2000000,1000000")
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

    keys, svals = [], []
    for vi in range(len(cameras)):
        cam = cameras[vi]
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vv = out_val.reshape(-1)[keep]
        accumulate_view_pairs(cols, vv, slots_used, P, keys, svals)
        del out_col, out_val, cols, vv
        torch.cuda.empty_cache()
        maybe_merge(keys, svals, limit=20_000_000)
    maybe_merge(keys, svals, force=True)
    k_all, v_all = keys[0], svals[0]
    print(f"[exact] {k_all.numel():,} edges", flush=True)

    g = torch.Generator(device=device).manual_seed(0)
    U = F.normalize(torch.randn(P, K, D, device=device, generator=g), dim=-1)
    H_exact = build_blocks(k_all, v_all, U, P, K)
    probes = [torch.randn(P, K, device=device, generator=g) for _ in range(3)]
    ref = [H_exact.matvec(x) for x in probes]
    del H_exact
    torch.cuda.empty_cache()

    print(f"\n{'budget':>12}{'edges':>14}{'mass kept':>12}{'rel err on Ha':>16}{'blocks GiB':>12}")
    print("-" * 66)
    for b in [int(x) for x in a.budgets.split(",")]:
        kk, vv, mass = prune_edges(k_all, v_all, P, b, verbose=False)
        Hp = build_blocks(kk, vv, U, P, K)
        errs = []
        for x, r in zip(probes, ref):
            y = Hp.matvec(x)
            errs.append(float((y - r).norm() / r.norm()))
        print(f"{b:>12,}{kk.numel():>14,}{mass*100:>11.3f}%{sum(errs)/len(errs):>16.3e}"
              f"{Hp.bytes()/2**30:>12.2f}")
        del Hp
        torch.cuda.empty_cache()
    print("\n  interpretation: S_jl is a sum of non-negative products, so small entries are")
    print("  genuinely weak couplings, not large ones that cancelled. A budget whose relative")
    print("  error on Ha is well below the ~1 mIoU seed noise is safe to use on big scenes.")


if __name__ == "__main__":
    main()
