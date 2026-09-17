"""Build the top-K co-visibility graph for ANY reconstruction arm, without a gram cache.

WHY. `covis_graph.covis_graph` reads `artifacts/scannet/<scene>/gram_cache_*.pt` and asserts
`gram P == model P`. The only caches on disk are the nonfrozen ones (`P=154830` on scene0062),
while the REPORTED configuration is `truefrozen` (`P=51610`), so covis diffusion -- the one
score-side idea that survives unit-norm features -- cannot be tested on the reported arm at all.
This builds the graph straight from the renderer by streaming `export_feature_operator`, the same
way `measure_cross_surface_streaming.py` does, and writes it in the shape `ppr_diffuse` wants.

WHAT IT COMPUTES. For each view, `export_feature_operator` gives per-pixel hit lists
`(row, col, val)`; a ray contributes `A_ij A_il` to `G_jl` for every pair `(j, l)` of primitives it
crosses. G is accumulated as an upper-triangular COO in int64-keyed form and reduced to the top-K
neighbours per node at the end -- never materialising the dense P x P.

MEMORY. Pairs per ray are `k(k-1)/2` with `k <= max_hits`, so a 640x480 view at k=6 yields up to
4.6M pairs. Those are reduced per view (`torch.unique` on the int64 key) before being merged into
the running accumulator, which keeps the peak bounded by one view's pairs plus the accumulated
sparsity rather than growing with view count. An earlier attempt in this project built a 764M-nnz
COO in one shot and OOMed; hence the per-view reduction.

KEY EXACTNESS. Keys are `lo * P + hi` in int64. This is exact for P up to ~3e9. A float32
composite key was used earlier in this project and is exact only to 2^24, which would have
silently mis-sorted at P=1.1M -- do not "optimise" this back to float.
"""
from __future__ import annotations

import argparse
import os
import time

import configargparse
import torch
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def view_pairs(cols: torch.Tensor, vals: torch.Tensor, slots_used: torch.Tensor,
               P: int, max_hits: int, pair_budget: int = 1 << 24):
    """Upper-triangular (key, weight) contributions of one view.

    GROUPED BY HIT COUNT. A dense `(n_rays, max_hits)` expansion costs
    `n_rays * max_hits*(max_hits-1)/2` regardless of how many hits the rays actually have, and at
    max_hits=64 that is 2016 pairs for every ray -- 619M for one 640x480 view, which OOMs a 48 GB
    card. Mean hits per ray in this data is ~2.7, so the dense form wastes almost all of that.

    Instead rays are grouped by their exact hit count k (at most `max_hits` distinct groups) and
    each group is paired with a k x k triu. Total work becomes `sum_k n_k * k(k-1)/2` -- exactly
    the number of pairs that exist. Groups are further chunked to `pair_budget` pairs so a single
    huge group cannot spike memory either.
    """
    device = cols.device
    # ragged -> per-ray start offsets
    k = slots_used.to(torch.long)
    starts = torch.zeros_like(k)
    starts[1:] = torch.cumsum(k, 0)[:-1]

    out_k, out_w = [], []
    for kk in torch.unique(k):
        kk_i = int(kk)
        if kk_i < 2:                          # a single-hit ray has no co-visibility
            continue
        rows = (k == kk).nonzero(as_tuple=True)[0]
        ii, jj = torch.triu_indices(kk_i, kk_i, offset=1, device=device)
        n_pairs = ii.numel()
        chunk = max(1, pair_budget // max(n_pairs, 1))
        for s in range(0, rows.numel(), chunk):
            r = rows[s:s + chunk]
            base = starts[r]                                  # (n,)
            idx = base[:, None] + torch.arange(kk_i, device=device)[None, :]
            c = cols[idx]                                     # (n, kk)
            v = vals[idx]
            a_c, b_c = c[:, ii].reshape(-1), c[:, jj].reshape(-1)
            w = (v[:, ii] * v[:, jj]).reshape(-1)
            nz = w > 0
            if not bool(nz.any()):
                continue
            a_c, b_c, w = a_c[nz], b_c[nz], w[nz]
            lo = torch.minimum(a_c, b_c)
            hi = torch.maximum(a_c, b_c)
            key = lo * P + hi
            uk, inv = torch.unique(key, return_inverse=True)
            out_k.append(uk)
            out_w.append(torch.zeros(uk.numel(), device=device).index_add_(0, inv, w))

    if not out_k:
        return (torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, device=device))
    ck = torch.cat(out_k)
    cw = torch.cat(out_w)
    ukey, inv = torch.unique(ck, return_inverse=True)
    uw = torch.zeros(ukey.numel(), device=device).index_add_(0, inv, cw)
    return ukey, uw


def merge(acc_k, acc_w, key, w):
    if acc_k is None:
        return key, w
    ck = torch.cat([acc_k, key])
    cw = torch.cat([acc_w, w])
    uk, inv = torch.unique(ck, return_inverse=True)
    uw = torch.zeros(uk.numel(), device=cw.device).index_add_(0, inv, cw)
    return uk, uw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--max-hits", type=int, default=6, help="matches the gram cache's kmax=6")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--out", default=None, help="default artifacts/scannet/<scene>/covis_<variant>.pt")
    a = ap.parse_args()

    dev = "cuda"
    wp.init()
    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt}/model.pt")

    P = model.points.shape[0]
    cameras = dh.cameras
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    ar = torch.arange(a.max_hits, device=dev)
    acc_k = acc_w = None
    t0 = time.time()

    for vi in range(n_views):
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cameras[vi], max_intersections=1024, max_hits_per_pixel=a.max_hits)
        slots_used = slots.reshape(-1).clamp(max=a.max_hits)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        k, w = view_pairs(cols, vals, slots_used, P, a.max_hits)
        acc_k, acc_w = merge(acc_k, acc_w, k, w)
        del out_col, out_val, cols, vals, slots_used, keep, k, w
        torch.cuda.empty_cache()
        if (vi + 1) % 10 == 0 or vi == n_views - 1:
            print(f"  view {vi+1}/{n_views}: {acc_k.numel():,} unique pairs "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = a.out or f"artifacts/scannet/{a.scene}/covis_{a.variant}.pt"
    torch.save({"P": P, "S_keys": acc_k.cpu(), "S_vals": acc_w.cpu(),
                "kmax": a.max_hits, "n_views": n_views, "variant": a.variant}, out)
    print(f"{a.scene}/{a.variant}: P={P:,} pairs={acc_k.numel():,} -> {out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
