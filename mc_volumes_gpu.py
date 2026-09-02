"""GPU Monte-Carlo power-cell volumes. Same membership rule, ~ two orders of magnitude faster.

WHY THE CPU VERSION IS SLOW. `mc_cell_volumes` chunks 500k samples at a time and calls
`assign_points_to_power_cells`, which runs a k=64 nearest-neighbour query against 700,000 centres in
FLOAT64 on the CPU, then evaluates exact power distances among those candidates. At 24M samples that
is 48 chunks x (500k x 64) double-precision candidate evaluations plus 48 KD-tree queries. The tree
query dominates and none of it touches the GPU.

THE GPU VERSION. Everything stays in VRAM:
  * cells are bucketed into a uniform grid sized so a bucket holds ~`per_bucket` centres;
  * buckets are stored CSR-style (sorted by bucket id + offsets), so no ragged gathers;
  * each sample visits its own bucket and the 26 around it, computing the EXACT power distance
    `||x-c||^2 - r^2` against those candidates and taking the true argmin.

The candidate filter is spatial, exactly as on the CPU, and the final decision is the exact power
distance -- so this is the same approximation the CPU path already makes, not a new one. The grid
radius is what bounds it: a cell whose centre is more than one bucket away can only win if its
radius is large relative to bucket size, which `--rings` can widen if ever needed.
"""
import argparse

import numpy as np
import torch


@torch.no_grad()
def mc_cell_volumes_gpu(centers, radii, valid, n_samples, device="cuda",
                        per_bucket=8.0, chunk=250_000, rings=1, seed=0, verbose=False,
                        cand_block=128):
    c = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=device)
    r = torch.as_tensor(np.asarray(radii), dtype=torch.float32, device=device).reshape(-1)
    P = c.shape[0]
    vmask = (torch.as_tensor(np.asarray(valid), device=device).bool() if valid is not None
             else torch.ones(P, dtype=torch.bool, device=device))
    lo, hi = c.min(0).values, c.max(0).values
    ext = (hi - lo).clamp_min(1e-6)
    nb = max(1, int(round((float(vmask.sum()) / per_bucket) ** (1 / 3))))
    res = torch.tensor([nb, nb, nb], device=device)
    cell_of = lambda x: ((x - lo) / ext * (res - 1e-6)).floor().long().clamp_(min=torch.zeros(3, dtype=torch.long, device=device), max=res - 1)

    idx_valid = torch.nonzero(vmask, as_tuple=False).squeeze(1)
    b = cell_of(c[idx_valid])
    bid = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    order = torch.argsort(bid)
    bid_s, cells_s = bid[order], idx_valid[order]
    counts = torch.bincount(bid_s, minlength=nb ** 3)
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
    if verbose:
        print(f"  grid {nb}^3, mean {float(vmask.sum())/nb**3:.1f} cells/bucket, "
              f"max {int(counts.max())}", flush=True)

    off = torch.arange(-rings, rings + 1, device=device)
    dz, dy, dx = torch.meshgrid(off, off, off, indexing="ij")
    nbr = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], 1)   # (R,3)

    vol = torch.zeros(P, dtype=torch.float64, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    done = 0
    while done < n_samples:
        n = min(chunk, n_samples - done)
        pts = torch.rand(n, 3, generator=g, device=device) * ext + lo
        pb = cell_of(pts)
        best = torch.full((n,), float("inf"), device=device)
        arg = torch.full((n,), -1, dtype=torch.long, device=device)
        for k in range(nbr.shape[0]):
            q = (pb + nbr[k]).clamp_(min=torch.zeros(3, dtype=torch.long, device=device),
                                     max=res - 1)
            qid = (q[:, 0] * nb + q[:, 1]) * nb + q[:, 2]
            s, e = offs[qid], offs[qid + 1]
            m = int((e - s).max())
            if m == 0:
                continue
            # Cells lie on SURFACES, so a uniform grid over the bounding box is wildly uneven:
            # most buckets are empty and the densest holds thousands. Materialising `chunk x m`
            # candidates asked for 73 GiB. Block over the candidate axis instead, so peak memory
            # is chunk x block regardless of how skewed the occupancy is.
            for b0 in range(0, m, cand_block):
                b1 = min(b0 + cand_block, m)
                ar = torch.arange(b0, b1, device=device)
                take = (s[:, None] + ar[None, :]).clamp(max=cells_s.numel() - 1)
                ok = ar[None, :] < (e - s)[:, None]
                if not bool(ok.any()):
                    del take, ok
                    continue
                ci = cells_s[take]
                d = ((pts[:, None, :] - c[ci]) ** 2).sum(-1) - r[ci] ** 2
                d = torch.where(ok, d, torch.full_like(d, float("inf")))
                v, j = d.min(1)
                upd = v < best
                best = torch.where(upd, v, best)
                arg = torch.where(upd, ci.gather(1, j[:, None]).squeeze(1), arg)
                del take, ok, ci, d
        hitm = arg >= 0
        vol.index_add_(0, arg[hitm], torch.ones(int(hitm.sum()), dtype=torch.float64, device=device))
        done += n
        del pts, pb, best, arg
    return vol.float().cpu()


if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    from build_true_facet_graph import load_points_radii
    from run_lambda_derivation_eval import mc_cell_volumes
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=r"D:\Downloads\spp_results\full\spp_pf_unfroz_27dd4da69e")
    p.add_argument("--samples", type=int, default=4_000_000)
    p.add_argument("--check-cpu", type=int, default=400_000)
    a = p.parse_args()
    c, r = load_points_radii(a.ckpt)
    valid = np.ones(len(c), dtype=bool)
    t0 = time.time()
    vg = mc_cell_volumes_gpu(c, r, valid, a.samples, verbose=True)
    tg = time.time() - t0
    print(f"GPU: {a.samples:,} samples in {tg:.1f}s  ({a.samples/tg/1e6:.2f} M/s), "
          f"cells hit {int((vg>0).sum()):,}")
    if a.check_cpu:
        t0 = time.time()
        vc = mc_cell_volumes(c, r, valid, a.check_cpu, "cpu")
        tc = time.time() - t0
        print(f"CPU: {a.check_cpu:,} samples in {tc:.1f}s  ({a.check_cpu/tc/1e6:.3f} M/s)"
              f"   -> GPU speedup {(a.samples/tg)/(a.check_cpu/tc):.0f}x")
        # agreement on the shared statistic: both estimate the same volume distribution
        gv = (vg / vg.sum()).numpy(); cv = (vc / vc.sum()).numpy()
        both = (vg.numpy() > 0) & (vc.numpy() > 0)
        print(f"  corr on cells hit by both ({both.sum():,}): "
              f"{np.corrcoef(gv[both], cv[both])[0,1]:.4f}")
