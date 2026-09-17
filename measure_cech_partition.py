"""Does the Cech-filtered facet set still give the renderer a PARTITION? (RENDERER_SPEC.md 8.1)

The rasteriser clips a ray by the bounding sphere, then by the power-facet half-spaces of the
ALPHA/CECH neighbours only (`build_adjacency(..., alpha_complex=True)` keeps edges with
||c_i - c_j|| < r_i + r_j). A true power cell is the intersection of half-spaces over ALL
regular-triangulation neighbours, so dropping facets can only ENLARGE the region. The renderer's
rendered cell is therefore

    R_i = sphere(c_i, r_i)  AND  { x : (c_j - c_i).x <= offset_ij  for all j in Cech(i) }

which contains the true sphere-clipped power cell. If two R_i overlap, foam does NOT partition
space at render time and the A34 "partition vs mixture" argument fails locally.

MEASUREMENT. Sample points uniformly inside each sampled primitive's bounding sphere, then for
each point ask two questions:

  rendered  : does it satisfy every Cech half-space of i?   (the renderer would shade it as cell i)
  owned     : is i the argmin of the power distance over ALL P sites?  (it truly belongs to cell i)

  rendered AND NOT owned  -> STOLEN: the renderer puts matter where another cell owns the space,
                             i.e. two rendered cells overlap there.
  owned AND NOT rendered  -> LOST:   the cell owns the space but the sphere/Cech test rejects it,
                             i.e. a gap (usually the sphere cull, which is legitimate).

The argmin is brute-forced over every site -- no k-NN shortcut -- because the whole question is
whether a restricted candidate set is adequate, and `assign_points_to_power_cells`'s own k=64 filter
is already known to disagree with the exact answer on 1.4-2.2% of points.
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler


def run(scene, arm, n_prims, n_samples, seed, site_chunk, dev="cuda"):
    import warp as wp
    wp.init()
    from powerfoam.scene import PowerfoamScene

    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{arm}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device=dev)
    m.load_pt(f"output/scannet_{scene}_{arm}/model.pt")

    C = m.points.detach().float()
    R = m.get_radii().detach().float().reshape(-1)
    adj = m.adjacency.detach().long()
    off = m.adjacency_offsets.detach().long()
    P = C.shape[0]

    g = torch.Generator(device=dev).manual_seed(seed)
    sel = torch.randperm(P, generator=g, device=dev)[:min(n_prims, P)]

    # uniform points in the unit ball, scaled per primitive
    u = torch.randn(sel.numel(), n_samples, 3, generator=g, device=dev)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    rad = torch.rand(sel.numel(), n_samples, 1, generator=g, device=dev) ** (1.0 / 3.0)
    X = C[sel][:, None, :] + u * rad * R[sel][:, None, None]      # (S, n, 3)

    # ---- rendered: all Cech half-spaces of i satisfied ----
    rendered = torch.ones(sel.numel(), n_samples, dtype=torch.bool, device=dev)
    pm = 0.5 * ((C * C).sum(-1) - R * R)                           # power moment
    for a in range(sel.numel()):
        i = int(sel[a])
        js = adj[off[i]:off[i + 1]]
        if js.numel() == 0:
            continue
        fn = C[js] - C[i]                                          # (k,3)
        fo = pm[js] - pm[i]                                        # (k,)  == face_offset
        # pow(x,i) <= pow(x,j)  <=>  (c_j - c_i).x <= 0.5(|c_j|^2 - |c_i|^2 + r_i^2 - r_j^2)
        rendered[a] = ((X[a] @ fn.T - fo[None, :]) <= 0).all(dim=-1)

    # ---- owned: exact argmin of power distance over ALL sites ----
    flat = X.reshape(-1, 3)
    best = torch.full((flat.shape[0],), float("inf"), device=dev)
    arg = torch.zeros(flat.shape[0], dtype=torch.long, device=dev)
    R2 = R * R
    for s0 in range(0, P, site_chunk):
        cc = C[s0:s0 + site_chunk]
        pd = torch.cdist(flat, cc).pow_(2).sub_(R2[s0:s0 + site_chunk][None, :])
        v, k = pd.min(1)
        upd = v < best
        best[upd] = v[upd]; arg[upd] = k[upd] + s0
        del pd
    owned = (arg.reshape(sel.numel(), n_samples) == sel[:, None])

    stolen = rendered & ~owned
    lost = owned & ~rendered
    n = rendered.numel()
    # per-primitive: what fraction of a rendered cell's sampled volume is stolen?
    per_prim = (stolen.sum(1).float() / rendered.sum(1).clamp_min(1).float())
    return dict(
        scene=scene, arm=arm, P=int(P), prims=int(sel.numel()), samples=int(n),
        r_ratio=float(R.max() / R.min().clamp_min(1e-20)),
        rendered_frac=float(rendered.float().mean()),
        owned_frac=float(owned.float().mean()),
        stolen_frac=float(stolen.float().mean()),
        lost_frac=float(lost.float().mean()),
        stolen_of_rendered=float(stolen.sum()) / max(float(rendered.sum()), 1.0),
        prims_with_any_steal=float((stolen.any(1)).float().mean()),
        per_prim_steal_p50=float(per_prim.median()),
        per_prim_steal_p90=float(torch.quantile(per_prim, 0.9)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00")
    ap.add_argument("--arms", default="truefrozen,nonfrozen")
    ap.add_argument("--prims", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--site-chunk", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"{'scene':<14}{'arm':<11}{'P':>10}{'r max/min':>11}"
          f"{'rendered':>10}{'owned':>8}{'STOLEN':>9}{'lost':>8}"
          f"{'stolen/rendered':>17}{'prims hit':>11}")
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = run(sc, arm, a.prims, a.samples, a.seed, a.site_chunk)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            print(f"{r['scene']:<14}{r['arm']:<11}{r['P']:>10,}{r['r_ratio']:>11.3g}"
                  f"{r['rendered_frac']:>10.1%}{r['owned_frac']:>8.1%}"
                  f"{r['stolen_frac']:>9.2%}{r['lost_frac']:>8.1%}"
                  f"{r['stolen_of_rendered']:>17.2%}{r['prims_with_any_steal']:>11.1%}",
                  flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
