"""Convert the TRUE power-diagram facet graph (true_facet_graph.npz) into the CSR
(adjacent, offsets) format every adjacency-consuming script in this repo expects.

WHY THIS EXISTS
---------------
Everything in this project that claimed to use "the power-diagram adjacency graph" in
fact consumed artifacts/scannet/<scene>/adjacency_<variant>.pt, which is populated by
powerfoam/scene.py::rebuild_adjacency from AABBTree.build_cech_complex(). That routine
tests  d(c_i, c_j) < r_i + r_j  with r = 0.5*(aabb.max.x - aabb.min.x), i.e. BOUNDING-BALL
OVERLAP -- a Cech complex, not facet sharing. On scene0347_00 it has mean degree 21.84 and
recalls only 54.2% of the true power-diagram facets, so it is both too dense and missing
half the real structure.

The true graph is stored as an undirected edge list with facet geometry:
    i, j    (int64) cell indices, one entry per undirected facet
    area    (float64) facet area
    dist    (float64) |c_i - c_j|
This script emits the symmetric (both-directions) CSR the scripts consume, plus per-edge
weights aligned to the CSR entries:

    adjacent     (int32) neighbour list, rows sorted ascending
    offsets      (int32) prefix offsets, length P+1
    num_primitives
    area, dist   (float32) facet geometry, aligned with `adjacent`
    weight       (float32) raw TPFA transmissibility  w = area / dist
    weight_sym   (float32) symmetrically normalised  D^-1/2 W D^-1/2,  D = diag(sum_j w_ij)

USE weight_sym FOR ANY SMOOTHING/DIFFUSION. The raw TPFA weights on scene0347_00 span a
2.9e19 dynamic range (min 3.8e-15, median 6.4e-03, max 1.1e+05); plugging them into an
iterative solver un-normalised makes it stall on the stiff rows.

Usage:
  python convert_true_facet_graph.py --scene scene0347_00
  python convert_true_facet_graph.py --npz <in.npz> --output <out.pt> --num-primitives N
"""
import argparse
import os

import numpy as np
import torch


def build_csr(i, j, area, dist, num_primitives):
    """Undirected edge list -> symmetric CSR with aligned per-edge weights."""
    # duplicate each undirected facet into both directions
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    ar = np.concatenate([area, area])
    di = np.concatenate([dist, dist])

    order = np.lexsort((dst, src))          # sort by src, then dst -> rows ascending
    src, dst, ar, di = src[order], dst[order], ar[order], di[order]

    deg = np.bincount(src, minlength=num_primitives)
    offsets = np.zeros(num_primitives + 1, dtype=np.int64)
    np.cumsum(deg, out=offsets[1:])

    w = ar / di                              # TPFA transmissibility
    # symmetric normalisation D^-1/2 W D^-1/2 (row sums of the RAW weights)
    rowsum = np.zeros(num_primitives, dtype=np.float64)
    np.add.at(rowsum, src, w)
    inv_sqrt = np.zeros(num_primitives, dtype=np.float64)
    nz = rowsum > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(rowsum[nz])
    w_sym = w * inv_sqrt[src] * inv_sqrt[dst]

    return {
        "adjacent": torch.from_numpy(dst.astype(np.int32)),
        "offsets": torch.from_numpy(offsets.astype(np.int32)),
        "num_primitives": int(num_primitives),
        "area": torch.from_numpy(ar.astype(np.float32)),
        "dist": torch.from_numpy(di.astype(np.float32)),
        "weight": torch.from_numpy(w.astype(np.float32)),
        "weight_sym": torch.from_numpy(w_sym.astype(np.float32)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="shorthand for the standard artifact paths")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--npz", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--num-primitives", type=int, default=None,
                   help="defaults to the count in the existing (wrong) adjacency file")
    a = p.parse_args()

    npz = a.npz
    out = a.output
    P = a.num_primitives
    if a.scene:
        base = f"artifacts/scannet/{a.scene}"
        npz = npz or f"{base}/true_facet_graph.npz"
        out = out or f"{base}/adjacency_true_facet.pt"
        if P is None:
            ref = f"{base}/adjacency_{a.variant}.pt"
            if os.path.exists(ref):
                P = int(torch.load(ref, map_location="cpu", weights_only=True)["num_primitives"])
    d = np.load(npz)
    i, j, area, dist = d["i"], d["j"], d["area"], d["dist"]
    if P is None:
        P = int(max(i.max(), j.max())) + 1

    csr = build_csr(i, j, area, dist, P)
    deg = (csr["offsets"][1:].long() - csr["offsets"][:-1].long()).numpy()
    w = csr["weight"].numpy()
    ws = csr["weight_sym"].numpy()
    print(f"[convert] {npz}")
    print(f"  primitives      : {P}")
    print(f"  undirected facets: {i.size}   directed CSR entries: {csr['adjacent'].numel()}")
    print(f"  degree          : mean={deg.mean():.2f} median={np.median(deg):.0f} "
          f"max={deg.max()} isolated={(deg == 0).sum()}")
    print(f"  raw w=area/dist : min={w.min():.4g} median={np.median(w):.4g} max={w.max():.4g} "
          f"ratio={w.max()/max(w.min(), 1e-300):.3g}")
    print(f"  D^-1/2 W D^-1/2 : min={ws.min():.4g} median={np.median(ws):.4g} max={ws.max():.4g}")
    torch.save(csr, out)
    print(f"[convert] wrote {out}")


if __name__ == "__main__":
    main()
