"""Build the TRUE power-diagram facet graph for an arbitrary scene.

The regular (weighted Delaunay) triangulation dual to the power diagram is obtained by
lifting every cell i to 4D with  w_i = |x_i|^2 - r_i^2  and taking the LOWER hull of the
4D convex hull.  Two cells share a power-diagram facet iff they share an edge of that
triangulation.

This replaces the lost script that produced artifacts/scannet/scene0347_00/
true_facet_graph.npz, and it fixes the duplicate-edge bug in
benchmark.py::build_power_adjacency, which calls torch.unique(edges, dim=0) WITHOUT
canonicalising orientation, so (i,j) and (j,i) both survive and the later flip(1)
doubles them (measured 30.4% duplicate directed entries).  Here the edge array is
sorted along dim=1 before the unique.

Output: a CSR .pt with keys adjacent/offsets/num_primitives/dist (+ area/weight/
weight_sym only when --with-area is given; facet area is NOT needed for region growing,
which reads the adjacency structure and feature cosines only).

Usage:
  python build_true_facet_graph.py --scene scene0347_00 --validate
  python build_true_facet_graph.py --scene scene0140_00 --output <path>
"""
import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial import ConvexHull


def load_points_radii(ckpt_dir):
    """points (N,3) float64 and radii (N,) float64 straight from model.pt.

    PowerfoamScene.get_radii() is softplus(raw, beta=100); reproduced here so the whole
    warp/dataset stack does not have to be constructed just to read two tensors."""
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu",
                    weights_only=False)
    pts = sd["points"].float()
    radii = torch.nn.functional.softplus(sd["radii"].float(), beta=100)
    return pts.numpy().astype(np.float64), radii.numpy().astype(np.float64)


def regular_triangulation_edges(points, radii, qhull_options=None):
    """Unique undirected edges (i<j) of the regular triangulation of the weighted sites."""
    w = (points ** 2).sum(axis=1) - radii ** 2
    lifted = np.concatenate([points, w[:, None]], axis=1)
    t0 = time.time()
    hull = ConvexHull(lifted) if qhull_options is None else ConvexHull(lifted, qhull_options=qhull_options)
    t_hull = time.time() - t0
    is_lower = hull.equations[:, 3] < 0
    simplices = hull.simplices[is_lower]           # (T, 4) tetrahedra
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    edges = np.concatenate([simplices[:, list(p)] for p in pairs], axis=0)
    # CANONICALISE ORIENTATION before unique -- without this (i,j) and (j,i) both survive
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return edges, simplices.shape[0], t_hull


def build_csr(i, j, num_primitives, dist=None, area=None):
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    deg = np.bincount(src, minlength=num_primitives)
    offsets = np.zeros(num_primitives + 1, dtype=np.int64)
    np.cumsum(deg, out=offsets[1:])
    out = {
        "adjacent": torch.from_numpy(dst.astype(np.int32)),
        "offsets": torch.from_numpy(offsets.astype(np.int32)),
        "num_primitives": int(num_primitives),
    }
    if dist is not None:
        di = np.concatenate([dist, dist])[order]
        out["dist"] = torch.from_numpy(di.astype(np.float32))
    if area is not None:
        ar = np.concatenate([area, area])[order]
        out["area"] = torch.from_numpy(ar.astype(np.float32))
        w = ar / np.maximum(di, 1e-30)
        rowsum = np.zeros(num_primitives, dtype=np.float64)
        np.add.at(rowsum, src, w)
        inv = np.zeros(num_primitives, dtype=np.float64)
        nz = rowsum > 0
        inv[nz] = 1.0 / np.sqrt(rowsum[nz])
        out["weight"] = torch.from_numpy(w.astype(np.float32))
        out["weight_sym"] = torch.from_numpy((w * inv[src] * inv[dst]).astype(np.float32))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--npz-output", default=None)
    p.add_argument("--qhull-options", default=None)
    p.add_argument("--validate", action="store_true",
                   help="compare against an existing true_facet_graph.npz for this scene")
    a = p.parse_args()

    ckpt_dir = a.ckpt_dir or f"output/scannet_{a.scene}_{a.variant}"
    out = a.output or f"artifacts/scannet/{a.scene}/adjacency_true_facet.pt"

    t0 = time.time()
    points, radii = load_points_radii(ckpt_dir)
    P = points.shape[0]
    print(f"[build] {a.scene}: P={P} cells  (load {time.time()-t0:.1f}s)", flush=True)

    edges, n_tets, t_hull = regular_triangulation_edges(points, radii, a.qhull_options)
    print(f"[build] 4D hull: {t_hull:.1f}s, lower-hull tets={n_tets}, "
          f"undirected facets={edges.shape[0]}", flush=True)

    i, j = edges[:, 0], edges[:, 1]
    dist = np.linalg.norm(points[i] - points[j], axis=1)
    csr = build_csr(i, j, P, dist=dist)
    deg = (csr["offsets"][1:].long() - csr["offsets"][:-1].long()).numpy()
    print(f"[build] degree: mean={deg.mean():.2f} median={np.median(deg):.0f} "
          f"max={deg.max()} isolated={(deg == 0).sum()}", flush=True)

    if a.validate:
        ref_path = f"artifacts/scannet/{a.scene}/true_facet_graph.npz"
        d = np.load(ref_path)
        ref = np.sort(np.stack([d["i"], d["j"]], axis=1), axis=1)
        ref = np.unique(ref, axis=0)
        mine = edges
        setr = set(map(tuple, ref.tolist()))
        setm = set(map(tuple, mine.tolist()))
        print(f"[validate] ref facets={d['i'].size} (unique {ref.shape[0]}), "
              f"mine={mine.shape[0]}, identical={setr == setm}, "
              f"|ref\\mine|={len(setr - setm)}, |mine\\ref|={len(setm - setr)}", flush=True)

    if a.npz_output:
        np.savez_compressed(a.npz_output, i=i, j=j, dist=dist)
    torch.save(csr, out)
    print(f"[build] wrote {out}  ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
