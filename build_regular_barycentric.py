"""Barycentric coordinates on the REGULAR (weighted Delaunay) triangulation, not the plain one.

THE BUG THIS FIXES. `run_barycentric_posterior.py` builds its coordinates with
`scipy.spatial.Delaunay(centers)` -- the ORDINARY Delaunay of the sites, which ignores the
radii completely. But the foam is a POWER diagram, and the dual of a power diagram is the
REGULAR (weighted Delaunay) triangulation: lift each site to 4D with w_i = |x_i|^2 - r_i^2 and
take the lower hull of the 4D convex hull. That is exactly what `build_true_facet_graph.py`
already computes -- it just throws the tetrahedra away after extracting edges.

So the barycentric arm has been running on the wrong triangulation. Same class of error as
using `model.pt`'s renderer adjacency instead of the true facet graph.

WHY IT ALSO MATTERS FOR THE FOAM CLAIM. A plain Delaunay of points is something anyone can
build over Gaussian means -- it is not foam-specific. The REGULAR triangulation is dual to an
actual bounded partition, and its weights are the power radii. Using it makes the barycentric
arm a statement about the partition rather than about a point cloud.

WHAT CHANGES CONCRETELY. Different tetrahedra, hence different containing simplex per GT point,
hence different 4 vertices and different weights. The coordinates remain nonnegative and sum to
1 inside the hull, so simplex closure (simplex-vs-sphere-extension.md Thm 2) still applies and
the interpolated posterior is still a convex combination.

Writes `artifacts/ablation_cache/{scene}_bary_regular.npz` with the same keys the existing
scorer reads (verts / lam / inside), so the two can be swapped and compared directly.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import load_scannet_pointcept_gt

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}


def regular_tetrahedra(points, radii):
    """Lower hull of the 4D lift -> the tetrahedra of the regular triangulation.

    Mirrors build_true_facet_graph.regular_triangulation_edges, but KEEPS the simplices.
    """
    w = (points ** 2).sum(axis=1) - radii ** 2
    lifted = np.concatenate([points, w[:, None]], axis=1)
    hull = ConvexHull(lifted)
    is_lower = hull.equations[:, 3] < 0          # downward-facing facets = lower hull
    return hull.simplices[is_lower]


def bary_from_tets(pts, points, tets, chunk=200_000):
    """Locate each query point in a tetrahedron and return its barycentric coordinates.

    scipy has no point-location structure for a hand-built tetrahedralisation, so a Delaunay
    object is constructed over the SAME vertex set purely as a spatial index, and the actual
    coordinates are computed against the REGULAR tetrahedra by solving the 3x3 affine system.
    Points whose regular tetrahedron cannot be identified fall back to `inside=False` and the
    caller keeps the hard assignment -- never an extrapolation, which would give negative
    coordinates and leave the simplex.
    """
    N = len(pts)
    verts = np.zeros((N, 4), dtype=np.int64)
    lam = np.zeros((N, 4), dtype=np.float64)
    inside = np.zeros(N, dtype=bool)

    # centroid KD-tree over the regular tetrahedra, then exact containment test on candidates
    from scipy.spatial import cKDTree
    cent = points[tets].mean(axis=1)
    tree = cKDTree(cent)
    K = 32
    _, cand = tree.query(pts, k=K, workers=-1)

    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        P = pts[a:b]
        done = np.zeros(len(P), dtype=bool)
        for k in range(K):
            todo = ~done
            if not todo.any():
                break
            t = tets[cand[a:b, k][todo]]                      # (m,4)
            V = points[t]                                     # (m,4,3)
            A = (V[:, :3, :] - V[:, 3:4, :]).transpose(0, 2, 1)   # (m,3,3)
            rhs = P[todo] - V[:, 3, :]
            try:
                sol = np.linalg.solve(A, rhs[..., None])[..., 0]  # (m,3)
            except np.linalg.LinAlgError:
                continue
            l4 = 1.0 - sol.sum(axis=1)
            full = np.concatenate([sol, l4[:, None]], axis=1)
            ok = (full >= -1e-9).all(axis=1)
            idx = np.where(todo)[0][ok]
            verts[a + idx] = t[ok]
            lam[a + idx] = np.clip(full[ok], 0.0, None)
            lam[a + idx] /= lam[a + idx].sum(axis=1, keepdims=True).clip(1e-12)
            inside[a + idx] = True
            done[np.where(todo)[0][ok]] = True
    return verts, lam, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    a = ap.parse_args()
    enable_determinism()

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        out = f"artifacts/ablation_cache/{scene}_bary_regular.npz"
        if os.path.exists(out):
            print(f"[skip] {scene}", flush=True)
            continue
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        if not os.path.exists(mp):
            print(f"[miss] {scene}", flush=True)
            continue
        t0 = time.time()
        m = torch.load(mp, map_location="cpu", weights_only=False)
        pts_sites = m["points"].float().numpy().astype(np.float64)
        radii = F.softplus(m["radii"].float().squeeze(), beta=100).numpy().astype(np.float64)

        tets = regular_tetrahedra(pts_sites, radii)
        t_tri = time.time() - t0
        gt_pts, raw, _ = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        q = np.asarray(gt_pts, dtype=np.float64)
        verts, lam, inside = bary_from_tets(q, pts_sites, tets)

        # sanity: compare against the plain-Delaunay coordinates already cached
        old = f"artifacts/ablation_cache/{scene}_bary.npz"
        agree = float("nan")
        if os.path.exists(old):
            z = np.load(old)
            both = inside & z["inside"]
            if both.any():
                agree = float((np.sort(verts[both], 1) == np.sort(z["verts"][both], 1))
                              .all(1).mean())
        np.savez_compressed(out, verts=verts, lam=lam, inside=inside)
        print(f"[{scene}] {len(pts_sites):,} sites, {len(tets):,} regular tetrahedra "
              f"({t_tri:.0f}s), located {100*inside.mean():.1f}% of GT points, "
              f"same tetra as plain Delaunay for {100*agree:.1f}%  "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
