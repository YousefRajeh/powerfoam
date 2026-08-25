"""Adjacency graphs over primitives, for every representation and all three complexes.

THREE COMPLEXES, and they are not interchangeable:

  delaunay  The EXACT dual of the cell decomposition. Lift site i to 4D with
            w_i = |x_i|^2 - r_i^2 and take the lower hull; two cells share a power-diagram
            facet iff they share an edge of that regular triangulation. For powerfoam this is
            the true facet graph. For radfoam (unweighted, r=0) the lift degenerates to the
            ordinary paraboloid and this is the ordinary Delaunay triangulation -- the exact
            dual of the Voronoi diagram radfoam actually traverses. For Gaussians there is no
            cell structure at all, so this is the Delaunay triangulation OF THE MEANS: a
            geometric neighbour graph, not a facet graph. That distinction is recorded rather
            than papered over, because "adjacent" means something weaker on the Gaussian arms.

  alpha     Delaunay filtered to edges with |x_i - x_j| < r_i + r_j, i.e. neighbours whose
            influence regions actually meet. Strictly a subgraph of delaunay. For Gaussians,
            r_i is taken as the largest Gaussian axis (max scale), the natural analogue of a
            reach radius.

  cech      AABB-overlap adjacency from powerfoam's own BVH: the graph the renderer builds,
            and what earlier experiments in this project used. It is NOT a superset of the
            true facet graph, despite the intuition that sharing a facet implies overlapping
            bounding boxes. Measured on scene0062_00 pf_nonfroz: cech has 1,187,759 undirected
            edges and delaunay 1,188,330 -- nearly identical counts -- yet 577,930 facet edges
            (48.6%) are ABSENT from cech. A power cell extends far beyond its own radius, so
            two cells can share a facet while their bounded volumes never meet. The two are
            different graphs of similar size, not a refinement of one another, which is why
            the ablation carries them as separate arms rather than treating cech as a cheap
            approximation. Powerfoam only.

CORRECTNESS. The delaunay path is build_true_facet_graph.regular_triangulation_edges, which is
validated by test_unweighted_delaunay_adjacency.py against an analytic bipyramid, against
weighted/unweighted invariants, and against scipy.spatial.Delaunay on random clouds (100%
edge-set agreement). Note benchmark.py::build_power_adjacency has a duplicate-edge bug -- it
calls torch.unique WITHOUT canonicalising orientation, so (i,j) and (j,i) both survive and the
later flip doubles them (measured 30.4% duplicate directed entries). That path is NOT used
here.
"""
import time

import numpy as np
import torch

from build_true_facet_graph import regular_triangulation_edges


def _csr_from_edges(edges, n):
    """Undirected edge list (E,2) with i<j -> symmetric CSR (adjacent, offsets)."""
    if len(edges) == 0:
        return (torch.zeros(0, dtype=torch.int32),
                torch.zeros(n + 1, dtype=torch.int32))
    e = np.asarray(edges, dtype=np.int64)
    both = np.concatenate([e, e[:, ::-1]], axis=0)
    order = np.argsort(both[:, 0], kind="stable")
    both = both[order]
    counts = np.bincount(both[:, 0], minlength=n)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return (torch.from_numpy(both[:, 1].astype(np.int32)),
            torch.from_numpy(offsets.astype(np.int32)))


def build_delaunay(centers, radii=None):
    """Exact facet dual (weighted if radii given, ordinary Delaunay if not)."""
    pts = np.asarray(centers, dtype=np.float64)
    r = None if radii is None else np.asarray(radii, dtype=np.float64)
    # radfoam is unweighted; a constant radius also leaves the lower hull unchanged, so
    # passing zeros and passing None are equivalent (asserted in the validation test).
    if r is not None and float(np.ptp(r)) == 0.0:
        r = None
    # regular_triangulation_edges returns (edges, n_simplices, hull_seconds)
    edges, _, _ = regular_triangulation_edges(pts, r)
    return edges


def build_alpha(centers, radii, edges=None):
    """Delaunay edges whose endpoints' influence radii overlap: |xi-xj| < ri+rj."""
    if edges is None:
        edges = build_delaunay(centers, radii)
    e = np.asarray(edges, dtype=np.int64)
    if len(e) == 0:
        return e
    c = np.asarray(centers, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    d = np.linalg.norm(c[e[:, 0]] - c[e[:, 1]], axis=1)
    return e[d < (r[e[:, 0]] + r[e[:, 1]])]


GAUSSIAN_SIGMA = 3.0     # gsplat's own bounding convention


def gaussian_reach_radii(scales, n_sigma=GAUSSIAN_SIGMA):
    """Reach radius for a Gaussian: n_sigma x its longest axis.

    A Gaussian has no boundary, so 'radius' is a convention, and the raw max scale (1 sigma)
    is far too tight: measured on scene0062_00 gs_froz it yields an alpha complex of 1,187
    edges at mean degree 0.05 -- an essentially empty graph, useless as an ablation arm.
    3 sigma is gsplat's OWN extent convention for tile bounds, so it is the value the renderer
    already treats as the splat's reach rather than something tuned here.
    """
    return n_sigma * np.asarray(scales, dtype=np.float64).max(axis=1)


def build_for(recon_kind, prim, complex_name):
    """-> (adjacent, offsets, stats dict). prim is ablation_assign.load_primitives output."""
    t0 = time.time()
    centers = prim["centers"].detach().cpu().numpy().astype(np.float64)
    n = centers.shape[0]

    if recon_kind == "gaussian":
        radii = gaussian_reach_radii(prim["scales"].detach().cpu().numpy())
    else:
        radii = prim["radii"].detach().cpu().numpy().astype(np.float64)

    if complex_name == "delaunay":
        # Gaussian means carry no weights -- their Delaunay is the ordinary (unweighted) one.
        edges = build_delaunay(centers, None if recon_kind == "gaussian" else radii)
    elif complex_name == "alpha":
        # radfoam sites have NO radii (the foam is unweighted), so |xi-xj| < ri+rj can never
        # hold and the alpha complex is the empty graph. It is genuinely undefined there
        # rather than merely inconvenient, so it is refused rather than silently returned
        # empty and mistaken for a measured result.
        if recon_kind == "radfoam" or float(np.max(radii)) == 0.0:
            raise ValueError("alpha complex is undefined for an unweighted (radfoam) foam: "
                             "every site has radius 0, so no two influence regions meet")
        edges = build_alpha(centers, radii)
    elif complex_name == "cech":
        raise ValueError("cech is built from the powerfoam BVH, not from centres; "
                         "see build_cech_powerfoam()")
    else:
        raise ValueError(f"unknown complex {complex_name}")

    adjacent, offsets = _csr_from_edges(edges, n)
    deg = (offsets[1:] - offsets[:-1]).float()
    return adjacent, offsets, {
        "n_edges": int(len(edges)),
        "mean_degree": float(deg.mean()) if n else 0.0,
        "max_degree": int(deg.max()) if n else 0,
        "seconds": time.time() - t0,
    }


CECH_CACHE = {          # powerfoam arm -> cached CSR built by the renderer's BVH
    "pf_nonfroz": "artifacts/scannet/{scene}/adjacency_nonfrozen.pt",
    "pf_tfroz":   "artifacts/scannet/{scene}/adjacency_truefrozen.pt",
}


def load_cech_powerfoam(recon, scene):
    """AABB-overlap adjacency from powerfoam's own BVH, as cached by earlier runs.

    Rebuilding it needs a full PowerfoamScene (warp kernels + dataset), which is far more
    machinery than reading the CSR the renderer already wrote. Returns None when absent, so
    the caller records a gap instead of silently substituting a different complex.
    """
    import os
    tmpl = CECH_CACHE.get(recon)
    if tmpl is None:
        return None
    path = tmpl.format(scene=scene)
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=True)
    adjacent = d["adjacent"] if "adjacent" in d else d["adjacency"]
    offsets = d["offsets"]
    deg = (offsets[1:].long() - offsets[:-1].long()).float()
    return adjacent, offsets, {
        "n_edges": int(adjacent.numel() // 2),
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "seconds": 0.0,
        "source": path,
    }
