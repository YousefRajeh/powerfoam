"""Validation of the UNWEIGHTED (radfoam) path of build_true_facet_graph.py.

radfoam's foam has no per-site radii, so the regular-triangulation lift
    w_i = |x_i|^2 - r_i^2
degenerates to the ordinary paraboloid lift w_i = |x_i|^2, whose lower convex hull is the
ordinary Delaunay triangulation -- the exact dual of the ordinary Voronoi diagram that
radfoam actually traverses.  This file proves that claim three ways:

  T1  ANALYTIC, hand-checkable: a triangular bipyramid whose Delaunay triangulation is
      NOT the complete graph, so it discriminates a true-facet graph from any
      "everything nearby is a neighbour" graph (such as our old Cech construction).
  T2  INVARIANTS: r=0 must reproduce the unweighted answer exactly, and any CONSTANT
      radius must too (a constant r^2 shifts every lift by the same amount and therefore
      cannot change the lower hull).
  T3  ORACLE: agreement with scipy.spatial.Delaunay -- an independent implementation --
      on random point clouds, plus the identity
      assign_points_to_power_cells(radii=0) == assign_points_to_nearest_center,
      which is what makes nearest-centre membership EXACT for radfoam.

Run:  python test_unweighted_delaunay_adjacency.py
"""
import sys

import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, r"D:\Downloads\powerfoam")

from build_true_facet_graph import regular_triangulation_edges, build_csr
from point_cloud_query import (assign_points_to_power_cells,
                               assign_points_to_nearest_center)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def edge_set(points, radii=None):
    e, _, _ = regular_triangulation_edges(points, radii)
    return set(map(tuple, e.tolist()))


def delaunay_edge_set(points):
    """Independent oracle: edges of scipy's Delaunay tetrahedralisation."""
    tri = Delaunay(points)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    e = np.concatenate([tri.simplices[:, list(p)] for p in pairs], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return set(map(tuple, e.tolist()))


def t1_analytic():
    """Triangular bipyramid, verified by hand.

    Sites (index: coordinate)
      0 A = ( 0.0,      0.0, 1.0)      apex above
      1 B = ( 0.0,      0.0,-2.0)      apex below (deliberately NOT -1.0, which would put
                                       B exactly on the circumsphere of ACDE -- a
                                       cospherical degeneracy with no unique answer)
      2 C = ( 1.0,      0.0, 0.0)
      3 D = (-0.5,  0.86603, 0.0)      equilateral triangle on the unit circle in z = 0
      4 E = (-0.5, -0.86603, 0.0)

    Hand derivation of the Delaunay tetrahedralisation:
      tet (A,C,D,E): C,D,E lie on the unit circle in z=0, so the circumcentre is on the
        z-axis at (0,0,z) with 1 + z^2 = (1 - z)^2  =>  z = 0; circumsphere = unit sphere
        at the origin, radius 1.  B is at distance 2 > 1, hence OUTSIDE.  Delaunay. OK
      tet (B,C,D,E): centre (0,0,z) with 1 + z^2 = (z + 2)^2  =>  z = -0.75, radius 1.25.
        A is at distance |1 - (-0.75)| = 1.75 > 1.25, hence OUTSIDE.  Delaunay. OK
      These two tets tile the bipyramid and share the face (C,D,E), so the triangulation is
      exactly {ACDE, BCDE} and the edge set is their 6+6 edges minus the 3 shared ones:
        A-C A-D A-E  B-C B-D B-E  C-D C-E D-E   ->  NINE edges.
      A-B is ABSENT: the triangle CDE separates the two apices, so their Voronoi cells
      touch only through it and share no facet.  K5 would have ten edges, so this case
      distinguishes the true facet graph from a complete/proximity graph.
    """
    print("T1  analytic triangular bipyramid")
    s3 = np.sqrt(3.0) / 2.0
    pts = np.array([[0.0, 0.0, 1.0],
                    [0.0, 0.0, -2.0],
                    [1.0, 0.0, 0.0],
                    [-0.5, s3, 0.0],
                    [-0.5, -s3, 0.0]], dtype=np.float64)
    expected = {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
    got = edge_set(pts)  # unweighted path
    check("bipyramid edges == hand-derived 9 edges", got == expected,
          f"got {sorted(got)}")
    check("apex-apex edge (0,1) absent", (0, 1) not in got)
    check("agrees with scipy.spatial.Delaunay oracle", got == delaunay_edge_set(pts))

    # CSR round-trip: degrees must be A:3 B:3 C:4 D:4 E:4  (sum = 18 directed entries)
    e = np.array(sorted(expected))
    csr = build_csr(e[:, 0], e[:, 1], 5)
    deg = (csr["offsets"][1:].long() - csr["offsets"][:-1].long()).numpy().tolist()
    check("CSR degrees == [3,3,4,4,4]", deg == [3, 3, 4, 4, 4], f"got {deg}")

    # Positive control -- a case whose answer IS the complete graph, so that T1 is not just
    # testing "the builder drops edges".  Regular tetrahedron + its centroid: the centroid
    # is strictly interior, so the ONLY tetrahedralisation of these 5 sites is the 4-way
    # split about it, hence that split is the Delaunay one.  Its edges are the 6 original
    # tet edges plus the 4 centroid spokes = all 10 pairs = K5.
    # (A bare tetrahedron cannot be tested through this route at all: the 4D lift of 4
    # points is degenerate and Qhull needs 5 points for an initial simplex in 4D.)
    tet_c = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                      [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
                      [0.0, 0.0, 0.0]], dtype=np.float64)
    got = edge_set(tet_c)
    k5 = {(i, j) for i in range(5) for j in range(i + 1, 5)}
    check("regular tetrahedron + centroid == K5 (10 edges)", got == k5,
          f"got {len(got)} edges")


def t2_invariants():
    """r=0 and any constant r must reproduce the unweighted lift exactly."""
    print("T2  weighted/unweighted invariants")
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(400, 3))
    base = edge_set(pts, None)
    check("radii=None == radii=0", base == edge_set(pts, np.zeros(400)))
    for c in (0.5, 3.0):
        check(f"radii=None == radii={c} (constant)",
              base == edge_set(pts, np.full(400, c)),
              "constant r^2 shifts every lift equally, so the lower hull is unchanged")
    # Sanity in the other direction: NON-constant radii must be able to change the answer,
    # otherwise the weighted path would be silently ignoring its weights.
    r = rng.uniform(0.0, 0.6, size=400)
    check("non-constant radii DO change the graph", base != edge_set(pts, r))


def t3_oracle():
    print("T3  oracle agreement on random clouds + exact membership")
    rng = np.random.default_rng(7)
    for n, tag in ((500, "gaussian"), (2000, "uniform-cube")):
        pts = (rng.normal(size=(n, 3)) if tag == "gaussian"
               else rng.uniform(-1, 1, size=(n, 3)))
        mine, oracle = edge_set(pts), delaunay_edge_set(pts)
        inter = len(mine & oracle)
        frac = inter / max(len(oracle), 1)
        check(f"{tag} n={n}: edge sets identical vs scipy.Delaunay", mine == oracle,
              f"|mine|={len(mine)} |oracle|={len(oracle)} overlap={frac*100:.4f}%")

    # Membership: with all radii equal, power distance ||x-c||^2 - r^2 is Euclidean
    # distance squared minus a CONSTANT, so its argmin is the Euclidean argmin.  This is
    # exactly why nearest-centre is EXACT (not approximate) for radfoam's unweighted foam.
    pts = rng.uniform(-1, 1, size=(3000, 3))
    q = rng.uniform(-1.2, 1.2, size=(5000, 3))
    a_pow = assign_points_to_power_cells(q, pts, np.zeros(3000), k=64)
    a_nn = assign_points_to_nearest_center(q, pts)
    check("power-cell(radii=0) assignment == nearest-centre assignment",
          bool(np.array_equal(a_pow, a_nn)),
          f"{int((a_pow != a_nn).sum())} / {q.shape[0]} disagreements")
    # brute force truth
    d = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    check("nearest-centre == brute-force argmin", bool(np.array_equal(a_nn, d.argmin(1))))


if __name__ == "__main__":
    t1_analytic()
    t2_invariants()
    t3_oracle()
    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)
