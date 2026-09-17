"""Verify the union-boundary extraction on cases with a known answer."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foam_union_mesh import _sphere_dirs, cell_polytope, union_boundary  # noqa: E402


def _csr(pairs, n):
    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[a].append(b); adj[b].append(a)
    off = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        off[i + 1] = off[i] + len(adj[i])
    flat = np.array([j for row in adj for j in row], dtype=np.int64)
    return flat, off


def test_isolated_cell_is_a_half_ball_area():
    """One cell, no neighbours: the boundary is a half-ball -- a disc plus a hemisphere.
    Area = pi r^2 + 2 pi r^2 = 3 pi r^2, up to the circumscribed-polytope over-estimate."""
    r = 0.5
    c = np.array([[0.0, 0.0, 0.0]]); rr = np.array([r])
    n = np.array([[0.0, 0.0, 1.0]])
    adj, off = _csr([], 1)
    V, F, C, st = union_boundary(c, rr, n, adj, off, np.array([True]), np.array([1]),
                                 n_sphere_dirs=256)
    exact = 3.0 * np.pi * r * r
    assert st["n_faces"] > 0
    assert abs(st["area_m2"] - exact) / exact < 0.05, (st["area_m2"], exact)
    assert st["n_interior_dropped"] == 0
    # The sphere is CIRCUMSCRIBED by tangent half-spaces, so vertices sit slightly OUTSIDE the
    # ball by construction (that is the documented over-estimate). The exact invariant is that
    # every vertex satisfies every tangent constraint u.(v - c) <= r, and that the area errs high.
    dirs = _sphere_dirs(256)
    assert ((V @ dirs.T) <= r + 1e-6).all(), "vertex violates a tangent half-space"
    assert st["area_m2"] >= exact - 1e-9, "circumscribed polytope must not UNDER-estimate"
    assert (V[:, 2] <= 1e-6).all()


def test_shared_face_is_dropped_when_both_cells_are_occupied():
    """THE CONNECTIVITY TEST. Two overlapping cells: the shared radical face must be classified
    interior and removed, so the union has strictly less area than the two solids separately."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    r = np.array([0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    adj, off = _csr([(0, 1)], 2)

    both = union_boundary(c, r, n, adj, off, np.array([True, True]), np.array([1, 1]),
                          n_sphere_dirs=128)
    only0 = union_boundary(c, r, n, adj, off, np.array([True, False]), np.array([1, 1]),
                           n_sphere_dirs=128)
    assert both[3]["n_interior_dropped"] > 0, "shared face was not detected as interior"
    # with the neighbour absent, cell 0's shared face is exposed instead of dropped
    assert only0[3]["n_interior_dropped"] == 0
    assert only0[3]["n_exposed_neighbour"] > 0


def test_face_kinds_are_identified():
    """A cell with one neighbour must produce all three face kinds: neighbour, dipole, sphere."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    r = np.array([0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out = cell_polytope(0, c, r, n, np.array([1]), _sphere_dirs(96))
    assert out is not None
    _, _, kind, nbr = out
    assert set(np.unique(kind)) == {0, 1, 2}, np.unique(kind)
    assert (nbr[kind == 0] == 1).all()
    assert (nbr[kind != 0] == -1).all()


def test_polytope_vertices_satisfy_every_halfspace():
    """No extracted vertex may violate any constraint that defines the solid."""
    rng = np.random.default_rng(4)
    c = rng.normal(0, 0.3, (14, 3))
    r = rng.uniform(0.12, 0.25, 14)
    nn = rng.normal(0, 1, (14, 3)); nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    nb = np.array([j for j in range(14) if j != 0])
    out = cell_polytope(0, c, r, nn, nb, _sphere_dirs(64))
    if out is None:
        pytest.skip("degenerate cell for this seed")
    V, _, _, _ = out
    dirs = _sphere_dirs(64)
    assert (((V - c[0]) @ dirs.T) <= r[0] + 1e-6).all(), "vertex violates a tangent half-space"
    assert (((V - c[0]) @ nn[0]) <= 1e-6).all()
    for j in nb:
        A = 2.0 * (c[j] - c[0])
        b = (c[j] @ c[j] - r[j] ** 2) - (c[0] @ c[0] - r[0] ** 2)
        assert ((V @ A) <= b + 1e-6).all(), f"violates neighbour {j}"


def test_occupied_subset_only():
    """Unoccupied cells contribute nothing at all."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    r = np.array([0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    adj, off = _csr([(0, 1)], 2)
    V, F, C, st = union_boundary(c, r, n, adj, off, np.array([False, False]),
                                 np.array([1, 1]), n_sphere_dirs=32)
    assert len(F) == 0 and st["n_cells"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


def test_weld_reduces_vertices_without_changing_area():
    """Welding must be topology-only: same surface area, strictly fewer vertices."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.12, 0.22, 0.0]])
    r = np.array([0.2, 0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0]] * 3)
    adj, off = _csr([(0, 1), (1, 2), (0, 2)], 3)
    occ = np.array([True, True, True])
    raw = union_boundary(c, r, n, adj, off, occ, np.array([1, 1, 1]),
                         n_sphere_dirs=64, weld_tol=0.0)
    wel = union_boundary(c, r, n, adj, off, occ, np.array([1, 1, 1]),
                         n_sphere_dirs=64, weld_tol=1e-6)
    assert abs(raw[3]["area_m2"] - wel[3]["area_m2"]) < 1e-9, "welding changed the area"
    assert wel[3]["n_verts"] < wel[3]["n_verts_before_weld"], "welding removed nothing"


def test_face_merge_gives_minimal_triangles_on_a_known_face():
    """A polytope face with k vertices must yield exactly k-2 triangles after merging."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    r = np.array([0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out = cell_polytope(0, c, r, n, np.array([1]), _sphere_dirs(96))
    assert out is not None
    verts, faces, kind, nbr = out
    # the single neighbour face: count its triangles and its distinct vertices
    m = kind == 0
    tris = faces[m]
    nv = len(np.unique(tris))
    assert len(tris) == nv - 2, (len(tris), nv)
