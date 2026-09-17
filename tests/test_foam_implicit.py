"""Verify the implicit-union construction on cases with a known answer."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foam_implicit import implicit_mesh, implicit_volume  # noqa: E402


def _grid(lo, hi, h):
    lo = np.asarray(lo, float) - 3 * h
    hi = np.asarray(hi, float) + 3 * h
    return lo, np.ceil((hi - lo) / h).astype(np.int64)


def _area(V, F):
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                      axis=1).sum())


def test_zero_displacement_single_cell_is_a_half_ball():
    """ONLY with zero displacement does one cell reduce to a half-ball: disc + hemisphere = 3 pi r^2.
    With displacement the interface is curved, which the next test covers."""
    r = 0.30
    c = np.array([[0.0, 0.0, 0.0]]); rr = np.array([r])
    n = np.array([[0.0, 0.0, 1.0]])
    sites = np.zeros((1, 4, 3)); heights = np.zeros((1, 4))
    h = 0.004
    lo, dims = _grid([-r, -r, -r], [r, r, r], h)
    f, win = implicit_volume(c, rr, n, sites, heights, np.array([1]), lo, dims, h)
    V, F, C = implicit_mesh(f, win, lo, h)
    assert len(F) > 0
    exact = 3.0 * np.pi * r * r
    assert abs(_area(V, F) - exact) / exact < 0.03, (_area(V, F), exact)


def test_displacement_changes_the_interface():
    """Detail-site displacement must move and curve the interface -- the area cannot stay at the
    flat-plane value, and a positive displacement must enlarge the solid."""
    r = 0.30
    c = np.array([[0.0, 0.0, 0.0]]); rr = np.array([r])
    n = np.array([[0.0, 0.0, 1.0]])
    sites = np.array([[[0.15, 0.0, 0.0], [-0.15, 0.0, 0.0],
                       [0.0, 0.15, 0.0], [0.0, -0.15, 0.0]]], dtype=float)
    h = 0.004
    lo, dims = _grid([-r, -r, -r], [r, r, r], h)

    flat = np.zeros((1, 4))
    f0, w0 = implicit_volume(c, rr, n, sites, flat, np.array([1]), lo, dims, h)
    V0, F0, _ = implicit_mesh(f0, w0, lo, h)
    vol0 = float((f0 > 0).sum())

    bumpy = np.array([[0.10, -0.10, 0.10, -0.10]])
    f1, w1 = implicit_volume(c, rr, n, sites, bumpy, np.array([1]), lo, dims, h)
    V1, F1, _ = implicit_mesh(f1, w1, lo, h)
    assert abs(_area(V1, F1) - _area(V0, F0)) / _area(V0, F0) > 0.01, "displacement did nothing"

    lifted = np.full((1, 4), 0.12)                    # push the whole interface outward
    f2, _ = implicit_volume(c, rr, n, sites, lifted, np.array([1]), lo, dims, h)
    assert float((f2 > 0).sum()) > vol0, "a positive displacement must enlarge the solid"


def test_two_overlapping_cells_give_ONE_connected_component():
    """THE POINT OF THE CONSTRUCTION. Under `max` the two displaced interfaces interpenetrate, so
    the zero-set is a single manifold -- not two shells meeting at a step."""
    import open3d as o3d
    r = 0.25
    c = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])     # 0.30 < 2r -> balls overlap
    rr = np.array([r, r])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    sites = np.zeros((2, 4, 3)); heights = np.zeros((2, 4))
    h = 0.006
    lo, dims = _grid([-r, -r, -r], [0.30 + r, r, r], h)
    f, win = implicit_volume(c, rr, n, sites, heights, np.array([1, 1]), lo, dims, h)
    V, F, _ = implicit_mesh(f, win, lo, h)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    lab = np.asarray(m.cluster_connected_triangles()[0])
    assert len(np.unique(lab)) == 1, f"{len(np.unique(lab))} components, expected 1"


def test_non_overlapping_cells_give_two_components():
    """The control: far apart, they must NOT be fused -- the union really is a union."""
    import open3d as o3d
    r = 0.15
    c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rr = np.array([r, r])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    sites = np.zeros((2, 4, 3)); heights = np.zeros((2, 4))
    h = 0.006
    lo, dims = _grid([-r, -r, -r], [1.0 + r, r, r], h)
    f, win = implicit_volume(c, rr, n, sites, heights, np.array([1, 1]), lo, dims, h)
    V, F, _ = implicit_mesh(f, win, lo, h)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    lab = np.asarray(m.cluster_connected_triangles()[0])
    assert len(np.unique(lab)) == 2, f"{len(np.unique(lab))} components, expected 2"


def test_min_max_semantics():
    """f must be the max over cells of the min of (ball, half-space)."""
    r = 0.2
    c = np.array([[0.0, 0.0, 0.0]]); rr = np.array([r])
    n = np.array([[0.0, 0.0, 1.0]])
    sites = np.zeros((1, 4, 3)); heights = np.zeros((1, 4))
    h = 0.01
    lo, dims = _grid([-r, -r, -r], [r, r, r], h)
    f, _ = implicit_volume(c, rr, n, sites, heights, np.array([1]), lo, dims, h)
    ix = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1)
    P = lo + (ix + 0.5) * h
    want = np.minimum(r - np.linalg.norm(P - c[0], axis=-1), -(P - c[0]) @ n[0])
    inside = f > -1e8
    assert np.abs(f[inside] - want[inside]).max() < 1e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
