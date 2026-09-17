"""Verify the chart parameterisation, its analytic derivatives, and the hand-off test."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foam_exact_surface import SV_TEMP, soft_voronoi_height  # noqa: E402
from foam_trace import chart_frames, exposed, height_uv, sample_chart  # noqa: E402


def _cell(seed=0, r=0.05, k=8):
    rng = np.random.default_rng(seed)
    sites_uv = rng.uniform(-0.6, 0.6, (k, 2)) * r
    heights = rng.normal(0.0, 0.25, k) * r
    return sites_uv, heights, r


def test_height_uv_matches_the_verified_3d_evaluator():
    """The in-plane form must agree with foam_exact_surface.soft_voronoi_height, which was itself
    checked against the render kernel's own loop to 3e-18."""
    sites_uv, heights, r = _cell(1)
    p = np.array([0.3, -0.2, 0.7])
    n = np.array([0.0, 0.0, 1.0]); t = np.array([1.0, 0.0, 0.0]); b = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(5)
    u = rng.uniform(-r, r, 200); v = rng.uniform(-r, r, 200)
    h_plane = height_uv(u, v, sites_uv, heights, r)
    base = p[None, :] + u[:, None] * t[None, :] + v[:, None] * b[None, :]
    sites_w = p[None, :] + sites_uv[:, 0:1] * t[None, :] + sites_uv[:, 1:2] * b[None, :]
    h_3d = soft_voronoi_height(base, sites_w, heights, r)
    assert np.abs(h_plane - h_3d).max() < 1e-12, np.abs(h_plane - h_3d).max()


def test_analytic_gradient_matches_finite_differences():
    sites_uv, heights, r = _cell(2)
    rng = np.random.default_rng(7)
    u = rng.uniform(-0.6 * r, 0.6 * r, 60); v = rng.uniform(-0.6 * r, 0.6 * r, 60)
    _, hu, hv = height_uv(u, v, sites_uv, heights, r, grad=True)
    e = 1e-7
    hu_fd = (height_uv(u + e, v, sites_uv, heights, r)
             - height_uv(u - e, v, sites_uv, heights, r)) / (2 * e)
    hv_fd = (height_uv(u, v + e, sites_uv, heights, r)
             - height_uv(u, v - e, sites_uv, heights, r)) / (2 * e)
    assert np.abs(hu - hu_fd).max() < 1e-5, np.abs(hu - hu_fd).max()
    assert np.abs(hv - hv_fd).max() < 1e-5, np.abs(hv - hv_fd).max()


def test_flat_chart_reduces_to_a_disc_of_the_right_area():
    """Zero displacement -> the chart is a flat disc; its triangulated area must approach pi r^2."""
    r = 0.05
    sites_uv = np.zeros((8, 2)); heights = np.zeros(8)
    c = np.array([[0.0, 0.0, 0.0]]); rr = np.array([r])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    N, T, B = chart_frames(q)
    P, Ns, inside, k = sample_chart(0, c, rr, N, T, B, sites_uv[None], heights[None], r / 20.0)
    ok = inside.reshape(k, k)
    a = np.arange(k * k).reshape(k, k)
    qd = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    i00 = a[:-1, :-1][qd]; i10 = a[1:, :-1][qd]; i01 = a[:-1, 1:][qd]; i11 = a[1:, 1:][qd]
    F = np.concatenate([np.stack([i00, i10, i11], 1), np.stack([i00, i11, i01], 1)], 0)
    t_ = P[F]
    area = float(0.5 * np.linalg.norm(np.cross(t_[:, 1] - t_[:, 0], t_[:, 2] - t_[:, 0]),
                                      axis=1).sum())
    exact = np.pi * r * r
    # A quad is emitted only where all FOUR lattice corners survive, so the chart rim loses a band
    # of width ~delta and the area is systematically LOW by roughly perimeter*delta/area = 2*delta/r.
    # That under-coverage is a real property of the tracer, so the test bounds it rather than
    # hiding it: never over-estimate, and never miss by more than the predicted rim.
    delta = r / 20.0
    assert area <= exact + 1e-12, "a trimmed lattice cannot exceed the disc"
    assert (exact - area) / exact < 2.5 * delta / r, ((exact - area) / exact, 2.5 * delta / r)
    # a flat chart's normals must all equal the cell normal
    assert np.abs(np.abs(Ns[inside] @ N[0]) - 1.0).max() < 1e-9

    # refining the lattice must shrink the rim deficit -- i.e. it converges to the disc
    def _area(d):
        P2, _, ins2, k2 = sample_chart(0, c, rr, N, T, B, sites_uv[None], heights[None], d)
        ok2 = ins2.reshape(k2, k2); a2 = np.arange(k2 * k2).reshape(k2, k2)
        q2 = ok2[:-1, :-1] & ok2[1:, :-1] & ok2[:-1, 1:] & ok2[1:, 1:]
        if not q2.any():
            return 0.0
        z = [a2[:-1, :-1][q2], a2[1:, :-1][q2], a2[:-1, 1:][q2], a2[1:, 1:][q2]]
        F2 = np.concatenate([np.stack([z[0], z[1], z[3]], 1),
                             np.stack([z[0], z[3], z[2]], 1)], 0)
        tt = P2[F2]
        return float(0.5 * np.linalg.norm(np.cross(tt[:, 1] - tt[:, 0], tt[:, 2] - tt[:, 0]),
                                          axis=1).sum())
    assert (exact - _area(r / 60.0)) < (exact - _area(r / 15.0)), "refinement did not converge"


def test_normals_are_perpendicular_to_the_surface():
    """The analytic normal must be orthogonal to numerically-estimated surface tangents."""
    sites_uv, heights, r = _cell(3)
    c = np.array([[0.1, 0.2, -0.3]]); rr = np.array([r])
    q = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
    N, T, B = chart_frames(q)
    delta = r / 40.0
    P, Ns, inside, k = sample_chart(0, c, rr, N, T, B, sites_uv[None], heights[None], delta)
    P = P.reshape(k, k, 3); Ns = Ns.reshape(k, k, 3); ok = inside.reshape(k, k)
    worst = 0.0
    for i in range(1, k - 1):
        for j in range(1, k - 1):
            if not (ok[i, j] and ok[i + 1, j] and ok[i - 1, j] and ok[i, j + 1] and ok[i, j - 1]):
                continue
            du = P[i + 1, j] - P[i - 1, j]
            dv = P[i, j + 1] - P[i, j - 1]
            du /= np.linalg.norm(du); dv /= np.linalg.norm(dv)
            worst = max(worst, abs(Ns[i, j] @ du), abs(Ns[i, j] @ dv))
    # the reference tangents are lattice SECANTS, which deviate from the true tangent by O(delta);
    # the analytic normal is exact, so this bounds the discretisation of the CHECK, not the normal
    assert worst < 6e-3, worst


def test_handoff_drops_points_buried_in_a_neighbour():
    """A chart point that lies inside a neighbour's solid must be dropped, and one outside kept."""
    r = 0.05
    c = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]])       # overlapping balls
    rr = np.array([r, r])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    N, T, B = chart_frames(q)
    sites_uv = np.zeros((2, 8, 2)); heights = np.zeros((2, 8))
    # q = [1,0,0,0] is the identity, and get_normals takes the rotation matrix's FIRST COLUMN,
    # so n = +X (not +Z). Cell 1's matter is therefore {x <= 0.04} inside its ball.
    inside_pt = np.array([[0.02, 0.0, 0.0]])                # strictly inside cell 1's solid
    outside_pt = np.array([[-0.04, 0.0, 0.0]])              # beyond cell 1's ball entirely
    keep_in = exposed(inside_pt, 0, np.array([1]), c, rr, N, T, B, sites_uv, heights)
    keep_out = exposed(outside_pt, 0, np.array([1]), c, rr, N, T, B, sites_uv, heights)
    assert not keep_in[0], "a point buried in the neighbour must be dropped"
    assert keep_out[0], "a point outside the neighbour must be kept"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
