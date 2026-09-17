"""Correctness tests for surface_extract, on inputs whose answers are known in closed form.

Every one of these would pass silently if the code were wrong in a way that only shifts or scales
the extracted surface -- which is exactly the failure mode that would bias every distance in the
metric without producing an error.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from surface_extract import grid_from_bbox, isosurface_samples, foam_volume


def test_grid_from_bbox_dims_and_origin():
    """A 1 m cube at 10 cm with a 5 cm margin: origin shifts by the margin, 11 cells per axis."""
    lo, dims = grid_from_bbox([0, 0, 0], [1, 1, 1], h=0.1, margin=0.05)
    assert np.allclose(lo, [-0.05, -0.05, -0.05])
    # span 1.1 m / 0.1 = 11 cells
    assert list(dims) == [11, 11, 11]


def _sphere_volume(R=0.30, h=0.02, pad=0.12):
    """Analytic sphere as a density field: 1 inside, 0 outside, centred in its own grid."""
    lo, dims = grid_from_bbox([-R - pad] * 3, [R + pad] * 3, h, margin=0.0)
    nx, ny, nz = [int(d) for d in dims]
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    p = lo + (np.stack([ix, iy, iz], -1) + 0.5) * h
    d = np.linalg.norm(p, axis=-1)
    dens = (d <= R).astype(np.float32)
    cls = np.ones((nx, ny, nz), dtype=np.int64)
    return dens, cls, lo, h, R


def test_isosurface_area_matches_analytic_sphere_on_a_SMOOTH_field():
    """On a smooth field marching cubes must recover 4*pi*R^2 essentially exactly.

    This is the test that pins the implementation: an offset error leaves the area right but moves
    the surface, a scale error changes the area quadratically, so area AND radius together fix both.
    A SMOOTH field is used because that is the only case in which marching cubes is supposed to be
    accurate -- see the next test for what a step function does.
    """
    dens, cls, lo, h, R = _sphere_volume()
    nx, ny, nz = dens.shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    d = np.linalg.norm(lo + (np.stack([ix, iy, iz], -1) + 0.5) * h, axis=-1)
    smooth = (R - d).astype(np.float32)                     # signed distance, level 0
    pts, lab, areas = isosurface_samples(smooth, cls, lo, h, iso=0.0, samples_per_m2=20000)
    got, exact = sum(areas.values()), 4 * np.pi * R ** 2
    assert abs(got - exact) / exact < 0.01, f"area {got:.4f} vs analytic {exact:.4f}"
    r = np.linalg.norm(pts, axis=1)
    assert abs(r.mean() - R) < 0.05 * h, f"mean radius {r.mean():.4f} vs {R}"
    assert r.std() < 0.05 * h, f"radius spread {r.std():.4f} too large"


def test_step_field_bias_is_real_and_does_not_converge():
    """A PIECEWISE-CONSTANT field inflates the area by ~10% at every resolution.

    Measured: 7.7% at h=4cm, 9.8% at 2cm, 8.9% at 1cm -- it does not shrink, because a step
    function gives the interpolator nothing to interpolate and every crossing snaps to a voxel
    midpoint. A foam's density is exactly this kind of field, which is why the foam is extracted
    analytically by foam_exact_surface instead of through this path. Pinned here so that the bias
    stays a measured, documented quantity rather than being rediscovered as a mystery.
    """
    exact_area = None
    errs = {}
    for h in (0.04, 0.02):
        dens, cls, lo, _, R = _sphere_volume(h=h)
        exact_area = 4 * np.pi * R ** 2
        _, _, areas = isosurface_samples(dens, cls, lo, h, iso=0.5, samples_per_m2=5000)
        errs[h] = (sum(areas.values()) - exact_area) / exact_area
    assert all(0.03 < e < 0.20 for e in errs.values()), errs
    # and it does NOT converge: halving h must not halve the error
    assert errs[0.02] > 0.5 * errs[0.04], errs


def test_isosurface_is_empty_below_and_above_the_level():
    dens, cls, lo, h, R = _sphere_volume()
    pts, _, areas = isosurface_samples(dens, cls, lo, h, iso=1.5)     # nothing reaches 1.5
    assert len(pts) == 0 and areas == {}


def test_class_labels_follow_geometry():
    """Two hemispheres with different class ids: every sample must carry its own side's label."""
    dens, cls, lo, h, R = _sphere_volume()
    nx, ny, nz = dens.shape
    ix = np.arange(nx)[:, None, None] * np.ones((1, ny, nz))
    xw = lo[0] + (ix + 0.5) * h
    cls = np.where(xw < 0, 1, 2).astype(np.int64)
    pts, lab, areas = isosurface_samples(dens, cls, lo, h, iso=0.5, samples_per_m2=20000)
    assert set(np.unique(lab)) <= {1, 2}
    # allow a thin band of mislabelling at the seam (faces straddling x=0 take a majority vote)
    for c, sign in ((1, -1), (2, +1)):
        sel = lab == c
        assert sel.sum() > 0
        frac_right_side = float((np.sign(pts[sel, 0]) == sign).mean())
        assert frac_right_side > 0.9, f"class {c}: only {frac_right_side:.2%} on its own side"


def test_area_sampling_is_proportional_to_area():
    """Sample counts must follow area, so a large region is not out-sampled by a small one."""
    dens, cls, lo, h, R = _sphere_volume()
    nx, ny, nz = dens.shape
    ix = np.arange(nx)[:, None, None] * np.ones((1, ny, nz))
    xw = lo[0] + (ix + 0.5) * h
    # split at x = +0.15R so class 2 is a small cap and class 1 the large remainder
    cls = np.where(xw < 0.15 * R, 1, 2).astype(np.int64)
    pts, lab, areas = isosurface_samples(dens, cls, lo, h, iso=0.5,
                                         samples_per_m2=20000, min_per_class=1)
    n1, n2 = (lab == 1).sum(), (lab == 2).sum()
    assert areas[1] > areas[2]
    ratio_n, ratio_a = n1 / n2, areas[1] / areas[2]
    assert abs(ratio_n - ratio_a) / ratio_a < 0.05


def test_foam_volume_matches_hand_computed_power_cells():
    """Two cells, equal radii: the split plane is the perpendicular bisector, so each voxel takes
    the density and class of whichever centre is nearer. Hand-checkable, no solver involved."""
    centers = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    radii = np.array([0.0, 0.0])
    density = np.array([10.0, 20.0])
    prim_class = np.array([1, 2])
    lo, dims = grid_from_bbox([0, 0, 0], [1, 1, 1], h=0.1, margin=0.0)
    dens, cls = foam_volume(centers, radii, density, prim_class, lo, dims, 0.1)

    nx, ny, nz = dens.shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    xw = lo[0] + (ix + 0.5) * 0.1
    left = xw < 0.5
    assert np.all(cls[left] == 1) and np.all(cls[~left] == 2)
    assert np.allclose(dens[left], 10.0) and np.allclose(dens[~left], 20.0)


def test_foam_volume_radius_moves_the_boundary():
    """A larger radius on one site must claim MORE space: the power-cell boundary is the radical
    plane, which shifts toward the smaller-radius site. Pins that radii are actually used."""
    centers = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    prim_class = np.array([1, 2])
    density = np.array([10.0, 20.0])
    lo, dims = grid_from_bbox([0, 0, 0], [1, 1, 1], h=0.05, margin=0.0)
    _, cls_eq = foam_volume(centers, np.array([0.0, 0.0]), density, prim_class, lo, dims, 0.05)
    _, cls_big = foam_volume(centers, np.array([0.2, 0.0]), density, prim_class, lo, dims, 0.05)
    assert (cls_big == 1).sum() > (cls_eq == 1).sum()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-x"]))
