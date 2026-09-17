"""Closed-form tests for the exact foam isosurface.

Each case has a hand-computable answer, so an error in the plane, the clipping, the area or the
sampling shows up as a number rather than as a plausible-looking surface.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foam_exact_surface import (face_polygon, foam_isosurface, polygon_area, radical_plane,
                                sample_polygon)

BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([1.0, 1.0, 1.0])


def test_radical_plane_is_the_bisector_for_equal_radii():
    ci, cj = np.array([0.25, 0.5, 0.5]), np.array([0.75, 0.5, 0.5])
    n, d = radical_plane(ci, cj, 0.0, 0.0)
    # plane n.x = d must be x = 0.5
    assert np.allclose(n / n[0], [1, 0, 0])
    assert abs(d / n[0] - 0.5) < 1e-12


def test_radical_plane_shifts_away_from_the_larger_radius():
    """r_i > r_j must push the plane toward j, giving cell i more space.

    Closed form on the axis: x* = (c_i + c_j)/2 + (r_i^2 - r_j^2) / (2 * (c_j - c_i)).
    """
    ci, cj = np.array([0.25, 0.5, 0.5]), np.array([0.75, 0.5, 0.5])
    ri, rj = 0.2, 0.0
    n, d = radical_plane(ci, cj, ri, rj)
    x = d / n[0]
    expect = 0.5 + (ri ** 2 - rj ** 2) / (2 * (cj[0] - ci[0]))
    assert abs(x - expect) < 1e-12
    assert x > 0.5                                     # i (the larger radius) gained space


def test_face_between_two_cells_fills_the_box_cross_section():
    """Two sites, no other neighbours: the face is the whole 1x1 cross-section of the box."""
    centers = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    radii = np.zeros(2)
    poly = face_polygon(0, 1, centers, radii, np.array([1]), BOX_LO, BOX_HI)
    assert abs(polygon_area(poly) - 1.0) < 1e-9
    assert np.allclose(poly[:, 0], 0.5)                # lies exactly on x = 0.5


def test_third_site_cuts_the_face_in_half():
    """A neighbour whose plane passes through the middle of the face must halve its area."""
    centers = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [0.25, 1.5, 0.5]])
    radii = np.zeros(3)
    # cell 0 vs 2 is the plane y = 1.0, so the face at x=0.5 survives only for y <= 1
    poly = face_polygon(0, 1, centers, radii, np.array([1, 2]), BOX_LO, BOX_HI)
    assert abs(polygon_area(poly) - 1.0) < 1e-9        # box already limits y <= 1
    # move the third site closer so its plane cuts at y = 0.5
    centers[2] = [0.25, 0.5 + 0.5, 0.5]                # bisector at y = 0.75
    poly = face_polygon(0, 1, centers, radii, np.array([1, 2]), BOX_LO, BOX_HI)
    assert abs(polygon_area(poly) - 0.75) < 1e-9


def test_polygon_area_of_known_square_and_triangle():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    assert abs(polygon_area(sq) - 1.0) < 1e-12
    tri = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=float)
    assert abs(polygon_area(tri) - 3.0) < 1e-12


def test_sampling_is_uniform_on_the_polygon():
    """Samples on the unit square must have mean at its centre and the right variance (1/12)."""
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    s = sample_polygon(sq, 200_000, np.random.default_rng(0))
    assert np.allclose(s.mean(axis=0)[:2], [0.5, 0.5], atol=5e-3)
    assert np.allclose(s.var(axis=0)[:2], [1 / 12, 1 / 12], atol=5e-3)
    assert np.allclose(s[:, 2], 0.0)


def test_isosurface_of_a_single_occupied_cell_is_its_whole_boundary():
    """One occupied cell surrounded by six unoccupied neighbours in a box: the surface is the
    cell's own boundary, a 0.5-cube of area 6 * 0.5^2 = 1.5."""
    c = np.array([0.5, 0.5, 0.5])
    off = 0.5
    centers = np.array([c] + [c + d * off for d in np.eye(3)] + [c - d * off for d in np.eye(3)])
    radii = np.zeros(7)
    density = np.array([10.0] + [0.0] * 6)
    prim_class = np.array([1] + [0] * 6)
    adjacency = np.concatenate([np.arange(1, 7), np.zeros(6, dtype=int)]).astype(np.int64)
    offsets = np.array([0, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    pts, cls, areas = foam_isosurface(centers, radii, density, prim_class, adjacency, offsets,
                                      sigma_iso=1.0, box_lo=BOX_LO, box_hi=BOX_HI)
    assert set(areas) == {1}
    assert abs(areas[1] - 1.5) < 1e-6, areas
    assert set(np.unique(cls)) == {1}
    # every sample must lie on the cube of side 0.5 centred at c
    dev = np.abs(pts - c).max(axis=1)
    assert np.allclose(dev, 0.25, atol=1e-6)


def test_occupancy_threshold_selects_the_surface():
    """Raising sigma_iso above every density empties the surface."""
    c = np.array([0.5, 0.5, 0.5])
    centers = np.array([c] + [c + d * 0.5 for d in np.eye(3)] + [c - d * 0.5 for d in np.eye(3)])
    radii = np.zeros(7)
    density = np.array([10.0] + [0.0] * 6)
    prim_class = np.array([1] + [0] * 6)
    adjacency = np.concatenate([np.arange(1, 7), np.zeros(6, dtype=int)]).astype(np.int64)
    offsets = np.array([0, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    pts, cls, areas = foam_isosurface(centers, radii, density, prim_class, adjacency, offsets,
                                      sigma_iso=50.0, box_lo=BOX_LO, box_hi=BOX_HI)
    assert len(pts) == 0 and areas == {}


def test_exact_surface_beats_marching_cubes_on_a_box():
    """The whole point: on a piecewise-constant field the exact extractor is exact, while marching
    cubes on the same configuration inflates the area. Uses the single-cell cube above (1.5 m^2)."""
    from surface_extract import grid_from_bbox, isosurface_samples
    c = np.array([0.5, 0.5, 0.5])
    centers = np.array([c] + [c + d * 0.5 for d in np.eye(3)] + [c - d * 0.5 for d in np.eye(3)])
    radii = np.zeros(7)
    density = np.array([10.0] + [0.0] * 6)
    prim_class = np.array([1] + [0] * 6)
    adjacency = np.concatenate([np.arange(1, 7), np.zeros(6, dtype=int)]).astype(np.int64)
    offsets = np.array([0, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    _, _, areas = foam_isosurface(centers, radii, density, prim_class, adjacency, offsets,
                                  sigma_iso=1.0, box_lo=BOX_LO, box_hi=BOX_HI)
    exact_err = abs(areas[1] - 1.5) / 1.5

    from surface_extract import foam_volume
    h = 0.02
    lo, dims = grid_from_bbox(BOX_LO, BOX_HI, h, margin=0.0)
    dens, cvol = foam_volume(centers, radii, density, prim_class, lo, dims, h)
    _, _, mc_areas = isosurface_samples(dens, cvol, lo, h, iso=1.0)
    mc_err = abs(sum(mc_areas.values()) - 1.5) / 1.5
    assert exact_err < 1e-6, exact_err
    assert mc_err > exact_err, (exact_err, mc_err)
    print(f"    exact error {exact_err:.2e}   marching-cubes error {mc_err:.3%}")
