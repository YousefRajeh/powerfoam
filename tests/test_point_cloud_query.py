import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center


def test_power_cell_assignment_prefers_larger_radius_when_centers_equidistant():
    """Two primitives equidistant (by Euclidean distance) from a query point, but one has a
    much larger radius -- the power-distance-minimizing (correct) cell owner is the larger one,
    NOT the nearest-by-Euclidean-distance one. This is exactly the case nearest-center matching
    (used for the 3DGS baseline) gets WRONG and power-cell matching gets right -- the whole
    point of using the real power-diagram formula instead of plain nearest-neighbor for
    PowerFoam."""
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.1, 0.9])  # second primitive much larger
    query = np.array([[0.0, 0.0, 0.0]])  # equidistant (dist=1) from both centers

    assigned = assign_points_to_power_cells(query, centers, radii, k=2)
    assert assigned[0] == 1  # larger-radius primitive wins despite equal Euclidean distance

    # Nearest-center matching, by contrast, can't distinguish them (tie) -- confirms the two
    # correspondence mechanisms are genuinely different, not just two names for the same thing.
    nearest = assign_points_to_nearest_center(query, centers)
    assert nearest[0] in (0, 1)  # a tie either way, unlike the power-cell case above


def test_power_cell_assignment_matches_brute_force():
    """Randomized cross-check: the k-NN-candidate-filtered result must match an exact
    brute-force argmin over ALL primitives (k = num_primitives here, so the candidate filter
    can't drop the true winner)."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-5, 5, size=(200, 3))
    radii = rng.uniform(0.05, 0.5, size=200)
    query = rng.uniform(-5, 5, size=(50, 3))

    assigned = assign_points_to_power_cells(query, centers, radii, k=200)

    diff = query[:, None, :] - centers[None, :, :]
    power_dist = (diff ** 2).sum(-1) - radii[None, :] ** 2
    brute_force = power_dist.argmin(axis=1)

    assert np.array_equal(assigned, brute_force)


def test_power_cell_assignment_excludes_invalid_primitives():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    radii = np.array([0.1, 0.1])
    valid = np.array([False, True])
    query = np.array([[0.0, 0.0, 0.0]])  # right on top of the INVALID primitive

    assigned = assign_points_to_power_cells(query, centers, radii, valid=valid, k=2)
    assert assigned[0] == 1  # must skip the invalid one even though it's much closer


def test_power_cell_assignment_max_power_dist_flags_far_points():
    centers = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([0.1])
    query = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    assigned = assign_points_to_power_cells(query, centers, radii, k=1, max_power_dist=1.0)
    assert assigned[0] == 0       # close point: real assignment
    assert assigned[1] == -1     # far point: flagged as outside the reconstruction


def test_nearest_center_assignment_basic():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    query = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])

    assigned = assign_points_to_nearest_center(query, centers)
    assert assigned.tolist() == [0, 1]


def test_nearest_center_assignment_max_dist_flags_far_points():
    centers = np.array([[0.0, 0.0, 0.0]])
    query = np.array([[0.1, 0.0, 0.0], [50.0, 0.0, 0.0]])

    assigned = assign_points_to_nearest_center(query, centers, max_dist=1.0)
    assert assigned[0] == 0
    assert assigned[1] == -1
