"""Point <-> primitive correspondence for 3D point-level mIoU evaluation (OpenGaussian-style
protocol: github.com/yanmin-wu/OpenGaussian, scripts/eval_scannet.py).

OpenGaussian gets an exact GT-point <-> Gaussian correspondence only by forcing it: they
initialize one Gaussian per GT mesh vertex and pass --frozen_init_pts (disables 3DGS
densification entirely) so the Gaussian set never changes count or order for the whole
90k-iteration run (verified directly against their scripts/train_scannet.sh). Vanilla 3DGS has
no other way to answer "which primitive owns this exact 3D point" -- Gaussians overlap and
blend, so there is no disjoint ownership to query.

PowerFoam does not need this trick. Its power diagram partitions ALL of 3D space into disjoint
cells by construction (every point belongs to exactly one cell, via the power/Laguerre distance
`||x - center||^2 - radius^2`, minimized over primitives -- the exact formula used for hard
cell-boundary determination in the real ray-cell traversal kernels, `powerfoam/rasterize.py`'s
`pow_dist = wp.dot(v, v) - radius * radius`). So PowerFoam can train completely normally (full
densification/pruning, no artificial freezing) and answer "which cell contains this GT point?"
directly and exactly at eval time, for any point, regardless of how primitive count/positions
evolved during training.

The Splat Feature Solver (3DGS) baseline has no such disjoint structure, so it gets its own
natural equivalent: nearest-Gaussian-center by Euclidean distance -- the standard convention in
the wider 3D open-vocab literature when a method isn't using OpenGaussian's frozen-point trick.
Same evaluation logic afterward for both (see evaluate_point_cloud_miou.py); only the
correspondence mechanism differs, one per method's own natural query, matching this project's
established principle of never reimplementing one method's mechanism on the other's terms.
"""
import numpy as np
import torch
from scipy.spatial import KDTree


def assign_points_to_power_cells(query_points, centers, radii, valid=None, k=64, max_power_dist=None):
    """For each query point, find the primitive whose power cell contains it: argmin over
    primitives of `||x - center||^2 - radius^2`.

    Exact for any k >= num_valid_primitives; for large P this uses a Euclidean k-NN spatial
    candidate filter first (a primitive minimizing power distance is essentially always among
    the k nearest by Euclidean distance unless radii vary wildly relative to local primitive
    spacing -- k=64 default is generous), then computes the EXACT power distance only among
    those k candidates and takes the true argmin. This is an approximate CANDIDATE filter with
    an exact final decision, not an approximate power-distance computation.

    Parameters
    ----------
    query_points : (Q, 3) float array/tensor, world-space points to assign (e.g. a GT mesh's
        labeled vertices).
    centers : (P, 3) primitive centers.
    radii : (P,) primitive radii.
    valid : (P,) bool, primitives eligible to own a point (e.g. `support > 0`). Invalid
        primitives are excluded from the KD-tree entirely.
    k : number of Euclidean nearest-neighbor candidates to consider per query point.
    max_power_dist : if given, query points whose best power distance still exceeds this
        (i.e. clearly outside every nearby cell -- can happen for GT points outside the
        reconstructed volume) get assigned index -1 instead of a nearest-anyway guess.

    Returns
    -------
    assigned_idx : (Q,) int64 array, index into the ORIGINAL (unfiltered) centers/radii arrays
        (i.e. already mapped back through `valid`), or -1 for points with no primitive within
        `max_power_dist` (if given).
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    if valid is None:
        valid_idx = np.arange(centers.shape[0])
    else:
        valid_idx = np.nonzero(np.asarray(valid))[0]
    if valid_idx.size == 0:
        raise ValueError("no valid primitives to assign points to")

    kdtree = KDTree(centers[valid_idx])
    k_eff = min(k, valid_idx.size)
    _, cand_local = kdtree.query(query_points, k=k_eff)
    if k_eff == 1:
        cand_local = cand_local[:, None]
    cand_global = valid_idx[cand_local]  # (Q, k_eff)

    cand_centers = centers[cand_global]                     # (Q, k_eff, 3)
    cand_radii = radii[cand_global]                          # (Q, k_eff)
    diff = query_points[:, None, :] - cand_centers            # (Q, k_eff, 3)
    power_dist = np.einsum("qkd,qkd->qk", diff, diff) - cand_radii ** 2  # (Q, k_eff)

    best_local = np.argmin(power_dist, axis=1)               # (Q,)
    best_power_dist = power_dist[np.arange(len(query_points)), best_local]
    assigned_idx = cand_global[np.arange(len(query_points)), best_local]

    if max_power_dist is not None:
        assigned_idx = np.where(best_power_dist <= max_power_dist, assigned_idx, -1)

    return assigned_idx.astype(np.int64)


def assign_points_to_nearest_center(query_points, centers, valid=None, max_dist=None):
    """Splat Feature Solver's correspondence: nearest Gaussian center by Euclidean distance
    (no disjoint-ownership structure to query exactly, unlike PowerFoam's power cells).

    Parameters mirror `assign_points_to_power_cells` (radii unused -- Gaussians have no hard
    partition boundary to test against, so only proximity is used).

    Returns
    -------
    assigned_idx : (Q,) int64 array, index into the ORIGINAL centers array, or -1 for points
        farther than `max_dist` from every valid primitive (if given).
    """
    centers = np.asarray(centers, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    if valid is None:
        valid_idx = np.arange(centers.shape[0])
    else:
        valid_idx = np.nonzero(np.asarray(valid))[0]
    if valid_idx.size == 0:
        raise ValueError("no valid primitives to assign points to")

    kdtree = KDTree(centers[valid_idx])
    dist, nearest_local = kdtree.query(query_points, k=1)
    assigned_idx = valid_idx[nearest_local]

    if max_dist is not None:
        assigned_idx = np.where(dist <= max_dist, assigned_idx, -1)

    return assigned_idx.astype(np.int64)
