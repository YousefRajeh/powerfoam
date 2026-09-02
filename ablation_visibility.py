"""Visibility-based GT->primitive assignment: attribute a point to the cell that ABSORBED it.

THE PROBLEM THIS SOLVES. ScanNet GT points are mesh vertices reconstructed from the RGB-D scans,
so every one of them was seen by at least one frame -- there is no unseen-depth guesswork on the
GT side. Yet 42% of them land in RadFoam cells with no feature. The reason is not missing
observation, it is MISFILED observation:

    1. the GT surface point is visible in many cameras
    2. rays toward it pass THROUGH a near-transparent cell (alpha ~ 0, contributing nothing)
       and are absorbed by a dense cell behind or beside it
    3. the photometric signal was captured -- it landed in the ABSORBING cell
    4. but Voronoi ownership hands the point to the TRANSPARENT cell, which received nothing

OpenGaussian's opacity mask responds by DELETING such points from the metric. That is defensible
for comparability, but it discards information that exists: on scene0062_00 it removes 60.7% of
RadFoam's GT points. This module instead REASSIGNS the point to the cell that could actually have
absorbed the light -- the nearest primitive whose opacity is above threshold.

WHY NEAREST-OPAQUE IS THE RIGHT APPROXIMATION. The exact quantity would be argmax over cells of
accumulated alpha*T along rays from every camera to that point, which needs a full re-trace per
view. Nearest-opaque is the same statement evaluated locally: among the cells in the immediate
neighbourhood, the one that can absorb light AND is closest to the point is the one a ray
terminating at that surface would have deposited into. It requires only positions and per-
primitive opacity, both already on disk.

WHAT THIS IS NOT. It is not a way to score points nothing observed. A point whose entire
neighbourhood is transparent has no absorbing cell within the search radius and is left unowned,
scoring as a miss -- exactly as an unobservable point should. The reassignment only moves a point
from a transparent owner to a nearby opaque one; it never invents evidence.

THREE PROTOCOLS, and the ablation reports all three:
    geometric   argmin |x-c|^2 - r^2                      -- the partition, unmodified
    masked      geometric, then delete low-opacity points  -- OpenGaussian/NormLift
    visibility  nearest primitive with alpha >= threshold  -- this module
"""
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_THRESHOLD = 0.1      # same threshold OpenGaussian masks at, for a like-for-like contrast


def assign_visibility(points, centers, alpha, threshold=DEFAULT_THRESHOLD, max_dist=None,
                      workers=-1):
    """-> (assignment int64 (P,), stats dict). -1 where no opaque primitive is within max_dist.

    points   (P,3)   GT points
    centers  (N,3)   primitive centres
    alpha    (N,)    per-primitive opacity (ablation_opacity.primitive_alpha)
    max_dist         search cap in metres; None -> the 99th percentile of the geometric
                     nearest-centre distance, so the cap adapts to primitive density rather than
                     being a magic constant.
    """
    centers = np.asarray(centers, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    opaque = np.asarray(alpha) >= threshold
    idx_opaque = np.where(opaque)[0]

    if idx_opaque.size == 0:
        return np.full(len(points), -1, dtype=np.int64), {
            "n_opaque": 0, "frac_assigned": 0.0, "max_dist": 0.0, "threshold": threshold}

    if max_dist is None:
        # Scale the cap by the spacing of the OPAQUE cells, not of all cells. Basing it on
        # all-cell spacing is degenerate for a frozen arm whose sites ARE the GT points: the
        # nearest-centre distance is then ~0, the cap collapses to ~0, and two thirds of points
        # fail it and go unowned (measured: rf_froz fell to 13.86 mIoU with only 33.8% assigned).
        # A ray terminating at a surface deposits into a cell within roughly a few cell
        # diameters, so the median nearest-neighbour spacing AMONG opaque cells is the right
        # unit, and 5x it is a generous but still finite reach.
        oc = centers[idx_opaque]
        probe = oc if len(oc) <= 20000 else oc[np.random.default_rng(0).choice(len(oc), 20000,
                                                                              replace=False)]
        dnn, _ = cKDTree(oc).query(probe, k=2, workers=workers)   # k=2: skip self
        max_dist = float(np.median(dnn[:, 1]) * 5.0)

    tree = cKDTree(centers[idx_opaque])
    d, j = tree.query(points, k=1, workers=workers)
    out = idx_opaque[j].astype(np.int64)
    too_far = d > max_dist
    out[too_far] = -1
    return out, {
        "n_opaque": int(idx_opaque.size),
        "frac_opaque": float(opaque.mean()),
        "frac_assigned": float((out >= 0).mean()),
        "median_dist": float(np.median(d[~too_far])) if (~too_far).any() else float("nan"),
        "max_dist": float(max_dist),
        "threshold": threshold,
    }
