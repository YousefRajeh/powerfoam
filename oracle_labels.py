"""Assign GROUND-TRUTH labels to foam cells, to build an oracle surface.

WHY. Every surface number we produce mixes two things: how well the semantics were lifted, and how
faithfully the extractor turns a labelled foam into a surface. An oracle replaces the first with
perfect information, so whatever error remains is purely the second -- the ceiling of the metric on
this representation. Without it there is no way to say whether SCD 25 cm is a good semantic result
or simply the best the geometry can do.

HOW. Each GT mesh vertex is assigned to the power cell that OWNS it, using the same exact
membership test the renderer uses (`argmin ||x - c_i||^2 - r_i^2`, via point_cloud_query), and each
cell takes the majority label of the GT vertices it owns. A cell owning no GT vertex gets label 0
and contributes nothing -- it has no oracle answer, and inventing one (nearest vertex, say) would
smuggle a guess into the ceiling.

That last point matters more than it looks: roughly 90% of cells own no GT point, so the oracle
surface is built from a small, surface-adjacent minority of cells. That is itself the answer to
"how much of the tessellation is even scoreable".

FOR GAUSSIANS the same idea needs a different ownership query, because a Gaussian mixture has no
disjoint partition of space to ask "which primitive contains this point". The convention used
everywhere else in this project applies here too: nearest centre by Euclidean distance. On a
frozen-init arm that is not even an approximation -- there is one Gaussian per GT vertex, so the
nearest centre is at distance ~0 and the oracle is an exact relabelling. That makes the 3DGS oracle
the most generous case available to it, which is the point: whatever haze survives it is not a
labelling failure.
"""
import numpy as np


def oracle_labels_by_nearest_gt(centers, gt_points, gt_labels, max_dist=0.10):
    """Every primitive takes the label of the NEAREST GT point, or 0 if that point is far away.

    WHEN TO USE THIS INSTEAD OF THE OWNERSHIP ORACLE. Ownership answers "what is the ceiling of a
    metric on this representation", and for that, refusing to label a cell that owns no GT vertex is
    the honest choice. But it is the wrong oracle for COMPARING arms with different primitive
    counts: a densified arm has ~3x more primitives than there are GT vertices, so most of its cells
    own nothing and would be silenced -- making it look cleaner purely because fewer primitives were
    labelled. Here every primitive gets a label, so the arms differ only in WHERE THEIR MASS SITS,
    which is the thing the figure is about.

    `max_dist` is not a tuning knob but the definition of "near the true surface": a primitive
    farther than this from any labelled GT point has no defensible class, and the fraction that
    survives is itself the diagnostic -- it measures how much of the representation sits on the
    real surface rather than floating in empty space.
    """
    from scipy.spatial import cKDTree

    lab = np.asarray(gt_labels)
    keep = lab > 0
    tree = cKDTree(np.asarray(gt_points, dtype=np.float64)[keep])
    d, j = tree.query(np.asarray(centers, dtype=np.float64), k=1)
    out = lab[keep][j].astype(np.int64)
    out[d > max_dist] = 0
    stats = {
        "n_cells": int(len(centers)),
        "n_labelled": int((out > 0).sum()),
        "frac_labelled": float((out > 0).mean()),
        "median_dist_m": float(np.median(d)),
        "frac_beyond_max": float((d > max_dist).mean()),
    }
    return out, stats


def oracle_labels_nearest(centers, gt_points, gt_labels, n_classes):
    """Gaussian-side oracle: majority GT label among the points whose NEAREST centre is this one."""
    from point_cloud_query import assign_points_to_nearest_center

    owner = np.asarray(assign_points_to_nearest_center(np.asarray(gt_points, dtype=np.float64),
                                                       np.asarray(centers, dtype=np.float64)))
    return _majority(owner, gt_labels, len(centers), n_classes)


def _majority(owner, gt_labels, P, n_classes):
    ok = (owner >= 0) & (np.asarray(gt_labels) > 0)
    votes = np.zeros((P, n_classes), dtype=np.int64)
    np.add.at(votes, (owner[ok], np.asarray(gt_labels)[ok]), 1)
    lab = votes.argmax(axis=1)
    lab[votes.max(axis=1) == 0] = 0
    stats = {
        "n_cells": int(P),
        "n_cells_with_gt": int((lab > 0).sum()),
        "frac_cells_with_gt": float((lab > 0).mean()),
        "n_gt_points": int(len(owner)),
        "n_gt_assigned": int(ok.sum()),
        "frac_gt_assigned": float(ok.mean()) if len(owner) else 0.0,
        "mean_gt_per_labelled_cell": float(votes.sum() / max((lab > 0).sum(), 1)),
        "vote_purity": float(votes.max(axis=1)[lab > 0].sum() / max(votes.sum(), 1)),
    }
    return lab, stats


def oracle_labels(centers, radii, gt_points, gt_labels, n_classes, valid=None):
    """(P,) cell labels in 1..K, 0 where the cell owns no GT vertex.

    Returns (labels, stats) with stats reporting how much of the tessellation the oracle covers.
    """
    from point_cloud_query import assign_points_to_power_cells

    owner = np.asarray(assign_points_to_power_cells(np.asarray(gt_points, dtype=np.float64),
                                                    np.asarray(centers, dtype=np.float64),
                                                    np.asarray(radii, dtype=np.float64),
                                                    valid=valid, k=8))
    P = len(centers)
    ok = (owner >= 0) & (np.asarray(gt_labels) > 0)
    # votes[cell, class]; class 0 (unlabelled GT) is never counted
    votes = np.zeros((P, n_classes), dtype=np.int64)
    np.add.at(votes, (owner[ok], np.asarray(gt_labels)[ok]), 1)
    lab = votes.argmax(axis=1)
    lab[votes.max(axis=1) == 0] = 0                    # owns no labelled GT vertex

    stats = {
        "n_cells": int(P),
        "n_cells_with_gt": int((lab > 0).sum()),
        "frac_cells_with_gt": float((lab > 0).mean()),
        "n_gt_points": int(len(gt_points)),
        "n_gt_assigned": int(ok.sum()),
        "frac_gt_assigned": float(ok.mean()) if len(gt_points) else 0.0,
        "mean_gt_per_labelled_cell": float(votes.sum() / max((lab > 0).sum(), 1)),
        # purity: of the GT votes inside a labelled cell, what fraction agree with the majority.
        # < 1 means the cell straddles a class boundary, so even a perfect method cannot win it.
        "vote_purity": float(votes.max(axis=1)[lab > 0].sum() / max(votes.sum(), 1)),
    }
    return lab, stats
