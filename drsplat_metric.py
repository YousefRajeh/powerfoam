"""Dr-Splat's 3D evaluation protocol (arXiv 2502.16652v1, Eq. 7-8 and the IoU definition).

WHY A SEPARATE MODULE. Our own protocol assigns each GT POINT to its nearest Gaussian and computes an
unweighted point IoU. Dr-Splat assigns a label to each GAUSSIAN by aggregating over GT points, then
computes a SIGNIFICANCE-WEIGHTED IoU. These are different metrics, not two implementations of one, so
they live side by side rather than one being edited into the other.

Their evaluation code is NOT released (README "## Evaluation - TBA"; the string `mahalanobis` does not
appear in their repo), so this is implemented from the paper text alone. Every deviation forced by
that is marked DEVIATION below and must be reported.

    Eq. 7   d_mahal(p, theta) = (p - mu)^T Sigma^-1 (p - mu)
    Eq. 8   s_theta_i = argmax_{s in S} ( sum_{p_k in P} 1{s_p_k = s} . d_mahal(p_k, theta_i) )

    significance      d_i = s_ix . s_iy . s_iz . alpha_i     (relative ellipsoid volume x opacity)
    Intersection_i    d . (l_pred  (x)  l_gt)
    Union_i           d . (l_pred + l_gt - l_pred (x) l_gt)

NOTE ON EQ. 8, recorded because it looks like an error in the paper: the RAW Mahalanobis distance is
the weight inside an argmax, so points FURTHER from a Gaussian contribute MORE to its label. A kernel
(exp(-d/2)) or a reciprocal would weight near points more, which is the obvious intent. We implement
what is PUBLISHED and expose `weight=` to test the alternative, because a protocol comparison that
silently "fixes" the reference is not a comparison.

DEVIATION 1 (unavoidable): Eq. 8 sums over all p_k in P. Evaluated literally that is
|P| x |Gaussians| = 2e5 x 2e6 = 4e11 terms per scene. We restrict each GT point's contribution to its
`k` Euclidean-nearest Gaussians, which is the only tractable reading and is consistent with their
"Gaussians which are not assigned any weight ... are pruned". `k` is a reported parameter.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def _rotation(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def assign_gaussian_labels(pts, labels, means, scales, quats, n_classes,
                           k=16, chunk=100_000, weight="raw"):
    """Dr-Splat Eq. 8: one GT label per GAUSSIAN, by weighted vote over nearby GT points.

    Returns (gt_label per Gaussian, has_gt mask). Gaussians receiving no contribution are left
    unlabelled and excluded -- their "not assigned any weight ... are pruned".

    `weight`: "raw" reproduces the published Eq. 8 exactly; "kernel" uses exp(-d/2) and "inv" uses
    1/(1+d), both of which weight NEAR points more. Provided to test the suspected sign error
    without silently substituting it.
    """
    M = means.shape[0]
    acc = torch.zeros((M, n_classes), dtype=torch.float64)
    tree = cKDTree(means)
    R = _rotation(quats)
    inv_s2 = 1.0 / np.maximum(scales, 1e-8) ** 2

    for s in range(0, pts.shape[0], chunk):
        e = min(s + chunk, pts.shape[0])
        p, lab = pts[s:e], labels[s:e]
        ok = lab >= 0
        if not ok.any():
            continue
        p, lab = p[ok], lab[ok]
        _, idx = tree.query(p, k=min(k, M), workers=-1)
        idx = np.atleast_2d(idx)
        d = p[:, None, :] - means[idx]
        loc = np.einsum('nkij,nkj->nki', R[idx].transpose(0, 1, 3, 2), d)
        m = (loc ** 2 * inv_s2[idx]).sum(-1)                       # Eq. 7, squared Mahalanobis
        if weight == "kernel":
            w = np.exp(-0.5 * m)
        elif weight == "inv":
            w = 1.0 / (1.0 + m)
        else:
            w = m                                                  # Eq. 8 as published
        flat = torch.from_numpy((idx * n_classes + lab[:, None]).reshape(-1).astype(np.int64))
        acc.view(-1).index_add_(0, flat, torch.from_numpy(w.reshape(-1).astype(np.float64)))

    has_gt = acc.sum(1) > 0
    gt_lab = torch.full((M,), -1, dtype=torch.long)
    gt_lab[has_gt] = acc[has_gt].argmax(1)
    return gt_lab, has_gt


def significance(scales, opacity_logit):
    """d_i = s_ix . s_iy . s_iz . alpha_i -- relative ellipsoid volume times opacity.

    `opacity_logit` is the raw stored value; alpha is its sigmoid, matching how 3DGS stores opacity.
    """
    vol = np.prod(np.maximum(scales, 0.0), axis=1)
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit))
    return torch.from_numpy((vol * alpha).astype(np.float64))


def weighted_miou(pred, gt_lab, has_gt, d, n_classes, keep=None):
    """Significance-weighted IoU/accuracy over classes PRESENT in that scene's GT.

    Present-class-only averaging follows the OpenGaussian/NormLift convention already used elsewhere
    in this project; scoring absent classes as 0 would make the number depend on the query list
    rather than the scene.
    """
    m = has_gt.clone()
    if keep is not None:
        m &= keep
    p, g, w = pred[m], gt_lab[m], d[m]
    ious, accs = [], []
    for c in range(n_classes):
        pc, gc = (p == c), (g == c)
        if not bool(gc.any()):
            continue                                   # class absent from this scene's GT
        inter = float((w * (pc & gc)).sum())
        union = float((w * (pc | gc)).sum())           # d.(l_pred + l_gt - l_pred (x) l_gt)
        denom = float((w * gc).sum())
        ious.append(inter / union if union > 0 else 0.0)
        accs.append(inter / denom if denom > 0 else 0.0)
    if not ious:
        return 0.0, 0.0, 0
    return 100.0 * float(np.mean(ious)), 100.0 * float(np.mean(accs)), len(ious)
