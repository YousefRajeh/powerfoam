"""Exact Mahalanobis GT-point -> Gaussian assignment, tiled to fit in VRAM.

WHY EXACT. The kNN-restricted version is not an approximation, it is wrong: measured against
an exhaustive argmin on real checkpoints, Euclidean-kNN candidates disagree on 11% of points
at k=256 for the frozen arm and 49.75% for the unfrozen one (measure_mahalanobis_real.py).
Real gsplat anisotropy is extreme -- median axis ratio 9-15x, p99 6,000-34,000x -- so the
Gaussian that best explains a point is frequently far from the nearest centre.

DISTANCE, NOT LIKELIHOOD. argmin_i (x-mu_i)^T Sigma_i^-1 (x-mu_i), with no +log|Sigma| term.
Adding it would make this a Gaussian log-likelihood, but real checkpoints contain Gaussians
with a scale of EXACTLY 0 (0.40% frozen, 5.75% unfrozen), whose log|Sigma| is -inf; a single
one would win every point in the scene. Pure distance is the query the user asked for and is
also the robust choice here: a degenerate axis makes Sigma^-1 huge, so slivers repel points
instead of attracting them.

Sigma^-1 = R diag(s^-2) R^T, so m2 = || diag(1/s) R^T (x - mu) ||^2 -- computed in that
factored form (no 3x3 inverse, no large-minus-large cancellation).
"""
import time

import torch

SCALE_FLOOR_REL = 1e-6      # relative to scene extent; only guards exact zeros


def quat_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def prepare(means, scales_log, quats):
    """-> means, scales (floored), R, extent, n_degenerate."""
    means = means.float()
    scales = torch.exp(scales_log.float())
    extent = (means.max(0).values - means.min(0).values).norm().item()
    floor = SCALE_FLOOR_REL * max(extent, 1e-6)
    n_deg = int((scales < floor).any(1).sum())
    scales = scales.clamp_min(floor)
    q = quats.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return means, scales, quat_to_R(q), extent, n_deg


def assign_exact(points, means, scales, R, pt_chunk=1024, g_tile=131072, device="cuda",
                 progress=None):
    """Exact argmin of Mahalanobis distance. Returns (idx int64 (P,), m2 float (P,)).

    Tiled both ways so peak memory is pt_chunk x g_tile floats regardless of scene size.
    Work is P*N, done once per (scene, recon) and cached.

    TILE SIZES ARE LOAD-BEARING. The (pt_chunk x g_tile x 3) intermediate must stay small
    enough to live in cache: at 8192x131072 it is 12 GiB and the kernel thrashes at 0.13e9
    pairs/s, while 1024x131072 is 1.5 GiB and runs at 3.10e9 -- a 24x speedup for
    bit-identical output, verified across six tile configurations on a real checkpoint.
    """
    P = points.shape[0]
    N = means.shape[0]
    idx = torch.full((P,), -1, dtype=torch.long, device=device)
    best = torch.full((P,), float("inf"), device=device)
    # pre-scale the rotation rows once: M_i = diag(1/s_i) R_i^T, so m2 = ||M_i (x-mu_i)||^2
    M = R.transpose(1, 2) / scales[:, :, None]          # (N,3,3)

    t0 = time.time()
    for p0 in range(0, P, pt_chunk):
        x = points[p0:p0 + pt_chunk].to(device)          # (C,3)
        c = x.shape[0]
        cb = torch.full((c,), float("inf"), device=device)
        ci = torch.full((c,), -1, dtype=torch.long, device=device)
        for a in range(0, N, g_tile):
            b = min(a + g_tile, N)
            diff = x[:, None, :] - means[a:b][None, :, :]            # (C,T,3)
            loc = torch.einsum("tij,ctj->cti", M[a:b], diff)         # (C,T,3)
            m2 = (loc * loc).sum(-1)                                  # (C,T)
            v, j = m2.min(1)
            upd = v < cb
            cb[upd] = v[upd]
            ci[upd] = j[upd] + a
            del diff, loc, m2
        idx[p0:p0 + pt_chunk] = ci
        best[p0:p0 + pt_chunk] = cb
        if progress and (p0 // pt_chunk) % 5 == 0:
            done = min(p0 + pt_chunk, P)
            el = time.time() - t0
            progress(f"    maha {done}/{P} pts  {el:.0f}s  eta {el/done*(P-done):.0f}s")
    return idx, best


def winner_stats(idx, N):
    """Concentration of the assignment -- a broken query sends most points to a few splats."""
    cnt = torch.bincount(idx[idx >= 0], minlength=N).float()
    used = int((cnt > 0).sum())
    top = cnt.sort(descending=True).values
    tot = cnt.sum().clamp_min(1)
    return {
        "n_gaussians": N,
        "n_used": used,
        "frac_used": used / max(N, 1),
        "top1_share": float(top[0] / tot),
        "top1pct_share": float(top[:max(1, N // 100)].sum() / tot),
        "max_points_on_one": int(top[0]),
    }
