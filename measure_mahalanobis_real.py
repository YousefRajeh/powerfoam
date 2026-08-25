"""Pick the Gaussian point-query on REAL checkpoints, not synthetic assumptions.

Two decisions have to be settled before the Gaussian arms of the ablation mean anything:

  (1) DISTANCE or LIKELIHOOD? Adding +log|Sigma| turns Mahalanobis distance into a Gaussian
      log-likelihood. On real gsplat checkpoints the minimum scale is EXACTLY 0.0 and
      log|Sigma| spans 40-87 nats, so a single degenerate Gaussian has log|Sigma| -> -inf and
      would win every point in the scene. The likelihood form is unusable here without
      arbitrary regularisation, and the user asked for Mahalanobis DISTANCE, so that is what
      this measures: argmin_i (x-mu_i)^T Sigma_i^-1 (x-mu_i).

  (2) HOW MANY kNN CANDIDATES? The kNN restriction is only valid if the Mahalanobis winner is
      usually among the k Euclidean-nearest centres. On synthetic heavy-anisotropy data that
      assumption failed badly (56% disagreement at k=32), so it must be measured on the real
      scale/rotation distribution rather than assumed.

Degenerate scales are floored at SCALE_FLOOR_REL x the scene extent: a zero-scale Gaussian is
a rendering degeneracy, and without a floor its Sigma^-1 is infinite, making m2 infinite in
the off-axis directions and 0/0 along the axis.

Run:  D:\\conda\\envs\\powerfoam\\python.exe measure_mahalanobis_real.py [scene]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from ablation_assign import _quat_to_R

SCALE_FLOOR_REL = 1e-6


def load(tag, scene):
    p = f"recon_remote/{tag}/{scene}/ckpt.pt"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sp = sd.get("splats", sd)
    means = sp["means"].float()
    scales = torch.exp(sp["scales"].float())
    q = sp["quats"].float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    extent = (means.max(0).values - means.min(0).values).norm().item()
    floor = SCALE_FLOOR_REL * extent
    n_deg = int((scales < floor).any(1).sum())
    scales = scales.clamp_min(floor)
    return means, scales, q, extent, floor, n_deg


def maha_exact(pts, means, scales, quats, tile=20000):
    """Exact argmin of pure Mahalanobis distance, tiled over Gaussians."""
    R = _quat_to_R(quats)
    best = torch.full((pts.shape[0],), float("inf"), device=pts.device)
    idx = torch.full((pts.shape[0],), -1, dtype=torch.long, device=pts.device)
    for a in range(0, means.shape[0], tile):
        b = min(a + tile, means.shape[0])
        diff = pts[:, None, :] - means[None, a:b, :]
        local = torch.einsum("nij,pni->pnj", R[a:b], diff)
        m2 = ((local / scales[None, a:b]) ** 2).sum(-1)
        v, j = m2.min(1)
        upd = v < best
        best[upd] = v[upd]
        idx[upd] = j[upd] + a
    return idx, best


def maha_knn(pts, means, scales, quats, k):
    R = _quat_to_R(quats)
    d = torch.cdist(pts, means)
    kk = min(k, means.shape[0])
    nn = d.topk(kk, largest=False).indices
    diff = pts[:, None, :] - means[nn]
    local = torch.einsum("pkij,pki->pkj", R[nn], diff)
    m2 = ((local / scales[nn]) ** 2).sum(-1)
    return nn.gather(1, m2.argmin(1, keepdim=True)).squeeze(1)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0062_00"
    dev = "cuda"
    g = torch.Generator().manual_seed(0)

    for tag in ("gs_froz", "gs_unfroz"):
        if not os.path.exists(f"recon_remote/{tag}/{scene}/ckpt.pt"):
            print(f"{tag}: not downloaded yet"); continue
        means, scales, quats, extent, floor, n_deg = load(tag, scene)
        N = means.shape[0]
        print(f"\n=== {tag} {scene}: N={N:,}  extent={extent:.2f}  "
              f"floor={floor:.2e}  degenerate={n_deg:,} ({100*n_deg/N:.2f}%)")

        # exact is O(P*N); subsample points so the reference is affordable
        P = 1500 if N < 300_000 else 400
        sel = torch.randperm(N, generator=g)[:P]
        pts = (means[sel] + torch.randn(P, 3, generator=g) * extent * 0.01).to(dev)

        m, s, q = means.to(dev), scales.to(dev), quats.to(dev)
        ref, _ = maha_exact(pts, m, s, q)
        nearest = torch.cdist(pts, m).argmin(1)
        print(f"  exact-maha vs nearest-centre differ: "
              f"{(ref != nearest).float().mean()*100:.1f}% of points")
        for k in (16, 32, 64, 128, 256):
            if k > N:
                continue
            a = maha_knn(pts, m, s, q, k)
            dis = (a != ref).float().mean().item() * 100
            mark = "  OK" if dis < 1 else ("  marginal" if dis < 5 else "  <-- UNUSABLE")
            print(f"  k={k:<4} disagreement vs exact {dis:6.2f}%{mark}")


if __name__ == "__main__":
    main()
