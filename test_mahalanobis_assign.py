"""Is the kNN-restricted Mahalanobis assignment correct, and is it the right query?

The Gaussian arms have no disjoint cell structure, so the GT-point -> primitive
correspondence is a modelling choice. This asserts three things about the one we made:

  T1  EXACTNESS vs brute force. With k = N the kNN restriction is vacuous, so the result must
      equal an exhaustive argmin over every Gaussian. Any disagreement is an indexing or
      einsum bug, not an approximation.
  T2  THE APPROXIMATION IS MEASURED, not assumed. At realistic k the disagreement rate
      against brute force is reported on deliberately anisotropic data. A silent 5% would
      corrupt every Gaussian row of the ablation.
  T3  IT IS NOT NEAREST-CENTRE IN DISGUISE. On anisotropic Gaussians the two must differ
      substantially -- otherwise the extra machinery buys nothing and we should say so. And
      on ISOTROPIC equal-scale Gaussians they must coincide exactly, since Sigma^-1 and
      log|Sigma| are then the same constant for every primitive.

Run:  D:\\conda\\envs\\powerfoam\\python.exe test_mahalanobis_assign.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from ablation_assign import _quat_to_R, assign_mahalanobis


def brute_force(points, means, scales, quats):
    R = _quat_to_R(quats)
    logdet = 2.0 * torch.log(scales.clamp_min(1e-12)).sum(-1)
    diff = points[:, None, :] - means[None, :, :]              # (P,N,3)
    local = torch.einsum("nij,pni->pnj", R, diff)              # (P,N,3)
    m2 = ((local / scales[None].clamp_min(1e-12)) ** 2).sum(-1)
    return (m2 + logdet[None]).argmin(1)


def rand_quats(n, g):
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    ok = True

    print("T1 exactness with k=N (restriction vacuous)")
    for N, P, aniso in ((60, 400, 8.0), (120, 300, 25.0)):
        means = torch.randn(N, 3, generator=g) * 2
        scales = torch.rand(N, 3, generator=g) * aniso + 0.05
        quats = rand_quats(N, g)
        pts = torch.randn(P, 3, generator=g) * 3
        a = assign_mahalanobis(pts, means.to(dev), scales.to(dev), quats.to(dev),
                               k=N, device=dev).cpu()
        b = brute_force(pts, means, scales, quats)
        same = torch.equal(a, b)
        ok &= same
        print(f"  N={N:<4} P={P:<4} aniso={aniso:<5} "
              f"{'identical' if same else f'DIFFER on {(a != b).sum().item()}'}")

    print("\nT2 disagreement at realistic k (anisotropic, worst case for kNN)")
    N, P = 4000, 3000
    means = torch.randn(N, 3, generator=g) * 5
    scales = (torch.rand(N, 3, generator=g) ** 2) * 12 + 0.02      # heavy-tailed anisotropy
    quats = rand_quats(N, g)
    pts = torch.randn(P, 3, generator=g) * 5
    ref = brute_force(pts, means, scales, quats)
    for k in (8, 16, 32, 64):
        a = assign_mahalanobis(pts, means.to(dev), scales.to(dev), quats.to(dev),
                               k=k, device=dev).cpu()
        dis = (a != ref).float().mean().item() * 100
        flag = "" if dis < 1.0 else "   <-- high"
        print(f"  k={k:<3} disagreement vs brute force {dis:6.2f}%{flag}")

    print("\nT3 relationship to nearest-centre")
    nearest = torch.cdist(pts, means).argmin(1)
    aniso_diff = (ref != nearest).float().mean().item() * 100
    print(f"  anisotropic: mahalanobis vs nearest-centre differ on {aniso_diff:.1f}% of points")
    if aniso_diff < 5:
        print("  [WARN] barely differs -- the extra machinery would not be buying anything")
        ok = False
    iso_s = torch.full((N, 3), 0.7)
    iso_q = torch.zeros(N, 4); iso_q[:, 0] = 1.0
    a_iso = assign_mahalanobis(pts, means.to(dev), iso_s.to(dev), iso_q.to(dev),
                               k=N, device=dev).cpu()
    same_iso = torch.equal(a_iso, nearest)
    ok &= same_iso
    print(f"  isotropic equal-scale == nearest-centre: {same_iso}")

    print("\nVERDICT:", "SAFE" if ok else "PROBLEM -- do not use for the ablation")


if __name__ == "__main__":
    main()
