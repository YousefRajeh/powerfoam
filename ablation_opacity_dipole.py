"""Dipole-aware crossing length for PowerFoam's opacity proxy, and what it changes.

THE DEFECT. `ablation_opacity.primitive_alpha` uses, for powerfoam,

    alpha = 1 - exp(-sigma * 2 * radii)

i.e. it takes the crossing length to be the BOUNDING-SPHERE DIAMETER, treating the whole cell
as occupied. The renderer does not: `rasterize.py::export_operator_kernel` clips the segment
at the displaced dipole surface (`t_far = min(t_surf,t_far)` if dp>=0 else
`t_near = max(t_surf,t_near)`), so the occupied half is the side the normal points AWAY from.

Measured on scene0347_00 (20,000 cells x 64 Monte-Carlo samples, exact power-cell membership):
    power cell as a fraction of its bounding sphere : 44.6%
    occupied (dipole-inside) fraction of the cell   : 54.7%
So `2r` overestimates the crossing length by ~1.8x, biasing the OpenGaussian opacity mask
PERMISSIVE -- it keeps cells the renderer treats as nearly transparent. That matters because
the mask deletes GT points from the metric, and under the coverage law (mIoU ~= 0.53 x
classifiable fraction) the kept fraction drives the score.

THE CORRECTION. Compute the actual occupied chord by exact half-space clipping. A power cell
is the intersection of half-spaces, one per facet neighbour:
    x in cell i  <=>  2 x.(c_j - c_i) <= (|c_j|^2 - r_j^2) - (|c_i|^2 - r_i^2)  for all j
which is LINEAR in x, so along a ray x = p + t*d each neighbour gives one bound on t. The
dipole adds one more: (x - p).n < 0  =>  t (d.n) < 0. Averaging the clipped segment length
over random directions through the site gives a per-cell crossing length that respects both
the cell's real shape and the dipole.

CONVENTION, stated plainly: this averages over random directions through the SITE, matching
the existing proxy's notion of "crossing the cell through its middle". It is not the
uniform-random-line mean chord (4V/S). The comparison against `2r` is like-for-like because
both are centre-crossing conventions; only the shape and dipole terms change.

This is a DIAGNOSTIC. It reports what the corrected mask would do; it does not silently
change the ablation default.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_opacity import mask_low_opacity
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}


def quat_normal(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def occupied_chord(P, radii, Nrm, adjacent, offsets, dev, n_dir=32, chunk=4096, seed=0):
    """Mean length of the ray segment that is inside the power cell AND on the occupied
    (dipole-inside) side, averaged over random directions through the site."""
    n_prim = P.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    D = torch.randn(n_dir, 3, generator=g, device=dev)
    D = D / D.norm(dim=-1, keepdim=True)
    out = torch.empty(n_prim, device=dev)
    w2 = (P * P).sum(-1) - radii ** 2                       # |c|^2 - r^2
    for s in range(0, n_prim, chunk):
        e = min(s + chunk, n_prim)
        ix = torch.arange(s, e, device=dev)
        p, n = P[ix], Nrm[ix]
        M = e - s
        # start with a generous interval, then clip
        BIG = (radii[ix] * 8).clamp_min(1e-6)
        tlo = -BIG[:, None].expand(M, n_dir).clone()
        thi = BIG[:, None].expand(M, n_dir).clone()
        deg = offsets[ix + 1] - offsets[ix]
        for k in range(int(deg.max().item())):
            sel = k < deg
            if not sel.any():
                break
            rows = torch.nonzero(sel).squeeze(1)
            nb = adjacent[offsets[ix[rows]] + k]
            aj = 2.0 * (P[nb] - p[rows])                     # (m,3)
            bj = w2[nb] - w2[ix[rows]]                       # (m,)
            ad = aj @ D.T                                    # (m,n_dir)
            ao = (aj * p[rows]).sum(-1)                      # (m,)  ray origin is the site
            rhs = (bj - ao)[:, None]
            pos, neg = ad > 1e-12, ad < -1e-12
            tb = torch.where(ad.abs() > 1e-12, rhs / torch.where(ad.abs() > 1e-12, ad,
                                                                 torch.ones_like(ad)),
                             torch.zeros_like(ad))
            thi[rows] = torch.where(pos, torch.minimum(thi[rows], tb), thi[rows])
            tlo[rows] = torch.where(neg, torch.maximum(tlo[rows], tb), tlo[rows])
        # dipole: (p + t d - p).n < 0  =>  t (d.n) < 0
        dn = n @ D.T                                         # (M,n_dir)
        thi = torch.where(dn > 1e-12, torch.minimum(thi, torch.zeros_like(thi)), thi)
        tlo = torch.where(dn < -1e-12, torch.maximum(tlo, torch.zeros_like(tlo)), tlo)
        out[s:e] = (thi - tlo).clamp_min(0).mean(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00,scene0070_00,scene0062_00")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--class-set", default="opengaussian19")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"

    print(f"{'scene':<15}{'chord':>22}{'kept%':>16}{'mIoU raw':>10}"
          f"{'masked(2r)':>12}{'masked(dip)':>13}")
    for scene in a.scenes.split(","):
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy"
        if not all(os.path.exists(q) for q in (mp, fp, apth)):
            print(f"{scene:<15} missing artifact")
            continue
        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        radii = F.softplus(m["radii"].float().squeeze().to(dev), beta=100)
        sigma = F.softplus(m["density"].float().squeeze().to(dev), beta=100)
        Nrm = quat_normal(m["quaternions"].float().to(dev))
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)

        dt_old = 2.0 * radii
        dt_new = occupied_chord(P, radii, Nrm, adjacent, offsets, dev)
        a_old = (1 - torch.exp(-sigma * dt_old)).cpu().numpy()
        a_new = (1 - torch.exp(-sigma * dt_new)).cpu().numpy()

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
        valid = d["valid_mask"].cpu().numpy()
        assign = np.load(apth)
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        gt = remap_gt_labels(raw, [n2i[n] for n in names])
        nc = len(names) + 1
        text = embed_class_names(names, dev)
        cls = (feats @ text.T).argmax(-1).cpu().numpy() + 1

        owned = assign >= 0
        sc = owned.copy()
        sc[owned] = valid[assign[owned]]
        pred = np.zeros(len(gt), dtype=np.int64)
        pred[sc] = cls[assign[sc]]

        def sc_of(g_):
            _, mi, _, _ = calculate_metrics(torch.from_numpy(g_).long(),
                                            torch.from_numpy(pred).long(), nc)
            return float(mi) * 100

        raw_mi = sc_of(gt)
        g_old, _, kf_old = mask_low_opacity(gt, assign, a_old, a.threshold)
        g_new, _, kf_new = mask_low_opacity(gt, assign, a_new, a.threshold)
        print(f"{scene:<15}{f'{dt_old.mean()*100:.2f}->{dt_new.mean()*100:.2f} cm':>22}"
              f"{f'{kf_old*100:.1f}->{kf_new*100:.1f}':>16}"
              f"{raw_mi:>10.2f}{sc_of(g_old):>12.2f}{sc_of(g_new):>13.2f}")


if __name__ == "__main__":
    main()
