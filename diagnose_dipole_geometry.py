"""Do PowerFoam's DIPOLES carry usable signal for 3D segmentation?

WHAT A DIPOLE IS (paper Sec 3.3/3.4, verified against scene.py + rasterize.py).
Each cell's site is an ORIENTED point: face centre p_i plus normal n_i (quaternion column 0).
That internal face bisects the power cell; the "inside" half carries learnable density and
radiance, the "outside" half is FIXED to zero density. On the face sit k=8 detail sites
s_i (texel_sites, 2D in the tangent/bitangent frame, in units of the cell radius), each with
a displacement d_i (texel_height, also radius units) and a directional radiance. The actual
reconstructed surface is the dipole plane DISPLACED along n_i by a soft-Voronoi blend:

    w_i(x) = exp(-tau * ||x - site3d_i||^2 / r^2),  tau = 10   [rasterize.py:88-98]
    d(x)   = sum_i w_i d_i / sum_i w_i

NOTE the implementation uses SQUARED radius-normalised distance; the paper's Eq.3 writes an
un-squared norm. We follow the code, since the code is what produced the checkpoint.

WHY THIS MIGHT MATTER. Our GT-point -> cell assignment is argmin ||x-c||^2 - r^2 over the
WHOLE power cell. But the model declares up to half of every cell to be void. So we may be
assigning ground-truth points into space PowerFoam itself says is empty, and we have never
looked at the per-cell oriented surface that says where the geometry actually is.

WHAT THIS SCRIPT MEASURES (all on artifacts that already exist, no training):
  1. signed distance s(x) = <x - p_i, n_i> - d(x_proj) of every GT point to the displaced
     dipole surface of its assigned cell, in units of that cell's radius.
  2. the sign split -- does the model's own occupancy convention put GT points on one side?
  3. |s| -- does the dipole surface actually pass NEAR the GT points? (validates dipoles as
     a surface representation at all, which our density-threshold extraction failed to do:
     634 m^2 of "surface" for a 23.5 m^2 room)
  4. whether |s| predicts CLASSIFICATION CORRECTNESS -- the T2 reliability theory.

FALSIFIERS, stated before running:
  - If the |s| distribution is broad and centred far from 0 (median |s| > ~1 radius), the
    dipole surface does not track the GT surface and the whole family is dead.
  - If correctness is FLAT across |s| deciles, dipole geometry carries no segmentation
    signal and T2 is dead -- exactly as bubble geometry (radius/degree/centre-offset) was
    already measured to be non-predictive, and as the reliability-weighting idea turned out
    inert under the narrow CLIP cone.
  - A sign split near 50/50 would mean the dipole orientation is not consistently aligned
    with the surface, killing the "reassign points out of void half-spaces" variant (T1).

The prize if it survives: a per-cell ORIENTED SURFACE with an analytic height field is
something a Gaussian simply does not have. No splat has a normal, a bisecting face, or a
declared void half-space.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0062_00": "train", "scene0000_00": "train", "scene0645_00": "val"}


def quat_frame(q):
    """normal, tangent, bitangent = columns of R(q).  Mirrors scene.py get_normals/get_tangents."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    t = torch.stack([2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)], -1)
    b = torch.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)], -1)
    f = lambda v: v / v.norm(dim=-1, keepdim=True)
    return f(n), f(t), f(b)


def signed_distance_to_dipole(pts, idx, P, N, T, B, sites2d, heights, radii, tau=10.0,
                              chunk=200_000):
    """s(x) = <x-p,n> - d(x_proj), in units of the owning cell's radius."""
    out = torch.empty(len(pts), dtype=torch.float32, device=pts.device)
    for a in range(0, len(pts), chunk):
        sl = slice(a, min(a + chunk, len(pts)))
        i = idx[sl]
        p, n, t, b, r = P[i], N[i], T[i], B[i], radii[i]                 # (M,3)... (M,)
        v = pts[sl] - p
        h = (v * n).sum(-1)                                             # (M,) height above plane
        x_proj = pts[sl] - h[:, None] * n                               # in-plane projection

        # 3D detail sites: p + r*(s0*t + s1*b)          [scene.py:397-402]
        off = sites2d[i] * r[:, None, None]                             # (M,8,2)
        site3 = p[:, None, :] + off[..., 0:1] * t[:, None, :] + off[..., 1:2] * b[:, None, :]

        d2 = ((x_proj[:, None, :] - site3) ** 2).sum(-1)                # (M,8)
        w = torch.exp(-tau * d2 / (r[:, None] ** 2).clamp_min(1e-20))
        hw = heights[i] * r[:, None]                                    # [scene.py:410]
        disp = (w * hw).sum(-1) / w.sum(-1).clamp_min(1e-20)
        out[sl] = (h - disp) / r.clamp_min(1e-20)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="pf_nonfroz")
    ap.add_argument("--model", default=None)
    ap.add_argument("--class-set", default="opengaussian19")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mp = a.model or f"output/scannet_{a.scene}_nonfrozen/model.pt"
    m = torch.load(mp, map_location="cpu", weights_only=False)
    P = m["points"].float().to(dev)
    radii = F.softplus(m["radii"].float().to(dev), beta=100)            # get_radii()
    N, T, B = quat_frame(m["quaternions"].float().to(dev))
    sites2d = m["texel_sites"].float().to(dev)
    heights = m["texel_height"].float().to(dev)
    dens = F.softplus(m["density"].float().to(dev), beta=100)
    n_prim = P.shape[0]
    print(f"[model] {mp}\n  {n_prim:,} cells  radius med={radii.median():.5f}  "
          f"density med={dens.median():.4f}")

    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    assign = np.load(f"artifacts/ablation_cache/{a.scene}_{a.recon}_assign.npy")
    owned = assign >= 0
    X = torch.from_numpy(np.asarray(gt_pts, dtype=np.float32)).to(dev)
    idx = torch.from_numpy(assign).long().to(dev)
    print(f"[gt] {len(gt_pts):,} points, {owned.sum():,} owned by a cell "
          f"({100*owned.mean():.1f}%)")

    s = signed_distance_to_dipole(X[owned], idx[owned], P, N, T, B, sites2d, heights, radii)
    sn = s.cpu().numpy()

    print("\n=== (1)(2) signed distance to DISPLACED dipole surface, in cell radii ===")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:<3} {np.percentile(sn, q):+8.3f}")
    print(f"  mean {sn.mean():+.3f}   median|s| {np.median(np.abs(sn)):.3f}")
    print(f"  sign split: {100*(sn > 0).mean():.1f}% positive / {100*(sn < 0).mean():.1f}% negative")
    for thr in (0.25, 0.5, 1.0, 2.0):
        print(f"  |s| < {thr:<4} : {100*(np.abs(sn) < thr).mean():5.1f}% of owned GT points")

    # ---- (4) does |s| predict correctness?
    sol = f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt"
    if not os.path.exists(sol):
        print(f"\n[skip] no solved features at {sol}")
        return
    d = torch.load(sol, map_location=dev, weights_only=True)
    feats = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()
    if feats.shape[0] != n_prim:
        print(f"\n[skip] feats {feats.shape[0]} vs cells {n_prim}")
        return
    n2i = {n: i for i, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt = remap_gt_labels(raw, [n2i[n] for n in names])
    text = embed_class_names(names, dev)
    cls = (F.normalize(feats, dim=-1) @ text.T).argmax(-1).cpu().numpy() + 1

    ow = np.where(owned)[0]
    keep = (gt[ow] > 0) & valid[assign[ow]]          # scorable: labelled AND cell has a feature
    correct = (cls[assign[ow][keep]] == gt[ow][keep]).astype(np.float64)
    absr = np.abs(sn[keep])
    print(f"\n=== (4) correctness vs |signed distance|  ({a.class_set}, "
          f"{keep.sum():,} scorable points) ===")
    order = np.argsort(absr)
    dec = np.array_split(order, 10)
    print(f"  {'decile':<8}{'|s| range':<22}{'accuracy':>9}{'n':>9}")
    for k, ix in enumerate(dec):
        print(f"  {k+1:<8}{f'{absr[ix].min():.3f} - {absr[ix].max():.3f}':<22}"
              f"{correct[ix].mean()*100:8.2f}%{len(ix):9,}")
    lo, hi = correct[dec[0]].mean(), correct[dec[-1]].mean()
    print(f"  monotone gap (decile1 - decile10): {100*(lo-hi):+.2f} pts")

    print(f"\n=== sign vs correctness ===")
    sk = sn[keep]
    for lab, msk in (("inside  (s<0)", sk < 0), ("outside (s>0)", sk > 0)):
        if msk.sum():
            print(f"  {lab}: acc {correct[msk].mean()*100:6.2f}%  n={msk.sum():,} "
                  f"({100*msk.mean():.1f}%)")


if __name__ == "__main__":
    main()
