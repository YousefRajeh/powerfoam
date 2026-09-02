"""Extract an explicit SURFACE from PowerFoam's dipoles, and measure it against ScanNet GT.

WHY. Surface extraction is the one dipole property that survived testing. Measured: the
displaced dipole surface passes within a median 1.2 cm of ScanNet GT points (0.525 cell
radii; 79% within one radius, 98% within two). Every semantic use of the dipole was
falsified -- void-side assignment, distance-as-reliability, normals-as-boundary-detector,
the joint feature predicate, and (against our real best) the macro-geometry coverage gain,
which posterior diffusion already subsumes at +0.00.

Our previous attempt at surface extraction did NOT use dipoles and failed badly: density
thresholding produced 634 m^2 of "surface" for a 23.5 m^2 room. That is the number to beat.

WHAT A GAUSSIAN CANNOT DO. A splat has no normal, no bisecting face, and no facets. 3DGS
surface extraction needs TSDF fusion or marching cubes over a *rendered depth field* -- an
image-space detour. Here the surface is a primitive-space object read directly off the
parameters, with orientation included for free.

THE GEOMETRY (paper Sec 3.3/3.4, verified in scene.py:394-410 and rasterize.py:68-135):
each cell carries an oriented face (centre p_i, normal n_i) bisecting its power cell, plus
k=8 detail sites s_i in the face plane, each with displacement d_i. The surface is the plane
displaced along n_i by a soft-Voronoi blend of the d_i:

    site3d_i = p + r*(s_i0 * tangent + s_i1 * bitangent)
    w_i(x)   = exp(-tau * ||x - site3d_i||^2 / r^2),   tau = 10
    disp(x)  = sum_i w_i d_i r / sum_i w_i

METHOD. Sample the face plane on a jittered grid over [-r, r]^2, displace each sample along
n, and KEEP only samples that actually lie in the cell's own power cell. The power cell is
convex and its nearest competitor is always a facet neighbour, so testing the power distance
against the cell's adjacency neighbours is EXACT, not an approximation.

AREA, which we have never computed before. Facet areas used in the P2/TPFA diffusion weights
are power-diagram facet polygons -- the boundary BETWEEN two cells -- a different surface
entirely. This is the dipole surface itself. Area is estimated as
    accepted_fraction * (2r)^2 * stretch,   stretch = sqrt(1 + ||grad disp||^2)
with the stretch term from the height field's finite differences, since a displaced surface
is larger than its base plane.

METRICS
  * total surface area (m^2)  -- vs the density-threshold failure of 634 m^2 / 23.5 m^2 room
  * Chamfer, both directions, against the ScanNet GT point cloud
  * fraction of GT points within 1/2/5 cm of the extracted surface
Writes an oriented-point PLY (x y z nx ny nz) for Poisson reconstruction or visualisation.
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
from evaluate_point_cloud_miou import load_scannet_pointcept_gt

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}


def quat_frame(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    t = torch.stack([2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)], -1)
    b = torch.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)], -1)
    f = lambda v: v / v.norm(dim=-1, keepdim=True)
    return f(n), f(t), f(b)


def chamfer_halves(A, B, chunk=4096):
    """min distance from every row of A to the set B (both on GPU)."""
    out = torch.empty(len(A), device=A.device)
    for s in range(0, len(A), chunk):
        e = min(s + chunk, len(A))
        out[s:e] = torch.cdist(A[s:e], B).min(1).values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--model", default=None)
    ap.add_argument("--grid", type=int, default=6, help="grid**2 samples per cell face")
    ap.add_argument("--density-thresh", type=float, default=1e-2,
                    help="skip cells whose inside-half density is below this (void cells)")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--max-gt-chamfer", type=int, default=60000)
    ap.add_argument("--min-support-pct", type=float, default=None,
                    help="keep only cells above this PERCENTILE of accumulated rendering "
                         "support (A^T 1, i.e. sum of alpha*T over all rays). This is the "
                         "principled surface filter: a wall is modelled by MANY cells "
                         "stacked along the view direction, and only those where rays "
                         "actually terminate are on the visible surface. Density and "
                         "visibility do not separate them (90%% of cells pass both).")
    ap.add_argument("--require-visible", action="store_true",
                    help="emit surface ONLY for cells the lifting actually observed. Most "
                         "cells are interior (only 7.5%% carry a GT label; 95.5%% of "
                         "unobserved cells lie INSIDE the volume), and interior cells emit "
                         "surface behind walls -- the accuracy failure mode.")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mp = a.model or f"output/scannet_{a.scene}_nonfrozen/model.pt"
    m = torch.load(mp, map_location="cpu", weights_only=False)
    P = m["points"].float().to(dev)
    radii = F.softplus(m["radii"].float().to(dev), beta=100)
    dens = F.softplus(m["density"].float().to(dev), beta=100)
    N, T, B = quat_frame(m["quaternions"].float().to(dev))
    sites2d = m["texel_sites"].float().to(dev)
    heights = m["texel_height"].float().to(dev)
    adjacent = m["adjacency"].long().to(dev)
    offsets = m["adjacency_offsets"].long().to(dev)
    n_prim = P.shape[0]

    live = dens > a.density_thresh
    if a.min_support_pct is not None:
        sys.path.insert(0, r"D:\Downloadseature-foam-lifting\src")
        from feature_foam_lifting.operator import AccumulatedFeatureStats
        sup = AccumulatedFeatureStats.load(
            f"artifacts/scannet/{a.scene}/stats_nonfrozen_ogl3.pt").support.to(dev)
        thr = torch.quantile(sup.float(), a.min_support_pct / 100.0)
        live = live & (sup > thr)
        print(f"  [support] p{a.min_support_pct:g} threshold={thr:.4g}, "
              f"{int((sup > thr).sum()):,} cells above it")
    if a.require_visible:
        vp = f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        vm = torch.load(vp, map_location="cpu", weights_only=True)["valid_mask"].to(dev)
        live = live & vm
        print(f"  [visible-only] {int(vm.sum()):,} cells were observed by some view")
    print(f"[model] {mp}\n  {n_prim:,} cells, {int(live.sum()):,} above density "
          f"{a.density_thresh} ({100*live.float().mean():.1f}%)")

    g = a.grid
    lin = (torch.arange(g, device=dev).float() + 0.5) / g * 2 - 1          # (-1, 1)
    uu, vv = torch.meshgrid(lin, lin, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], -1)                 # (g^2, 2)
    nS = uv.shape[0]

    pts_all, nrm_all, area_all = [], [], []
    idx_live = torch.nonzero(live).squeeze(1)
    CH = max(1, int(4e7 // (nS * 8)))
    for s in range(0, len(idx_live), CH):
        ix = idx_live[s:s + CH]
        p, n, t, b, r = P[ix], N[ix], T[ix], B[ix], radii[ix]
        M = len(ix)
        loc = uv[None, :, :] * r[:, None, None]                            # (M,g^2,2)
        base = p[:, None, :] + loc[..., 0:1] * t[:, None, :] + loc[..., 1:2] * b[:, None, :]

        off3 = sites2d[ix] * r[:, None, None]
        site3 = p[:, None, :] + off3[..., 0:1] * t[:, None, :] + off3[..., 1:2] * b[:, None, :]
        d2 = ((base[:, :, None, :] - site3[:, None, :, :]) ** 2).sum(-1)   # (M,g^2,8)
        w = torch.exp(-10.0 * d2 / (r[:, None, None] ** 2).clamp_min(1e-20))
        hw = heights[ix] * r[:, None]
        disp = (w * hw[:, None, :]).sum(-1) / w.sum(-1).clamp_min(1e-20)   # (M,g^2)
        surf = base + disp[..., None] * n[:, None, :]

        # EXACT power-cell membership: the nearest competitor is always a facet neighbour.
        own = (surf - p[:, None, :]).pow(2).sum(-1) - (r ** 2)[:, None]
        keep = torch.ones(M, nS, dtype=torch.bool, device=dev)
        deg = (offsets[ix + 1] - offsets[ix])
        for k in range(int(deg.max().item()) if M else 0):
            sel = k < deg
            if not sel.any():
                break
            nb = adjacent[offsets[ix[sel]] + k]
            comp = ((surf[sel] - P[nb][:, None, :]).pow(2).sum(-1)
                    - (radii[nb] ** 2)[:, None])
            keep[sel] &= own[sel] <= comp

        # area: accepted fraction of the (2r)^2 patch, scaled by the height-field stretch
        gd = disp.view(M, g, g)
        du = torch.zeros_like(gd); dv = torch.zeros_like(gd)
        if g > 1:
            step = (2.0 * r / g).clamp_min(1e-20)[:, None, None]
            du[:, 1:, :] = (gd[:, 1:, :] - gd[:, :-1, :]) / step
            dv[:, :, 1:] = (gd[:, :, 1:] - gd[:, :, :-1]) / step
        stretch = torch.sqrt(1.0 + du ** 2 + dv ** 2).view(M, nS)
        cell_area = ((2 * r) ** 2 / nS) * (keep.float() * stretch).sum(1)

        pts_all.append(surf[keep])
        nrm_all.append(n[:, None, :].expand(M, nS, 3)[keep])
        area_all.append(cell_area)

    pts = torch.cat(pts_all)
    nrm = torch.cat(nrm_all)
    area = torch.cat(area_all).sum().item()
    print(f"  extracted {len(pts):,} oriented surface samples "
          f"({100*len(pts)/(len(idx_live)*nS):.1f}% of samples inside their own cell)")
    print(f"\n=== SURFACE AREA ===\n  dipole surface area = {area:.2f} m^2")
    print(f"  (density-threshold extraction previously gave 634 m^2 for a 23.5 m^2 room)")

    gt_pts, raw, _ = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    G = torch.from_numpy(np.asarray(gt_pts, dtype=np.float32)).to(dev)
    ext = np.ptp(np.asarray(gt_pts), axis=0)
    print(f"  GT bbox {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m "
          f"(floor footprint ~{ext[0]*ext[1]:.1f} m^2)")

    gsub = G if len(G) <= a.max_gt_chamfer else G[torch.randperm(
        len(G), generator=torch.Generator(device=dev).manual_seed(0), device=dev)[:a.max_gt_chamfer]]
    psub = pts if len(pts) <= a.max_gt_chamfer else pts[torch.randperm(
        len(pts), generator=torch.Generator(device=dev).manual_seed(0), device=dev)[:a.max_gt_chamfer]]
    d_gt = chamfer_halves(gsub, psub)      # GT -> surface (completeness)
    d_sf = chamfer_halves(psub, gsub)      # surface -> GT (accuracy)
    print(f"\n=== CHAMFER (on {len(gsub):,} GT / {len(psub):,} surface samples) ===")
    print(f"  completeness  GT->surface : mean {d_gt.mean()*100:6.2f} cm   "
          f"median {d_gt.median()*100:6.2f} cm")
    print(f"  accuracy      surface->GT : mean {d_sf.mean()*100:6.2f} cm   "
          f"median {d_sf.median()*100:6.2f} cm")
    print(f"  symmetric Chamfer          : {((d_gt.mean()+d_sf.mean())/2)*100:.2f} cm")
    for thr in (0.01, 0.02, 0.05):
        print(f"  GT within {thr*100:>2.0f} cm of surface: {100*(d_gt < thr).float().mean():5.1f}%"
              f"   surface within {thr*100:>2.0f} cm of GT: "
              f"{100*(d_sf < thr).float().mean():5.1f}%")

    out = a.ply or f"artifacts/scannet/{a.scene}/dipole_surface.ply"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pn = torch.cat([pts, nrm], 1).cpu().numpy().astype(np.float32)
    with open(out, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(pn)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\n"
                 "end_header\n").encode())
        f.write(pn.tobytes())
    print(f"\nwrote {out} ({len(pn):,} oriented points, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
