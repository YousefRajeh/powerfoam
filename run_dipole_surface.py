"""The "Ours + dipole" row: semantic surface metrics against an EXPLICIT EXTRACTED SURFACE.

WHAT THE OTHER ROWS ACTUALLY MEASURE, and why this is different. `semantic_surface_metrics`
compares two subsets OF THE SAME GT POINT CLOUD: the predicted region for class c is
`gt_points[pred == c]`. That is all any published baseline can offer, because a per-point label
vector is the whole of its output -- there is no geometry to ask. So those rows measure *region
agreement on ScanNet's cloud*, and our reconstruction's own geometry never enters the number.

This row replaces the predicted side with the surface WE reconstruct:

    PRED_c = dipole surface samples whose primitive's argmax class is c

so `mae_gt2pred(c)` becomes "how far is a true class-c point from the surface we actually built
for class c", and `mae_pred2gt(c)` becomes "how far does the surface we built stray from the true
class-c region". Those are questions about a reconstruction, not about a labelling. The class
assignment is unchanged plain argmax -- identical to the `Ours` rows -- so the ONLY difference is
that the predicted geometry is ours instead of ScanNet's points. mIoU/mAcc are therefore identical
by construction and are not re-reported here.

THE SURFACE (extract_dipole_surface.py, paper Sec 3.3/3.4; scene.py:394-410, rasterize.py:68-135).
Each cell carries an oriented face (centre p, normal n) bisecting its power cell plus k=8 detail
sites with displacements, and the surface is that plane displaced by a soft-Voronoi blend:

    site3d_i = p + r*(s_i0*t + s_i1*b);  w_i(x) = exp(-10*||x-site3d_i||^2/r^2)
    disp(x)  = sum_i w_i d_i r / sum_i w_i

Samples are kept only if they lie in the cell's OWN power cell. That test is EXACT rather than
approximate: the power cell is convex, so its nearest competitor is always a facet neighbour, and
the neighbours are enumerated from the stored adjacency.

WHICH CELLS EMIT SURFACE. Only cells that are (a) above the density floor and (b) carry a lifted
feature. (b) is not an optimisation -- a cell with no feature has no class, so it could not join
any PRED_c anyway. It also happens to be the documented accuracy fix: most cells are interior, and
interior cells emit surface behind walls.

A GAUSSIAN HAS NO ANALOGUE. A splat has no normal, no bisecting face and no facets; 3DGS surface
extraction requires TSDF fusion or marching cubes over a rendered depth field, an image-space
detour. So this row is not something the baselines are being denied -- it is a property of the
representation, which is the claim the table exists to support.

SAMPLING DENSITY IS A KNOB AND IS REPORTED. `--grid` controls samples per cell face (grid^2).
Denser sampling can only help the GT->pred direction, so the grid is stated with the numbers and
swept with --grid-sweep rather than quietly chosen.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import TAU, GTSurfaceIndex
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from run_percell_masked import OPACITY_THRESH, SPLIT, primitive_alpha

SCENES = list(SPLIT)
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
DENSITY_THRESH = 1e-2       # verbatim from extract_dipole_surface.py

ARMS = {
    "pf_tfroz":   ("output/scannet_{s}_truefrozen/model.pt",
                   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt"),
    "pf_nonfroz": ("output/scannet_{s}_nonfrozen/model.pt",
                   "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt"),
}


def quat_frame(q):
    """Normal, tangent, bitangent from the primitive quaternion (verbatim from
    extract_dipole_surface.py so the surface is byte-for-byte the one measured there)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], -1)
    t = torch.stack([2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)], -1)
    b = torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)], -1)
    f = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    return f(n), f(t), f(b)


def extract_surface(m, live, grid, dev):
    """Returns (points (S,3), owner primitive index (S,), total area m^2)."""
    P = m["points"].float().to(dev)
    radii = F.softplus(m["radii"].float().to(dev), beta=100)
    N, T, B = quat_frame(m["quaternions"].float().to(dev))
    sites2d = m["texel_sites"].float().to(dev)
    heights = m["texel_height"].float().to(dev)
    adjacent = m["adjacency"].long().to(dev)
    offsets = m["adjacency_offsets"].long().to(dev)

    lin = (torch.arange(grid, device=dev).float() + 0.5) / grid * 2 - 1
    uu, vv = torch.meshgrid(lin, lin, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], -1)
    nS = uv.shape[0]

    pts_all, own_all, area_tot = [], [], 0.0
    idx_live = torch.nonzero(live).squeeze(1)
    CH = max(1, int(4e7 // (nS * 8)))
    for s in range(0, len(idx_live), CH):
        ix = idx_live[s:s + CH]
        p, n, t, b, r = P[ix], N[ix], T[ix], B[ix], radii[ix]
        M = len(ix)
        loc = uv[None, :, :] * r[:, None, None]
        base = p[:, None, :] + loc[..., 0:1]*t[:, None, :] + loc[..., 1:2]*b[:, None, :]

        off3 = sites2d[ix] * r[:, None, None]
        site3 = p[:, None, :] + off3[..., 0:1]*t[:, None, :] + off3[..., 1:2]*b[:, None, :]
        d2 = ((base[:, :, None, :] - site3[:, None, :, :]) ** 2).sum(-1)
        w = torch.exp(-10.0 * d2 / (r[:, None, None] ** 2).clamp_min(1e-20))
        hw = heights[ix] * r[:, None]
        disp = (w * hw[:, None, :]).sum(-1) / w.sum(-1).clamp_min(1e-20)
        surf = base + disp[..., None] * n[:, None, :]

        # EXACT power-cell membership: nearest competitor is always a facet neighbour
        own = (surf - p[:, None, :]).pow(2).sum(-1) - (r ** 2)[:, None]
        keep = torch.ones(M, nS, dtype=torch.bool, device=dev)
        deg = offsets[ix + 1] - offsets[ix]
        for k in range(int(deg.max().item()) if M else 0):
            sel = k < deg
            if not sel.any():
                break
            nb = adjacent[offsets[ix[sel]] + k]
            comp = (surf[sel] - P[nb][:, None, :]).pow(2).sum(-1) - (radii[nb] ** 2)[:, None]
            keep[sel] &= own[sel] <= comp

        gd = disp.view(M, grid, grid)
        du, dv = torch.zeros_like(gd), torch.zeros_like(gd)
        if grid > 1:
            step = (2.0 * r / grid).clamp_min(1e-20)[:, None, None]
            du[:, 1:, :] = (gd[:, 1:, :] - gd[:, :-1, :]) / step
            dv[:, :, 1:] = (gd[:, :, 1:] - gd[:, :, :-1]) / step
        stretch = torch.sqrt(1.0 + du**2 + dv**2).view(M, nS)
        area_tot += float((((2*r)**2 / nS) * (keep.float()*stretch).sum(1)).sum())

        pts_all.append(surf[keep])
        own_all.append(ix[:, None].expand(M, nS)[keep])
    return (torch.cat(pts_all).cpu().numpy().astype(np.float64),
            torch.cat(own_all).cpu().numpy(), area_tot)


def surface_metrics_vs_extracted(index, surf_pts, surf_cls, tau=TAU):
    """Same definitions as ablation_surface.semantic_surface_metrics, but PRED_c is the
    extracted surface rather than a subset of the GT cloud."""
    per_class, workers = {}, -1
    for c in index.classes():
        pm = surf_cls == c
        n_pred = int(pm.sum())
        if n_pred == 0:
            per_class[c] = {"n_gt": index.n_gt[c], "n_pred": 0, "missed": True}
            continue
        ppts, gpts = surf_pts[pm], index.gt_pts[c]
        d_p2g, _ = index.gt_trees[c].query(ppts, k=1, workers=workers)
        d_g2p, _ = cKDTree(ppts).query(gpts, k=1, workers=workers)
        prec, rec = float((d_p2g <= tau).mean()), float((d_g2p <= tau).mean())
        per_class[c] = {
            "n_gt": index.n_gt[c], "n_pred": n_pred, "missed": False,
            "mae_pred2gt": float(d_p2g.mean()), "mae_gt2pred": float(d_g2p.mean()),
            "scd": float((d_p2g.mean() + d_g2p.mean()) / 2),
            "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
            "boundary_f1": float(2*prec*rec / max(prec + rec, 1e-9)),
        }
    live = [m for m in per_class.values() if not m["missed"]]
    n_missed = sum(1 for m in per_class.values() if m["missed"])
    if not live:
        return {"n_missed": n_missed, "n_scored": 0}
    agg = {k: float(np.mean([m[k] for m in live]))
           for k in ("mae_pred2gt", "mae_gt2pred", "scd", "hd95", "boundary_f1")}
    agg.update({"n_missed": n_missed, "n_scored": len(live)})
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(ARMS))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--grid", type=int, default=6, help="grid^2 surface samples per cell face")
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--out", default="artifacts/scannet/dipole_surface.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, skipped = [], []

    for recon in a.recons.split(","):
        mtmpl, ftmpl = ARMS[recon]
        for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
            mp, fp = mtmpl.format(s=scene), ftmpl.format(s=scene)
            apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            if not all(os.path.exists(p) for p in (mp, fp, apth)):
                skipped.append(f"{recon}/{scene}: missing artifact")
                continue
            t0 = time.time()
            m = torch.load(mp, map_location="cpu", weights_only=False)
            dens = F.softplus(m["density"].float().to(dev), beta=100)
            d = torch.load(fp, map_location=dev, weights_only=True)
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].to(dev)
            if feats.shape[0] != dens.shape[0]:
                skipped.append(f"{recon}/{scene}: feats vs model mismatch")
                continue
            unit = torch.zeros_like(feats)
            unit[valid] = F.normalize(feats[valid], dim=-1)

            # a cell emits surface only if it is solid AND carries a class
            live = (dens > DENSITY_THRESH) & valid
            surf_pts, surf_own, area = extract_surface(m, live, a.grid, dev)
            del m
            print(f"[{recon}/{scene}] {len(surf_pts):,} surface samples from "
                  f"{int(live.sum()):,} cells, area={area:.1f} m^2", flush=True)

            alpha = primitive_alpha(recon, scene, a.mult)
            assign = np.load(apth)
            owned = assign >= 0
            gt_pts, raw, names_all = load_scannet_pointcept_gt(
                rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
            pts = np.asarray(gt_pts, dtype=np.float64)
            n2i = {n: q for q, n in enumerate(names_all)}
            present = set(np.unique(raw).tolist())
            low = np.zeros(len(pts), dtype=bool)
            low[owned] = alpha[assign[owned]] < OPACITY_THRESH

            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                gt[low] = 0                      # same masked GT as every other row
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1     # plain argmax, unchanged
                sm = surface_metrics_vs_extracted(GTSurfaceIndex(pts, gt, nc),
                                                  surf_pts, cls[surf_own])
                rec = {"recon": recon, "scene": scene, "class_set": cs, "grid": a.grid,
                       "n_surface_pts": int(len(surf_pts)), "area_m2": area, **sm}
                rows.append(rec)
                print(f"   {cs[11:]:>3} SCD={sm.get('scd', float('nan')):.4f} "
                      f"HD95={sm.get('hd95', float('nan')):.4f} "
                      f"BF1={sm.get('boundary_f1', float('nan'))*100:.2f} "
                      f"missed={sm.get('n_missed')}", flush=True)
            print(f"[done] {recon}/{scene} {(time.time()-t0)/60:.1f} min", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")
    for s in skipped:
        print("  SKIPPED", s)


if __name__ == "__main__":
    main()
