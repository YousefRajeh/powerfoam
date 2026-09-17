"""Adjacency built FROM the surface, not filtered from the volumetric facet graph.

MOTIVATION. Filtering the volumetric graph by visibility was measured and its ceiling is low:
dropping the 77.5% of edges never observed pixel-adjacent COSTS 0.021 mIoU (interior cells are
invisible by construction, so "unobserved" is not "unsupported"), and the depth-continuity test
that does work touches only 0.4% of edges, so paired against the control it came out flat
(-0.0001, p=0.995 at n=4). The problem is that the facet graph is a partition of SPACE: most of its
vertices are interior cells that no GT point ever lands on, and most of its edges join two lumps of
air or two lumps of wall. No reweighting or filtering of that object yields a surface.

THE CONSTRUCTION. Take the foam's own isosurface definition (foam_exact_surface.py): a cell is
OCCUPIED when its density is above the iso level, and the surface is exactly the set of faces
separating an occupied cell from an unoccupied one. Then:

  * a SURFACE CELL is an occupied cell with at least one face to an unoccupied cell -- i.e. a cell
    the isosurface actually passes through;
  * two surface cells are ADJACENT when they share a face and both are occupied -- that face lies
    along the material, so walking it moves ALONG the surface rather than through it.

The result is the induced subgraph on surface cells. It is a 2D-manifold-like graph embedded in the
3D diagram, which is what a segmentation of visible geometry should be partitioning. Cells that are
not surface cells keep no edges and become singletons; they are overwhelmingly interior or air, and
the diagnostic below reports exactly what fraction of GT-owning cells they account for, so the cost
of excluding them is measured rather than assumed.

OCCUPANCY uses the same alpha the evaluation's opacity mask uses, alpha = 1 - exp(-density*radius*2),
so "occupied" here means the same thing as "not culled" there. Foam alpha is strongly bimodal (52%
of cells above 0.9, 35% below 0.1), so this threshold is not a sensitive knob -- `--report` prints
the surface-cell count so a sweep can confirm that rather than take it on faith.

OUTPUT is the same CSR layout (`offsets`, `adjacent`, `dist`) as every other adjacency here, so it
drops into cutpursuit_facet.py unchanged.
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adjacency", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alpha-threshold", type=float, default=0.1,
                    help="occupied when 1-exp(-density*radius*2) >= this (the eval's opacity mask)")
    ap.add_argument("--gt-points", default=None, help="optional, to report GT-owning-cell coverage")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    centers = ck["points"].float()
    radii = ck["radii"].float().reshape(-1)
    density = ck["density"].float().reshape(-1)
    alpha = 1.0 - torch.exp(-density * radii * 2.0)
    occ = alpha >= a.alpha_threshold

    ad = torch.load(a.adjacency, map_location="cpu", weights_only=False)
    P = int(ad["num_primitives"])
    off = ad["offsets"].to(torch.int64)
    adj = ad["adjacent"].to(torch.int64)
    E = adj.numel()
    deg = off[1:] - off[:-1]
    src = torch.repeat_interleave(torch.arange(P, dtype=torch.int64), deg)

    # a surface cell: occupied, and touching at least one unoccupied neighbour
    touches_air = torch.zeros(P, dtype=torch.bool)
    air_edge = occ[src] & (~occ[adj])
    touches_air.index_put_((src[air_edge],), torch.ones(int(air_edge.sum()), dtype=torch.bool))
    surf = occ & touches_air

    # keep faces joining two SURFACE cells (both occupied => the face lies along the material)
    keep = surf[src] & surf[adj]
    new_deg = torch.zeros(P, dtype=torch.int64)
    if int(keep.sum()):
        new_deg.index_add_(0, src[keep], torch.ones(int(keep.sum()), dtype=torch.int64))
    new_off = torch.zeros(P + 1, dtype=torch.int64)
    new_off[1:] = torch.cumsum(new_deg, 0)
    out = {"num_primitives": P,
           "offsets": new_off.to(torch.int32),
           "adjacent": adj[keep].to(torch.int32)}
    if "dist" in ad:
        out["dist"] = ad["dist"][keep]
    torch.save(out, a.output)

    stats = {"P": P, "occupied": int(occ.sum()), "surface_cells": int(surf.sum()),
             "surface_frac": float(surf.float().mean()),
             "directed_in": E, "directed_out": int(keep.sum()),
             "kept_frac": float(keep.float().mean()),
             "mean_degree_surface": float(new_deg[surf].float().mean()) if int(surf.sum()) else 0.0,
             "isolated_surface_cells": int((new_deg[surf] == 0).sum()),
             "alpha_threshold": a.alpha_threshold}

    # how much of what the metric actually scores does this graph cover?
    if a.gt_points:
        import sys
        sys.path.insert(0, "D:/Downloads/powerfoam")
        from point_cloud_query import assign_points_to_power_cells
        from evaluate_point_cloud_miou import load_scannet_pointcept_gt
        gt_pts, raw, _ = load_scannet_pointcept_gt(a.gt_points, "segment20")
        cell = np.asarray(assign_points_to_power_cells(torch.as_tensor(gt_pts).float(),
                                                       centers, radii))
        lab_ok = np.asarray(raw) > 0
        owners = np.unique(cell[lab_ok])
        on_surf = surf.numpy()[owners]
        stats["gt_owning_cells"] = int(owners.size)
        stats["gt_owning_cells_on_surface"] = int(on_surf.sum())
        stats["gt_owning_surface_frac"] = float(on_surf.mean())
        pts_on = surf.numpy()[cell[lab_ok]]
        stats["gt_points_in_surface_cells_frac"] = float(pts_on.mean())

    print("[surface graph]")
    print("  occupied cells        %d/%d (%.1f%%)" % (stats["occupied"], P, 100 * stats["occupied"] / P))
    print("  SURFACE cells         %d/%d (%.1f%%)" % (stats["surface_cells"], P, 100 * stats["surface_frac"]))
    print("  edges kept            %d/%d (%.1f%%)" % (stats["directed_out"], E, 100 * stats["kept_frac"]))
    print("  mean degree (surface) %.2f" % stats["mean_degree_surface"])
    print("  isolated surface cells %d" % stats["isolated_surface_cells"])
    if a.gt_points:
        print("  GT-owning cells on the surface: %d/%d (%.1f%%)"
              % (stats["gt_owning_cells_on_surface"], stats["gt_owning_cells"],
                 100 * stats["gt_owning_surface_frac"]))
        print("  GT POINTS whose cell is a surface cell: %.1f%%"
              % (100 * stats["gt_points_in_surface_cells_frac"]))
    if a.report:
        json.dump(stats, open(a.report, "w"), indent=2)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
