"""Can the EXISTING partition express the ground truth, without splitting anything?

THE REFRAME. A power diagram is already a polyhedral complex. A semantic region is a UNION OF CELLS
and its boundary is a SUBSET OF EXISTING FACETS -- an arbitrary polyhedron, not a simplex, and not
something that has to be built by cutting cells. So before asking "where do I split this cell?", ask
whether any splitting is needed at all:

    CEILING = the accuracy an oracle would get by labelling every cell with the majority GT label
              of the points inside it.

Ceiling ~ 100%  -> the partition ALREADY represents the ground truth. Every error in the real system
                   is a FEATURE error, not a geometry error. Sub-cell splitting, semantic
                   triangulation and finer subdivision are all pointless: they add resolution the
                   labelling does not lack. The lever is entirely on the accumulator/feature side.
Ceiling << 100% -> cells genuinely straddle semantic boundaries, the geometry is the bottleneck, and
                   splitting has headroom equal to (ceiling - current).

Also reported, because it is the "which facets are the boundary" question directly:
  * BOUNDARY EDGE FRACTION -- adjacency edges whose two cells carry different GT labels. That is the
    size of the true cut in the dual graph, i.e. how sparse the semantic boundary actually is.
  * IMPURE CELLS -- cells whose own GT points disagree. These are the only cells a split could ever
    help, and their share bounds the entire splitting programme.

Cheap: no rendering, no operator, no views. Membership + labels only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import gsplat_env_gsview  # noqa: F401

import configargparse
import numpy as np
import torch

OUT = "artifacts/scannet/partition_ceiling"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--arm", default="truefrozen", choices=("truefrozen", "nonfrozen"))
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    from configs import Params, add_group
    from data_loader import DataHandler
    import warp as wp
    from powerfoam.scene import PowerfoamScene
    from point_cloud_query import assign_points_to_power_cells
    from evaluate_point_cloud_miou import (load_scannet_pointcept_gt, remap_gt_labels,
                                           OPENGAUSSIAN_CLASS_SETS, SCANNET20_CLASS_NAMES)

    cfg = f"output/scannet_{a.scene}_{a.arm}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    wp.init()
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"output/scannet_{a.scene}_{a.arm}/model.pt")
    P = int(model.points.shape[0])
    centres = model.points.detach().float()
    radii = model.get_radii().detach().float().reshape(-1)

    names = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    C_ = len(names)
    for sub in ("train", "val", "test"):
        gt_dir = os.path.join(a.gt_root, sub, a.scene)
        if os.path.isdir(gt_dir):
            break
    gt_pts, gt_raw, _ = load_scannet_pointcept_gt(gt_dir)
    target_ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]
    gl_np = remap_gt_labels(gt_raw, target_ids) - 1
    owner_np = assign_points_to_power_cells(gt_pts.astype(np.float64),
                                            centres.cpu().numpy(), radii.cpu().numpy())
    keep = (owner_np >= 0) & (gl_np >= 0)
    own = torch.from_numpy(owner_np[keep]).long().to(dev)
    gl = torch.from_numpy(gl_np[keep]).long().to(dev)
    n_pts = int(own.numel())

    votes = torch.zeros((P, C_), device=dev)
    votes.index_put_((own, gl), torch.ones(n_pts, device=dev), accumulate=True)
    tot = votes.sum(1)
    has = tot > 0
    maj = votes.argmax(1)
    maj_cnt = votes.max(1).values

    # CEILING: label each cell by its own majority, score the points.
    correct = int(maj_cnt.sum())
    ceiling = correct / max(n_pts, 1)
    # cells whose points disagree -- the only ones a split could help
    impure = has & (maj_cnt < tot - 1e-6)
    pts_in_impure = float(tot[impure].sum())
    lost = float((tot[impure] - maj_cnt[impure]).sum())

    # BOUNDARY: which facets carry a GT label change -- the true cut in the dual graph
    model.aabb_tree.update(model.points.detach(), model.get_radii().detach())
    adjacent, offsets = model.aabb_tree.build_cech_complex()
    adjacent = adjacent.long().to(dev)
    offsets = offsets.long().to(dev)
    src = torch.repeat_interleave(torch.arange(P, device=dev), (offsets[1:] - offsets[:-1]))
    dst = adjacent
    both = has[src] & has[dst]
    diff = both & (maj[src] != maj[dst])
    res = {"scene": a.scene, "arm": a.arm, "P": P, "gt_points": n_pts,
           "cells_with_gt": int(has.sum()),
           "pts_per_cell": n_pts / max(int(has.sum()), 1),
           "CEILING_majority_label": ceiling,
           "impure_cells": int(impure.sum()),
           "impure_cell_frac_of_gt_cells": float(impure.sum()) / max(int(has.sum()), 1),
           "pts_in_impure_cells_frac": pts_in_impure / max(n_pts, 1),
           "pts_lost_to_impurity_frac": lost / max(n_pts, 1),
           "adj_edges": int(both.sum()),
           "boundary_edge_frac": float(diff.sum()) / max(int(both.sum()), 1)}
    json.dump(res, open(f"{OUT}/{a.scene}_{a.arm}.json", "w"), indent=1)
    print(f"[{a.scene}/{a.arm}] P={P:,}  GT pts {n_pts:,}  cells with GT "
          f"{res['cells_with_gt']:,}  ({res['pts_per_cell']:.2f} pts/cell)", flush=True)
    print(f"  CEILING (majority label per cell) = {ceiling*100:.3f}%", flush=True)
    print(f"  impure cells {res['impure_cells']:,} "
          f"({res['impure_cell_frac_of_gt_cells']*100:.2f}% of GT cells); "
          f"points lost to impurity {res['pts_lost_to_impurity_frac']*100:.3f}%", flush=True)
    print(f"  boundary facets {res['boundary_edge_frac']*100:.2f}% of "
          f"{res['adj_edges']:,} adjacency edges", flush=True)


if __name__ == "__main__":
    main()
