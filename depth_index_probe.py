"""Two label-free quantities, tested against GT before anything is built on them.

(1) DEPTH INDEX -- how far a cell is behind the visible surface, counted in facet hops.
    A cell is depth 0 when some view sees it as the FRONT primitive (`front_prim_idx_out`); every
    other cell takes its BFS distance to that set over the facet graph. No GT is used, and the
    counter is well-defined only because the power diagram's cells are disjoint and ray-ordered --
    a kNN graph over Gaussian centres has no such ordering, and overlapping Gaussians make
    "how many did the ray pass through first" ill-posed.
    CLAIM TO TEST: error concentrates with depth (our diagnostics say 83% of wrong points are
    interior). If true, a depth-ordered, one-directional propagation -- surface informs interior,
    never the reverse -- is well founded.

(2) CONSENSUS -- the fraction of a region's cells voting for its majority class, computed from
    pre-pooling predictions alone. This is the label-free stand-in for region purity, which is what
    separates the regions Cut Pursuit fixes (purity 0.627) from the ones it breaks (0.316). If
    consensus tracks purity, it can gate pooling so only trustworthy regions are averaged.

Nothing here changes any feature; it only measures whether the two signals carry what the proposed
mechanism needs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import warp as wp
import configargparse

sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")
sys.path.insert(0, "D:/Downloads/powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, apply_gt_opacity_mask,
                                       classify_primitives, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from point_cloud_query import assign_points_to_power_cells


def bfs_depth(P, off, adj, seed_mask):
    """Hops from the seed set over the facet graph; -1 where unreachable."""
    depth = np.full(P, -1, np.int32)
    frontier = np.nonzero(seed_mask)[0]
    depth[frontier] = 0
    d = 0
    while frontier.size:
        d += 1
        nb = np.concatenate([adj[off[v]:off[v + 1]] for v in frontier]) if frontier.size else np.array([], np.int64)
        nb = np.unique(nb)
        nb = nb[depth[nb] < 0]
        if nb.size == 0:
            break
        depth[nb] = d
        frontier = nb
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--recon", default="nonfrozen")
    ap.add_argument("--adjacency", default="adjacency_true_facet")
    ap.add_argument("--base", default="solved_geometric_median_nonfrozen_ogl3")
    ap.add_argument("--regions", default="solved_nf_cp0.03")
    ap.add_argument("--classes", default="opengaussian19")
    ap.add_argument("--alpha-eps", type=float, default=0.5)
    # A GRAZING ray clips a cell's corner and would otherwise count as a full front hit. The
    # rasterizer returns the front cell's entry and exit t, so the chord through the cell is
    # (t_surf - t_entry); requiring it to be a real fraction of the cell's own size keeps only
    # rays that actually pass THROUGH the cell rather than skim it.
    ap.add_argument("--min-chord", type=float, default=0.5,
                    help="chord must exceed this x the cell radius to count as a meaningful hit")
    ap.add_argument("--min-front-px", type=int, default=4,
                    help="meaningful front pixels needed before a cell counts as surface")
    ap.add_argument("--output", default="D:/Downloads/claude_logs/depth_index_probe.json")
    a = ap.parse_args()
    dev = "cuda"
    wp.init()

    ck = "output/scannet_%s_%s" % (a.scene, a.recon)
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", "%s/config.yaml" % ck])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args)
    m.initialize_from_dataset(dh, device=dev)
    m.load_pt("%s/model.pt" % ck)
    m.update_vis_cache()

    A = "artifacts/scannet/%s" % a.scene
    ad = torch.load("%s/%s.pt" % (A, a.adjacency), map_location="cpu", weights_only=False)
    P = int(ad["num_primitives"])
    off = ad["offsets"].to(torch.int64).numpy()
    adj = ad["adjacent"].to(torch.int64).numpy()

    # --- depth 0: ever the front primitive in any view
    c = m._vis_cache
    rgbz = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)
    front_px = torch.zeros(P, dtype=torch.float32, device=dev)
    naive_front = torch.zeros(P, dtype=torch.bool, device=dev)
    for vi in range(len(dh.cameras)):
        with torch.no_grad():
            out = m.rasterizer.visualize(dh.cameras[vi], c["points"], c["radii"], c["density"],
                                         c["normals"], c["texel_sites"], rgbz, c["texel_height"],
                                         c["adjacency"], c["adjacency_offsets"])
        alpha, fpi = out[3].reshape(-1), out[7].long().reshape(-1)
        t_surf, t_entry = out[8].reshape(-1), out[9].reshape(-1)
        ok = (alpha >= a.alpha_eps) & (fpi >= 0)
        if bool(ok.any()):
            idx = fpi[ok]
            chord = (t_surf[ok] - t_entry[ok]).abs()
            meaningful = chord >= a.min_chord * c["radii"].reshape(-1)[idx]
            if bool(meaningful.any()):
                front_px.index_add_(0, idx[meaningful],
                                    torch.ones(int(meaningful.sum()), device=dev))
            naive_front[idx] = True
    surf = (front_px >= a.min_front_px).cpu().numpy()
    naive = naive_front.cpu().numpy()
    depth = bfs_depth(P, off, adj, surf)
    print("[depth] surface cells: naive(any front pixel) %d (%.1f%%) -> meaningful(chord>=%.1fr, "
          ">=%d px) %d (%.1f%%)  -- grazing hits removed: %d"
          % (naive.sum(), 100 * naive.mean(), a.min_chord, a.min_front_px,
             surf.sum(), 100 * surf.mean(), int(naive.sum() - surf.sum())), flush=True)

    # --- GT, predictions
    cand = [q for q in glob.glob("D:/Downloads/scannet_pointcept/*/%s" % a.scene) if os.path.isdir(q)][0]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand, "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[a.classes] if n2i[n] in pres]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    centers = torch.load("%s/model.pt" % ck, map_location="cpu", weights_only=False)["points"].float()
    radii = c["radii"].detach().cpu().float().reshape(-1)
    density = c["density"].detach().cpu().float().reshape(-1)
    cell = np.asarray(assign_points_to_power_cells(torch.as_tensor(gt_pts).float(), centers, radii))
    alpha_p = 1.0 - np.exp(-density.numpy() * radii.numpy() * 2.0)
    gt_lab, _ = apply_gt_opacity_mask(gt_lab, cell, alpha_p, 0.1, a.scene)
    keep = gt_lab > 0
    y = gt_lab[keep]
    text = embed_class_names(kept, dev)
    sb = torch.load("%s/%s.pt" % (A, a.base), map_location="cpu", weights_only=True)
    pb = classify_primitives(sb["primitive_features"].float().to(dev), text).cpu().numpy()[cell][keep] + 1
    ok_b = pb == y
    dpt = depth[cell[keep]]

    print("\n(1) ERROR vs DEPTH INDEX")
    print("  %-10s %-10s %-9s %s" % ("depth", "points", "share", "accuracy"))
    rows_d = []
    for d in range(0, 6):
        sel = dpt == d
        if sel.sum() < 50:
            continue
        rows_d.append({"depth": d, "n": int(sel.sum()), "acc": float(ok_b[sel].mean())})
        print("  %-10d %-10d %-9.3f %.4f" % (d, sel.sum(), sel.mean(), ok_b[sel].mean()))
    deep = dpt >= 6
    if deep.sum() > 50:
        rows_d.append({"depth": "6+", "n": int(deep.sum()), "acc": float(ok_b[deep].mean())})
        print("  %-10s %-10d %-9.3f %.4f" % ("6+", deep.sum(), deep.mean(), ok_b[deep].mean()))
    unre = dpt < 0
    if unre.sum() > 50:
        print("  %-10s %-10d %-9.3f %.4f" % ("unreach", unre.sum(), unre.mean(), ok_b[unre].mean()))

    # --- (2) consensus vs purity
    sc = torch.load("%s/%s.pt" % (A, a.regions), map_location="cpu", weights_only=True)
    lab = sc["labels"].numpy()
    reg = lab[cell][keep]
    R = int(lab.max()) + 1
    C = len(kept)
    prim_pred = classify_primitives(sb["primitive_features"].float().to(dev), text).cpu().numpy()
    vm = sb.get("valid_mask")
    vm = vm.numpy() if vm is not None else np.ones(P, bool)
    Hc = np.zeros((R, C), np.int64)
    np.add.at(Hc, (lab[vm], prim_pred[vm]), 1)           # label-free: cells, not GT points
    tot = Hc.sum(1).clip(min=1)
    consensus = Hc.max(1) / tot                           # fraction voting for the majority
    cnt = np.bincount(reg, minlength=R).astype(float)
    cor = np.bincount(reg, weights=ok_b.astype(float), minlength=R)
    purity = np.divide(cor, np.maximum(cnt, 1))
    has_pts = cnt >= 5
    cs, pu = consensus[has_pts], purity[has_pts]
    print("\n(2) CONSENSUS (label-free) vs PURITY (needs GT)")
    print("  regions with >=5 scored points: %d" % has_pts.sum())
    if cs.size > 2:
        r = float(np.corrcoef(cs, pu)[0, 1])
        print("  corr(consensus, purity) = %+.3f" % r)
        for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
            s = (cs >= lo) & (cs < hi)
            if s.sum() > 5:
                print("    consensus [%.1f,%.2f): n=%-6d mean purity=%.3f" % (lo, hi, s.sum(), pu[s].mean()))
    json.dump({"scene": a.scene, "surface_frac": float(surf.mean()), "depth_rows": rows_d,
               "consensus_purity_corr": r if cs.size > 2 else None},
              open(a.output, "w"), indent=2)
    print("\nwrote", a.output)


if __name__ == "__main__":
    main()
