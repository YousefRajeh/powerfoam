"""QUERY-ADAPTIVE GRANULARITY: cache pooled embeddings at every level of a spatial hierarchy and
let the query choose its own scale.

THE IDEA (user's). Open-world queries live at different semantic scales -- "the room" vs "the desk"
vs "the coffee mug on the desk". A single per-cell embedding commits to one scale (the finest), so a
query whose natural referent is a whole object must be assembled from hundreds of independent cell
decisions. Instead: build a space-partitioning hierarchy over the cells, pool an embedding at EVERY
node, and evaluate the query from the root down, stopping descent when similarity stops improving.
That dynamically cuts the tree and yields variable-sized clusters chosen by the query itself.

WHY IT IS PLAUSIBLE HERE, from our own measurements:
  * scale demonstrably matters -- `run_attribution_diag.py` found a 43-point accuracy spread across
    mask-area deciles, peaking at ~0.18 of the image and collapsing at both extremes;
  * 83% of errors are INTERIOR cells of coherent regions, i.e. whole regions taking one wrong label,
    which is a scale failure, not a boundary failure;
  * feature-similarity region growing failed for a diagnosed reason (adjacent cells agree at median
    cosine 0.996, so no threshold has a natural scale). A SPATIAL hierarchy sidesteps that entirely:
    the scale is set by geometry, not by a feature threshold.

CONSTRUCTION. Recursive spatial k-means over cell centres gives a balanced tree whose levels are
genuine spatial partitions (an octree on raw coordinates would be mostly empty, since foam cells lie
on surfaces -- measured: mean 8.2 cells/bucket, max 4925). Each node's embedding is the
support-weighted mean of its members, renormalised: support is the accumulated render weight, so a
node is dominated by the cells actually observed rather than by cell count.

ARMS.
  `H_leaf`    -- baseline, per-cell only (the current stack).
  `H_max`     -- per (cell, class), the best score over all levels containing that cell. The query
                 picks its own scale independently for each class.
  `H_bestlvl` -- per cell, choose ONE level (the one whose top-1 score is highest) and take all class
                 scores from it. Keeps the classes commensurable, unlike H_max.
  `H_descend` -- the user's traversal: start at the root, descend while the best class score does not
                 decrease, stop otherwise, and read the label off the node where traversal halted.
  `H_blend_w` -- convex blend of leaf and ancestor scores, the partial version, since partial
                 corrections have beaten full ones repeatedly in this project.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from run_overnight import RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


def build_hierarchy(pos, weights, branch=8, min_size=32, max_levels=5, seed=0):
    """Recursive spatial k-means. Returns a list of (assignment, n_nodes), coarse -> fine.

    Level 0 is `branch` nodes over the whole scene; each subsequent level subdivides every node
    into at most `branch` children. Assignment arrays are global node ids per cell.
    """
    n = pos.shape[0]
    g = torch.Generator(device=pos.device).manual_seed(seed)
    levels = []
    parent = torch.zeros(n, dtype=torch.long, device=pos.device)
    for lvl in range(max_levels):
        new = torch.full((n,), -1, dtype=torch.long, device=pos.device)
        nxt = 0
        for p in torch.unique(parent):
            m = parent == p
            k = int(m.sum())
            if k <= min_size:
                new[m] = nxt; nxt += 1
                continue
            sub = pos[m]
            b = min(branch, max(2, k // min_size))
            c = sub[torch.randperm(k, generator=g, device=pos.device)[:b]].clone()
            for _ in range(8):
                lab = torch.cdist(sub, c).argmin(1)
                for j in range(b):
                    mm = lab == j
                    if int(mm.sum()) > 0:
                        c[j] = sub[mm].mean(0)
            new[m] = nxt + lab
            nxt += b
        parent = new
        levels.append((parent.clone(), nxt))
        if nxt >= n // min_size:
            break
    return levels


def pool_nodes(feats, weights, assign, n_nodes):
    """Support-weighted mean per node, renormalised to the sphere."""
    D = feats.shape[1]
    num = torch.zeros(n_nodes, D, device=feats.device)
    den = torch.zeros(n_nodes, device=feats.device)
    num.index_add_(0, assign, feats * weights[:, None])
    den.index_add_(0, assign, weights)
    return F.normalize(num / den.clamp_min(1e-12)[:, None], dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/hierarchy.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
    for scene in a.scenes.split(","):
        art = f"artifacts/scannetpp/{scene}"
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"{art}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.exists(sp) and os.path.isdir(ck)):
            continue
        centers, radii = load_points_radii(ck)
        sv = torch.load(sp, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        st = AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
        support = st.support.to(device).float().clamp_min(1e-6)
        R = st.reliability()["reliability"].to(device).float() * vm
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R, st
        torch.cuda.empty_cache()

        cells = cen[vm]; w = support[vm]; pv = pos[vm]
        levels = build_hierarchy(pv, w)
        log(f"  {scene}: hierarchy levels " + " -> ".join(str(n) for _, n in levels)
            + f" over {cells.shape[0]:,} cells")
        node_feats = [pool_nodes(cells, w, asg, n) for asg, n in levels]

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        row = {}
        for K in sizes:
            pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
            if not pres: continue
            nm = [top[:K][i] for i in pres]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt = embed_class_names(nm, device); C = len(nm)

            leaf = cells @ txt.T
            rK = leaf.topk(min(CSLS_K, leaf.shape[0]), dim=0).values.mean(0)
            leaf = leaf - 0.5 * rK[None, :]
            # per-level scores broadcast back to cells, coarse -> fine, then the leaf itself
            # CROSS-LEVEL COMMENSURABILITY. Raw cosines are NOT comparable across levels: pooling
            # averages features toward the global mean direction, which has higher cosine to EVERY
            # text embedding, so coarse levels score systematically higher (measured on
            # f9f95681fd: L2 top1 0.1661 vs leaf 0.1196). Taking a max/argmax over raw scores
            # therefore always selects the coarsest level regardless of semantics -- which is why
            # H_max and H_bestlvl returned identical numbers and collapsed to ~15 mIoU.
            # Z-scoring each level's scores PER CELL makes the comparison one of PEAKEDNESS: how
            # far the best class stands out from its competitors at that scale, which is the
            # quantity "the query matches this granularity" actually means.
            def zc(x):
                return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-8)

            lvl_scores = []
            for (asg, n), nf in zip(levels, node_feats):
                s = nf @ txt.T
                s = s - 0.5 * s.topk(min(CSLS_K, n), dim=0).values.mean(0)[None, :]
                lvl_scores.append(zc(s)[asg])
            stack = torch.stack(lvl_scores + [zc(leaf)], 0)      # (L+1, N, C), commensurable

            def finish(scores_v):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"H_leaf": finish(leaf)}  # unnormalised leaf == the current stack
            r["H_max"] = finish(stack.max(0).values)
            best = stack.max(-1).values.argmax(0)                # (N,) best level per cell
            r["H_bestlvl"] = finish(stack.gather(0, best[None, :, None].expand(1, *stack.shape[1:]))[0])
            # descend: keep the coarsest level from which the top score never improves again
            top1 = stack.max(-1).values                          # (L+1, N)
            improves = top1[1:] > top1[:-1]
            stop = improves.float().cumprod(0).sum(0).long()     # first level where it stops rising
            r["H_descend"] = finish(stack.gather(0, stop[None, :, None].expand(1, *stack.shape[1:]))[0])
            for b in (0.25, 0.5):
                r[f"H_blend_{b:g}"] = finish((1 - b) * zc(leaf) + b * stack[:-1].mean(0))
            row[f"top{K}"] = r
            del txt, leaf, stack, lvl_scores
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " ".join(f"{k}={v:.2f}" for k, v in row.get("top100", {}).items()))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, cells, src, dst, deg, pos, node_feats, levels
        torch.cuda.empty_cache()

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([x["H_leaf"] for x in ks])
        print(f"\n=== top{K} ({len(ks)} scenes), leaf-only {b:.2f} ===")
        for d, k, w_ in sorted(((np.mean([x[k] for x in ks]) - b, k,
                                 sum(1 for x in ks if x[k] > x["H_leaf"])) for k in ks[0]),
                               reverse=True):
            print(f"  {k:<16}{b+d:7.2f}  {d:+6.2f}  wins {w_}/{len(ks)}")


if __name__ == "__main__":
    main()
