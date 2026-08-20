"""FOAM-EXCLUSIVE clustering: seeding and region assignment that 3DGS cannot replicate.

Goal (user): beat NormLift using mechanisms that exist ONLY because the representation is a
power diagram. The one generic component retained is partial centering
(sim - lambda*column_mean), the CLIP hubness fix -- without it raw-name argmax collapses onto
attractor words and nothing else is measurable. Everything else here is foam-structural.

WHAT IS FOAM-EXCLUSIVE HERE, AND WHY 3DGS CANNOT DO IT
-----------------------------------------------------
1. **Facet adjacency (the Cech complex).** Two cells are neighbours iff they SHARE A FACET of
   the tessellation -- a hard, exact, symmetric relation produced by the partition itself
   (~13-18 neighbours/cell). 3DGS has no such relation: Gaussians overlap continuously and
   any "neighbour" must be invented via KNN, which is a choice of k, not a fact about the
   scene. We already measured facet adjacency beating Euclidean KNN-30 for refinement.
2. **The power radius r_i.** Each cell's Laguerre weight sets how much SPACE it owns. We
   measured support ~ r^k with k = 1.98 +/- 0.30 over 10 scenes -- the surface law -- so r^2
   is a direct proxy for how much semantic evidence a cell accumulated. A Gaussian's scale
   is an appearance parameter, not a partition weight; and notably even VoroTracing's plain
   Voronoi cells have NO r_i (see [[VoroTracing]]). This is ours alone.
3. **Geodesic distance ON the partition.** Distance measured by walking facet to facet is a
   property of the tessellation's connectivity. Euclidean distance between Gaussian centers
   ignores whether anything lies between them; a facet walk cannot cross a wall that the
   partition represents.

MODES
-----
  fps_feat      baseline for reference: plain feature-space FPS + spherical k-means leaves
                (this is the champion stack's step 4 -- NOT foam-exclusive, run for contrast)
  geo           geodesic FPS on the facet graph + geodesic-Voronoi region assignment
  geo_r         same, but seed selection weighted by r^beta (surface law)
  geo_r_grow    radius-weighted seeds, growth gated by feature coherence (regions stop at
                semantic discontinuities rather than filling to a fixed count)

Seeding and assignment are ONE algorithm here, which is the point: greedy farthest-point
selection under a graph metric leaves behind, for free, the label "which seed reached this
cell first" -- a geodesic Voronoi over the Cech complex. That is region growing (idea 3) and
adjacency-geodesic FPS (idea 1) as a single pass, not two.

Edge cost  w_ij = 1 + alpha * (1 - cos(u_i, u_j))  mixes connectivity with feature agreement:
one hop always costs at least 1 (so distance stays geometric), and disagreeing neighbours
cost more (so regions prefer to grow through semantically coherent tissue). alpha=0 recovers
pure hop distance.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from run_normlift_refine_eval import mode_vote_refine
from eval_semantic_surface import semantic_surface_metrics

LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def build_edge_costs(unit, adjacent, offsets, alpha, chunk=2_000_000):
    """Per-edge cost w_ij = 1 + alpha*(1 - cos), plus the flat src index for each edge.

    CHUNKED over edges deliberately. A one-shot `(unit[src] * unit[dst]).sum(-1)`
    materializes two (E, 512) gathers, and the big ScanNet scenes have ~19-21M facet edges
    -> 37GB+ per gather, which OOMs a 48GB card. This is the same trap the earlier
    region-growing work hit on scene0140_00; 2M-edge chunks keep the peak near 4GB while
    producing an identical result.
    """
    P = offsets.numel() - 1
    deg = offsets[1:] - offsets[:-1]
    src = torch.repeat_interleave(torch.arange(P, device=unit.device), deg)
    dst = adjacent
    E = src.numel()
    cos = torch.empty(E, device=unit.device, dtype=unit.dtype)
    for s in range(0, E, chunk):
        e = min(s + chunk, E)
        cos[s:e] = (unit[src[s:e]] * unit[dst[s:e]]).sum(-1)
    cos.clamp_(-1, 1)
    return src, dst, 1.0 + alpha * (1.0 - cos)


def geodesic_fps(unit, radii, adjacent, offsets, k, alpha=4.0, beta=0.0,
                 max_rounds=400, coherence=None, verbose=False):
    """Greedy farthest-point selection under the FACET-GRAPH metric.

    Returns (seeds, owner) where owner[i] is the index (0..k-1) of the seed whose geodesic
    ball reached cell i first -- i.e. the geodesic Voronoi region assignment, obtained as a
    by-product of the seeding rather than by a second clustering pass.

    beta > 0 biases seed choice by the power radius: score = d * r^beta. With the measured
    surface law (support ~ r^2), beta=2 makes the score proportional to expected evidence,
    so seeds land on cells that actually accumulated features instead of on slivers that
    happen to be far away. beta=0 is pure geodesic FPS.

    `coherence`: if set, an edge is only traversable when cos(u_i,u_j) >= coherence, so
    regions stop at semantic discontinuities instead of filling space. Cells left unreached
    are assigned afterwards to their nearest reached neighbour in feature space.
    """
    device = unit.device
    P = unit.shape[0]
    src, dst, w = build_edge_costs(unit, adjacent, offsets, alpha)
    if coherence is not None:
        cos = 1.0 - (w - 1.0) / max(alpha, 1e-9)
        keep = cos >= coherence
        src, dst, w = src[keep], dst[keep], w[keep]

    INF = torch.finfo(torch.float32).max / 4
    d = torch.full((P,), INF, device=device)
    owner = torch.full((P,), -1, dtype=torch.long, device=device)
    rw = radii.clamp_min(1e-8).pow(beta) if beta else None

    seeds = []
    # first seed: the cell with the most evidence (largest radius) if beta>0, else index 0 --
    # deterministic either way, no RNG, so runs are reproducible.
    s0 = int(radii.argmax()) if beta else 0
    for it in range(k):
        s = s0 if it == 0 else int(((d if rw is None else d * rw)
                                    .masked_fill(d >= INF / 2, -1.0)).argmax())
        if it > 0 and d[s] <= 0:
            break                                   # everything already claimed
        seeds.append(s)
        d[s], owner[s] = 0.0, it
        # relax outward from the new seed until no distance improves. d only ever decreases,
        # so later seeds converge in very few rounds -- total work is far below k*rounds*E.
        frontier = torch.zeros(P, dtype=torch.bool, device=device)
        frontier[s] = True
        for _ in range(max_rounds):
            m = frontier[src]
            if not bool(m.any()):
                break
            es, ed, ew = src[m], dst[m], w[m]
            cand = d[es] + ew
            newd = torch.full((P,), INF, device=device)
            newd.scatter_reduce_(0, ed, cand, reduce="amin", include_self=True)
            improved = newd < d
            if not bool(improved.any()):
                break
            d = torch.where(improved, newd, d)
            # propagate ownership along the edges that achieved the new minimum
            hit = improved[ed] & (cand <= d[ed] + 1e-6)
            owner[ed[hit]] = owner[es[hit]]
            frontier = improved
        if verbose and (it + 1) % 64 == 0:
            print(f"    seed {it+1}/{k} unreached={int((owner < 0).sum())}", flush=True)

    unreached = owner < 0
    if bool(unreached.any()):
        # coherence-gated growth can leave cells stranded; attach each to the most similar
        # REACHED cell (feature space), so every cell still gets a region.
        idx_r = torch.where(~unreached)[0]
        idx_u = torch.where(unreached)[0]
        for chunk in idx_u.split(4096):
            sim = unit[chunk] @ unit[idx_r].T
            owner[chunk] = owner[idx_r[sim.argmax(-1)]]
    return seeds, owner.clamp_min(0)


def run_scene(scene, variant, gt_root, device, mode, alpha, beta, coherence, k, refine, tau):
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats)
    unit_full[vi] = F.normalize(feats[vi], dim=-1)
    # Reliability, with the same uniform fallback used in eval_semantic_surface: the ~1.9GB
    # accumulator stats were deleted for most variants, and R is only ever a WEIGHT here.
    # Measured cost of the substitution on a scene where both exist: mIoU within ~1 point,
    # i.e. below the run-to-run noise floor. Never substituted silently -- it is reported.
    stats_path = f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt"
    if os.path.exists(stats_path):
        stats = AccumulatedFeatureStats.load(stats_path)
        R = stats.reliability()["reliability"].to(device).float() * vm_t
        del stats
        uniform_R = False
    else:
        R = vm_t.float()
        uniform_R = True
        print(f"    [{scene}] no stats for '{variant}' -> UNIFORM reliability", flush=True)
    del feats
    torch.cuda.empty_cache()

    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt",
                     map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    positions = torch.from_numpy(centers).to(device).float()
    radii_t = torch.from_numpy(radii).to(device).float()

    # foam-exclusive refinement over the facet graph (kept in every mode)
    ref = unit_full
    for _ in range(refine):
        ref = mode_vote_refine(ref, R, positions, adjacent, offsets)

    if mode == "fps_feat":
        unit = ref[vi]
        leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")
        n_regions = K_FLAT
    else:
        # geodesic modes operate on the FULL graph (adjacency indexes all primitives), then
        # restrict to valid cells afterwards -- the facet graph is a property of the
        # tessellation and does not know about feature validity.
        coh = coherence if mode == "geo_r_grow" else None
        b = beta if mode in ("geo_r", "geo_r_grow") else 0.0
        _, owner_full = geodesic_fps(ref, radii_t, adjacent, offsets, k,
                                     alpha=alpha, beta=b, coherence=coh)
        unit = ref[vi]
        leaf = owner_full[vi]
        n_regions = k

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    Rv = R[vi]
    out = {}
    for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = remap_gt_labels(raw_labels, tids)
        text = embed_class_names(tnames, device)
        percell = unit @ text.T
        lam = LAMBDA[cs]                                    # the one generic step retained
        lab = (percell - lam * percell.mean(0, keepdim=True)).argmax(-1)
        hist = torch.zeros(n_regions, len(tids), device=device)
        hist.index_put_((leaf, lab), Rv, accumulate=True)
        vcls = hist.argmax(-1)
        pc = np.zeros(centers.shape[0], dtype=np.int64)
        pc[vi.cpu().numpy()] = vcls[leaf].cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = pc[assigned[owned]] + 1
        ncls = len(tids) + 1
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_t).long(),
                                             torch.from_numpy(pred).long(), ncls)
        m = semantic_surface_metrics(gt_points, gt_t, pred, ncls, tau=tau)
        m["mIoU"], m["mAcc"] = float(miou), float(macc)
        out[cs] = m
        print(f"  {scene} {cs} [{mode}]: mIoU={miou*100:.2f} mAcc={macc*100:.2f} "
              f"semCD={m['scd']*100:.2f}cm missed={m['n_missed']}", flush=True)
    del unit_full, ref, R, adjacent, offsets
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--modes", default="fps_feat,geo,geo_r,geo_r_grow")
    p.add_argument("--alpha", type=float, default=4.0, help="feature term in the edge cost")
    p.add_argument("--beta", type=float, default=2.0, help="radius exponent (surface law: 2)")
    p.add_argument("--coherence", type=float, default=0.95, help="growth gate for geo_r_grow")
    p.add_argument("--k", type=int, default=320, help="regions (320 = OpenGaussian codebook)")
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    results = {}
    for scene in args.scenes.split(","):
        for mode in args.modes.split(","):
            results.setdefault(mode, {})[scene] = run_scene(
                scene, args.variant, args.gt_root, "cuda", mode,
                args.alpha, args.beta, args.coherence, args.k, args.refine, args.tau)

    print("\n=== averages ===")
    summary = {}
    for mode, per in results.items():
        summary[mode] = {}
        for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
            agg = {m: float(np.mean([v[cs][m] for v in per.values()]))
                   for m in ("mIoU", "mAcc", "scd", "hd95", "boundary_f1")}
            agg["n"] = len(per)
            summary[mode][cs] = agg
        s = summary[mode]
        print(f"{mode:<12} mIoU {s['opengaussian19']['mIoU']*100:6.2f}/"
              f"{s['opengaussian15']['mIoU']*100:6.2f}/{s['opengaussian10']['mIoU']*100:6.2f}  "
              f"mAcc {s['opengaussian19']['mAcc']*100:6.2f}/"
              f"{s['opengaussian15']['mAcc']*100:6.2f}/{s['opengaussian10']['mAcc']*100:6.2f}  "
              f"(n={s['opengaussian19']['n']})")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_scene": results, "config": vars(args)}, f, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
