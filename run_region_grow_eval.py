"""Adjacency-graph region-growing cluster-then-classify (user-proposed idea, tested as the
next single idea after position-aware k-means).

Clustering: instead of any k-means, grow regions over PowerFoam's REAL power-diagram
adjacency graph (export_adjacency_graph.py's CSR structure -- facet-sharing neighbors from
bvh.py::build_cech_complex):
  1. Seed selection: the unassigned primitive with the highest mean cosine similarity to
     its unassigned graph neighbors (most locally-coherent point -- a wall interior beats
     a boundary primitive).
  2. BFS flood-fill from the seed: a frontier neighbor joins the region iff
     cos(feature, running region mean) >= threshold. The mean is updated after every BFS
     level (online spherical mean), so the region tracks its own average semantics rather
     than the (possibly noisy) seed alone.
  3. Repeat until k=320 regions (OpenGaussian's codebook size, same as the k-means runs
     for a controlled comparison) or no unassigned primitives remain.
  4. Leftover primitives (unreached or rejected everywhere): iterative label propagation
     -- each takes the label of its most-feature-similar already-labeled graph neighbor
     (5 rounds), then any still-orphaned fall back to the globally nearest region centroid.

Classification: identical to run_cluster_classify_eval.py (mean-pool unit features per
region, hubness-corrected argmax vs class-set text embeddings, broadcast), so any metric
delta is attributable to the clustering alone.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS,
    calculate_metrics, remap_gt_labels, embed_class_names,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import pool_classify_broadcast, SCENES, CLASS_SETS

K_REGIONS = 320


def region_grow(adjacent, offsets, unit_feats, valid_mask_t, threshold, k=K_REGIONS):
    """CSR graph region growing. Returns (labels, num_regions); labels[i] = -1 for
    primitives never reached (handled by caller's propagation), region id otherwise.
    All tensors on the same device. unit_feats: (P, C) unit-normalized (invalid rows
    are zeros and never join any region since valid_mask_t excludes them)."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    assignable = valid_mask_t.clone()

    # Per-node local coherence (mean cosine to graph neighbors), computed once.
    # Edge similarities are computed in chunks: materializing (E, 512) gathers whole
    # (scene0140_00: ~21M edges -> ~40GB per gather) OOMs; 2M-edge chunks stay <5GB.
    src = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    coherence = torch.zeros(P, device=device)
    E = adjacent.numel()
    for s in range(0, E, 2_000_000):
        e = min(s + 2_000_000, E)
        chunk_sim = (unit_feats[src[s:e]] * unit_feats[adjacent[s:e]]).sum(-1)
        coherence.index_add_(0, src[s:e], chunk_sim)
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    coherence /= deg

    # Per-node MAX neighbor similarity (chunked like coherence): a seed whose best
    # neighbor is already below threshold can never grow -- fast-path it to a singleton
    # region without a BFS. In k=None (exhaustive) mode this kills the dominant cost:
    # the low-coherence junk tail that would otherwise be tens of thousands of one-node
    # BFS setups (scene0645_00: ~700k such primitives).
    max_nsim = torch.full((P,), float("-inf"), device=device)
    for s in range(0, E, 2_000_000):
        e = min(s + 2_000_000, E)
        chunk_sim = (unit_feats[src[s:e]] * unit_feats[adjacent[s:e]]).sum(-1)
        max_nsim.scatter_reduce_(0, src[s:e], chunk_sim, reduce="amax")

    # Seeds in descending local-coherence order, consumed via a pointer (CPU-side list
    # to avoid one GPU sync per skipped/checked entry).
    seed_order = torch.argsort(coherence, descending=True).cpu().tolist()
    can_grow_cpu = (max_nsim >= threshold).cpu()
    assignable_cpu = valid_mask_t.cpu().clone()
    seed_ptr = 0

    region_id = 0
    while k is None or region_id < k:
        while seed_ptr < P and not assignable_cpu[seed_order[seed_ptr]]:
            seed_ptr += 1
        if seed_ptr >= P:
            return labels, region_id
        seed_i = seed_order[seed_ptr]
        if not can_grow_cpu[seed_i]:
            labels[seed_i] = region_id
            assignable[seed_i] = False
            assignable_cpu[seed_i] = False
            region_id += 1
            continue
        seed = torch.tensor(seed_i, device=device)
        labels[seed] = region_id
        assignable[seed] = False
        assignable_cpu[seed_i] = False
        region_sum = unit_feats[seed].clone()
        frontier = seed.unsqueeze(0)
        while frontier.numel() > 0:
            starts, ends = offsets[frontier], offsets[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            # flat CSR gather: for each frontier node, adjacent[start:end], fully vectorized
            flat = torch.repeat_interleave(starts, counts) + (
                torch.arange(total, device=device)
                - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts))
            neigh = torch.unique(adjacent[flat])
            neigh = neigh[assignable[neigh]]
            if neigh.numel() == 0:
                break
            mean = F.normalize(region_sum, dim=0)
            sim = unit_feats[neigh] @ mean
            accepted = neigh[sim >= threshold]
            if accepted.numel() == 0:
                break
            labels[accepted] = region_id
            assignable[accepted] = False
            assignable_cpu[accepted.cpu()] = False
            region_sum += unit_feats[accepted].sum(0)
            frontier = accepted
        region_id += 1
    return labels, region_id


def region_grow_exhaustive(adjacent, offsets, unit_feats, valid_mask_t, threshold):
    """Exhaustive variant: every valid primitive ends up in a region (no k cap, no
    propagation). Vectorized as threshold connected-components -- keep edges whose
    endpoint features agree (cos >= threshold), then iterated min-label propagation
    until fixpoint. Semantics differ slightly from the k-capped BFS (pairwise edge
    threshold instead of cos-to-running-region-mean, so long low-drift chains can
    merge -- transitivity leak), but it needs no per-region Python loop, which matters
    when large scenes produce 10^5 regions. Returns (labels compacted to 0..R-1, R)."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    src = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    E = adjacent.numel()
    keep = torch.zeros(E, dtype=torch.bool, device=device)
    for s in range(0, E, 2_000_000):
        e = min(s + 2_000_000, E)
        sim = (unit_feats[src[s:e]] * unit_feats[adjacent[s:e]]).sum(-1)
        keep[s:e] = sim >= threshold
    keep &= valid_mask_t[src] & valid_mask_t[adjacent]
    ksrc, kdst = src[keep], adjacent[keep]

    labels = torch.arange(P, device=device)
    labels[~valid_mask_t] = -1
    while True:
        pulled = labels.clone()
        pulled.scatter_reduce_(0, kdst, labels[ksrc], reduce="amin")
        pulled.scatter_reduce_(0, ksrc, labels[kdst], reduce="amin")
        # pointer-jumping (label -> label's current label) accelerates convergence
        upd = torch.minimum(pulled, torch.where(pulled >= 0, pulled.clamp_min(0), pulled))
        valid_rows = upd >= 0
        upd[valid_rows] = torch.minimum(upd[valid_rows], labels[upd[valid_rows]])
        if (upd == labels).all():
            break
        labels = upd
    # compact region ids to 0..R-1
    valid_rows = labels >= 0
    uniq, compact = torch.unique(labels[valid_rows], return_inverse=True)
    out = torch.full((P,), -1, dtype=torch.long, device=device)
    out[valid_rows] = compact
    return out, int(uniq.numel())


def batched_region_grow(adjacent, offsets, unit_feats, valid_mask_t, threshold,
                        normals=None, normal_tau=0.0, positions=None, coplanar_eps=0.0,
                        feature_gate=True, batch_size=4096):
    """Batched mean-anchored region growing: up to `batch_size` regions grow
    SIMULTANEOUSLY, one vectorized BFS level per Python iteration, so wall-clock is
    O(#batches x depth) kernel launches instead of O(#regions x depth) -- the sequential
    grower spent minutes per large scene in per-region launch overhead at <5% GPU
    utilization.

    Semantics match the sequential mean-anchored grower except for conflict handling:
    when several concurrently-growing regions claim the same primitive in the same BFS
    level, the highest-similarity claimant wins (sequential growing resolved this by
    coherence-rank order instead). Validated to land within noise of the sequential
    pilot numbers before adoption.

    Pluggable acceptance criterion (a candidate must pass ALL enabled gates vs the
    claiming region's running state):
      - feature_gate:   cos(feature, region feature mean) >= threshold
      - normal_tau > 0: |dot(normal, region normal mean)| >= normal_tau      (idea B)
      - coplanar_eps>0: |dot(region normal mean, x - region centroid)| <= coplanar_eps
                        (idea C's plane-growing stage; use with feature_gate=False)
    After all growable seeds are exhausted, every remaining valid primitive becomes its
    own singleton region in one vectorized shot (the junk tail never enters a batch).
    Returns (labels, num_regions)."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    assignable = valid_mask_t.clone()

    src = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    E = adjacent.numel()
    coherence = torch.zeros(P, device=device)
    max_nsim = torch.full((P,), float("-inf"), device=device)
    for s in range(0, E, 2_000_000):
        e = min(s + 2_000_000, E)
        chunk_sim = (unit_feats[src[s:e]] * unit_feats[adjacent[s:e]]).sum(-1)
        coherence.index_add_(0, src[s:e], chunk_sim)
        max_nsim.scatter_reduce_(0, src[s:e], chunk_sim, reduce="amax")
    coherence /= (offsets[1:] - offsets[:-1]).clamp_min(1).float()

    # Growable = could ever accept a neighbor. With feature gating that needs a
    # neighbor above threshold; geometry-only growing (idea C) has no such shortcut.
    growable = (max_nsim >= threshold) if feature_gate else valid_mask_t.clone()
    seed_order = torch.argsort(coherence, descending=True)

    next_region = 0
    while True:
        cand_seeds = seed_order[assignable[seed_order] & growable[seed_order]]
        if cand_seeds.numel() == 0:
            break
        seeds = cand_seeds[:batch_size]
        B = seeds.numel()
        rid = torch.arange(B, device=device)
        labels[seeds] = next_region + rid
        assignable[seeds] = False
        feat_sum = unit_feats[seeds].clone()          # (B, C)
        norm_sum = normals[seeds].clone() if normals is not None else None
        pos_sum = positions[seeds].clone() if positions is not None else None
        count = torch.ones(B, device=device)
        frontier_nodes, frontier_region = seeds, rid
        while frontier_nodes.numel() > 0:
            starts = offsets[frontier_nodes]
            counts_n = offsets[frontier_nodes + 1] - starts
            total = int(counts_n.sum())
            if total == 0:
                break
            flat = torch.repeat_interleave(starts, counts_n) + (
                torch.arange(total, device=device)
                - torch.repeat_interleave(torch.cumsum(counts_n, 0) - counts_n, counts_n))
            cand = adjacent[flat]
            cand_region = torch.repeat_interleave(frontier_region, counts_n)
            ok = assignable[cand]
            cand, cand_region = cand[ok], cand_region[ok]
            if cand.numel() == 0:
                break

            keep = torch.ones(cand.numel(), dtype=torch.bool, device=device)
            region_fmean = F.normalize(feat_sum, dim=-1)
            sim = (unit_feats[cand] * region_fmean[cand_region]).sum(-1)
            if feature_gate:
                keep &= sim >= threshold
            if normal_tau > 0.0 and normals is not None:
                region_nmean = F.normalize(norm_sum, dim=-1)
                keep &= (normals[cand] * region_nmean[cand_region]).sum(-1).abs() >= normal_tau
            if coplanar_eps > 0.0 and positions is not None and normals is not None:
                region_nmean = F.normalize(norm_sum, dim=-1)
                centroid = pos_sum / count[:, None]
                offset_d = ((positions[cand] - centroid[cand_region])
                            * region_nmean[cand_region]).sum(-1).abs()
                keep &= offset_d <= coplanar_eps
            cand, cand_region, sim = cand[keep], cand_region[keep], sim[keep]
            if cand.numel() == 0:
                break

            # conflict resolution: per-node max-similarity claim wins
            best = torch.full((P,), float("-inf"), device=device)
            best.scatter_reduce_(0, cand, sim, reduce="amax")
            win = sim >= best[cand] - 1e-7
            cand, cand_region = cand[win], cand_region[win]
            # exact ties: last write wins, then re-read the final assignment
            labels[cand] = next_region + cand_region
            assignable[cand] = False
            final_nodes = torch.unique(cand)
            final_region = labels[final_nodes] - next_region
            feat_sum.index_add_(0, final_region, unit_feats[final_nodes])
            if norm_sum is not None:
                # sign-align normals to the region mean before accumulating (n and -n
                # are the same plane); otherwise opposite-signed wall halves cancel out
                region_nmean = F.normalize(norm_sum, dim=-1)
                n = normals[final_nodes]
                sign = torch.sign((n * region_nmean[final_region]).sum(-1, keepdim=True))
                sign[sign == 0] = 1.0
                norm_sum.index_add_(0, final_region, n * sign)
            if pos_sum is not None:
                pos_sum.index_add_(0, final_region, positions[final_nodes])
            count.index_add_(0, final_region, torch.ones_like(final_region, dtype=count.dtype))
            frontier_nodes, frontier_region = final_nodes, final_region
        next_region += B

    # leftover junk tail -> singleton regions, one shot
    rest = torch.where(assignable)[0]
    labels[rest] = next_region + torch.arange(rest.numel(), device=device)
    next_region += rest.numel()
    return labels, next_region


def ransac_planes(positions, valid_mask_t, adjacent, offsets, inlier_eps=0.05,
                  min_plane_size=500, max_planes=32, hypotheses=512, seed=0):
    """Multi-plane RANSAC on primitive POSITIONS (trained quaternion normals measured
    unusable -- diagnose_normals.py: floor normals ~random at 62deg median spread, while
    floor center positions are planar to 2.5cm p50 -- so planes are detected from
    positions alone). Iteratively: vectorized RANSAC (all hypotheses scored in parallel)
    -> best plane's inliers within inlier_eps -> SVD refine -> split inliers into
    adjacency-connected components (coplanar-but-disconnected surfaces stay separate)
    -> keep components >= min_plane_size as plane regions -> remove and repeat.
    Returns (labels with plane-region ids or -1, num_plane_regions)."""
    device = positions.device
    P = positions.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    remaining = valid_mask_t.clone()
    next_id = 0
    for _ in range(max_planes):
        idx = torch.where(remaining)[0]
        if idx.numel() < min_plane_size:
            break
        pick = idx[torch.randint(0, idx.numel(), (hypotheses, 3), generator=g, device=device)]
        p0, p1, p2 = positions[pick[:, 0]], positions[pick[:, 1]], positions[pick[:, 2]]
        n_raw = torch.cross(p1 - p0, p2 - p0, dim=-1)
        degenerate = n_raw.norm(dim=-1) < 1e-8
        n = F.normalize(n_raw, dim=-1)
        d = ((positions[idx][None, :, :] - p0[:, None, :]) * n[:, None, :]).sum(-1).abs()
        inl = (d <= inlier_eps).sum(1)
        inl[degenerate] = 0
        best = int(inl.argmax())
        if int(inl[best]) < min_plane_size:
            break
        # refine on inliers via SVD, then re-collect inliers once
        for _refine in range(2):
            mask_in = ((positions[idx] - p0[best]) * n[best]).sum(-1).abs() <= inlier_eps
            pts = positions[idx[mask_in]]
            centroid = pts.mean(0)
            _, _, V = torch.linalg.svd(pts - centroid, full_matrices=False)
            nb = V[2]
            p0 = p0.clone(); n = n.clone()
            p0[best] = centroid; n[best] = nb
        inlier_idx = idx[mask_in]
        # adjacency-connected components within the inlier set (threshold-CC machinery,
        # but connectivity-only: both endpoints inliers)
        in_set = torch.zeros(P, dtype=torch.bool, device=device)
        in_set[inlier_idx] = True
        src = torch.repeat_interleave(
            torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
        emask = in_set[src] & in_set[adjacent]
        ksrc, kdst = src[emask], adjacent[emask]
        comp = torch.full((P,), 2 ** 62, dtype=torch.long, device=device)
        comp[inlier_idx] = inlier_idx
        while True:
            pulled = comp.clone()
            pulled.scatter_reduce_(0, kdst, comp[ksrc], reduce="amin")
            pulled.scatter_reduce_(0, ksrc, comp[kdst], reduce="amin")
            sel = pulled < 2 ** 62
            pulled[sel] = torch.minimum(pulled[sel], comp[pulled[sel]])
            if (pulled == comp).all():
                break
            comp = pulled
        roots, counts = torch.unique(comp[inlier_idx], return_counts=True)
        big_roots = roots[counts >= min_plane_size]
        claimed_any = False
        for r in big_roots.tolist():
            members = inlier_idx[comp[inlier_idx] == r]
            labels[members] = next_id
            remaining[members] = False
            next_id += 1
            claimed_any = True
        # small disconnected slivers of this plane stay 'remaining' for stage 2, but the
        # plane hypothesis itself is consumed either way to guarantee progress
        if not claimed_any:
            # all components below min size: burn these primitives from plane DETECTION
            # (guarantees progress; labels stay -1 so they remain available to stage 2)
            remaining[inlier_idx] = False
    return labels, next_id


def propagate_leftovers(labels, adjacent, offsets, unit_feats, valid_mask_t, num_regions, rounds=5):
    """Attach labels[i]==-1 primitives: most-similar labeled graph neighbor (iterated),
    then global nearest region centroid for anything still orphaned."""
    device = unit_feats.device
    P = unit_feats.shape[0]
    src_all = torch.repeat_interleave(
        torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
    for _ in range(rounds):
        todo_mask = (labels == -1) & valid_mask_t
        if not todo_mask.any():
            break
        # every edge (i -> j) with i unlabeled+valid and j labeled; per-i argmax over sim
        emask = todo_mask[src_all] & (labels[adjacent] >= 0)
        src, dst = src_all[emask], adjacent[emask]
        if src.numel() == 0:
            break
        sim = torch.empty(src.numel(), device=device)
        for s in range(0, src.numel(), 2_000_000):
            e = min(s + 2_000_000, src.numel())
            sim[s:e] = (unit_feats[src[s:e]] * unit_feats[dst[s:e]]).sum(-1)
        best = torch.full((P,), float("-inf"), device=device)
        best.scatter_reduce_(0, src, sim, reduce="amax")
        winners = sim >= best[src] - 1e-7
        new_labels = labels.clone()
        new_labels[src[winners]] = labels[dst[winners]]
        if (new_labels == labels).all():
            break
        labels = new_labels
    todo = torch.where((labels == -1) & valid_mask_t)[0]
    if todo.numel() > 0:
        centroids = torch.zeros(num_regions, unit_feats.shape[1], device=device)
        labeled = labels >= 0
        centroids.index_add_(0, labels[labeled], unit_feats[labeled])
        centroids = F.normalize(centroids, dim=-1)
        labels[todo] = (unit_feats[todo] @ centroids.T).argmax(dim=1)
    return labels


def eval_scene(scene, split, threshold, device, class_sets, text_cache, mode="capped",
               normal_tau=0.8, plane_tau=0.9, coplanar_eps=0.05, min_plane_size=500,
               restrict_owners=False, point_weight=False, consolidate_min=200):
    ckpt_dir = f"output/scannet_{scene}_nonfrozen"
    features_path = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen.pt"
    adjacency_path = f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt"
    gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
    need_geometry = mode in ("batched_normal", "plane_first", "plane_ransac")
    if need_geometry:
        centers, radii, normals_np = load_foam(ckpt_dir, device, return_normals=True)
        normals = F.normalize(torch.from_numpy(normals_np).to(device).float(), dim=-1)
        positions = torch.from_numpy(centers).to(device).float()
    else:
        centers, radii = load_foam(ckpt_dir, device)
        normals, positions = None, None
    solved = torch.load(features_path, map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()
    adj = torch.load(adjacency_path, map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

    unit_all = torch.zeros_like(feats)
    vm_t = torch.from_numpy(valid_mask).to(device)
    unit_all[vm_t] = F.normalize(feats[vm_t], dim=-1)

    # Assignment BEFORE clustering: per-cell GT-point ownership counts drive the
    # owner-restriction/point-weighting options. ~90% of valid cells own zero GT points
    # (measured: 87.4% scene0000_00, 90.8% scene0645_00) -- the metric never reads their
    # predictions, and OpenGaussian's frozen 1:1 protocol has no such cells at all.
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
    owned_counts_np = np.bincount(assigned[assigned >= 0], minlength=centers.shape[0])
    owned_counts = torch.from_numpy(owned_counts_np).to(device).float()
    if restrict_owners:
        vm_t = vm_t & (owned_counts > 0)
        n_owners = int(vm_t.sum())
        print(f"  [{scene}] owner restriction: clustering over {n_owners} point-owning cells", flush=True)
    pool_weights = owned_counts.clamp_min(1.0) if point_weight else None

    if mode == "batched_consolidated":
        # pool-size test: exhaustive growth, then merge every region smaller than
        # consolidate_min into its feature-nearest large region (region-centroid level,
        # NOT per-primitive flooding) -- recovers big pools + full coverage
        labels, num_regions = batched_region_grow(adjacent, offsets, unit_all, vm_t, threshold)
        sizes = torch.bincount(labels[labels >= 0], minlength=num_regions)
        cents = torch.zeros(num_regions, unit_all.shape[1], device=device)
        li = labels >= 0
        cents.index_add_(0, labels[li], unit_all[li])
        cents = F.normalize(cents, dim=-1)
        if consolidate_min < 0:
            # top-K mode (-K): keep the K largest regions, merge everything else into
            # them -- pool granularity adapts to scene size exactly like k-means' k=320
            topk = min(-consolidate_min, int((sizes > 0).sum()))
            cut = sizes[torch.argsort(sizes, descending=True)[topk - 1]].clamp_min(2)
            big = torch.where(sizes >= cut)[0]
            small = torch.where((sizes > 0) & (sizes < cut))[0]
        else:
            big = torch.where(sizes >= consolidate_min)[0]
            small = torch.where((sizes > 0) & (sizes < consolidate_min))[0]
        if big.numel() > 0 and small.numel() > 0:
            tgt = big[(cents[small] @ cents[big].T).argmax(dim=1)]
            remap = torch.arange(num_regions, device=device)
            remap[small] = tgt
            labels[li] = remap[labels[li]]
        kept = torch.unique(labels[li])
        compact = torch.full((num_regions,), -1, dtype=torch.long, device=device)
        compact[kept] = torch.arange(kept.numel(), device=device)
        labels[li] = compact[labels[li]]
        n_before = num_regions
        num_regions = int(kept.numel())
        print(f"  [{scene} thr={threshold} batched_consolidated min={consolidate_min}] "
              f"regions {n_before} -> {num_regions}, valid={int(vm_t.sum())}", flush=True)
    elif mode == "kmeans_pos":
        from run_cluster_classify_eval import two_level_position_aware, K_FLAT
        vi = torch.where(vm_t)[0]
        pos_v = torch.from_numpy(centers[vi.cpu().numpy()]).to(device).float()
        leaf = two_level_position_aware(pos_v, unit_all[vi], seed=0)
        labels = torch.full((unit_all.shape[0],), -1, dtype=torch.long, device=device)
        labels[vi] = leaf
        num_regions = K_FLAT
        print(f"  [{scene} kmeans_pos] 64x5 over {int(vm_t.sum())} cells", flush=True)
    elif mode == "batched":
        labels, num_regions = batched_region_grow(adjacent, offsets, unit_all, vm_t, threshold)
        nonsingleton = int((torch.bincount(labels[labels >= 0], minlength=num_regions) > 1).sum())
        print(f"  [{scene} thr={threshold} batched] regions={num_regions} "
              f"({nonsingleton} non-singleton), valid={int(vm_t.sum())}", flush=True)
    elif mode == "batched_normal":
        labels, num_regions = batched_region_grow(
            adjacent, offsets, unit_all, vm_t, threshold, normals=normals, normal_tau=normal_tau)
        nonsingleton = int((torch.bincount(labels[labels >= 0], minlength=num_regions) > 1).sum())
        print(f"  [{scene} thr={threshold} batched_normal tau={normal_tau}] regions={num_regions} "
              f"({nonsingleton} non-singleton), valid={int(vm_t.sum())}", flush=True)
    elif mode == "plane_first":
        # Stage 1 (geometry only): grow planar regions -- aligned normals + coplanarity,
        # NO feature gate, so contaminated wall/floor CLIP features can't fragment them.
        plane_labels, n_plane_regions = batched_region_grow(
            adjacent, offsets, unit_all, vm_t, threshold, normals=normals,
            normal_tau=plane_tau, positions=positions, coplanar_eps=coplanar_eps,
            feature_gate=False)
        sizes = torch.bincount(plane_labels[plane_labels >= 0], minlength=n_plane_regions)
        big = sizes >= min_plane_size
        old_to_new = torch.full((n_plane_regions,), -1, dtype=torch.long, device=device)
        old_to_new[big] = torch.arange(int(big.sum()), device=device)
        labels = torch.full_like(plane_labels, -1)
        in_plane = (plane_labels >= 0) & (old_to_new[plane_labels.clamp_min(0)] >= 0)
        labels[in_plane] = old_to_new[plane_labels[in_plane]]
        n_planes = int(big.sum())
        # Stage 2 (semantics): feature-gated growing over the non-plane remainder.
        vm_stage2 = vm_t & ~in_plane
        labels2, n2 = batched_region_grow(adjacent, offsets, unit_all, vm_stage2, threshold)
        has2 = labels2 >= 0
        labels[has2] = n_planes + labels2[has2]
        num_regions = n_planes + n2
        plane_frac = float(in_plane.sum()) / max(float(vm_t.sum()), 1)
        print(f"  [{scene} thr={threshold} plane_first ptau={plane_tau} eps={coplanar_eps} "
              f"min={min_plane_size}] planes={n_planes} (cover {plane_frac*100:.1f}%), "
              f"feature regions={n2}, valid={int(vm_t.sum())}", flush=True)
    elif mode == "plane_ransac":
        # corrected idea C: planes from POSITIONS via multi-plane RANSAC (trained
        # normals measured unusable, see diagnose_normals.py), then feature growing
        # on the remainder
        plane_labels, n_planes = ransac_planes(
            positions, vm_t, adjacent, offsets,
            inlier_eps=coplanar_eps, min_plane_size=min_plane_size)
        in_plane = plane_labels >= 0
        labels = plane_labels.clone()
        vm_stage2 = vm_t & ~in_plane
        labels2, n2 = batched_region_grow(adjacent, offsets, unit_all, vm_stage2, threshold)
        has2 = labels2 >= 0
        labels[has2] = n_planes + labels2[has2]
        num_regions = n_planes + n2
        plane_frac = float(in_plane.sum()) / max(float(vm_t.sum()), 1)
        print(f"  [{scene} thr={threshold} plane_ransac eps={coplanar_eps} min={min_plane_size}] "
              f"planes={n_planes} (cover {plane_frac*100:.1f}%), feature regions={n2}, "
              f"valid={int(vm_t.sum())}", flush=True)
    elif mode == "exhaustive":
        labels, num_regions = region_grow_exhaustive(adjacent, offsets, unit_all, vm_t, threshold)
        print(f"  [{scene} thr={threshold} exhaustive] regions={num_regions}, "
              f"valid={int(vm_t.sum())}", flush=True)
    elif mode == "exhaustive_mean":
        labels, num_regions = region_grow(adjacent, offsets, unit_all, vm_t, threshold, k=None)
        nonsingleton = int((torch.bincount(labels[labels >= 0], minlength=num_regions) > 1).sum())
        print(f"  [{scene} thr={threshold} exhaustive_mean] regions={num_regions} "
              f"({nonsingleton} non-singleton), valid={int(vm_t.sum())}", flush=True)
    else:
        labels, num_regions = region_grow(adjacent, offsets, unit_all, vm_t, threshold)
        grown = int((labels >= 0).sum())
        labels = propagate_leftovers(labels, adjacent, offsets, unit_all, vm_t, num_regions)
        print(f"  [{scene} thr={threshold}] regions={num_regions}, grown={grown}, "
              f"propagated={int((labels >= 0).sum()) - grown}, valid={int(vm_t.sum())}", flush=True)

    owned = assigned >= 0

    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    out = {}
    valid_idx = torch.where(vm_t)[0]
    for cs in class_sets:
        kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
        target_ids = [i for i, _ in kept]
        target_names = [n for _, n in kept]
        gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
        key = (cs, tuple(target_names))
        if key not in text_cache:
            text_cache[key] = embed_class_names(target_names, device)
        text_feats = text_cache[key]

        prim_cls_valid = pool_classify_broadcast(
            labels[valid_idx], unit_all[valid_idx], num_regions, text_feats,
            weights=pool_weights[valid_idx] if pool_weights is not None else None).cpu().numpy()
        prim_class = np.zeros(centers.shape[0], dtype=np.int64)
        prim_class[valid_idx.cpu().numpy()] = prim_cls_valid
        pred = np.zeros(gt_points.shape[0], dtype=np.int64)
        pred[owned] = prim_class[assigned[owned]] + 1
        _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), len(target_ids) + 1)
        out[cs] = {"mIoU": miou, "mAcc": macc, "overall_acc": acc}
        print(f"  {scene} {cs} thr={threshold}: mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="all", help="'all' or comma-separated scene names")
    p.add_argument("--thresholds", default="0.85", help="comma-separated cosine thresholds")
    p.add_argument("--class-sets", default="all", help="'all' or comma-separated class set names")
    p.add_argument("--consolidate-min", type=int, default=200)
    p.add_argument("--restrict-owners", action="store_true")
    p.add_argument("--point-weight", action="store_true")
    p.add_argument("--normal-tau", type=float, default=0.8)
    p.add_argument("--plane-tau", type=float, default=0.9)
    p.add_argument("--coplanar-eps", type=float, default=0.05)
    p.add_argument("--min-plane-size", type=int, default=500)
    p.add_argument("--mode", choices=["capped", "exhaustive", "exhaustive_mean", "batched", "batched_normal", "plane_first", "plane_ransac", "kmeans_pos", "batched_consolidated"], default="capped")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scenes = SCENES if args.scenes == "all" else {s: SCENES[s] for s in args.scenes.split(",")}
    thresholds = [float(t) for t in args.thresholds.split(",")]
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
    text_cache = {}

    summary = {}
    for thr in thresholds:
        results = {cs: {} for cs in class_sets}
        for scene, split in scenes.items():
            per_cs = eval_scene(scene, split, thr, device, class_sets, text_cache, mode=args.mode,
                                normal_tau=args.normal_tau, plane_tau=args.plane_tau,
                                coplanar_eps=args.coplanar_eps, min_plane_size=args.min_plane_size,
                                restrict_owners=args.restrict_owners, point_weight=args.point_weight,
                                consolidate_min=args.consolidate_min)
            for cs, m in per_cs.items():
                results[cs][scene] = m
        summary[str(thr)] = {}
        line = [f"thr={thr}"]
        for cs in class_sets:
            mious = [m["mIoU"] for m in results[cs].values()]
            maccs = [m["mAcc"] for m in results[cs].values()]
            summary[str(thr)][cs] = {
                "num_scenes": len(mious),
                "mean_mIoU": float(np.mean(mious)),
                "mean_mAcc": float(np.mean(maccs)),
                "per_scene": results[cs],
            }
            line.append(f"{cs} {np.mean(mious)*100:.2f}/{np.mean(maccs)*100:.2f}")
        print("\n== " + "  ".join(line) + " ==\n", flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
