"""Growing CSLS: measure a class's crowding over connected REGIONS, not over individual cells.

THE DEFECT IN CSLS THIS ADDRESSES. `w_c = r_K(t_c)/2` is the mean cosine of class c to its K nearest
CELLS. But those K cells are spatially correlated -- if they are all one contiguous surface, that is
ONE observation of crowding repeated K times, not K independent ones. A class whose top-K neighbours
form a single blob is measured as strongly "hubby" as one whose top-K are scattered across the whole
scene, and they are not the same situation.

This is the same effective-sample-size problem NormLift's `N_eff = (sum W)^2 / sum W^2` solves for
VIEWS -- 400 rays from one direction is one measurement repeated -- applied here to a class's own
neighbourhood in feature space. It needs the exact adjacency, so it is foam-native: a Gaussian
mixture has no facet graph to grow on.

METHOD. Grow connected regions on the true-facet graph (union-find, edge kept when the two cells'
features agree above `tau`), then for each class take its top neighbours and reduce them to one
score PER REGION before averaging:

    cells    :  r_K(t_c) = mean of top-K cosines over CELLS           (incumbent)
    regions  :  r_G(t_c) = mean over the top-G REGIONS of that
                           region's representative cosine             (this script)

`rep` chooses the representative: `max` (the region's best-matching cell -- how strongly the class
claims that region at all) or `mean` (how strongly it claims it on average).

Two further arms:
  `n_eff`  -- weight each cell by 1/|its region among the neighbours|, the continuous version of the
              same correction, avoiding a hard region cut.
  strength -- swept, because seven FULL corrections have failed here and two PARTIAL ones have won.
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
from run_overnight import SPP, RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter


def grow_regions(unit, src, dst, vm, tau=0.9):
    """Connected components of the facet graph, keeping edges whose endpoints agree above tau.

    Union-find on GPU-resident tensors would be awkward; the edge list is ~10M so a simple
    label-propagation to a fixed point is both simpler and fast enough (it converges in tens of
    iterations on these graphs).
    """
    P = unit.shape[0]
    sim = (unit[src] * unit[dst]).sum(-1)
    keep = sim > tau
    s, d = src[keep], dst[keep]
    lab = torch.arange(P, device=unit.device)
    for _ in range(64):
        upd = lab.clone()
        upd.scatter_reduce_(0, s, lab[d], reduce="amin")
        upd.scatter_reduce_(0, d, lab[s], reduce="amin")
        upd = torch.minimum(upd, lab)
        if torch.equal(upd, lab):
            break
        lab = upd
    lab[~vm] = -1
    return lab


def csls_regions(cv, region, k_regions, rep="max", strength=0.5):
    """Radius averaged over the top-G REGIONS instead of the top-K cells."""
    C = cv.shape[1]
    valid = region >= 0
    r = region[valid]
    uniq, inv = torch.unique(r, return_inverse=True)
    G = uniq.numel()
    w = torch.zeros(C, device=cv.device)
    for c in range(C):
        v = cv[valid, c]
        acc = torch.full((G,), -1e9, device=cv.device)
        if rep == "max":
            acc.scatter_reduce_(0, inv, v, reduce="amax")
        else:
            acc = torch.zeros(G, device=cv.device).scatter_add_(0, inv, v)
            cnt = torch.zeros(G, device=cv.device).scatter_add_(0, inv, torch.ones_like(v))
            acc = acc / cnt.clamp_min(1)
        kk = min(k_regions, G)
        w[c] = acc.topk(kk).values.mean()
    return cv - strength * w[None, :]


def csls_neff(cv, region, k, strength=0.5):
    """Continuous version: down-weight neighbours by how many of them share a region."""
    C = cv.shape[1]
    out = cv.clone()
    w = torch.zeros(C, device=cv.device)
    for c in range(C):
        idx = cv[:, c].topk(min(k, cv.shape[0])).indices
        reg = region[idx]
        uniq, inv, cnt = torch.unique(reg, return_inverse=True, return_counts=True)
        ww = 1.0 / cnt[inv].float()                 # a K-cell blob contributes 1, not K
        w[c] = (cv[idx, c] * ww).sum() / ww.sum().clamp_min(1e-9)
    return out - strength * w[None, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SPP[:6]))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--out", default="artifacts/scannetpp/growing_csls.json")
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
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
             .reliability()["reliability"].to(device).float() * vm)
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        region = grow_regions(cen, src, dst, vm, tau=a.tau)
        n_reg = int(torch.unique(region[region >= 0]).numel())

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        row = {"n_regions": n_reg}
        for K in sizes:
            pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
            if not pres: continue
            nm = [top[:K][i] for i in pres]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt = embed_class_names(nm, device); C = len(nm)
            cv = cen[vm] @ txt.T
            reg_v = region[vm]

            def finish(scores_v):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base_csls_cells": finish(
                cv - 0.5 * cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)[None, :])}
            for g in (64, 256):
                r[f"reg_max_G{g}"] = finish(csls_regions(cv, reg_v, g, "max", 0.5))
                r[f"reg_mean_G{g}"] = finish(csls_regions(cv, reg_v, g, "mean", 0.5))
            r["neff"] = finish(csls_neff(cv, reg_v, CSLS_K, 0.5))
            row[f"top{K}"] = r
            del txt, cv
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene} ({n_reg:,} regions): " + " | ".join(
            f"top{K} " + " ".join(f"{k}={v:.2f}" for k, v in row[f'top{K}'].items())
            for K in sizes if f"top{K}" in row))
        del raw, cen, src, dst, deg, pos, region
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    for K in sizes:
        ks = [v for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([v[f"top{K}"]["base_csls_cells"] for v in ks])
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        for arm in ks[0][f"top{K}"]:
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base_csls_cells"])
            print(f"  {arm:<18}{m:7.2f}  {m-b:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
