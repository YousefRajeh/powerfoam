"""GATE: is within-object scatter separable from class structure LOCALLY, when it is not globally?

THE FINDING THIS FOLLOWS. Global whitening by the within-object scatter S_W failed at every strength
and every conditioning ([[Three-directions-closed-2026-08-31]] section 4), including a
subspace-restricted form that is well-conditioned by construction. The explanation: globally, the
directions along which one object's cells vary ARE the directions that separate classes, so
suppressing the former destroys the latter.

THE QUESTION. "Globally" is doing real work in that sentence. A single 512x512 scatter pools every
object in the scene: the illumination/viewpoint variation of a countertop and the variation of a
refrigerator door get averaged into one basis, and their union can easily span the class-separating
directions even if neither does alone. Locally -- within one region of the scene -- the nuisance
subspace may be low-dimensional and genuinely disjoint from what separates the few classes present
there. That is precisely the premise of local metric learning (LFDA, multi-metric LMNN).

THE MEASUREMENT, and it is falsifiable. Let V_k be the top-k eigenvectors of S_W (built UNSUPERVISED
from true-facet pairs) and S_B the between-class scatter (built WITH GT, for diagnosis only):

    leakage(k) = tr(V_k^T S_B V_k) / tr(S_B)

= the fraction of class-discriminative energy that lies inside the subspace whitening suppresses.
High leakage means whitening destroys signal, which is what the global result implies. If LOCAL
leakage is materially lower than GLOBAL leakage at matched k, local whitening has headroom and is
worth building. If they are equal, the local version dies here for the price of one diagnostic and
we do not write the method.

Regions are spatial k-means over cell centres. NOT feature-threshold region growing: that was
measured to produce one 632k-cell component out of 700k (adjacent cells agree at median cosine
0.996), so it yields no locality at all.
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
from evaluate_point_cloud_miou import remap_gt_labels
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges
from run_normlift_refine_eval import mode_vote_refine
from run_macro_iou_gap import cell_histograms
from run_overnight import RECON, LAM, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_whitening_eval import within_object_scatter


def between_class_scatter(feats, labels):
    """S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T / N, over cells with a GT label. Diagnostic only."""
    mu = feats.mean(0, keepdim=True)
    S = torch.zeros(feats.shape[1], feats.shape[1], device=feats.device)
    n_tot = 0
    for c in torch.unique(labels):
        m = labels == c
        n = int(m.sum())
        if n < 2:
            continue
        d = feats[m].mean(0, keepdim=True) - mu
        S += n * (d.T @ d)
        n_tot += n
    return S / max(n_tot, 1)


def leakage(S_W, S_B, k):
    """Fraction of class-discriminative energy inside the top-k eigen-subspace of S_W.

    1.0 means whitening those k directions removes ALL class signal; k/D means it removes no more
    than a random subspace would, i.e. the nuisance and signal subspaces are unrelated.
    """
    w, V = torch.linalg.eigh(S_W.double())
    Vk = V[:, torch.argsort(w, descending=True)[:k]].float()
    return float(torch.trace(Vk.T @ S_B @ Vk) / torch.trace(S_B).clamp_min(1e-12))


def kmeans_regions(pos, R, iters=15, seed=0):
    g = torch.Generator(device=pos.device).manual_seed(seed)
    C = pos[torch.randperm(pos.shape[0], generator=g, device=pos.device)[:R]].clone()
    for _ in range(iters):
        lab = torch.cdist(pos, C).argmin(1)
        for r in range(R):
            m = lab == r
            if int(m.sum()) > 0:
                C[r] = pos[m].mean(0)
    return torch.cdist(pos, C).argmin(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="f9f95681fd,c50d2d1d42,3864514494")
    p.add_argument("--regions", default="16,64,256")
    p.add_argument("--out", default="artifacts/scannetpp/local_scatter_gate.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
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
        R_rel = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
                 .reliability()["reliability"].to(device).float() * vm)
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R_rel, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        del adj, ad0, of0, R_rel
        torch.cuda.empty_cache()

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
        gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        H, _ = cell_histograms(assigned, gt_t, len(centers), len(pres))
        has_gt = torch.from_numpy(H.sum(1) > 0).to(device) & vm
        cell_lab = torch.from_numpy(H.argmax(1)).to(device)

        sub_idx = torch.full((P,), -1, dtype=torch.long, device=device)
        sub_idx[vm] = torch.arange(int(vm.sum()), device=device)
        cells = cen[vm]
        S_W_g = within_object_scatter(cells, sub_idx[src], sub_idx[dst])
        lab_idx = torch.nonzero(has_gt).squeeze(1)
        S_B_g = between_class_scatter(cen[lab_idx], cell_lab[lab_idx])
        row = {"global": {str(k): leakage(S_W_g, S_B_g, k) for k in (1, 2, 8, 32)}}
        log(f"  {scene} GLOBAL leakage k=1/2/8/32: "
            + "/".join(f"{row['global'][str(k)]:.3f}" for k in (1, 2, 8, 32)))

        for R in [int(x) for x in a.regions.split(",")]:
            reg = kmeans_regions(pos[vm], R)
            reg_full = torch.full((P,), -1, dtype=torch.long, device=device)
            reg_full[vm] = reg
            vals = {str(k): [] for k in (1, 2, 8, 32)}
            sizes = []
            for r in range(R):
                cm = reg_full == r
                lm = cm & has_gt
                if int(lm.sum()) < 500 or int(torch.unique(cell_lab[lm]).numel()) < 2:
                    continue
                em = (reg_full[src] == r) & (reg_full[dst] == r)
                if int(em.sum()) < 2000:
                    continue
                S_W_l = within_object_scatter(cells, sub_idx[src[em]], sub_idx[dst[em]])
                idx = torch.nonzero(lm).squeeze(1)
                S_B_l = between_class_scatter(cen[idx], cell_lab[idx])
                for k in (1, 2, 8, 32):
                    vals[str(k)].append(leakage(S_W_l, S_B_l, k))
                sizes.append(int(lm.sum()))
            if not sizes:
                continue
            wgt = np.array(sizes, dtype=float); wgt /= wgt.sum()
            row[f"R{R}"] = {k: float((np.array(v) * wgt).sum()) for k, v in vals.items()}
            row[f"R{R}"]["n_regions_used"] = len(sizes)
            log(f"  {scene} R={R:<4} ({len(sizes)} regions) leakage k=1/2/8/32: "
                + "/".join(f"{row[f'R{R}'][str(k)]:.3f}" for k in (1, 2, 8, 32)))
        res[scene] = row
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, cells, src, dst, pos
        torch.cuda.empty_cache()

    print(f"\n{'':>12}" + "".join(f"{'k='+str(k):>10}" for k in (1, 2, 8, 32)))
    for scope in ["global"] + [f"R{r}" for r in [int(x) for x in a.regions.split(",")]]:
        vals = [v[scope] for v in res.values() if scope in v]
        if not vals:
            continue
        print(f"{scope:>12}" + "".join(
            f"{np.mean([x[str(k)] for x in vals]):>10.3f}" for k in (1, 2, 8, 32)))
    print("\n  A RANDOM k-dim subspace of 512 would leak k/512 = 0.002/0.004/0.016/0.063.")
    print("  GATE: local leakage materially BELOW global at matched k -> local whitening has")
    print("  headroom and is worth building. Equal -> it dies here, for one diagnostic.")


if __name__ == "__main__":
    main()
