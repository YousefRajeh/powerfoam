"""GATE: can foam supply the BETWEEN-class term that every discriminative method needs?

WHAT THE PAPERS SAY (curled 2026-08-31). LFDA (Sugiyama, JMLR 2007) defines both scatters as sums
over PAIRS of difference vectors,

    S~(w) = 1/2 sum_ij W(w)_ij (x_i-x_j)(x_i-x_j)^T ,  W(w)_ij = A_ij/n_l if y_i=y_j, 0 otherwise
    S~(b) = 1/2 sum_ij W(b)_ij (x_i-x_j)(x_i-x_j)^T ,  W(b)_ij = 1/n      if y_i!=y_j
    T_LFDA = argmax_T tr[ (T^T S~(w) T)^-1 T^T S~(b) T ]

and every discriminative method in that family needs LABELS: LFDA "supervised dimensionality
reduction", LMNN (Weinberger & Saul JMLR 2009) "from labeled examples", NCA (Goldberger et al. NIPS
2004) "labeled data", Frome et al. (NIPS 2006) per-image functions from labeled triplets. Only LPP
(He & Niyogi NIPS 2003) is unsupervised, and it preserves neighbourhoods rather than discriminating.

WHAT WE DID WRONG. `run_whitening_eval.py` built only S~(w) -- from true-facet pairs, which are
must-link constraints -- and inverted it. That is RCA (Bar-Hillel et al.), must-link with no
cannot-link term, i.e. optimising LFDA's DENOMINATOR blind. The leakage measurement
([[Three-directions-closed-2026-08-31]] section 5) is exactly why that cannot work: within-object
directions carry 15-68% of the class signal, so with no between-term there is no way to know which
of them are safe to suppress.

THE FOAM PROPOSAL. Foam can supply W(b) without labels. Cells sharing a true facet whose features
DISAGREE sit on an object boundary -- a cannot-link pair. Cells sharing a facet and agreeing are
must-link. This is MFA's intrinsic/penalty graph construction with the graphs built from exact
geometry instead of class labels, and it is foam-native: a facet is a real shared boundary between
two disjoint cells, whereas Gaussian "neighbours" overlap and need not be distinct objects at all.

THE GATE, before building any method. Compute S_B_est from the low-agreement (cannot-link) facet
pairs and compare, against the TRUE GT between-class scatter S_B_true:

    capture(M, k) = tr(V_k^T S_B_true V_k) / tr(S_B_true),  V_k = top-k eigenvectors of M

If S_B_est captures the true class directions substantially better than S_W does, the Fisher ratio
has something to work with and LFDA-with-geometric-graphs is worth building. If S_B_est is no better
than S_W -- i.e. the cannot-link pairs point the same way as the must-link ones -- then foam cannot
supply the between-term either, and the whole discriminative-metric direction closes with it.
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
from run_local_scatter_gate import between_class_scatter, leakage
from run_overnight import RECON, LAM, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_whitening_eval import within_object_scatter


def pair_scatter(feats, i, j, chunk=1_000_000):
    """Chunked: the edge list is ~9M and feats is 512-d, so materialising all difference vectors at
    once asks for 17 GB. The Gram accumulation is exactly additive over chunks."""
    D = feats.shape[1]
    S = torch.zeros(D, D, device=feats.device)
    n = i.numel()
    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        d = feats[i[s0:e0]] - feats[j[s0:e0]]
        S += d.T @ d
        del d
    return S / (2.0 * max(n, 1))


def facet_agreement(feats, i, j, chunk=2_000_000):
    out = torch.empty(i.numel(), device=feats.device)
    for s0 in range(0, i.numel(), chunk):
        e0 = min(s0 + chunk, i.numel())
        out[s0:e0] = (feats[i[s0:e0]] * feats[j[s0:e0]]).sum(-1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="f9f95681fd,c50d2d1d42,3864514494")
    p.add_argument("--out", default="artifacts/scannetpp/fisher_gate.json")
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
        idx = torch.nonzero(has_gt).squeeze(1)
        S_B_true = between_class_scatter(cen[idx], cell_lab[idx])

        # facet agreement -> must-link / cannot-link split, entirely unsupervised
        agree = facet_agreement(cen, src, dst)
        q = torch.quantile(agree[torch.randperm(agree.numel(), device=device)[:200_000]].float(),
                           torch.tensor([0.05, 0.25, 0.5], device=device))
        log(f"  {scene}: facet agreement quantiles 5/25/50 = "
            f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}")

        sub_idx = torch.full((P,), -1, dtype=torch.long, device=device)
        sub_idx[vm] = torch.arange(int(vm.sum()), device=device)
        cells = cen[vm]
        S_W = within_object_scatter(cells, sub_idx[src], sub_idx[dst])

        row = {"agreement_q": [float(x) for x in q]}
        # how well do the TRUE class directions live inside each estimated subspace?
        row["S_W"] = {str(k): leakage(S_W, S_B_true, k) for k in (1, 2, 8, 32)}
        for name, thr in (("cl05", q[0]), ("cl25", q[1])):
            m = agree <= thr                       # cannot-link: neighbours that DISAGREE
            if int(m.sum()) < 5000:
                continue
            S_B_est = pair_scatter(cen, src[m], dst[m])
            row[name] = {str(k): leakage(S_B_est, S_B_true, k) for k in (1, 2, 8, 32)}
            row[name]["n_pairs"] = int(m.sum())
        # control: the same number of RANDOM facet pairs, to show any gain is from the SELECTION
        n = int((agree <= q[1]).sum())
        rp = torch.randperm(src.numel(), device=device)[:n]
        row["rand_pairs"] = {str(k): leakage(pair_scatter(cen, src[rp], dst[rp]), S_B_true, k)
                             for k in (1, 2, 8, 32)}
        res[scene] = row
        for nm in ("S_W", "cl05", "cl25", "rand_pairs"):
            if nm in row:
                log(f"    {nm:<11} capture k=1/2/8/32: "
                    + "/".join(f"{row[nm][str(k)]:.3f}" for k in (1, 2, 8, 32)))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, cells, src, dst, pos, S_W, S_B_true
        torch.cuda.empty_cache()

    print(f"\n{'':>12}" + "".join(f"{'k='+str(k):>10}" for k in (1, 2, 8, 32)))
    for nm in ("S_W", "cl05", "cl25", "rand_pairs"):
        vals = [v[nm] for v in res.values() if nm in v]
        if not vals:
            continue
        print(f"{nm:>12}" + "".join(
            f"{np.mean([x[str(k)] for x in vals]):>10.3f}" for k in (1, 2, 8, 32)))
    print("\n  capture = fraction of TRUE class-discriminative energy inside each estimated")
    print("  subspace. random 512-dim baseline: 0.002/0.004/0.016/0.063.")
    print("  GATE: cannot-link (cl05/cl25) must capture MORE than S_W and more than rand_pairs,")
    print("  or foam cannot supply LFDA's between-term and the discriminative direction closes.")


if __name__ == "__main__":
    main()
