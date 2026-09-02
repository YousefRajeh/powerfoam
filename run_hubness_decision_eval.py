"""Attack the argmax failure directly: the correct class is ranked 2-3, hubs take top-1.

EVIDENCE THIS IS BUILT ON ([[Argmax-failure-2026-08-30]]). On f9f95681fd, `refrigerator` is top-3 for
89% of its points and top-5 for 96%, yet scores 2.34 IoU; `kitchen cabinet` is top-3 for 50% and
scores 3.17. The top-1 thieves are systematic -- shelf, doorframe, refrigerator, wall win across many
true classes -- i.e. HUBS. Pairwise CLIP AUC between the confused classes is 0.87-0.97, so the
information is present and the flat 100-way argmax is what loses it.

ARMS (all label-free, all on top of the frozen stack's centred+refined features):

  base            current stack: CSLS(k=1000) once, then argmax
  csls_iter       iterate the CSLS correction to a fixed point. One pass removes each class's mean
                  similarity to its k nearest cells, but removing it CHANGES the neighbourhoods, so
                  a hub can survive one pass. Iterating re-measures and re-subtracts.
  csls_adaptive   per-class k proportional to how much of the cloud that class currently claims. A
                  class taking 30% of top-1 slots is measured against a correspondingly larger
                  neighbourhood, so its radius reflects its actual dominance rather than a constant.
  hub_penalty     subtract only the EXCESS top-1 share: c is penalised by log(share_c / uniform)
                  when it wins more than its share of argmaxes. Targets exactly the handful of hub
                  classes rather than reshaping every marginal -- full-marginal Sinkhorn was already
                  refuted at -0.89 in [[Prior-correction-derived-2026-08-29]].
  top5_pairwise   restrict to each point's top-5 classes and decide by pairwise margins among them,
                  exploiting the 0.87-0.97 pairwise AUC that the flat argmax cannot use.

Scored on all 12 ScanNet++ scenes, top100 and top20, against the frozen stack.
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


def csls_once(cv, k):
    r_t = cv.topk(min(k, cv.shape[0]), dim=0).values.mean(0)
    return cv - 0.5 * r_t[None, :]


def csls_iterated(cv, k, iters=5):
    """Re-measure the hubness radius after each correction: removing a hub's advantage changes
    which cells are its nearest, so one pass under-corrects the strongest hubs."""
    out = cv
    for _ in range(iters):
        out = csls_once(out, k)
    return out


def csls_adaptive(cv, k0):
    """Per-class k scaled by that class's current share of the argmax. A class winning 30% of the
    cloud is measured against a neighbourhood 0.30*N wide, not a constant 1000."""
    N, C = cv.shape
    share = torch.bincount(cv.argmax(1), minlength=C).float() / max(N, 1)
    out = cv.clone()
    for c in range(C):
        kc = int(max(k0, min(N, share[c].item() * N)))
        out[:, c] = cv[:, c] - 0.5 * cv[:, c].topk(kc).values.mean()
    return out


def hub_penalty(cv, strength=1.0, iters=3):
    """Penalise only classes taking MORE than a uniform share of top-1, by the log excess."""
    N, C = cv.shape
    out = cv.clone()
    unif = 1.0 / C
    for _ in range(iters):
        share = torch.bincount(out.argmax(1), minlength=C).float() / max(N, 1)
        excess = torch.clamp(torch.log((share + 1e-9) / unif), min=0.0)
        sd = out.std().clamp_min(1e-6)
        out = out - strength * sd * excess[None, :]
    return out


def top5_pairwise(cv, k=5):
    """Among each point's top-k candidates, score by summed pairwise margin -- a round-robin using
    the pairwise separability (AUC 0.87-0.97) that a flat argmax cannot exploit."""
    v, idx = cv.topk(k, dim=1)
    # round-robin score of candidate a = sum_b (v_a - v_b) = k*v_a - sum_b v_b
    wins = k * v - v.sum(1, keepdim=True)
    pick = wins.argmax(1)
    return idx[torch.arange(cv.shape[0], device=cv.device), pick]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/hubness_decision.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
    for scene in SPP:
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
            c0 = torch.zeros(P, C, device=device); c0[vm] = cen[vm] @ txt.T
            cv = c0[vm]

            def finish(scores_v, cls_v=None):
                full = torch.zeros(P, C, device=device)
                if cls_v is None:
                    full[vm] = scores_v
                    p0 = rank_encode(full, RANK_S, device)
                else:
                    oh = torch.zeros_like(cv); oh.scatter_(1, cls_v[:, None], 1.0)
                    full[vm] = oh
                    p0 = rank_encode(full, RANK_S, device)
                p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            row[f"top{K}"] = {
                "base":          finish(csls_once(cv, CSLS_K)),
                "csls_iter":     finish(csls_iterated(cv, CSLS_K, 5)),
                "csls_adaptive": finish(csls_adaptive(cv, CSLS_K)),
                "hub_penalty":   finish(hub_penalty(csls_once(cv, CSLS_K))),
                "top5_pairwise": finish(None, top5_pairwise(csls_once(cv, CSLS_K))),
            }
            del txt, c0, cv
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " | ".join(
            f"top{K} " + " ".join(f"{k}={v:.2f}" for k, v in row[f'top{K}'].items())
            for K in sizes if f"top{K}" in row))
        del raw, cen, src, dst, deg, pos
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    for K in sizes:
        ks = [v for v in res.values() if f"top{K}" in v]
        if not ks: continue
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        b = np.mean([v[f"top{K}"]["base"] for v in ks])
        for arm in ("base", "csls_iter", "csls_adaptive", "hub_penalty", "top5_pairwise"):
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base"])
            print(f"  {arm:<16}{m:7.2f}  {m-b:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
