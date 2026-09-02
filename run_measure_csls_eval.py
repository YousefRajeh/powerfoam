"""CSLS with a MEASURE instead of a count -- using render support, which needs no sampling.

THE ARGUMENT. `r_K(t_c) = mean of the top-K cosines over CELLS` measures a class's crowding with the
COUNTING measure the reconstruction happens to induce. Under refinement -- split a cell into n
sub-cells inheriting its feature -- a region's contribution grows n-fold, so `r_K` is a functional of
(field, partition) rather than of the scene. A measure-weighted radius is a functional of the scene
alone.

WHY SUPPORT RATHER THAN VOLUME. Volume is the geometrically natural measure and foam can compute it
exactly, but the Monte-Carlo estimate is expensive (a uniform grid over the bbox is pathological when
cells lie on surfaces: mean 8.2 cells/bucket, max 4925) and heavy-tailed, so the estimator has high
variance -- measured drift 0.00444 against the count's 0.00081 under resampling.

`support` = sum_r A[r,j], the accumulated render weight, is already computed by the lift, needs no
sampling, and is arguably the RIGHT measure here: it is how much each cell actually contributed to
the 2D observations the feature was derived from. A cell seen by many rays has a well-determined
feature; one seen by few does not, and weighting by support says so. It also answers "how much of the
OBSERVED scene does this class claim", which is the question CSLS is really asking.

ARMS. Partial variants are included because the strongest regularity in this project is that partial
corrections beat full ones (7 full corrections negative, 2 partial positive), and a raw measure
weight is the "full" end of that axis: sqrt/cbrt of support interpolate toward the count.
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


def r_count(cv, k):
    return cv.topk(min(k, cv.shape[0]), dim=0).values.mean(0)


def r_measure(cv, w, frac):
    """Radius over the top `frac` of MEASURE, averaged with measure weights.

    Selecting by cumulative measure rather than by count is what makes this a functional of the
    scene: refining a region splits its weight across more cells but does not change the total.
    """
    C = cv.shape[1]
    out = torch.zeros(C, device=cv.device)
    target = frac * w.sum()
    for c in range(C):
        order = torch.argsort(cv[:, c], descending=True)
        cum = torch.cumsum(w[order], 0)
        n = int(torch.searchsorted(cum, target).item()) + 1
        sel = order[:n]
        ws = w[sel]
        out[c] = (cv[sel, c] * ws).sum() / ws.sum().clamp_min(1e-12)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(
        ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/measure_csls.json")
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
        support = st.support.to(device).float().clamp_min(0)
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
            cv = cen[vm] @ txt.T
            sw = support[vm]
            N = cv.shape[0]
            frac = CSLS_K / N

            def finish(w_c):
                full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * w_c[None, :]
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base_count": finish(r_count(cv, CSLS_K))}
            r["support"] = finish(r_measure(cv, sw, frac))
            r["support_sqrt"] = finish(r_measure(cv, sw.clamp_min(1e-6) ** 0.5, frac))
            r["support_cbrt"] = finish(r_measure(cv, sw.clamp_min(1e-6) ** (1 / 3), frac))
            row[f"top{K}"] = r
            del txt, cv
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " | ".join(
            f"top{K} " + " ".join(f"{k}={v:.2f}" for k, v in row[f'top{K}'].items())
            for K in sizes if f"top{K}" in row))
        del raw, cen, src, dst, deg, pos, support
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    for K in sizes:
        ks = [v for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([v[f"top{K}"]["base_count"] for v in ks])
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        for arm in ks[0][f"top{K}"]:
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base_count"])
            print(f"  {arm:<16}{m:7.2f}  {m-b:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
