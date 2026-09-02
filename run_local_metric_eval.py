"""Two foam-derived generalisations of CSLS: a true RADIUS, and a local TANGENT metric.

WHERE THIS COMES FROM. `argmax_c <f,t_c>` is a spherical Voronoi diagram and a per-class offset
makes it a power diagram ([[Power-weights-on-the-CLIP-sphere-2026-08-30]]). CSLS is one choice of
weight, `w_c = r_K(t_c)/2` with `r_K` the MEAN cosine to the K nearest cells. In PowerFoam the
weights are radii that adapt to LOCAL density, and each cell also carries an orientation. Two things
follow that CSLS does not do:

ARM A -- RADIUS instead of mean. A power radius is not an average; it is the radius of the ball
containing K neighbours. On the sphere that is the K-th nearest cosine, i.e. a QUANTILE not a mean:

        w_c = quantile_K( {cos(x, t_c)} )        (arm `radius`)

This is the classical k-NN density estimator and is robust to the shape of the tail, which the mean
is not: one very close cell drags CSLS's mean but not the quantile.

ARM B -- LOCAL TANGENT METRIC. A scalar weight assumes the cloud around `t_c` is isotropic. It is
not: the CLIP cone is strongly anisotropic (mean cos(f,mu)=0.887, and rank-1 globally per
[[Geometric-bias-corrections-2026-08-29]] section 1). Around a class, cells spread along some
directions more than others, and displacement along a HIGH-variance direction is weak evidence of
dissimilarity while the same displacement along a low-variance direction is strong evidence.
So replace the scalar radius with a local Mahalanobis metric in the tangent space at `t_c`:

        v      = x - <x,t_c> t_c                 tangent displacement (radial part removed)
        Sigma_c= cov(v) over the K nearest cells, low-rank: sum_i s_i^2 u_i u_i^T + eps I
        score  = <x,t_c> - gamma * v^T Sigma_c^-1 v

This is the anisotropic generalisation of a single radius, and it is exactly the "local tangent
structure of the feature manifold" the scalar weight throws away.

PARTIAL, NOT FULL. Seven full corrections have failed in this project and two partial ones have won,
so both arms are applied with a strength parameter and swept, rather than driven to a fixed point.
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


def csls_mean(cv, k):
    """Incumbent: w_c = mean cosine to the K nearest cells."""
    return cv - 0.5 * cv.topk(min(k, cv.shape[0]), dim=0).values.mean(0)[None, :]


def csls_radius(cv, k, strength=0.5):
    """Arm A: w_c = the K-th nearest cosine -- the RADIUS of the ball holding K cells."""
    kk = min(k, cv.shape[0])
    r = cv.topk(kk, dim=0).values[-1]          # K-th largest == the ball radius
    return cv - strength * r[None, :]


def tangent_metric(u_cells, txt, k=CSLS_K, m=8, gamma=0.1, eps=1e-3, chunk=200_000):
    """Arm B: local Mahalanobis in the tangent space at each class embedding.

    Returns the PENALTY (n_cells, C) to subtract from the cosine. Low-rank so the 512x512 covariance
    is never formed: an SVD of the K x 512 tangent matrix gives the directions directly.
    """
    N, D = u_cells.shape
    C = txt.shape[0]
    cos_all = u_cells @ txt.T
    pen = torch.zeros(N, C, device=u_cells.device)
    for c in range(C):
        t = txt[c]
        idx = cos_all[:, c].topk(min(k, N)).indices
        X = u_cells[idx]
        V = X - (X @ t)[:, None] * t[None, :]          # tangent components of the neighbourhood
        # top-m directions of the local spread
        try:
            _, S, Vh = torch.linalg.svd(V - V.mean(0, keepdim=True), full_matrices=False)
        except Exception:
            continue
        mm = min(m, S.shape[0])
        U = Vh[:mm]                                     # (m, D)
        var = (S[:mm] ** 2) / max(V.shape[0] - 1, 1)
        inv = 1.0 / (var + eps)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Vx = u_cells[s:e] - (u_cells[s:e] @ t)[:, None] * t[None, :]
            proj = Vx @ U.T                             # (n, m)
            # Mahalanobis along the modelled directions, isotropic eps elsewhere
            quad = (proj ** 2 * inv[None, :]).sum(1)
            resid = (Vx ** 2).sum(1) - (proj ** 2).sum(1)
            pen[s:e, c] = gamma * (quad + resid / eps) / max(D, 1)
    return pen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SPP[:6]))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/local_metric.json")
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
            uc = cen[vm]
            cv = uc @ txt.T

            def finish(scores_v):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base_csls_mean": finish(csls_mean(cv, CSLS_K))}
            for st in (0.25, 0.5, 1.0):
                r[f"radius_{st:g}"] = finish(csls_radius(cv, CSLS_K, st))
            base = csls_mean(cv, CSLS_K)
            for g in (0.05, 0.2):
                pen = tangent_metric(uc, txt, k=CSLS_K, m=8, gamma=g)
                r[f"tangent_{g:g}"] = finish(base - pen)
                r[f"tangent_only_{g:g}"] = finish(cv - pen)
                del pen
            row[f"top{K}"] = r
            del txt, cv, uc
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
        b = np.mean([v[f"top{K}"]["base_csls_mean"] for v in ks])
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        for arm in ks[0][f"top{K}"]:
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base_csls_mean"])
            print(f"  {arm:<20}{m:7.2f}  {m-b:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
