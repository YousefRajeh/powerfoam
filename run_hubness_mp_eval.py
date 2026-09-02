"""Mutual Proximity: the one canonical hubness correction we never tested.

WHAT WE HAVE AND HAVE NOT TRIED. CSLS (Conneau et al. 2018) is a hubness correction and it is one of
only two components that ever helped (+1.98). Around it we tested and rejected: iterated CSLS
(-0.92), adaptive-k CSLS (-0.92), an explicit hub penalty (-7.27), capacity-constrained power
weights (-3.85), EM prior adaptation (-23.10), Sinkhorn (-0.89/-3.77), per-class z-score (-1.07),
divisive local scaling (-0.46/-2.30) and mutual-NN as a replacement criterion (-1.24/-2.78).

The hubness literature that CSLS itself cites (Radovanovic et al. JMLR 2010; Schnitzer et al. JMLR
2012) has THREE families: local scaling (= CSLS, tested), NICDM (= the divisive form, tested), and
MUTUAL PROXIMITY, which we have never implemented.

WHAT MAKES MP DIFFERENT. CSLS subtracts a per-class CONSTANT, so it shifts every cell's score for a
class by the same amount. MP instead reweights by an empirical CDF -- the probability that the pair
is mutually closer than chance:

    MP(i,c) = P(sim(i, .) < sim(i,c)) * P(sim(., c) < sim(i,c))

The first factor is cell i's rank of class c among all classes; the second is class c's rank of cell
i among all cells. The second factor is CLASS-CONDITIONAL and non-linear in the score, so unlike a
constant offset it can reorder classes differently for different cells -- which a shift cannot. It is
also rank-based rather than mean-based, so a single very close cell cannot drag it the way it drags
CSLS's mean.

Both factors survive rank_encode: the product varies by class, which is the invariance test that
killed the per-cell arms (see test_csls_paper_ideas.py).

ARMS. Empirical MP, Gaussian MP (Schnitzer's parametric variant, cheaper and smoother), and the
partial blends, because full corrections have failed in this project and partial ones have won.
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
from run_hierarchy_eval import SCENES


def mp_empirical(cv):
    """MP(i,c) = (rank of c within cell i) * (rank of i within class c), both as empirical CDFs."""
    N, C = cv.shape
    r_cell = cv.argsort(1).argsort(1).float() / max(C - 1, 1)      # per-cell CDF over classes
    r_cls = cv.argsort(0).argsort(0).float() / max(N - 1, 1)       # per-class CDF over cells
    return r_cell * r_cls


def mp_gaussian(cv):
    """Schnitzer's parametric MP: model each row and column as Gaussian and multiply the tail
    probabilities. Smoother than the empirical CDF and far cheaper than a full double sort."""
    nd = torch.distributions.Normal(0.0, 1.0)
    zc = (cv - cv.mean(1, keepdim=True)) / cv.std(1, keepdim=True).clamp_min(1e-8)
    zk = (cv - cv.mean(0, keepdim=True)) / cv.std(0, keepdim=True).clamp_min(1e-8)
    return nd.cdf(zc) * nd.cdf(zk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/hubness_mp.json")
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
        R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
             .reliability()["reliability"].to(device).float() * vm)
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
            cv = cen[vm] @ txt.T

            def finish(scores_v):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            base = cv - 0.5 * cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)[None, :]
            r = {"base_csls": finish(base)}
            mpe, mpg = mp_empirical(cv), mp_gaussian(cv)
            r["MP_empirical"] = finish(mpe)
            r["MP_gaussian"] = finish(mpg)
            # partial: blend MP with the CSLS baseline on a common scale (both z-scored per cell)
            def z(x):
                return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-8)
            zb = z(base)
            for w_ in (0.25, 0.5, 0.75):
                r[f"MP_emp_blend{w_:g}"] = finish((1 - w_) * zb + w_ * z(mpe))
                r[f"MP_gau_blend{w_:g}"] = finish((1 - w_) * zb + w_ * z(mpg))
            # MP applied ON TOP of CSLS rather than instead of it
            r["MP_gau_on_csls"] = finish(mp_gaussian(base))
            row[f"top{K}"] = r
            del txt, cv, mpe, mpg
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " ".join(f"{k}={v:.2f}" for k, v in row.get("top100", {}).items()))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, src, dst, deg, pos
        torch.cuda.empty_cache()

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([x["base_csls"] for x in ks])
        print(f"\n=== top{K} ({len(ks)} scenes), base_csls {b:.2f} ===")
        for d, k, w_ in sorted(((np.mean([x[k] for x in ks]) - b, k,
                                 sum(1 for x in ks if x[k] > x["base_csls"])) for k in ks[0]),
                               reverse=True):
            print(f"  {k:<20}{b+d:7.2f}  {d:+6.2f}  wins {w_}/{len(ks)}")


if __name__ == "__main__":
    main()
