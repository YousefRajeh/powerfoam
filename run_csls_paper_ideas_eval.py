"""Three attacks on CSLS derived from re-reading Conneau et al. (ICLR 2018) Section 2.3.

WHAT THE PAPER ACTUALLY CLAIMS. CSLS(x,y) = 2cos(x,y) - r_T(x) - r_S(y) exists to make matching
MUTUAL: "we want to improve the comparison metric such that the nearest neighbor of a source word,
in the target language, is more likely to have as a nearest neighbor this particular source word."
It is a surrogate for mutual-NN consistency, NOT a calibration of class marginals. The authors
explicitly reject one-sided updates (Dinu's reverse ranks, Smith's inverted softmax) because "the
similarity updates are different for the words of the source and target languages", and reject ISF
additionally because it "requires to cross-validate a parameter".

That last point retro-explains this project's seven negative marginal-matching results (EM -23.10,
hub_penalty -7.27, Rule G -5.33, capacity -3.85, Sinkhorn -3.77/-0.89, z-score -1.07): inverted
softmax IS the marginal-normalisation family, and it was already published as inferior to local
scaling. We were not finding an anomaly, we were reproducing a known result.

THE INVARIANCE THAT KILLS MOST OF OUR IDEAS. Our stack ends
   ... -> rank_encode -> diffuse -> argmax
and `rank_encode` sets p0[i, order[i,j]] = tmpl[j]: every cell receives the SAME template permuted
by its own class ranking. The pipeline is therefore invariant to ANY per-cell monotone transform of
the scores. Per-CLASS terms survive (CSLS, lambda-centering: the two things that ever worked);
per-CELL terms are erased before diffusion sees them -- reliability, c_intra (AUC 0.72 but +0.02
when acted upon), support, and Sinkhorn's row scalings. That is a better explanation of the
"partial beats full" regularity than partialness.

ARM A -- MUTUAL-NN DIRECTLY. If CSLS works because it approximates mutual-NN, compute mutual-NN
instead: reward cell i for lying in class c's own top-K neighbourhood N_S(t_c). Per-class-dependent,
so it survives rank_encode. Hard indicator and a soft reciprocal-rank form, beta scaled by the
per-cell spread of cosines so it is not a free parameter in absolute units.

ARM B -- PUT PER-CELL INFORMATION WHERE IT CANNOT BE ERASED. The discarded r_T(f_i) is exactly an
isolation measure. Rank-encoding destroys it, but DIFFUSION still sees per-node quantities, via
`anchor` (per-node fidelity: confident cells keep their own evidence) and `edge_w` (confident
neighbours send stronger messages). This re-injects the half of CSLS we dropped at the only stage
where it is not algebraically invisible, and simultaneously re-tests reliability and c_intra, which
were shelved on evidence that the invariance argument says was never valid.

ARM C -- DIVISIVE, NOT SUBTRACTIVE. The paper cites Zelnik-Manor & Perona, whose self-tuning local
scaling divides by a per-point sigma. We only ever tested the subtractive form. cos / r_K(t_c)^gamma
is the same "discount dense regions" intent in the functional form the hubness literature (NICDM,
mutual proximity) settled on.

BUILT-IN FALSIFICATION TEST. `C_nicdm_mutual` divides by sqrt(r_K(t_c) * r_T(f_i)) -- it differs
from `C_div_g0.5` only by a per-cell factor. If the invariance claim above is right these two arms
must score EXACTLY equal. If they differ, the claim is wrong and the whole analysis needs revisiting.
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


def norm01(x):
    """Rank-normalise to [0,1]. Rank rather than min-max because every per-cell statistic here
    (reliability, c_intra, r_T) is heavy-tailed, and a single outlier would otherwise flatten the
    whole scale into a corner."""
    o = torch.argsort(torch.argsort(x))
    return o.float() / max(x.numel() - 1, 1)


def r_class(cv, k):
    """r_S(t_c): mean cosine of class c to its K nearest CELLS. The term we already use."""
    return cv.topk(min(k, cv.shape[0]), dim=0).values.mean(0)


def r_cell(cv, k):
    """r_T(f_i): mean cosine of cell i to its K nearest CLASSES. The term we DISCARD."""
    return cv.topk(min(k, cv.shape[1]), dim=1).values.mean(1)


def mutual_rank(cv, k):
    """For each (i,c), cell i's rank within class c's own top-k cell list; k means 'not present'.

    This is the quantity CSLS approximates. Returned as a soft score in [0,1], 1 = class c's single
    nearest cell.
    """
    N, C = cv.shape
    kk = min(k, N)
    idx = cv.topk(kk, dim=0).indices                      # (kk, C) cell ids per class
    sc = torch.zeros_like(cv)
    ranks = torch.arange(kk, device=cv.device).float()
    val = 1.0 - ranks / kk                                # 1 .. ~0 down the list
    sc.scatter_(0, idx, val[:, None].expand(kk, C))
    return sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/csls_paper_ideas.json")
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
        rel = st.reliability()
        R = rel["reliability"].to(device).float() * vm
        c_intra = rel["c_intra"].to(device).float()
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
        del adj, ad0, of0, st, rel
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
            rK = r_class(cv, CSLS_K)
            rT = r_cell(cv, min(10, C))
            spread = cv.std(dim=1).mean()

            def finish(scores_v, anchor=None, edge_w=None):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS, edge_w=edge_w, anchor=anchor)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            base_scores = cv - 0.5 * rK[None, :]
            r = {"base": finish(base_scores)}

            # ---- ARM A: mutual-NN, the objective CSLS is a surrogate for ----------------------
            mr = mutual_rank(cv, CSLS_K)
            hard = (mr > 0).float()
            for b in (0.25, 1.0):
                r[f"A_mutual_hard_b{b:g}"] = finish(base_scores + b * spread * hard)
                r[f"A_mutual_soft_b{b:g}"] = finish(base_scores + b * spread * mr)
            r["A_mutual_only_b1"] = finish(cv + 1.0 * spread * mr)   # replaces CSLS entirely
            del mr, hard

            # ---- ARM B: per-cell signals injected into DIFFUSION, not the scores --------------
            full_anchor = torch.zeros(P, device=device)
            for name, sig in (("rT", rT), ("margin", cv.topk(2, dim=1).values.diff(dim=1).squeeze(1)),
                              ("rel", R[vm]), ("cintra", c_intra[vm])):
                conf = norm01(sig)
                full_anchor.zero_(); full_anchor[vm] = conf
                r[f"B_anchor_{name}"] = finish(base_scores, anchor=full_anchor)
                # confident NEIGHBOURS send stronger messages
                r[f"B_edgew_{name}"] = finish(base_scores, edge_w=full_anchor[dst].clamp_min(1e-3))
                del conf

            # ---- ARM C: divisive local scaling -----------------------------------------------
            rKp = rK.clamp_min(1e-6)
            for g in (0.5, 1.0):
                r[f"C_div_g{g:g}"] = finish(cv / rKp[None, :] ** g)
            # falsification test: differs from C_div_g0.5 ONLY by a per-cell factor
            r["C_nicdm_mutual"] = finish(cv / (rKp[None, :] ** 0.5 * rT.clamp_min(1e-6)[:, None] ** 0.5))

            row[f"top{K}"] = r
            del txt, cv, rK, rT, base_scores, full_anchor
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + ", ".join(
            f"top{K} base={row[f'top{K}']['base']:.2f}" for K in sizes if f"top{K}" in row))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, src, dst, deg, pos, R, c_intra
        torch.cuda.empty_cache()

    for K in sizes:
        ks = [v for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([v[f"top{K}"]["base"] for v in ks])
        print(f"\n=== top{K} ({len(ks)} scenes), base {b:.2f} ===")
        rows = []
        for arm in ks[0][f"top{K}"]:
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base"])
            rows.append((m - b, arm, m, w))
        for d, arm, m, w in sorted(rows, reverse=True):
            print(f"  {arm:<22}{m:7.2f}  {d:+6.2f}  wins {w}/{len(ks)}")
        e = [v[f"top{K}"] for v in ks]
        same = all(abs(x["C_div_g0.5"] - x["C_nicdm_mutual"]) < 1e-9 for x in e)
        print(f"  [invariance check] C_div_g0.5 == C_nicdm_mutual (per-cell factor erased): {same}")


if __name__ == "__main__":
    main()
