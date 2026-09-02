"""Two untouched levers: decorrelating the TEXT prototypes, and pseudo-labels as the missing S_B.

WHY THE TEXT SIDE. `run_degenerate_diag.py` measured the failing classes scoring 30.9% in a TWO-WAY
contest against their single top confuser -- below the 50% chance line. The evidence is not absent
(that would give ~50); the features actively prefer the wrong class one-on-one. And the confusers
are hypernyms of their victims, with near-collinear prototypes:

    kitchen cabinet -> cabinet  (text cos 0.829)     ceiling lamp -> ceiling (0.802)
    doorframe       -> door     (0.822)              office chair -> chair   (0.749)
    storage cabinet -> cabinet  (0.726)

Every normalisation tried so far (lambda-centering, CSLS, WCCN, ABTT, local scatter, Fisher) acted on
the CELL FEATURES. The text prototypes have never been normalised against each other, yet that is
where the measured redundancy lives. Decorrelating them is the classical treatment of correlated
class means, applied to the side we never touched.

NOTE ON DIRECTION, since it looks superficially like the whitening that failed. score_c = <f, M t_c>
= <M^T f, t_c>, so this is also a linear map on features -- but M is estimated from the TEXT
covariance (100 prototypes), whose top directions are the directions prototypes SPREAD along. Whitening
there EXPANDS the cramped directions that distinguish `kitchen cabinet` from `cabinet`. The failed
arms whitened by the WITHIN-OBJECT scatter, suppressing directions that carry class signal. Opposite
operation, opposite estimator, no contradiction.

WHY PSEUDO-LABELS (user's proposal). Every discriminative method in the curled papers needs labels:
LFDA (Sugiyama JMLR 2007), LMNN (Weinberger & Saul JMLR 2009), NCA (Goldberger NIPS 2004), Frome
(NIPS 2006). Foam's facet-disagreement substitute failed its gate -- cannot-link facet pairs captured
0.155 of true class structure against 0.150 for RANDOM facet pairs, i.e. nothing. But reliability is
strongly monotone with correctness (measured deciles 0.48 -> 0.87), so the argmax of the most
RELIABLE cells is a usable pseudo-label set, and that gives a real between-class scatter.

It is self-training, so it is partly circular -- pseudo-labels come from the same features. That is
exactly why S_B_pseudo is GATED with the same capture metric used for the facet version before any
method is built on it: does it capture the TRUE class directions better than random pairs?
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
from run_macro_iou_gap import cell_histograms
from run_local_scatter_gate import between_class_scatter, leakage
from run_overnight import RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_whitening_eval import within_object_scatter

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


# ------------------------------------------------------------------ text-prototype decorrelation

def text_center(T, beta):
    """Rank-1: subtract beta * the prototype mean. The text-side analogue of lambda-centering."""
    return F.normalize(T - beta * T.mean(0, keepdim=True), dim=-1)


def text_whiten(T, alpha, ridge=1e-3):
    """Whiten prototypes by their own covariance, EXPANDING the cramped directions that separate
    near-collinear names. C is only ~100 so the covariance is rank-deficient in 512-d; the ridge is
    what keeps the inverse defined, and alpha<1 keeps it partial."""
    Tc = T - T.mean(0, keepdim=True)
    S = (Tc.T @ Tc) / max(Tc.shape[0] - 1, 1)
    S = S + ridge * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=T.device)
    w, V = torch.linalg.eigh(S.double())
    w = w.clamp_min(1e-10 * float(w.max()))
    M = (V @ torch.diag(w ** (-alpha)) @ V.T).float()
    return F.normalize(Tc @ M.T, dim=-1), M


def text_lowdin(T):
    """Loewdin symmetric orthogonalisation: T' = (T T^T)^{-1/2} T, the orthonormal set CLOSEST to T
    in least squares. Maximally decorrelates the prototypes with minimum displacement, so no class
    name is arbitrarily privileged."""
    G = T @ T.T
    w, V = torch.linalg.eigh(G.double())
    w = w.clamp_min(1e-8 * float(w.max()))
    Gi = (V @ torch.diag(w ** -0.5) @ V.T).float()
    return F.normalize(Gi @ T, dim=-1)


def text_hypernym(T, beta):
    """Subtract beta * the projection onto each prototype's single most similar OTHER prototype.
    Directly targets the measured failure: `kitchen cabinet` keeps only what `cabinet` does not
    already explain."""
    S = T @ T.T
    S.fill_diagonal_(-2.0)
    nn = S.argmax(1)
    proj = (T * T[nn]).sum(-1, keepdim=True) * T[nn]
    return F.normalize(T - beta * proj, dim=-1)


# ------------------------------------------------------------------------------ pseudo-label LFDA

def pseudo_labels(scores, R_cells, frac):
    """argmax of the most reliable `frac` of cells. Returns (index, label)."""
    n = scores.shape[0]
    k = max(int(frac * n), 100)
    idx = torch.topk(R_cells, k).indices
    return idx, scores[idx].argmax(1)


def lfda_map(S_W, S_B, rank, ridge=1e-2):
    """Top-`rank` generalised eigenvectors of S_B v = lambda S_W v, as a projector applied to BOTH
    features and prototypes. This is LFDA's T = argmax tr[(T^T S_W T)^-1 T^T S_B T]."""
    D = S_W.shape[0]
    S_Wr = S_W + ridge * torch.trace(S_W) / D * torch.eye(D, device=S_W.device)
    w, V = torch.linalg.eigh(S_Wr.double())
    Wm = (V @ torch.diag(w.clamp_min(1e-12 * float(w.max())) ** -0.5) @ V.T)
    M = Wm @ S_B.double() @ Wm
    ew, EV = torch.linalg.eigh(0.5 * (M + M.T))
    U = (Wm @ EV[:, torch.argsort(ew, descending=True)[:rank]]).float()   # (D, rank)
    return U


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/text_pseudo.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res, gates = {}, {}
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
        del adj, ad0, of0
        torch.cuda.empty_cache()

        cells = cen[vm]
        R_cells = R[vm]
        sub_idx = torch.full((P,), -1, dtype=torch.long, device=device)
        sub_idx[vm] = torch.arange(int(vm.sum()), device=device)
        S_W = within_object_scatter(cells, sub_idx[src], sub_idx[dst])

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

            def finish(T, U=None):
                cf = cells if U is None else cells @ U
                tf = T if U is None else T @ U
                cv = F.normalize(cf, dim=-1) @ F.normalize(tf, dim=-1).T
                rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
                full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * rK[None, :]
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base": finish(txt)}
            # ---- TEXT-side decorrelation ----
            for b in (0.25, 0.5, 1.0):
                r[f"T_center_b{b:g}"] = finish(text_center(txt, b))
            for al in (0.125, 0.25, 0.5):
                r[f"T_white_a{al:g}"] = finish(text_whiten(txt, al)[0])
            r["T_lowdin"] = finish(text_lowdin(txt))
            for b in (0.25, 0.5):
                r[f"T_hyper_b{b:g}"] = finish(text_hypernym(txt, b))

            # ---- PSEUDO-LABEL LFDA ----
            cv0 = cells @ txt.T
            rK0 = cv0.topk(min(CSLS_K, cv0.shape[0]), dim=0).values.mean(0)
            base_scores = cv0 - 0.5 * rK0[None, :]
            for frac in (0.05, 0.2):
                pidx, plab = pseudo_labels(base_scores, R_cells, frac)
                S_B_p = between_class_scatter(cells[pidx], plab)
                for rk in (16, 64):
                    U = lfda_map(S_W, S_B_p, rk)
                    r[f"P_lfda_f{frac:g}_r{rk}"] = finish(txt, U)
                    del U
                if K == 100:
                    gates.setdefault(scene, {})[f"S_B_pseudo_f{frac:g}"] = None  # filled below
                    gates[scene][f"S_B_pseudo_f{frac:g}"] = S_B_p.clone()
                del S_B_p
            row[f"top{K}"] = r
            del txt, cv0
            torch.cuda.empty_cache()

        # ---- GATE: does the pseudo-label S_B capture TRUE class directions? ----
        pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
        gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        H, _ = cell_histograms(assigned, gt_t, len(centers), len(pres))
        has_gt = torch.from_numpy(H.sum(1) > 0).to(device) & vm
        cell_lab = torch.from_numpy(H.argmax(1)).to(device)
        gidx = torch.nonzero(has_gt).squeeze(1)
        S_B_true = between_class_scatter(cen[gidx], cell_lab[gidx])
        g = {"S_W": {str(k): leakage(S_W, S_B_true, k) for k in (1, 2, 8, 32)}}
        for nm_, S in list(gates.get(scene, {}).items()):
            if torch.is_tensor(S):
                g[nm_] = {str(k): leakage(S, S_B_true, k) for k in (1, 2, 8, 32)}
        gates[scene] = g
        log(f"  {scene} GATE " + " | ".join(
            f"{k}: " + "/".join(f"{v[str(kk)]:.3f}" for kk in (1, 2, 8, 32)) for k, v in g.items()))
        res[scene] = row
        log(f"  {scene}: " + " ".join(f"{k}={v:.2f}" for k, v in row.get("top100", {}).items()))
        json.dump({"res": res, "gates": {s: {k: v for k, v in gg.items()}
                                         for s, gg in gates.items()}},
                  open(a.out, "w"), indent=1, default=str)
        del raw, cen, cells, src, dst, deg, pos, S_W, R, R_cells
        torch.cuda.empty_cache()

    print(f"\n{'':>22}" + "".join(f"{'k='+str(k):>9}" for k in (1, 2, 8, 32)))
    for nm_ in ("S_W", "S_B_pseudo_f0.05", "S_B_pseudo_f0.2"):
        vals = [g[nm_] for g in gates.values() if nm_ in g]
        if vals:
            print(f"{nm_:>22}" + "".join(
                f"{np.mean([x[str(k)] for x in vals]):>9.3f}" for k in (1, 2, 8, 32)))
    print("  (random 512-d baseline: 0.002/0.004/0.016/0.063)")

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([x["base"] for x in ks])
        print(f"\n=== top{K} ({len(ks)} scenes), base {b:.2f} ===")
        rows = sorted(((np.mean([x[k] for x in ks]) - b, k,
                        sum(1 for x in ks if x[k] > x["base"])) for k in ks[0]), reverse=True)
        for d, k, w in rows:
            print(f"  {k:<22}{b+d:7.2f}  {d:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
