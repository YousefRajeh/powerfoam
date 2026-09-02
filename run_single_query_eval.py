"""SINGLE-QUERY (open-set) evaluation: does the pipeline survive outside the closed N-class benchmark?

THE OBJECTION THIS ANSWERS (coauthor, 2026-08-31). Closed-vocabulary segmentation -- "map the scene
onto these N words" -- is how open-vocabulary methods are EVALUATED, but not how they are used. The
practical query is "here is a scene, find the bench", with no other 30 class names supplied. A
contribution that only works when the other 99 classes are present is a benchmark artefact.

WHICH COMPONENTS ACTUALLY DEPEND ON THE CLASS LIST (derived, not guessed):

  * lambda-centering: a per-feature operation. NO dependence on the query set. Works unchanged.
  * CSLS `w_c = r_K(t_c)/2`: computed from the CELL distribution, not from other classes, so it is
    defined for a single query -- BUT it is then a constant shift, and a constant cannot change the
    ranking of cells within one query. It can only matter for calibrating a threshold ACROSS queries.
    That is a different (and separately testable) claim from "it improves argmax".
  * Loewdin `(T T^T)^{-1/2} T`: with C=1, `T T^T = 1`, so T' = T exactly. PROVABLY A NO-OP on a
    single query. It needs >= 2 prototypes.
  * rank-encode + diffusion: rank-encode builds a C-simplex per cell; with one query there is no
    simplex. The diffusion OPERATOR still applies to a scalar relevancy field, but the encoding does
    not carry over.

ADAPTATIONS TESTED HERE, one per broken component:

  A. Loewdin against CANONICAL NEGATIVES. A single query always carries an implicit background;
     LERF and OpenGaussian both score relevancy against fixed negatives ("object", "things", "stuff",
     "texture"). Decorrelating the matrix [query; negatives] is the single-query form of C3 and is
     NOT a no-op. This also makes the method's requirement honest: it needs a contrast set, not a
     labelled vocabulary.
  B. Scalar diffusion of the relevancy field on the same graph -- C2 with one channel.
  C. CSLS as CROSS-QUERY THRESHOLD CALIBRATION, measured by mIoU at a SINGLE global threshold shared
     by every query, which is the quantity a real system needs and the closed-set benchmark never
     tests.

TWO METRICS, deliberately.
  * per-query Average Precision -- RANK-BASED and threshold-free. A constant offset cannot change it,
    so this isolates the components that genuinely reorder (Loewdin-with-negatives, diffusion).
  * mIoU at one global threshold shared across queries -- this is where calibration (CSLS) shows up,
    and where a method that only works with a per-query tuned threshold would be exposed.

Each class present in a scene is issued as ONE query in isolation; no other class name is ever given
to the model.
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
from run_normlift_refine_eval import mode_vote_refine
from run_macro_iou_gap import cell_histograms
from run_overnight import RECON, LAM, CSLS_K, ALPHA, ITERS, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_text_and_pseudo_eval import text_lowdin

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]
# LERF / OpenGaussian canonical negatives -- a FIXED set, never tuned per query or per scene.
NEGATIVES = ["object", "things", "stuff", "texture"]


def average_precision(scores, labels):
    """AP for a binary retrieval task, computed exactly (no interpolation).

    Rank-based, so any strictly increasing transform of `scores` leaves it unchanged -- which is
    precisely why a constant CSLS offset cannot show up here.
    """
    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    if y.sum() == 0:
        return np.nan
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def iou_at(scores, labels, tau):
    pred = scores >= tau
    inter = np.logical_and(pred, labels).sum()
    union = np.logical_or(pred, labels).sum()
    return float(inter / union) if union > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--n-classes", type=int, default=100)
    p.add_argument("--out", default="artifacts/scannetpp/single_query.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    per_scene = {}
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

        # C1: lambda-centering -- no class list involved, so it is available in the open-set regime.
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        uncen = raw
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        pres = sorted(set(np.unique(lab).tolist()) & set(range(a.n_classes)))
        names = [top[:a.n_classes][i] for i in pres]
        gt_pt = remap_gt_labels(lab, pres)          # 0 = ignore, c+1 = class c
        neg_emb = embed_class_names(NEGATIVES, device)

        rows = []
        for ci, nm in enumerate(names):
            y = (gt_pt == ci + 1)
            if y.sum() < 200:
                continue
            t = embed_class_names([nm], device)[0]                   # ONE query, in isolation

            def to_points(cellscore):
                s = np.full(gt_pts.shape[0], -1e9, dtype=np.float64)
                s[owned] = cellscore[assigned[owned]]
                return s

            base_cell = (cen[vm] @ t).float()
            full = torch.full((P,), -1e9, device=device); full[vm] = base_cell
            r = {"class": nm, "n_pos": int(y.sum())}

            # --- ranking-based arms (AP) -------------------------------------------------------
            unc = torch.full((P,), -1e9, device=device); unc[vm] = (uncen[vm] @ t).float()
            r["AP_uncentred"] = average_precision(to_points(unc.cpu().numpy()), y)
            r["AP_C1_centred"] = average_precision(to_points(full.cpu().numpy()), y)

            # A: Loewdin on [query; canonical negatives] -- the single-query form of C3
            T = F.normalize(torch.cat([t[None, :], neg_emb], 0), dim=-1)
            t_low = text_lowdin(T)[0]
            low = torch.full((P,), -1e9, device=device); low[vm] = (cen[vm] @ t_low).float()
            r["AP_C3_lowdin_neg"] = average_precision(to_points(low.cpu().numpy()), y)

            # B: scalar diffusion of the relevancy field -- C2 with one channel
            def diffuse_scalar(vec):
                x = torch.zeros(P, 1, device=device)
                v = vec[vm]
                v = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-9)
                x[vm, 0] = v
                out = diffuse(x, src, dst, deg, ALPHA, ITERS)[:, 0]
                out[~vm] = -1e9
                return out
            r["AP_C2_diffused"] = average_precision(to_points(diffuse_scalar(full).cpu().numpy()), y)
            r["AP_C2_C3_stacked"] = average_precision(
                to_points(diffuse_scalar(low).cpu().numpy()), y)

            # --- C1 variants: the -1.62 comes from RENORMALISATION, not from the subtraction ----
            # <u - lam*mu, t> = <u,t> - lam<mu,t>: the second term is CONSTANT across cells, so the
            # subtraction cannot re-rank. Dividing by ||u - lam*mu|| = sqrt(1 - 2lam<u,mu> + lam^2)
            # DOES vary per cell, and it shrinks for mean-aligned cells -- so renormalising promotes
            # exactly the generic, hub-like cells. Verified in float64: without renormalisation the
            # ranking is bitwise identical to raw; with it, 98.5% of ranks move.
            nr = torch.full((P,), -1e9, device=device)
            nr[vm] = ((uncen[vm] - LAM * mu[0]) @ t).float()
            r["AP_C1_no_renorm"] = average_precision(to_points(nr.cpu().numpy()), y)

            # --- the OTHER half of CSLS: r_T(x), a PER-CELL hubness term -----------------------
            # Closed-set argmax keeps r_S(y) (per class) and drops r_T(x) as constant. Single-query
            # ranking is the mirror image: r_S(y) is the constant and r_T(x) is what can re-rank.
            # r_T(x) = how strongly a cell responds to text IN GENERAL, estimated on the fixed
            # canonical negatives -- a cell that matches everything should not win any query.
            rt = (cen[vm] @ neg_emb.T).mean(1).float()
            for g in (0.25, 0.5, 1.0):
                cc = torch.full((P,), -1e9, device=device)
                cc[vm] = base_cell - g * rt
                r[f"AP_C3cell_rT_g{g:g}"] = average_precision(to_points(cc.cpu().numpy()), y)
            # best cell-side variant stacked with scalar diffusion
            cc = torch.full((P,), -1e9, device=device); cc[vm] = base_cell - 0.5 * rt
            r["AP_C3cell_plus_C2"] = average_precision(
                to_points(diffuse_scalar(cc).cpu().numpy()), y)

            # --- calibration arms (global threshold) -------------------------------------------
            # CSLS offset from the CELL distribution only; a constant per query, so it cannot change
            # AP, but it re-scales scores so one threshold can serve every query.
            w = 0.5 * base_cell.topk(min(CSLS_K, base_cell.numel())).values.mean()
            cal = full.clone(); cal[vm] = base_cell - w
            r["_scores_plain"] = to_points(full.cpu().numpy())
            r["_scores_csls"] = to_points(cal.cpu().numpy())
            r["_y"] = y
            rows.append(r)

        # global-threshold sweep, shared across ALL queries of the scene
        for key, tag in (("_scores_plain", "IoU_global_tau_plain"),
                         ("_scores_csls", "IoU_global_tau_csls")):
            allv = np.concatenate([r[key][r[key] > -1e8] for r in rows])
            taus = np.quantile(allv, np.linspace(0.50, 0.999, 60))
            best, best_t = -1, None
            for tau in taus:
                v = np.nanmean([iou_at(r[key], r["_y"], tau) for r in rows])
                if v > best:
                    best, best_t = v, tau
            for r in rows:
                r[tag] = best * 100
                r[tag + "_tau"] = float(best_t)
        for r in rows:
            for k in ("_scores_plain", "_scores_csls", "_y"):
                r.pop(k, None)
        per_scene[scene] = rows
        log(f"  {scene}: {len(rows)} single queries")
        json.dump(per_scene, open(a.out, "w"), indent=1)
        del raw, cen, uncen, src, dst, deg, pos
        torch.cuda.empty_cache()

    allr = [r for rs in per_scene.values() for r in rs]
    print(f"\n=== SINGLE-QUERY OPEN-SET RETRIEVAL: {len(allr)} queries over "
          f"{len(per_scene)} scenes ===")
    print("\nRanking quality (per-query Average Precision, threshold-free):")
    base = np.nanmean([r["AP_C1_centred"] for r in allr]) * 100
    for k in ("AP_uncentred", "AP_C1_centred", "AP_C1_no_renorm", "AP_C3_lowdin_neg",
              "AP_C2_diffused", "AP_C2_C3_stacked", "AP_C3cell_rT_g0.25", "AP_C3cell_rT_g0.5",
              "AP_C3cell_rT_g1", "AP_C3cell_plus_C2"):
        m = np.nanmean([r[k] for r in allr]) * 100
        w = sum(1 for r in allr if r[k] > r["AP_C1_centred"])
        print(f"  {k:<22}{m:7.2f}  {m-base:+6.2f}   better than C1 on {w}/{len(allr)} queries")
    print("\nCalibration (mIoU at ONE global threshold shared by every query):")
    for k in ("IoU_global_tau_plain", "IoU_global_tau_csls"):
        vals = [rs[0][k] for rs in per_scene.values() if rs]
        print(f"  {k:<22}{np.mean(vals):7.2f}")
    json.dump(per_scene, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
