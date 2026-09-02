"""Established, parameter-free test-time prior corrections vs hand-tuned lambda-centering.

THE CRITIQUE THIS ANSWERS
-------------------------
lambda-centering reduces exactly to `score_c(j) = <f_j,t_c> - lam*<mu_hat,t_c>`, a per-class constant
offset. And `<mu_hat,t_c>` is proportional to `mean_j <f_j,t_c>`. So the incumbent is a HAND-TUNED
FRACTION OF PER-CLASS MEAN SUBTRACTION -- the crudest possible member of a family that computer
vision has studied for decades. Tuning `lam` per scene also makes it useless in deployment: a user
who types a query cannot sweep a hyperparameter against labels they do not have.

Every method below is derived at QUERY TIME from (a) the text embeddings of whatever classes the user
asked for and (b) the observed distribution of cell features. None uses labels. None has a knob tuned
against mIoU.

THE ESTABLISHED FAMILIES
------------------------
1. HUBNESS (Radovanovic et al., JMLR 2010). In high dimensions a few points become nearest-neighbour
   to everything. Cross-modal CLIP retrieval has this badly: some class embeddings are hubs that
   attract cells regardless of content. The standard correction is CSLS (Conneau et al., ICLR 2018,
   "Word translation without parallel data"):

       CSLS(f,t) = 2cos(f,t) - r_k(t) - r_k(f)

   `r_k(t)` = mean cosine of class t to its k nearest CELLS; `r_k(f)` is constant per cell and so
   cannot move an argmax over classes. What survives is `cos - r_k(t)/2`: our offset form, with the
   offset DERIVED PER CLASS rather than tuned. This is the closest established analogue to what
   lambda was doing by hand.

2. PER-CLASS STANDARDISATION. The textbook normalise-before-classify fix:

       z_c(j) = ( cos(f_j,t_c) - mean_j cos(f_j,t_c) ) / std_j cos(f_j,t_c)

   Note `mean_j cos(f_j,t_c) = a_bar*<mu_hat,t_c>`, so the numerator alone IS rule G -- which failed
   badly (28.16 vs 35.73). The untested ingredient is the DENOMINATOR: classes differ in how much
   their score varies across the scene, and a class with a narrow range is being compared unfairly
   against one with a wide range. Mean-subtraction without scale correction is exactly the mistake.

3. TEST-TIME PRIOR ADAPTATION (Saerens, Latinne & Decaestecker, Neural Computation 2002). The
   classic EM procedure: given a classifier's posteriors on UNLABELLED data, alternately re-estimate
   the class prior as the mean posterior and re-weight the posteriors by it. Converges to a fixed
   point, no free parameter. This is the principled version of "some classes get a head start".

4. DISTRIBUTION ALIGNMENT / SINKHORN. Scale the score matrix to satisfy a target class marginal
   (doubly-stochastic-style). Parameter-free once the target marginal is fixed. Two targets are
   tested: uniform over classes (what class-averaged mIoU implicitly rewards), and -- the
   foam-specific variant -- uniform over SPACE, using Monte-Carlo power-cell volumes, so the balance
   constraint is on volume occupied rather than on arbitrary cell counts. A Gaussian mixture cannot
   state that constraint: it has no disjoint partition and no volume per primitive.

Families 3 and 4 need posteriors, hence a temperature; both are reported over a small `s` grid so the
dependence is visible rather than hidden. Families 1 and 2 are fully parameter-free (CSLS has only
the neighbourhood size `k`, whose sensitivity is reported).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii


def em_prior(p0, iters=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (2002) test-time prior adaptation.

    p0 are the classifier's posteriors under its implicit training prior (here uniform, since the
    scores are bare cosine similarities with no class prior baked in). Alternate:
        pi   <- mean_j p(c|x_j)
        p(c|x_j) <- p0(c|x_j)*pi_c / sum_k p0(k|x_j)*pi_k
    Fixed point, no free parameter.
    """
    pi = torch.full((p0.shape[1],), 1.0 / p0.shape[1], device=p0.device)
    for _ in range(iters):
        w = p0 * pi
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        new = w.mean(0)
        if float((new - pi).abs().max()) < tol:
            pi = new; break
        pi = new
    w = p0 * pi
    return w / w.sum(1, keepdim=True).clamp_min(1e-30), pi


def iou_plugin(p0, v, iters=60):
    """Plug-in decision rule for MACRO-IoU (Nowozin CVPR 2014; Koyejo et al. NeurIPS 2014).

    mIoU is not accuracy, so posterior argmax is not its Bayes rule: IoU_c = TP/(TP+FP+FN) puts the
    prediction in its OWN denominator, so over-predicting a rare class destroys its precision. That
    is why FULL prior correction (rule G, z-score, EM, Sinkhorn) all lose while a PARTIAL offset
    wins -- lambda was never estimating a prior, it was setting a decision threshold.

    The consistent plug-in rule for a generalised metric is a cost-weighted argmax
    `argmax_c w_c * p(c|j)` whose weights depend on expected class size. Fixed point:

        pred_c   = argmax_c w_c p(c|j)
        S_c      = measure predicted as c          (foam: VOLUME, not primitive count)
        E_c      = expected measure of class c     = sum_j v_j p(c|j)
        TP_c     = sum_{j: pred=c} v_j p(c|j)
        w_c     <- 1 / (S_c + E_c - TP_c)          = 1 / E[union], the IoU denominator

    Uses only posteriors and the cell measure `v` -- no labels, no tuned constant. `v` is where foam
    contributes: the power diagram gives each cell an exact disjoint volume, so "how much of the
    scene is class c" is an integral. A Gaussian mixture has no such measure.
    """
    C = p0.shape[1]
    w = torch.ones(C, device=p0.device)
    vs = v / v.sum().clamp_min(1e-30)
    E = (p0 * vs[:, None]).sum(0)
    for _ in range(iters):
        pred = (p0 * w).argmax(1)
        oh = torch.zeros_like(p0).scatter_(1, pred[:, None], 1.0)
        S = (oh * vs[:, None]).sum(0)
        TP = (oh * p0 * vs[:, None]).sum(0)
        new = 1.0 / (S + E - TP).clamp_min(1e-12)
        new = new / new.mean()
        if float((new - w).abs().max()) < 1e-6:
            w = new; break
        w = 0.5 * w + 0.5 * new          # damped, the raw fixed point can oscillate
    return p0 * w, w


def sinkhorn(logits, target, iters=50, cell_w=None):
    """Scale exp(logits) so the (optionally volume-weighted) class marginal matches `target`."""
    P = torch.softmax(logits, dim=1)
    cw = torch.ones(P.shape[0], device=P.device) if cell_w is None else cell_w
    cw = cw / cw.sum().clamp_min(1e-30)
    for _ in range(iters):
        col = (P * cw[:, None]).sum(0)                    # current class marginal
        P = P * (target / col.clamp_min(1e-30))[None, :]  # scale toward the target
        P = P / P.sum(1, keepdim=True).clamp_min(1e-30)   # renormalise per cell
    return P


def mc_cell_volumes(centers, radii, valid, n_samples, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo = torch.as_tensor(centers, dtype=torch.float64).min(0).values
    hi = torch.as_tensor(centers, dtype=torch.float64).max(0).values
    vol = torch.zeros(centers.shape[0], dtype=torch.float64)
    done = 0
    while done < n_samples:
        n = min(500_000, n_samples - done)
        pts = (torch.rand(n, 3, generator=g, dtype=torch.float64) * (hi - lo) + lo).numpy()
        a = assign_points_to_power_cells(pts, centers, radii, valid=valid, k=64)
        a = a[a >= 0]
        vol.index_add_(0, torch.from_numpy(a).long(), torch.ones(a.shape[0], dtype=torch.float64))
        done += n
    return vol.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--csls-ks", default="10,100,1000,10000")
    p.add_argument("--scales", default="50,100,200")
    p.add_argument("--lam-ref", type=float, default=0.3)
    p.add_argument("--mc-samples", type=int, default=1_000_000)
    p.add_argument("--outdir", default="artifacts/scannet/prior_correct")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    ks = [int(x) for x in a.csls_ks.split(",")]
    scales = [float(x) for x in a.scales.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        art = f"artifacts/scannet/{scene}"
        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved
        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)
        vol = mc_cell_volumes(centers, radii, valid_mask, a.mc_samples).to(device) * vm.float()

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{SCENES[scene]}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "arms": {}}
        print(f"[{scene}] P={P:,} valid={int(vm.sum()):,}", flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            C = len(names)
            cos = torch.zeros(P, C, device=device); cos[vm] = unit[vm] @ text.T
            cv = cos[vm]                                   # only valid cells define the statistics

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            def emit(sc_mat, tag, b):
                full = torch.zeros(P, C, device=device); full[vm] = sc_mat
                v = score(full.argmax(-1).cpu().numpy(), tag)
                print(f"  {cs} [{tag}] {v:.2f} ({v-b:+.2f})", flush=True)
                del full
                return v

            b = score(cos.argmax(-1).cpu().numpy(), "base")
            print(f"  {cs} [base] {b:.2f}", flush=True)

            # reference: the hand-tuned incumbent
            off = (mu @ text.T).squeeze(0)
            emit(cv - a.lam_ref * off[None, :], f"center_lam{a.lam_ref:g}_TUNED", b)

            # --- family 2: per-class standardisation (parameter-free) ---------------------
            m_c, s_c = cv.mean(0, keepdim=True), cv.std(0, keepdim=True).clamp_min(1e-12)
            emit(cv - m_c, "meanonly_ruleG", b)                 # expected to fail (== rule G)
            emit((cv - m_c) / s_c, "zscore", b)                 # the missing SCALE term
            emit(cv / s_c, "scaleonly", b)                      # isolate the scale term

            # --- family 1: CSLS hubness correction (Conneau et al. 2018) ------------------
            for k in ks:
                kk = min(k, cv.shape[0])
                r_t = cv.topk(kk, dim=0).values.mean(0)          # per-class hubness radius
                emit(cv - 0.5 * r_t[None, :], f"csls_k{k}", b)

            # --- family 3 & 4: need posteriors, so report the s dependence openly ---------
            for s in scales:
                p0 = torch.softmax(s * cv, dim=1)
                pe, pi = em_prior(p0)
                emit(pe, f"em_prior_s{s:g}", b)
                if cs == CLASS_SETS[0] and s == scales[0]:
                    res["em_prior_pi"] = pi.cpu().tolist()
                    res["em_prior_names"] = names
                pl, wv = iou_plugin(p0, vol[vm])
                emit(pl, f"iou_plugin_vol_s{s:g}", b)
                pl, _ = iou_plugin(p0, torch.ones_like(vol[vm]))
                emit(pl, f"iou_plugin_cnt_s{s:g}", b)
                tgt = torch.full((C,), 1.0 / C, device=device)
                emit(sinkhorn(s * cv, tgt), f"sinkhorn_uni_s{s:g}", b)
                emit(sinkhorn(s * cv, tgt, cell_w=vol[vm]), f"sinkhorn_vol_s{s:g}", b)
            del cos, cv, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
