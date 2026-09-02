"""ISSUE F: why do centering, feature-consensus and diffusion underperform on 3DGS?

THE OBSERVATION (RESULTS_LEDGER.md, 12 scenes, 3DGS). Sequential deltas are centering +0.13/+0.15,
feature consensus +1.05/+1.02, CSLS +2.09/+3.54, rank-encode+diffusion +0.45/-0.02 -- whereas on foam
centering alone is reported at +2.25. Two of the three named contributions barely move on the
representation this paper is built on.

THE HYPOTHESIS (user's, and it is supported by three measured confounds): the constants are not
WRONG, they are expressed in UNITS THAT DO NOT TRANSFER.

  1. SOLVER. Foam features come from the streaming geometric-median solve, which returns UNIT vectors
     by construction (`gm_z` is renormalised every update), so ||f|| == 1 for every valid primitive.
     3DGS features come from the WEIGHTED solve, where ||f|| varies (median 0.866). Our 3DGS
     reliability is `R = ||f||`, which is therefore a live signal on 3DGS and a CONSTANT on foam --
     they are not the same quantity, and feature consensus consumes it.
  2. COUNT vs FRACTION. `CSLS_K = 1000` is an absolute count. Foam has 699,999 primitives (633k
     valid), 3DGS has 1,249,364 (1.148M valid). So the same K is the top 0.158% on foam and the top
     0.087% on 3DGS: CSLS measures local density at a DIFFERENT SCALE on each representation.
  3. CONCENTRATION. `lambda = 0.3` was chosen where `mean cos(f,mu) = 0.884`; on 3DGS that is 0.845.
     The amount of shared direction to remove is not the same, so a fixed lambda over-corrects or
     under-corrects.
  Graph `K = 30` over 1.8x more primitives is also a physically smaller neighbourhood.

WHAT THIS SCRIPT DOES. Coordinate-wise sweeps around the foam defaults on 3DGS, one constant at a
time, so each curve is readable. For every constant we report where the foam value sits on the 3DGS
curve. Two framings of CSLS_K are swept -- absolute count and FRACTION of valid primitives -- because
if the fraction transfers and the count does not, the fix is to re-express the constant, not to
re-tune it, and that is a much stronger result.

FALSIFIABLE PREDICTION. If the units hypothesis is right, the 3DGS optimum for CSLS_K should sit near
the FOAM-EQUIVALENT FRACTION (0.158% of valid primitives ~ 1800 on 3DGS), not near 1000. If the
optimum is near 1000 regardless, the hypothesis is wrong and the cause is elsewhere.
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
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign, knn_csr_safe

ART = "artifacts/scannetpp_gs"
SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494"]
FOAM_VALID = 632_478          # measured on foam f9f95681fd
FOAM_FRAC = CSLS_K / FOAM_VALID   # 0.158% -- the density scale CSLS was actually tuned at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-size", type=int, default=100)
    p.add_argument("--out", default="artifacts/scannetpp/hyperparam_transfer.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    res = {}
    for scene in a.scenes.split(","):
        solved = f"{ART}/{scene}/solved_weighted_gs_unfroz_ogl3.pt"
        if not os.path.exists(solved):
            log(f"  [miss] {scene}"); continue
        means, scales, quats = load_gaussians(scene)
        sv = torch.load(solved, map_location="cpu", weights_only=True)
        feats = sv["primitive_features"].float(); vmn = sv["valid_mask"].numpy()
        P = feats.shape[0]
        if means.shape[0] != P:
            log(f"  [skip] {scene}"); continue
        feats = feats.to(device); vm = torch.from_numpy(vmn).to(device)
        nvalid = int(vm.sum())
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        R = feats.norm(dim=-1) * vm          # 3DGS reliability = ||f||; on foam this is identically 1
        del feats, sv
        pos = torch.from_numpy(means).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        conc = float((raw[vm] @ mu.T).mean())

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
        assigned = np.where(vmn[assigned], assigned, -1)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, means, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        pres = sorted(set(np.unique(lab).tolist()) & set(range(a.class_size)))
        if not pres: continue
        nm = [top[:a.class_size][i] for i in pres]
        gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        txt = embed_class_names(nm, device); C = len(nm)
        log(f"  {scene}: P={P:,} valid={nvalid:,} cos(f,mu)={conc:.3f}  "
            f"foam-equivalent CSLS_K here = {int(FOAM_FRAC*nvalid)}")

        graphs = {}
        def get_graph(K):
            if K not in graphs:
                adj, off = knn_csr_safe(pos, vm, K=K)
                s, d, _ = csr_to_edges(adj, off, P, device)
                ke = vm[s] & vm[d]; s, d = s[ke], d[ke]
                deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
                    0, s, torch.ones_like(s))
                graphs[K] = (s, d, deg, adj, off)
            return graphs[K]

        def pipeline(lam=LAM, csls_k=CSLS_K, gK=30, alpha=ALPHA, iters=ITERS, rs=RANK_S,
                     use_consensus=True, use_diffusion=True):
            u = raw.clone()
            if lam > 0:
                u[vm] = F.normalize(raw[vm] - lam * mu, dim=-1)
            s, d, deg, adj, off = get_graph(gK)
            if use_consensus:
                Dm = int((off[1:] - off[:-1]).max()) + 1
                u = mode_vote_refine(u, R, pos, adj, off, chunk=max(256, 200_000 // max(Dm, 1)))
            cv = torch.zeros(P, C, device=device); cv[vm] = u[vm] @ txt.T
            if csls_k > 0:
                k = min(int(csls_k), nvalid)
                cv[vm] = cv[vm] - 0.5 * cv[vm].topk(k, dim=0).values.mean(0)
            if not use_diffusion:
                return score_pred(cv.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]
            p0 = rank_encode(cv, rs, device); p0[~vm] = 0.0
            x = diffuse(p0, s, d, deg, alpha, iters)
            return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                              gt_t, C, gt_pts.shape[0])[0]

        row = {"P": P, "valid": nvalid, "conc": conc,
               "foam_equiv_csls_k": int(FOAM_FRAC * nvalid), "default": pipeline()}
        row["lam"] = {str(v): pipeline(lam=v) for v in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)}
        row["csls_k_abs"] = {str(v): pipeline(csls_k=v) for v in (300, 1000, 3000, 10000, 30000)}
        row["csls_k_frac"] = {f"{f:g}": pipeline(csls_k=int(f * nvalid))
                              for f in (0.0005, 0.001, FOAM_FRAC, 0.005, 0.02)}
        row["graph_K"] = {str(v): pipeline(gK=v) for v in (8, 15, 30, 60)}
        row["alpha"] = {str(v): pipeline(alpha=v) for v in (0.5, 0.8, 0.95, 0.99)}
        row["iters"] = {str(v): pipeline(iters=v) for v in (10, 30, 100)}
        row["no_diffusion"] = pipeline(use_diffusion=False)
        row["no_consensus"] = pipeline(use_consensus=False)
        res[scene] = row
        log(f"    default={row['default']:.2f}  lam-curve="
            + "/".join(f"{row['lam'][k]:.1f}" for k in row['lam']))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, R, pos, txt, graphs
        torch.cuda.empty_cache()

    if not res: return
    print(f"\n=== 3DGS hyperparameter transfer, {len(res)} scenes, {a.class_size}-class ===")
    print(f"foam-equivalent CSLS_K on 3DGS ~ "
          f"{int(np.mean([r['foam_equiv_csls_k'] for r in res.values()]))} "
          f"(foam used {CSLS_K}; foam fraction {FOAM_FRAC*100:.3f}% of valid primitives)\n")
    for key, foam_val in (("lam", str(LAM)), ("csls_k_abs", str(CSLS_K)), ("csls_k_frac", None),
                          ("graph_K", "30"), ("alpha", str(ALPHA)), ("iters", str(ITERS))):
        curves = [r[key] for r in res.values() if key in r]
        if not curves: continue
        ks = list(curves[0])
        mean = {k: float(np.mean([c[k] for c in curves])) for k in ks}
        best = max(mean, key=mean.get)
        print(f"{key}:")
        for k in ks:
            tag = "  <-- foam default" if k == foam_val else ("  <-- BEST" if k == best else "")
            print(f"   {k:>10}  {mean[k]:7.2f}{tag}")
        if foam_val in mean:
            print(f"   foam default costs {mean[foam_val]-mean[best]:+.2f} vs the 3DGS optimum")
        print()
    d = float(np.mean([r["default"] for r in res.values()]))
    print(f"default pipeline {d:.2f}   |   without diffusion "
          f"{np.mean([r['no_diffusion'] for r in res.values()]):.2f}"
          f"   |   without feature consensus {np.mean([r['no_consensus'] for r in res.values()]):.2f}")


if __name__ == "__main__":
    main()
