"""Turn the CLIP Voronoi diagram into a capacity-constrained POWER diagram.

THE CORRESPONDENCE (user's observation, made precise). Cosine-argmax classification IS a spherical
Voronoi diagram: `argmax_c <f,t_c>` partitions S^(F-1) into one cell per class, sites `t_c`, NO
weights, every feature in exactly one cell -- the same kind of object as our power diagram of R^3.
Adding a per-class offset turns it into a POWER diagram:

    Voronoi:  argmax_c  <f, t_c>
    Power:    argmax_c [ <f, t_c> - w_c ]

Every component of the stack that has ever worked is an ad-hoc choice of `w_c`:
    lambda-centering   w_c = lam * <mu_hat, t_c>
    CSLS               w_c = r_k(t_c) / 2
    logit adjustment   w_c = log pi_c
CSLS works best because its weight is the one actually measured from local neighbourhood structure.

WHY WEIGHTS ARE THE RIGHT FIX HERE. [[Argmax-failure-2026-08-30]]: the true class is top-3 for 89%
of `refrigerator` points and 50% of `kitchen cabinet` points, yet they score 2.34 and 3.17. Top-1 is
taken by `shelf`, `doorframe`, `refrigerator`, `wall` -- hubs whose unweighted Voronoi cells swallow
their neighbours. Shrinking a hub's cell is exactly what a positive `w_c` does.

THE ALGORITHM. Aurenhammer, Hoffmann & Aronov (1998): for any prescribed cell capacities there exist
power weights realising them, and they minimise a CONVEX function whose gradient is the capacity
mismatch. So instead of inventing `w_c`, SOLVE for it:

    minimise  Phi(w) = sum_j max_c [ <f_j,t_c> - w_c ]  +  sum_c nu_c * w_c
    dPhi/dw_c = nu_c - (share of points currently assigned to c)

Plain gradient descent on this is just `w_c += eta * (assigned_share_c - nu_c)`: a class taking more
than its target grows its weight, shrinking its cell. Convex, no temperature, unique up to an
additive constant -- unlike the multiplicative Sinkhorn scaling refuted at -0.89 in
[[Prior-correction-derived-2026-08-29]].

TARGET CAPACITIES `nu`. Uniform is the obvious choice and is probably wrong -- real scenes are
dominated by wall/floor/ceiling. Three are tried:
  uniform   nu_c = 1/C                          -- what class-averaged mIoU nominally rewards
  sqrt      nu_c ∝ sqrt(current share)          -- partial correction, hubs shrink but keep mass
  cbrt      nu_c ∝ share^(1/3)                  -- stronger flattening, still not uniform
The `sqrt`/`cbrt` families interpolate between "leave it alone" and "force uniform", which is the
axis every prior-correction result in this project has turned on (partial beats full, every time).
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


def solve_power_weights(cv, nu, iters=300, eta=None, w0=None):
    """Aurenhammer gradient descent: w_c += eta * (share_c - nu_c).

    share_c is the fraction of points whose weighted argmax is c. The update is the exact gradient
    of the convex objective, so this converges to weights realising the capacities `nu` (up to an
    additive constant, which does not affect an argmax).
    """
    N, C = cv.shape
    w = torch.zeros(C, device=cv.device) if w0 is None else w0.clone()
    if eta is None:
        eta = float(cv.std()) * 2.0          # scale the step to the score spread
    for _ in range(iters):
        share = torch.bincount((cv - w).argmax(1), minlength=C).float() / max(N, 1)
        g = share - nu
        w += eta * g
        w -= w.mean()                        # fix the gauge
        if float(g.abs().max()) < 1e-4:
            break
    return w


def targets(cv, mode):
    C = cv.shape[1]
    if mode == "uniform":
        return torch.full((C,), 1.0 / C, device=cv.device)
    share = torch.bincount(cv.argmax(1), minlength=C).float() / max(cv.shape[0], 1)
    p = {"sqrt": 0.5, "cbrt": 1.0 / 3.0}[mode]
    t = share.clamp_min(1e-9) ** p
    return t / t.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/capacity_power.json")
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
            # CSLS first: it is itself a power weight, and the solver refines from there
            base_v = c0[vm] - 0.5 * c0[vm].topk(min(CSLS_K, int(vm.sum())), dim=0).values.mean(0)[None, :]

            def finish(scores_v):
                full = torch.zeros(P, C, device=device); full[vm] = scores_v
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            row[f"top{K}"] = {"base": finish(base_v)}
            for mode in ("uniform", "sqrt", "cbrt"):
                nu = targets(base_v, mode)
                w = solve_power_weights(base_v, nu)
                row[f"top{K}"][f"cap_{mode}"] = finish(base_v - w[None, :])
            del txt, c0, base_v
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
        b = np.mean([v[f"top{K}"]["base"] for v in ks])
        print(f"\n=== top{K} ({len(ks)} scenes) ===")
        for arm in ("base", "cap_uniform", "cap_sqrt", "cap_cbrt"):
            m = np.mean([v[f"top{K}"][arm] for v in ks])
            w = sum(1 for v in ks if v[f"top{K}"][arm] > v[f"top{K}"]["base"])
            print(f"  {arm:<14}{m:7.2f}  {m-b:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
