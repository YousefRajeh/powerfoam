"""Posterior DECONVOLUTION: invert the lift's mixing operator instead of smoothing it further.

THE MODEL
---------
For a foam the render weight is exact and closed form, because a ray decomposes into a finite
ordered set of DISJOINT segments through a bounded partition:

    A[r,j] = T_r(j) * (1 - exp(-sigma_j * l_rj)),   T_r(j) = exp(-sum_{k before j} sigma_k l_rk)

with l_rj the chord length (power radii + partition), sigma_j the density. Writing w_r = sum_l
A[r,l] for the total absorption along ray r, the lift returns

    f = M g,     M = D^-1 A^T W^-1 A,     D = diag(A^T 1),  W = diag(w_r)

M is row-stochastic and M_jl is literally "the fraction of cell j's evidence that came from cell l".
M_jj is the cell's PURITY.

WHY THIS MATTERS. Every method in this project so far applies ANOTHER stochastic operator to f --
diffusion (N f), region growing, mode-voting -- yielding N M g, i.e. more smoothing of an already
over-smoothed signal. They help mainly by transporting evidence into cells that have almost none.
The principled operation is the INVERSE. It is unavailable for Gaussians (overlapping unbounded
support admits no exact ray decomposition, so M is not computable); for a foam it is closed form.

Deconvolution cannot be done in CLIP space -- M^-1 f leaves the unit sphere, which is NormLift's
33.6% semantic-drift failure. On the SIMPLEX it is well posed:

    q* = argmin_q || M q - p ||^2   s.t.  q >= 0, 1^T q = 1

convex objective, convex feasible set, and the solution is always a valid posterior. The cheapest
form is one Richardson step, which is diffusion with the sign flipped:

    q = p + lam * (p - M p)        SHARPENING      (this file)
    p = (1-a) p0 + a S p           SMOOTHING       (run_simplex_diffusion_eval.py)

FALSIFIABLE PREDICTION. The two diseases are different and want opposite treatments:
  * INTERIOR cells are STARVED  -- T_r(j) ~ 0 for every ray, so sum_r A[r,j] ~ 0 and M_jj is 0/0.
    They need transport (smoothing), and sharpening should HURT them.
  * BOUNDARY cells are MIXED    -- ample evidence, but M_jl spread across classes.
    They need un-mixing, and sharpening should HELP them.
So a per-cell lam (positive where evidence is high and purity low, negative where evidence is
starved) should beat any global lam. That is derived from the operator, not tuned.

THE W^-1 DEFECT. The gram cache stores S = A^T A, NOT A^T W^-1 A, so the round-trip operator used
previously (D^-1 S) omitted the per-ray normalisation. This script measures the size of that
omission before correcting for it: rowsum(S)_j / support_j = the evidence-weighted mean of w_r over
the rays through j. If that ratio is ~1 the rays are fully absorbed, W ~ I, and the original
operator was already right; if it departs from 1, cells whose rays escape were being over-weighted,
which would explain the residual's degenerate low end.
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
from run_simplex_diffusion_eval import HARDEST_FIRST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0645_00,scene0140_00,scene0590_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--scales", default="50,200")
    p.add_argument("--lams", default="0.25,0.5,1.0")
    p.add_argument("--max-edges", type=int, default=140_000_000)
    p.add_argument("--class-aware", action="store_true",
                   help="weight M's off-diagonal mass by POSTERIOR DISAGREEMENT instead of using "
                        "raw purity. Low M_jj does not imply error: most off-diagonal mass comes "
                        "from cells along the same ray near the same surface, which carry the SAME "
                        "label -- that is redundancy, not contamination, and deconvolving it "
                        "destroys agreement (why global sharpening lost 0.88). Only mixing ACROSS a "
                        "class boundary is harmful. Using BC as the agreement kernel, "
                        "impurity_j = sum_l M_jl (1 - BC(p_j,p_l)) = 1 - <sqrt(p_j), (M sqrt(p))_j> "
                        "since M is row-stochastic -- one extra sparse matmul.")
    p.add_argument("--adaptive", action="store_true",
                   help="per-cell lam: sharpen where evidence is high and purity low, smooth "
                        "(negative lam) where evidence is starved -- the interior/boundary split")
    p.add_argument("--outdir", default="artifacts/scannet/deconv")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    scales = [float(x) for x in a.scales.split(",")]
    lams = [float(x) for x in a.lams.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        import glob
        cache_p = sorted(glob.glob(f"{art}/gram_cache_*.pt"))[0]
        cache = torch.load(cache_p, map_location="cpu", weights_only=False)
        P = int(cache["P"])
        keys, vals = cache["S_keys"], cache["S_vals"].float()
        support = cache["support"].float()
        if keys.numel() > a.max_edges:
            from gram_blocks import prune_edges
            keys, vals, _ = prune_edges(keys, vals, P, a.max_edges, verbose=False)
        idx = torch.stack([keys // P, keys % P]).to(device); del keys
        S = torch.sparse_coo_tensor(idx, vals.to(device), (P, P), device=device).coalesce()
        del idx, vals
        sup = support.to(device)

        # ---- the W^-1 diagnostic
        rowsum = torch.sparse.mm(S, torch.ones(P, 1, device=device)).squeeze(1)
        ok = sup > 0
        ratio = torch.zeros(P, device=device)
        ratio[ok] = rowsum[ok] / sup[ok]
        q = torch.quantile(ratio[ok].float(), torch.tensor([0.05, 0.5, 0.95], device=device))
        print(f"[{scene}] rowsum(S)/support  p5={q[0]:.4f}  median={q[1]:.4f}  p95={q[2]:.4f}",
              flush=True)
        print(f"           (=1 means rays fully absorbed, W=I, and D^-1 S was already correct)",
              flush=True)

        # W^-1-corrected operator: rescale each row so it is genuinely row-stochastic. This is the
        # exact correction when w_r is constant along a cell's rays, and the best available one
        # from the cache when it is not (the cache does not store per-ray w_r).
        Mrow = torch.zeros(P, device=device)
        Mrow[ok] = 1.0 / rowsum[ok].clamp_min(1e-30)

        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0

        # purity and evidence -> the interior/boundary split
        diag = torch.sparse.sum(S * torch.sparse_coo_tensor(
            torch.arange(P, device=device).repeat(2, 1),
            torch.ones(P, device=device), (P, P)), dim=1).to_dense() \
            if False else None   # (cheap path below instead)
        # M_jj without materialising the product: diag(S) * Mrow
        Sd = torch.zeros(P, device=device)
        si = S.indices(); sv = S.values()
        dm = si[0] == si[1]
        Sd[si[0][dm]] = sv[dm]
        purity = Sd * Mrow
        ev_rank = torch.empty(P, device=device)
        ev_rank[torch.argsort(sup)] = torch.linspace(0, 1, P, device=device)
        print(f"[{scene}] purity M_jj: median={float(purity[ok].median()):.4f}  "
              f"frac<0.5={float((purity[ok] < 0.5).float().mean()):.3f}", flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "P": P, "rowsum_over_support_median": float(q[1]),
               "purity_median": float(purity[ok].median()), "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            cos = torch.zeros(P, len(names), device=device); cos[vm] = unit[vm] @ text.T

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            b = score(cos.argmax(-1).cpu().numpy(), "base")
            print(f"  {cs} [base] mIoU={b:.2f}", flush=True)

            for s in scales:
                p0 = torch.softmax(s * cos, dim=-1); p0[~vm] = 0.0
                Mp = torch.sparse.mm(S, p0) * Mrow[:, None]        # M p, row-stochastic
                if a.class_aware:
                    sq = p0.sqrt()
                    agree = (sq * (torch.sparse.mm(S, sq) * Mrow[:, None])).sum(-1)
                    raw_imp = (1.0 - agree).clamp(0, 1)
                    # Rank-normalise. The raw scale is ~0.005 (only ~0.5% of a cell's borrowed
                    # evidence crosses a class boundary; the rest is same-label redundancy), so a
                    # lam tuned against raw purity (~0.8) is 150x too small here and the step is a
                    # no-op. Ranking makes lam mean the same thing for both signals.
                    impurity = torch.empty_like(raw_imp)
                    impurity[torch.argsort(raw_imp)] = torch.linspace(0, 1, raw_imp.numel(),
                                                                      device=device)
                    print(f"  class-aware impurity: median={float(impurity[vm].median()):.4f} "
                          f"(raw 1-purity median={float((1-purity)[vm].median()):.4f})", flush=True)
                else:
                    impurity = (1.0 - purity).clamp(0, 1)
                for lam in lams:
                    if a.adaptive:
                        # sharpen mixed-but-well-observed cells, SMOOTH starved ones.
                        # lam_j = lam * (2*ev_rank - 1) * (1 - purity): positive at high evidence,
                        # negative at low, scaled by how contaminated the cell is.
                        lj = (lam * (2 * ev_rank - 1) * impurity).unsqueeze(1)
                        tag = (f"ca_s{s:g}_l{lam:g}" if a.class_aware
                               else f"adapt_s{s:g}_l{lam:g}")
                    else:
                        lj = lam
                        tag = f"sharp_s{s:g}_l{lam:g}"
                    qd = (p0 + lj * (p0 - Mp)).clamp_min(0)
                    qd = qd / qd.sum(-1, keepdim=True).clamp_min(1e-30)   # back to the simplex
                    v = score(qd.argmax(-1).cpu().numpy(), tag)
                    print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
                    del qd
                del p0, Mp
            del cos, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s -> {out_path}", flush=True)
        del S


if __name__ == "__main__":
    main()
