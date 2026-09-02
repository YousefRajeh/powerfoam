"""POSTERIOR CONSENSUS solve: project each observation, then combine -- never mix CLIP features.

THE PROBLEM THIS ADDRESSES
--------------------------
Every solver in this project mixes CLIP features. The weighted mean obviously does; so does the
GEOMETRIC MEDIAN, which is the current best (room_0 SAM round: 0.6095 vs weighted's 0.4649) -- the
median of a set of unit vectors lies in their convex hull, so its direction is a mixture and
NormLift's drift argument (a 50/50 blend lands on a third unrelated class 33.6% of the time) applies
to it unchanged. Robust-to-outliers is not the same as on-manifold. So the unsafe operation is at
step ONE of the pipeline, before any grouping, and every arm in the results database inherits it.

It matters because cells really do see conflicting things. Measured on scene0590_00 from the gram
cache's own candidate sets:

    85.4% of cells have >= 2 candidate observations (mean 4.93)
    mean pairwise cosine among a cell's OWN candidates: 0.7248 (median 0.8067)
    46.3% of cells below 0.8      12.9% below 0.5

For nearly half of all cells the solver is being asked to reconcile observations that disagree, and
for 13% they disagree badly. Those are precisely the cells where a mixture direction is meaningless.

THE CHANGE: reorder the projection and the averaging.

    current  (average-then-project):  f_j = geomedian_r{ b_r }         then  softmax(s * W^T f_j)
    here     (project-then-average):  p_j = sum_r w_r * softmax(s * W^T b_r) / sum_r w_r

softmax is nonlinear, so these are genuinely different. The second one never forms a CLIP vector
that was not observed: every projection is applied to a real observation, and the only combination
happens on the SIMPLEX, which is closed under convex combination. Drift is not bounded, it is zero.

It also preserves information the median destroys. If a cell's candidates form two clusters, the
consensus posterior stays BIMODAL rather than collapsing to a midpoint -- and that bimodality is
exactly the "mixed / boundary cell" signature from [[Mixing-operator-framework]], so it can be read
off directly instead of being estimated from M.

The candidate sets are already cached: `U` (P, kmax, 512) holds the top-kmax per-view observations
per cell and `top_w` (P, kmax) their view weights, so no re-streaming is needed.

ARMS
  base_median      argmax cos(f_geomedian, text)          -- what the pipeline does today
  consensus_s{s}   argmax of the weighted posterior mix   -- project-then-average
  consensus_top1   argmax of the single highest-weight observation (no combination at all),
                   the strict "copy, never mix" control that says how much the combination adds
"""
import argparse
import glob
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


def weighted_geometric_median_simplex(pk, w, iters=12, eps=1e-8):
    """Weighted geometric median of per-candidate posteriors, by Weiszfeld.

    This is the exact simplex analogue of the solver the pipeline already uses. The first attempt
    at project-then-average replaced the feature-space GEOMETRIC MEDIAN with a simplex weighted
    MEAN, which is not robust, and lost 1.70 mIoU -- that measured the loss of robustness, not the
    reordering. The median restores it while still never mixing CLIP features: every input is the
    projection of a real observation and the iterate stays in their convex hull, hence on the
    simplex, hence always a valid posterior.

    pk: (c, k, K) posteriors, w: (c, k) weights.
    """
    wv = w.clamp_min(0).unsqueeze(-1)                      # (c,k,1)
    q = (pk * wv).sum(1) / wv.sum(1).clamp_min(eps)        # mean as the starting point
    for _ in range(iters):
        d = (pk - q.unsqueeze(1)).norm(dim=-1).clamp_min(eps)   # (c,k)
        a = (w.clamp_min(0) / d).unsqueeze(-1)
        q = (pk * a).sum(1) / a.sum(1).clamp_min(eps)
    return q / q.sum(-1, keepdim=True).clamp_min(eps)


def trimmed_mean_simplex(pk, w, eps=1e-8):
    """Weighted mean after dropping each cell's single most-deviant candidate."""
    wv = w.clamp_min(0)
    mu = (pk * wv.unsqueeze(-1)).sum(1) / wv.sum(1).clamp_min(eps).unsqueeze(-1)
    d = (pk - mu.unsqueeze(1)).norm(dim=-1)                # (c,k)
    d = d.masked_fill(wv <= 0, -1.0)
    drop = d.argmax(1)
    keep = wv.clone()
    keep[torch.arange(keep.shape[0], device=keep.device), drop] = 0.0
    kv = keep.unsqueeze(-1)
    out = (pk * kv).sum(1) / kv.sum(1).clamp_min(eps)
    # a cell with a single candidate loses everything -- fall back to the plain mean there
    single = (wv > 0).sum(1) <= 1
    out[single] = mu[single]
    return out / out.sum(-1, keepdim=True).clamp_min(eps)


def dominant_cluster_mean(U, TW, device, thresh=0.8, chunk=60000):
    """Select the dominant AGREEING cluster of a cell's observations, then average WITHIN it.

    Rationale. In 512-d, independent per-view noise is near-orthogonal, so averaging k observations
    of the same surface gains ~sqrt(k) in SNR -- high dimension is why averaging works, not an
    obstacle to it. Selecting a single observation forfeits that (measured: -6.93 mIoU). But the
    plain mean is contaminated when some views saw a DIFFERENT surface, which is why the geometric
    median beats it (room_0: 0.6095 vs 0.4649).

    The median handles that by DOWN-WEIGHTING outliers, so it never collects the full sqrt(k) gain
    on the inliers. This instead makes the choice discrete at the level of "which surface" and keeps
    averaging at the level of "observations of that surface": take the candidate whose agreement
    mass is largest, keep every candidate within `thresh` cosine of it, and average those in the
    full 512-d space. Selection where the modes are genuinely distinct, averaging where they are
    redundant -- no CLIP interpolation ACROSS a semantic boundary, full noise cancellation within.
    """
    P, K, D = U.shape
    out = torch.zeros(P, D, device=device)
    for st in range(0, P, chunk):
        en = min(st + chunk, P)
        u = F.normalize(U[st:en].to(device).float(), dim=-1)      # (c,k,D)
        w = TW[st:en].to(device)                                   # (c,k)
        g = torch.einsum("ckd,cld->ckl", u, u)                     # pairwise cosines
        valid = (w > 0).float()
        # agreement mass of each candidate: weight of the candidates it agrees with
        mass = ((g > thresh).float() * valid[:, None, :] * w[:, None, :]).sum(-1)
        mass = mass.masked_fill(valid <= 0, -1.0)
        anchor = mass.argmax(1)
        ga = g[torch.arange(en - st, device=device), anchor]        # (c,k) cos to the anchor
        keep = ((ga > thresh).float() * valid * w).unsqueeze(-1)
        out[st:en] = F.normalize((u * keep).sum(1), dim=-1)
        del u, w, g, ga, keep
    return out


def icm_select(U, TW, src, dst, text, lam, iters, device, chunk=200000):
    """ICM over per-cell candidate SELECTION -- the discrete, never-mix alternative to the solve.

    Each cell keeps its OWN observed CLIP vectors as its label space and picks one, minimising
      unary  = -w_c                                (how strongly the cell actually saw it)
      pair   = lam * sum_{n in N(j)} (1 - cos(c, chosen_n))
    Output is always an observed embedding, so drift is zero by construction. Mode-voting is the
    1-hop greedy special case of this; ICM is the multi-hop version, initialised at the unary
    optimum (= the `consensus_top1` arm).
    """
    P, K, D = U.shape
    w = TW.to(device)
    sel = w.argmax(1)                                   # unary optimum == top1
    for _ in range(iters):
        cur = torch.empty(P, D, device=device)
        for st in range(0, P, chunk):
            en = min(st + chunk, P)
            u = F.normalize(U[st:en].to(device).float(), dim=-1)
            cur[st:en] = u[torch.arange(en - st, device=device), sel[st:en]]
        # sum of neighbour features per cell -> pairwise term is a single dot product
        agg = torch.zeros(P, D, device=device).index_add_(0, src, cur[dst])
        deg = torch.zeros(P, device=device).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        changed = 0
        for st in range(0, P, chunk):
            en = min(st + chunk, P)
            u = F.normalize(U[st:en].to(device).float(), dim=-1)
            pair = torch.einsum("ckd,cd->ck", u, agg[st:en])          # sum_n cos(c, chosen_n)
            cost = -w[st:en] - lam * pair / deg[st:en].clamp_min(1)[:, None]
            cost = cost.masked_fill(w[st:en] <= 0, float("inf"))
            new = cost.argmin(1)
            changed += int((new != sel[st:en]).sum())
            sel[st:en] = new
        if changed == 0:
            break
    return sel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0645_00,scene0140_00,scene0590_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--scales", default="20,50,200")
    p.add_argument("--chunk", type=int, default=60000)
    p.add_argument("--icm-lams", default="",
                   help="comma-separated pairwise weights; enables the ICM selection arm")
    p.add_argument("--icm-iters", type=int, default=8)
    p.add_argument("--outdir", default="artifacts/scannet/posterior_consensus")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    scales = [float(x) for x in a.scales.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"

        cache = torch.load(sorted(glob.glob(f"{art}/gram_cache_*.pt"))[0],
                           map_location="cpu", weights_only=False)
        U = cache["U"]                      # (P, k, 512) fp16 unit-norm observations
        TW = cache["top_w"].float()         # (P, k) view weights
        P = int(cache["P"])
        del cache

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
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "P": P, "arms": {}}
        print(f"[{scene}] P={P:,} candidates/cell={float((TW>0).float().sum(1).mean()):.2f}",
              flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)          # (K, 512)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            if "dom_feats" not in locals():
                dom_feats = dominant_cluster_mean(U, TW, device)
            cos_med = torch.zeros(P, len(names), device=device)
            cos_med[vm] = unit[vm] @ text.T
            b = score(cos_med.argmax(-1).cpu().numpy(), "base_median")
            print(f"  {cs} [base_median] mIoU={b:.2f}", flush=True)

            # --- candidate cosines, chunked (P x k x K is large)
            outs = {f"consensus_s{s:g}": torch.zeros(P, dtype=torch.long) for s in scales}
            for s in scales:
                outs[f"median_s{s:g}"] = torch.zeros(P, dtype=torch.long)
                outs[f"trimmed_s{s:g}"] = torch.zeros(P, dtype=torch.long)
            outs["consensus_top1"] = torch.zeros(P, dtype=torch.long)
            # LINEAR consensus: average the class SCORES, no softmax before combining. The forward
            # model (ray tracing, lifting, diffusion) is linear, and a linear projection commutes
            # with a linear combination -- W^T(sum_r w_r b_r) = sum_r w_r (W^T b_r) -- so ordering
            # cannot matter here. Any gap to base_median is therefore attributable to the SOLVER
            # being a geometric median (itself nonlinear) rather than to the projection order,
            # which is what the softmax arms confounded.
            outs["linear_consensus"] = torch.zeros(P, dtype=torch.long)
            outs["dominant_cluster"] = (dom_feats @ text.T).argmax(-1).cpu()
            for st in range(0, P, a.chunk):
                en = min(st + a.chunk, P)
                u = F.normalize(U[st:en].to(device).float(), dim=-1)      # (c,k,512)
                w = TW[st:en].to(device)                                  # (c,k)
                cc = torch.einsum("ckd,md->ckm", u, text)                 # (c,k,K)
                wv = (w > 0).float() * w
                outs["consensus_top1"][st:en] = cc[
                    torch.arange(en - st, device=device), w.argmax(1)].argmax(-1).cpu()
                outs["linear_consensus"][st:en] = (
                    (cc * wv[..., None]).sum(1) / wv.sum(1).clamp_min(1e-8)[:, None]
                ).argmax(-1).cpu()
                for s in scales:
                    pk = torch.softmax(s * cc, dim=-1)                    # project EACH candidate
                    mix = (pk * wv[..., None]).sum(1)                     # then combine on simplex
                    outs[f"consensus_s{s:g}"][st:en] = mix.argmax(-1).cpu()
                    outs[f"median_s{s:g}"][st:en] = weighted_geometric_median_simplex(
                        pk, wv).argmax(-1).cpu()
                    outs[f"trimmed_s{s:g}"][st:en] = trimmed_mean_simplex(pk, wv).argmax(-1).cpu()
                del u, w, cc, wv
            if a.icm_lams:
                adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device,
                                 weights_only=True)
                from run_simplex_diffusion_eval import csr_to_edges
                esrc, edst, _ = csr_to_edges(adj["adjacent"].to(device).long(),
                                             adj["offsets"].to(device).long(), P, device)
                kp = vm[esrc] & vm[edst]; esrc, edst = esrc[kp], edst[kp]
                for lm in [float(x) for x in a.icm_lams.split(",")]:
                    sel = icm_select(U, TW, esrc, edst, text, lm, a.icm_iters, device)
                    cls_i = torch.zeros(P, dtype=torch.long)
                    for st in range(0, P, a.chunk):
                        en = min(st + a.chunk, P)
                        u = F.normalize(U[st:en].to(device).float(), dim=-1)
                        pick = u[torch.arange(en - st, device=device), sel[st:en]]
                        cls_i[st:en] = (pick @ text.T).argmax(-1).cpu()
                    outs[f"icm_lam{lm:g}"] = cls_i
                del adj, esrc, edst
            for tag, cls in outs.items():
                cls = cls.numpy().copy()
                cls[~valid_mask] = cos_med.argmax(-1).cpu().numpy()[~valid_mask]
                v = score(cls, tag)
                print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
            del cos_med, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
