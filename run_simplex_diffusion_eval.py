"""SOFT POSTERIOR (SIMPLEX) DIFFUSION on the true power-diagram facet graph.

THE IDEA
--------
Every prior graph-smoothing attempt in this project smoothed CLIP FEATURES.  That is
provably wrong for CLIP (NormLift Fig.2: a 50/50 linear mix of two CLIP directions lands
on a THIRD unrelated class 33.6% of the time) and was correctly rejected.  Mode-voting
(copy a neighbour's vector) was the accepted escape, but it is a 1-HOP, HARD copy -- and
83% of our errors are INTERIOR cells of coherent regions, which a 1-hop copy cannot reach
because the interior cell's neighbours are wrong in the same way it is.

This diffuses the CLASS POSTERIOR SIMPLEX instead:

    p0_i = softmax(s * cos(f_i, text))          (P x K, on the simplex)
    p    = (1-a) p0 + a S p    ->    p* = (1-a)(I - a S)^-1 p0        (label spreading)

The simplex is CLOSED under convex combination: no mixture of two posteriors can ever
point at a third class the way a mixture of two CLIP vectors can.  The graph Laplacian of
a power diagram is a PSD M-matrix (facets are radical planes, TPFA weights are positive),
so S is row-stochastic and the MAXIMUM PRINCIPLE holds: the diffused posterior stays in
the convex hull of the data.  No over-smoothing past the data range, no invented classes.

SOFTNESS-LAW STANCE: this is the SOFT direction, not a decisiveness move.  Nothing is
thresholded, nothing is copied, argmax happens once at the very end.  `s` is an explicit
softness dial -- s -> inf makes p0 one-hot and the diffusion becomes hard vote counting,
s -> 0 makes it uniform.  The softness law predicts an interior optimum in s, and s = inf
(hard) should LOSE.  That is a falsifiable prediction of the law on a new axis.

WHY IT IS FOAM-ONLY: the edge set is the exact dual of a disjoint bounded partition -- an
edge means the two cells literally share a boundary facet and compete for the same space.
The alpha complex of Gaussians is degenerate (mean degree 0.05 at gsplat's own 3-sigma
bound), so splats have no analogue: GaussianCut et al. must fabricate a kNN graph.

Arms (all deterministic -- no clustering, no seeds, so paired per-scene deltas carry NO
seed noise):
  base        argmax of p0 (per-primitive zero-shot)
  true_facet  diffusion on adjacency_true_facet.pt   (regular-triangulation dual)
  cech        diffusion on adjacency_<variant>.pt    (the WRONG graph, 51.4% shared edges)
  knn30       diffusion on a Euclidean 30-NN graph   (what a Gaussian method would build)

Usage:
  python run_simplex_diffusion_eval.py --scenes scene0140_00 --graphs true_facet,cech
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
from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from run_normlift_refine_eval import build_knn_csr
from build_true_facet_graph import load_points_radii

HARDEST_FIRST = ["scene0140_00", "scene0645_00", "scene0070_00", "scene0347_00",
                 "scene0590_00", "scene0400_00", "scene0200_00", "scene0000_00",
                 "scene0097_00", "scene0062_00"]



def load_roundtrip_rank(scene, device):
    """Rank-normalised round-trip residual in [0,1]; 1 = least self-consistent cell.

    Rank rather than raw value because r is heavy-tailed and its scale is not comparable across
    scenes (it depends on view count and cell size), so a fixed lambda would mean different things
    on different scenes.
    """
    p = f"artifacts/scannet/roundtrip_{scene}_signals.npz"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run roundtrip_consistency.py for {scene} first")
    r = torch.from_numpy(np.load(p)["r1"]).to(device).float()
    rank = torch.empty_like(r)
    rank[torch.argsort(r)] = torch.linspace(0, 1, r.numel(), device=device)
    return rank


def csr_to_edges(adjacent, offsets, P, device):
    deg = (offsets[1:] - offsets[:-1]).long()
    src = torch.repeat_interleave(torch.arange(P, device=device), deg)
    return src, adjacent.long(), deg


def diffuse(p0, src, dst, deg, alpha, iters, edge_w=None, anchor=None, chunk=8_000_000):
    """p <- (1-a_i) p0_i + a_i * sum_j S_ij p_j, S row-stochastic on the (weighted) graph.

    anchor: optional (P,) in [0,1] scaling the per-node fidelity.  a_i = alpha * (1-anchor_i)
    means highly-anchored (reliable) nodes hold onto their own evidence.
    """
    P, K = p0.shape
    if edge_w is None:
        w = torch.ones(src.numel(), device=p0.device)
    else:
        w = edge_w
    rowsum = torch.zeros(P, device=p0.device).index_add_(0, src, w)
    w = w / rowsum.clamp_min(1e-30)[src]
    a = torch.full((P, 1), alpha, device=p0.device) if anchor is None \
        else (alpha * (1.0 - anchor)).unsqueeze(1)
    # dead rows (deg 0) must keep their own evidence
    a = torch.where((deg > 0).unsqueeze(1), a, torch.zeros_like(a))
    p = p0.clone()
    for _ in range(iters):
        acc = torch.zeros_like(p)
        for s in range(0, src.numel(), chunk):
            e = min(s + chunk, src.numel())
            acc.index_add_(0, src[s:e], p[dst[s:e]] * w[s:e, None])
        p = (1 - a) * p0 + a * acc
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--graphs", default="true_facet,cech,knn30")
    p.add_argument("--scales", default="50,100,200,1000")
    p.add_argument("--alphas", default="0.5,0.8,0.9,0.95")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--edge-weight", default="uniform",
                   choices=["uniform", "invdist", "gauss"])
    p.add_argument("--anchor", default="none",
                   choices=["none", "reliability", "roundtrip", "roundtrip_x_support"])
    p.add_argument("--rank-encode", action="store_true",
                   help="replace softmax(s*cos) with a FIXED distribution shape mapped onto each "
                        "cell's class ranking. Isolated over 10 scenes, the softmax's whole "
                        "contribution to diffusion comes from the diffused quantity being "
                        "non-negative, bounded and unit-sum -- NOT from its values: stripping "
                        "per-cell confidence costs -0.10 mIoU and the stripped arm is invariant to "
                        "s (38.02 at both s=50 and s=200). If that holds inside the full stack, "
                        "`s` stops being a tunable and one knob leaves the tuning surface.")
    p.add_argument("--center-lam", type=float, default=0.0,
                   help="subtract lam * the common CLIP direction from the lifted features before "
                        "projecting. Removes a BIAS term (the cone effect: measured mean cos to the "
                        "common direction = 0.887 +/- 0.03 across all 10 scenes, i.e. a property of "
                        "CLIP not of the scene). Diffusion reduces VARIANCE by aggregating across "
                        "cells, so the two attack different error terms and should compose. Acts on "
                        "the REPRESENTATION, leaving the benchmark's bare cosine argmax intact -- "
                        "unlike similarity-space centering, which changes the decision rule and is "
                        "therefore a protocol violation.")
    p.add_argument("--conf-source", default="none", choices=["none", "roundtrip"],
                   help="per-cell confidence tempering of the posterior sharpness (decision-rule "
                        "lever); orthogonal to --anchor, which modulates transport instead")
    p.add_argument("--conf-lambda", type=float, default=0.5,
                   help="strength in [0,1]; 0 recovers the untempered posterior exactly")
    p.add_argument("--prerefine", action="store_true",
                   help="run one NormLift mode-vote pass on the true facet graph FIRST, "
                        "then diffuse -- tests whether the two are redundant")
    p.add_argument("--outdir", default="artifacts/scannet/simplex_diffusion")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    scales = [float(x) for x in a.scales.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    graphs = a.graphs.split(",")

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True)
            continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        ckpt_dir = f"output/scannet_{scene}_{a.variant}"

        centers, radii = load_points_radii(ckpt_dir)
        P = centers.shape[0]
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(valid_mask).to(device)
        unit = torch.zeros_like(feats)
        unit[vm_t] = F.normalize(feats[vm_t], dim=-1)
        del feats, solved
        if a.center_lam > 0:
            mu = F.normalize(unit[vm_t].mean(0, keepdim=True), dim=-1)
            unit[vm_t] = F.normalize(unit[vm_t] - a.center_lam * mu, dim=-1)
            print(f"[{scene}] centred features, lam={a.center_lam}", flush=True)
        positions = torch.from_numpy(centers).to(device).float()

        anchor = None
        conf = None
        if a.conf_source == "roundtrip" and a.conf_lambda > 0:
            conf = (1.0 - a.conf_lambda * load_roundtrip_rank(scene, device)).clamp(0.0, 1.0)
        if a.prerefine:
            from run_normlift_refine_eval import mode_vote_refine
            cands = [f"{art}/train_stats_sam_{a.variant}{a.suffix}.pt",
                     f"{art}/stats_{a.variant}{a.suffix}.pt"]
            sp = next(c for c in cands if os.path.exists(c))
            Rr = AccumulatedFeatureStats.load(sp).reliability()["reliability"].to(device).float() * vm_t
            adj0 = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device,
                              weights_only=True)
            ad0 = adj0["adjacent"].to(device).long(); of0 = adj0["offsets"].to(device).long()
            Dm = int((of0[1:] - of0[:-1]).max()) + 1
            unit_r = mode_vote_refine(unit, Rr, positions, ad0, of0,
                                      chunk=max(256, 200_000 // max(Dm, 1)))
            print(f"[{scene}] prerefine changed "
                  f"{float(((unit_r-unit).abs().sum(-1)>1e-6).float().mean())*100:.1f}%",
                  flush=True)
            unit = unit_r
            del adj0, ad0, of0, Rr
            torch.cuda.empty_cache()
        if a.anchor == "reliability":
            cands = [f"{art}/train_stats_sam_{a.variant}{a.suffix}.pt",
                     f"{art}/stats_{a.variant}{a.suffix}.pt"]
            sp = next(c for c in cands if os.path.exists(c))
            R = AccumulatedFeatureStats.load(sp).reliability()["reliability"].to(device).float()
            R = R * vm_t
            anchor = (R / R.max().clamp_min(1e-12)).clamp(0, 1)
            del R
        elif a.anchor in ("roundtrip", "roundtrip_x_support"):
            # Round-trip consistency as a LABEL-FREE fidelity signal (roundtrip_consistency.py).
            # The lift is A f = b and the cache stores S = A^T A, so rendering then re-lifting a
            # feature field is exactly S f; with the diagonal lift actually deployed the round-trip
            # operator is T = D^-1 S, and r = 1 - cos(T f, f) measures how far a cell's own
            # evidence fails to reproduce itself. High r = the cell's rays are dominated by
            # neighbours carrying different features, so it should hold its own value LESS and take
            # more from the graph -- i.e. small anchor. Hence anchor = 1 - normalised r.
            #
            # Measured on the three hardest scenes, r separates correct from incorrect cells inside
            # EVERY support quintile (15/15 bands positive, +0.02 to +0.21), so it is not simply
            # re-reading view count. But `support` alone is the stronger single signal on all three
            # (+0.156/+0.232/+0.334 vs +0.125/+0.121/+0.175), which is why the combined variant
            # exists: r is complementary evidence, not a replacement.
            rt = f"{art}/../roundtrip_{scene}_signals.npz"
            if not os.path.exists(rt):
                raise SystemExit(f"missing {rt}; run roundtrip_consistency.py for {scene} first")
            z = np.load(rt)
            fid = (1.0 - load_roundtrip_rank(scene, device)).clamp(0, 1)
            if a.anchor == "roundtrip_x_support":
                s = torch.from_numpy(z["support"]).to(device).float()
                srank = torch.empty_like(s)
                srank[torch.argsort(s)] = torch.linspace(0, 1, s.numel(), device=device)
                fid = (fid * srank).clamp(0, 1)
            anchor = (fid * vm_t).clamp(0, 1)
            del fid
        print(f"[{scene}] P={P} valid={int(valid_mask.sum())} ({time.time()-t0:.0f}s)",
              flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        print(f"[{scene}] assigned {gt_points.shape[0]} pts ({time.time()-t0:.0f}s)",
              flush=True)

        # ---- graphs ----
        G = {}
        for g in graphs:
            if g.startswith("knn"):
                # block sized so the (block x P_valid) cdist stays under ~2 GB
                blk = max(256, int(2e9 / (4 * max(int(valid_mask.sum()), 1))))
                adjacent, offsets = build_knn_csr(positions, vm_t, K=int(g[3:]),
                                                  block=blk)
            else:
                path = (f"{art}/adjacency_true_facet.pt"
                        if g.split("@")[0] == "true_facet"
                        else f"{art}/adjacency_{a.variant}.pt")
                adj = torch.load(path, map_location=device, weights_only=True)
                adjacent = adj["adjacent"].to(device).long()
                offsets = adj["offsets"].to(device).long()
            src, dst, deg = csr_to_edges(adjacent, offsets, P, device)
            # drop edges touching invalid cells: they carry no evidence
            keep = vm_t[src] & vm_t[dst]
            if "@" in g:
                # ALPHA FILTRATION of the true facet graph: keep the facet only if the two
                # power balls (scaled by t) actually reach each other.  t -> inf recovers the
                # full regular triangulation, t -> 0 empties it.  Foam-only: undefined for
                # radfoam (r=0) and degenerate for Gaussians (mean degree 0.05 at 3 sigma).
                t = float(g.split("@")[1])
                rr = torch.from_numpy(radii).to(device).float()
                d = (positions[src] - positions[dst]).norm(dim=-1)
                keep = keep & (d < t * (rr[src] + rr[dst]))
            src, dst = src[keep], dst[keep]
            deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
                0, src, torch.ones_like(src))
            if a.edge_weight == "uniform":
                ew = None
            else:
                d = (positions[src] - positions[dst]).norm(dim=-1).clamp_min(1e-9)
                ew = (1.0 / d) if a.edge_weight == "invdist" else \
                     torch.exp(-(d / d.median()) ** 2)
            G[g] = (src, dst, deg, ew)
            print(f"[{scene}] graph {g}: E={src.numel()} mean_deg="
                  f"{float(deg.float().mean()):.2f}", flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "num_primitives": P, "iters": a.iters,
               "edge_weight": a.edge_weight, "anchor": a.anchor, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names([n for _, n in kept], device)
            cos = unit @ text.T                                   # (P, K)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(
                    gt_t, torch.from_numpy(pred).long(), len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            b = score(cos.argmax(-1).cpu().numpy(), "base")
            print(f"  {cs} [base] mIoU={b:.2f}", flush=True)
            for s in scales:
                # DECISION-RULE LEVER: per-cell confidence tempering.
                #
                #     p0_i = softmax( s * (1 - lam * rhat_i) * cos(f_i, T) )
                #
                # rhat is the rank-normalised round-trip residual. This is deliberately NOT the
                # anchor: anchoring modulated a_i, i.e. TRANSPORT, and measurably lost to uniform
                # alpha (+0.20 vs +0.65 at 19cls on the 3 hardest) because damping "reliable" nodes
                # blocks the multi-hop consensus that fixes interior cells -- 83% of errors. Here
                # alpha stays uniform, so every cell still RELAYS at full strength; only what it
                # CONTRIBUTES is softened. A high-residual cell becomes a weak source rather than a
                # closed valve, and the diffusion fixed point p = (1-a)p0 + a S p then lets its
                # neighbours dominate it automatically -- no gate, no threshold.
                #
                # Per Theorem 1 (mIoU = Phi(a, S, L)) this acts on L, the labelling function --
                # which anchoring never did. It is also a SOFTNESS move, which the softness law
                # (7 confirmations that decisiveness loses) predicts should help. lam = 0 recovers
                # the current behaviour exactly, so it cannot lose to baseline by construction.
                eff = s if conf is None else (s * conf).unsqueeze(1)
                if a.rank_encode:
                    Kc = cos.shape[1]
                    tmpl = torch.softmax(s * torch.linspace(1.0, -1.0, Kc, device=device), 0)
                    order = cos.argsort(dim=-1, descending=True)
                    p0 = torch.zeros_like(cos).scatter_(1, order, tmpl.expand(cos.shape[0], -1))
                else:
                    p0 = torch.softmax(eff * cos, dim=-1)
                p0[~vm_t] = 0.0
                for g in graphs:
                    src, dst, deg, ew = G[g]
                    for al in alphas:
                        pd = diffuse(p0, src, dst, deg, al, a.iters,
                                     edge_w=ew, anchor=anchor)
                        tag = f"{g}_s{s:g}_a{al:g}"
                        v = score(pd.argmax(-1).cpu().numpy(), tag)
                        print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
                        del pd
                del p0
            del cos, text
            torch.cuda.empty_cache()

        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s -> {out_path}\n", flush=True)
        del unit, positions, G
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
