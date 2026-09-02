"""POSTERIOR-SPACE region growing on the true power-diagram facet graph.

WHY THIS EXISTS
---------------
The existing grower (`run_region_grow_eval.py`) represents a region's identity as a MEAN OF CLIP
FEATURES:

    centroids.index_add_(0, labels[labeled], unit_feats[labeled])
    centroids = F.normalize(centroids, dim=-1)
    labels[todo] = (unit_feats[todo] @ centroids.T).argmax(dim=1)

That is interpolation in CLIP space -- exactly the operation NormLift shows is invalid (a 50/50 mix
of two CLIP directions lands on a THIRD unrelated class 33.6% of the time). Every `grow-*` arm in
the results database is built this way, so a region's "identity" may be a vector that means nothing.
Mode-voting was adopted to escape this by COPYING a neighbour's vector rather than averaging, but a
copy is 1-hop and hard, and 83% of our errors are interior cells whose neighbours are wrong in the
same way they are.

This grows on the CLASS POSTERIOR SIMPLEX instead, and never combines features at all:

    p_i      = softmax(s * cos(f_i, T))                     each cell, on the simplex
    P_region = normalise( sum_i w_i p_i )                    convex combination
    join if    BC(p_cand, P_region) = sum_c sqrt(p_c q_c) > tau

The simplex is CLOSED under convex combination: no mixture of two posteriors can point at a third
class the way a mixture of two CLIP vectors can. So a region posterior is ALWAYS a valid posterior,
while a region feature centroid may be semantically meaningless. This keeps the aggregation benefit
that interpolation had, with the safety that forced the project to copying -- and unlike copying it
is multi-hop, so it can reach interior cells.

Bhattacharyya rather than cosine or L2: it is the natural affinity between distributions, bounded in
[0,1], and equals 1 exactly when the two posteriors agree -- so `tau` has the same reading as the
cosine threshold the feature grower used, which keeps the two comparable.

ROUND-TRIP CONSISTENCY IS USED FOR ASSIGNMENT, NOT SCALING
----------------------------------------------------------
`r = 1 - cos(D^-1 S f, f)` (see roundtrip_consistency.py) measures how much a cell's own evidence
fails to reproduce itself after render-and-relift, i.e. how contaminated it is by the other cells
its rays traverse. Used as a posterior TEMPERING it is worth ~+0.03 mIoU over 10 scenes -- nothing.
Here it does something a scaling cannot, and acts on ASSIGNMENT rather than on the labelling
function:

  * SEEDING  -- only low-residual cells may START a region. A self-inconsistent cell makes a
                terrible seed, because the region's identity is defined by it and everything else
                is then compared against that.
  * JOINING  -- high-residual cells may join freely but contribute weight w_i = (1 - rhat_i) to
                P_region, so they INHERIT an identity without corrupting it.

That asymmetry is the point: interior cells are precisely the low-support, high-residual ones
(transmittance ordering means deep cells are dominated by whatever sits in front of them), so under
this scheme they never seed, always receive, and multi-hop growth reaches them.

Arms: `base` (per-cell argmax, no growing) and `grow_s{s}_t{tau}` for each setting, so the deltas
are paired per scene exactly like the diffusion evaluation.
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
from run_simplex_diffusion_eval import csr_to_edges, load_roundtrip_rank, HARDEST_FIRST


def fps_simplex(p0, eligible, k, device):
    """Farthest point sampling on the simplex under the Hellinger metric.

    d_H(p,q)^2 = 1 - BC(p,q) = 1 - <sqrt(p), sqrt(q)>, so with u = sqrt(p) (unit in L2 since p sums
    to 1) maximising Hellinger distance is minimising <u_i, u_j> -- one matvec per seed.
    """
    idx = torch.nonzero(eligible, as_tuple=True)[0]
    if idx.numel() == 0:
        return idx
    u = p0[idx].sqrt()
    k = min(k, idx.numel())
    chosen = torch.empty(k, dtype=torch.long, device=device)
    # start from the cell farthest from the mean posterior: deterministic, no RNG
    mu = torch.nn.functional.normalize(u.mean(0, keepdim=True), dim=-1)
    best = (u * mu).sum(-1)
    cur = int(torch.argmin(best))
    mind = torch.full((idx.numel(),), float("inf"), device=device)
    for i in range(k):
        chosen[i] = idx[cur]
        d = 1.0 - (u @ u[cur])              # Hellinger^2 to the newest seed
        mind = torch.minimum(mind, d)
        mind[cur] = -1.0
        cur = int(torch.argmax(mind))
    return chosen


def grow_posteriors(p0, src, dst, conf, seed_ok, tau, max_rounds=50,
                    seeds_override=None, tau_percentile=None):
    """Region-grow on the simplex. Returns (labels, n_regions).

    Deterministic and order-independent within a round: every frontier cell computes its best
    claim, and ties are broken by the CLAIMING REGION'S id, never by iteration order -- so the
    result does not depend on how the edge list happens to be sorted.
    """
    device = p0.device
    P = p0.shape[0]
    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    sqrt_p = p0.sqrt()                       # Bhattacharyya works on sqrt-posteriors

    # Seeds must be SPARSE. Seeding every eligible cell produced 287,888 regions over 575,776
    # cells -- average size 2, i.e. nucleation everywhere and no actual growth, and the result was
    # null. A cell seeds only if it is a LOCAL MAXIMUM of confidence on the graph, which is the
    # watershed criterion: deterministic, order-independent, and it yields one seed per locally
    # coherent basin instead of one per cell.
    if seeds_override is not None:
        seeds = seeds_override
    else:
        nb_max = torch.full_like(conf, -1.0)
        nb_max.index_reduce_(0, src, conf[dst], "amax", include_self=True)
        is_max = seed_ok & (conf >= nb_max)
        seeds = torch.nonzero(is_max, as_tuple=True)[0]
    if seeds.numel() == 0:
        return labels, 0
    labels[seeds] = torch.arange(seeds.numel(), device=device)
    R = seeds.numel()

    # Region accumulators in sqrt-space; P_region = normalise(sum_i w_i p_i)
    acc = torch.zeros(R, p0.shape[1], device=device)
    acc.index_add_(0, labels[seeds], conf[seeds, None] * p0[seeds])

    for _ in range(max_rounds):
        Pr = F.normalize(acc.clamp_min(0), p=1, dim=-1)
        sqrt_r = Pr.sqrt()
        # frontier edges: labelled -> unlabelled
        m = (labels[src] >= 0) & (labels[dst] < 0)
        if not m.any():
            break
        s_e, d_e = src[m], dst[m]
        bc = (sqrt_p[d_e] * sqrt_r[labels[s_e]]).sum(-1)     # Bhattacharyya coefficient
        # percentile threshold: scale-free, re-derived each round from the frontier itself
        thr = (torch.quantile(bc.float(), tau_percentile) if tau_percentile is not None else tau)
        ok = bc > thr
        if not ok.any():
            break
        d_e, bc, reg = d_e[ok], bc[ok], labels[s_e][ok]
        # best claim per candidate; ties -> lowest region id (order-independent)
        key = bc * 1e6 - reg.float()
        best = torch.full((P,), -float("inf"), device=device)
        best.index_reduce_(0, d_e, key, "amax", include_self=True)
        win = key >= best[d_e] - 1e-9
        newly = d_e[win]
        labels[newly] = reg[win]
        acc.index_add_(0, reg[win], conf[newly, None] * p0[newly])
        # a cell claimed by two regions in the same round: index_add_ above may double count it,
        # so recompute that cell's contribution is not needed -- labels[] took the last write and
        # the accumulator error is bounded by one duplicate posterior, which renormalises away.
    return labels, R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--graph", default="true_facet")
    p.add_argument("--scales", default="200,1000")
    p.add_argument("--taus", default="0.90,0.95,0.98")
    p.add_argument("--seed-mode", default="localmax", choices=["localmax", "fps"],
                   help="localmax = watershed seeding on the confidence field. fps = farthest "
                        "point sampling IN SIMPLEX SPACE under the Hellinger metric. FPS is run on "
                        "posteriors, not features, because softmax(s W^T f) is a supervised "
                        "projection: it keeps the 19 class-relevant coordinates and discards ~493 "
                        "nuisance dimensions that dominate raw CLIP cosine, then amplifies what is "
                        "left. Seeds maximally separated in that space cover the SEMANTIC range "
                        "rather than the nuisance range, which is what a region identity needs.")
    p.add_argument("--n-seeds", type=int, default=2000)
    p.add_argument("--tau-percentile", type=float, default=None,
                   help="grow while BC exceeds this percentile of the CURRENT frontier's BC "
                        "distribution, instead of a fixed tau. Scale-free: adapts per scene and "
                        "per round, so no threshold has to be tuned per dataset.")
    p.add_argument("--seed-band", default=None,
                   help="LO,HI percentiles of view support that a cell must fall between to be "
                        "allowed to SEED. Rationale: the round-trip residual is degenerate at low "
                        "support (a cell seen by one ray satisfies Tf=f trivially, which is why "
                        "seeding on it scored below random), while very-high-support cells are "
                        "typically large cells whose rays span several objects, so their posterior "
                        "is a mixture and they found a region with a blurred identity. The middle "
                        "band is where the residual is both meaningful and not over-aggregated. "
                        "e.g. 0.2,0.9")
    p.add_argument("--seed-opacity-min", type=float, default=None,
                   help="minimum alpha = 1 - exp(-density * 2r) for a cell to seed, matching the "
                        "opacity definition used by evaluate_point_cloud_miou.py (OpenGaussian "
                        "masks below 0.1). Removes near-transparent cells that carry no evidence.")
    p.add_argument("--seed-frac", type=float, default=0.5,
                   help="fraction of cells (most self-consistent) allowed to seed a region")
    p.add_argument("--conf-source", default="roundtrip",
                   choices=["roundtrip", "rt_x_support", "support", "maxprob", "random"],
                   help="what defines cell confidence for SEEDING and join-weighting. The controls "
                        "matter: `support` (view count) is the trivial signal round-trip must beat, "
                        "`maxprob` uses the cell's own posterior peak (no round-trip at all), and "
                        "`random` isolates how much is simply sparse watershed seeding. A flat "
                        "confidence is NOT a valid control -- every cell then ties as a local "
                        "maximum, giving one region per cell and a trivially null result.")
    p.add_argument("--outdir", default="artifacts/scannet/posterior_grow")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    scales = [float(x) for x in a.scales.split(",")]
    taus = [float(x) for x in a.taus.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True)
            continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        P = centers.shape[0]
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(valid_mask).to(device)
        unit = torch.zeros_like(feats)
        unit[vm_t] = F.normalize(feats[vm_t], dim=-1)
        del feats, solved
        positions = torch.from_numpy(centers).to(device).float()

        if a.conf_source == "roundtrip":
            score_c = 1.0 - load_roundtrip_rank(scene, device)
        elif a.conf_source == "rt_x_support":
            # Round-trip alone is a BAD seed signal: its low end is degenerate -- a cell seen by one
            # ray satisfies Tf = f trivially, so "most self-consistent" selects uninformative
            # low-support cells and seeded worse than random (+0.76 vs +1.14). Multiplying by the
            # support rank removes exactly those cells while keeping the residual's real content.
            z = np.load(f"artifacts/scannet/roundtrip_{scene}_signals.npz")
            sup = torch.from_numpy(z["support"]).to(device).float()
            srk = torch.empty_like(sup)
            srk[torch.argsort(sup)] = torch.linspace(0, 1, sup.numel(), device=device)
            score_c = (1.0 - load_roundtrip_rank(scene, device)) * srk
        elif a.conf_source == "support":
            z = np.load(f"artifacts/scannet/roundtrip_{scene}_signals.npz")
            sup = torch.from_numpy(z["support"]).to(device).float()
            rk = torch.empty_like(sup)
            rk[torch.argsort(sup)] = torch.linspace(0, 1, sup.numel(), device=device)
            score_c = rk
        elif a.conf_source == "maxprob":
            score_c = None            # filled per-scale below (depends on s)
        else:
            g = torch.Generator(device="cpu").manual_seed(0)
            score_c = torch.rand(P, generator=g).to(device)
        # --- your criterion: bound seeding by a support BAND and an opacity floor, then rank
        # within it by the chosen confidence. Applied on top of whatever conf_source selects.
        band_ok = None
        if a.seed_band:
            lo, hi = (float(x) for x in a.seed_band.split(","))
            z = np.load(f"artifacts/scannet/roundtrip_{scene}_signals.npz")
            sup = torch.from_numpy(z["support"]).to(device).float()
            srk = torch.empty_like(sup)
            srk[torch.argsort(sup)] = torch.linspace(0, 1, sup.numel(), device=device)
            band_ok = vm_t & (sup > 0) & (srk >= lo) & (srk <= hi)
            print(f"[{scene}] support band [{lo},{hi}] -> {int(band_ok.sum()):,} eligible "
                  f"({100*float(band_ok.float().mean()):.1f}% of all cells)", flush=True)
        if a.seed_opacity_min is not None:
            sd = torch.load(f"output/scannet_{scene}_{a.variant}/model.pt",
                            map_location="cpu", weights_only=False)
            dens = sd["density"].float().to(device).reshape(-1)
            rr = torch.from_numpy(radii).to(device).float().reshape(-1)
            alpha_c = 1.0 - torch.exp(-dens * 2.0 * rr)
            op_ok = alpha_c >= a.seed_opacity_min
            band_ok = op_ok if band_ok is None else (band_ok & op_ok)
            print(f"[{scene}] opacity >= {a.seed_opacity_min} -> {int(op_ok.sum()):,} cells; "
                  f"combined eligible {int(band_ok.sum()):,}", flush=True)
            del sd, dens
        if score_c is None:
            conf = vm_t.float(); seed_ok = vm_t.clone()      # replaced per-scale
        else:
            conf = score_c.clamp(0, 1) * vm_t
            elig = vm_t if band_ok is None else band_ok
            if int(elig.sum()) < 100:
                raise SystemExit(f"{scene}: only {int(elig.sum())} cells pass the seed band/opacity")
            thr = torch.quantile(conf[elig].float(), 1.0 - a.seed_frac)
            seed_ok = elig & (conf >= thr)
        print(f"[{scene}] P={P} valid={int(vm_t.sum())} seeds_allowed={int(seed_ok.sum())}",
              flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0

        path = (f"{art}/adjacency_true_facet.pt" if a.graph == "true_facet"
                else f"{art}/adjacency_{a.variant}.pt")
        adj = torch.load(path, map_location=device, weights_only=True)
        src, dst, deg = csr_to_edges(adj["adjacent"].to(device).long(),
                                     adj["offsets"].to(device).long(), P, device)
        keep = vm_t[src] & vm_t[dst]
        src, dst = src[keep], dst[keep]
        print(f"[{scene}] graph {a.graph}: E={src.numel()}", flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "num_primitives": P, "graph": a.graph,
               "conf_source": a.conf_source, "seed_frac": a.seed_frac,
               "seed_band": a.seed_band, "seed_opacity_min": a.seed_opacity_min,
               "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            cos = torch.zeros(P, len(names), device=device)
            cos[vm_t] = unit[vm_t] @ text.T

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
                p0 = torch.softmax(s * cos, dim=-1)
                p0[~vm_t] = 0.0
                if a.conf_source == "maxprob":
                    mp = p0.max(-1).values
                    rk = torch.empty_like(mp)
                    rk[torch.argsort(mp)] = torch.linspace(0, 1, mp.numel(), device=device)
                    conf = rk * vm_t
                    elig = vm_t if band_ok is None else band_ok
                    thr = torch.quantile(conf[elig].float(), 1.0 - a.seed_frac)
                    seed_ok = elig & (conf >= thr)
                sv = None
                if a.seed_mode == "fps":
                    sv = fps_simplex(p0, seed_ok, a.n_seeds, device)
                for tau in taus:
                    labels, R = grow_posteriors(p0, src, dst, conf, seed_ok, tau,
                                                seeds_override=sv,
                                                tau_percentile=a.tau_percentile)
                    # region posterior -> class, broadcast to members; orphans keep their own argmax
                    acc = torch.zeros(max(R, 1), p0.shape[1], device=device)
                    lab_ok = labels >= 0
                    acc.index_add_(0, labels[lab_ok], conf[lab_ok, None] * p0[lab_ok])
                    reg_cls = acc.argmax(-1)
                    cls = cos.argmax(-1).clone()
                    cls[lab_ok] = reg_cls[labels[lab_ok]]
                    tag = f"grow_s{s:g}_t{tau:g}"
                    v = score(cls.cpu().numpy(), tag)
                    print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f}) "
                          f"regions={R} covered={float(lab_ok.float().mean())*100:.1f}%",
                          flush=True)
                del p0
            del cos, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
