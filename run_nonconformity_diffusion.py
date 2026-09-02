"""Non-conformity and bipartite evidence flow, transferred from ANoCo (CVPR 2026).

SOURCE. Seo et al., "Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy
Minimization", CVPR 2026. Two ideas there are directly transferable, plus one implementation
correction:

  (i)  NON-CONFORMITY. They "do not use the optimized features themselves -- the score is the
       MAGNITUDE OF THE UPDATE required to satisfy the constraints, reframing the graph
       Laplacian as a non-conformity operator rather than a smoothing prior." We currently
       compute p_diffused and throw away p_diffused - p0. That residual is a graph-derived
       confidence: a cell whose posterior must move a long way to agree with its neighbours is
       an isolated disagreement.

       WHY IT MAY WORK WHERE NORMLIFT RELIABILITY DID NOT. Reliability weighting was measured
       inert here because the narrow CLIP cone makes feature-space signals nearly constant
       (facet feature cosine p50 0.9872, p90 0.9995). The residual lives in POSTERIOR space,
       where softmax(1000*sim) is near one-hot and disagreement is ~sqrt(2) across a boundary
       and ~0 within a region. Same distinction that motivated the bilateral arm.

  (ii) BIPARTITE / DIRECTED FLOW. They "explicitly remove query-query and normal-normal edges
       to prevent evidence dilution." Our diffusion lets two low-confidence cells trade mass
       and mutually reinforce their errors. Restricting evidence to flow only from more
       confident to less confident cells removes that channel.

  (iii) CLOSED FORM. They solve the convex Laplacian energy exactly rather than by message
       passing. Our 60 power iterations approximate the resolvent (I + lam L)^{-1} p0; solving
       it with CG removes the iteration count as a hyperparameter.

ALL THREE STAY CONVEX, so the simplex-closure theorem still applies:
  H  per-cell alpha_i in [0,1]                      -> convex combination per row
  I  masked edge set, renormalised                  -> nonneg weights summing to 1
  J  (I + lam L)^{-1} is nonneg with unit row sums  -> a convex averaging operator

FALSIFIER, stated before running: each arm must beat plain diffusion by >= +0.5 mIoU at 19cls
on the pilot. Prior is poor -- seven convex-operation arms have already failed this bar, and
the only survivor (co-area TV) cleared it at 10cls only.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def diffuse_adaptive(p0, src, dst, deg, alpha_vec, iters=60):
    """Arm H: per-cell alpha. Cells whose posterior disagrees most with their neighbourhood
    (high non-conformity) lean harder on that neighbourhood."""
    p = p0.clone()
    av = alpha_vec[:, None]
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - av) * p0 + av * (agg / deg[:, None])
    return p


def diffuse_bipartite(p0, i, j, conf, alpha=0.9, iters=60, strict=True):
    """Arm I: evidence flows only from MORE confident to LESS confident cells.

    Each undirected facet contributes at most one directed edge. Isolated cells (no incoming
    edge) keep p0, which is the correct fallback -- they have no more-confident neighbour to
    learn from.
    """
    hi_to_lo = conf[i] > conf[j]
    s = torch.where(hi_to_lo, j, i)          # receiver = the LESS confident endpoint
    d = torch.where(hi_to_lo, i, j)          # sender   = the MORE confident endpoint
    if not strict:
        s = torch.cat([s, d]); d = torch.cat([d, s[:len(d)]])
    n = p0.shape[0]
    deg = torch.zeros(n, device=p0.device).index_add_(
        0, s, torch.ones(len(s), device=p0.device))
    has_in = deg > 0
    degc = deg.clamp_min(1.0)
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, s, p[d])
        upd = (1 - alpha) * p0 + alpha * (agg / degc[:, None])
        p = torch.where(has_in[:, None], upd, p0)
    return p


def anchored_closed_form(p0, i, j, conf, lam=0.1, w=None):
    """Arm K: ANoCo's ACTUAL closed form -- one weighted average, no iteration, exact.

    Their solve is closed-form because two things hold together: the reference nodes are
    ANCHORED (their values are fixed) and reference-reference / query-query edges are REMOVED.
    With both, the quadratic energy

        E(p) = sum_{j in N_conf(i)} w_ij ||p_i - p_j||^2 + lam ||p_i - p0_i||^2

    DECOUPLES across i, so setting the gradient to zero gives, per cell,

        p_i* = ( sum_j w_ij p_j + lam p0_i ) / ( sum_j w_ij + lam ).

    So the bipartite restriction is not an independent trick -- it is the precondition that
    makes the closed form exist. Anchoring the confident cells is what removes the coupling
    that forces plain diffusion to iterate.

    Convex by inspection: the coefficients are nonnegative and sum to 1, so simplex closure
    (simplex-vs-sphere-extension.md Thm 2) applies and no class can appear that no contributor
    supported.
    """
    hi_to_lo = conf[i] > conf[j]
    recv = torch.where(hi_to_lo, j, i)
    send = torch.where(hi_to_lo, i, j)
    n = p0.shape[0]
    ew = torch.ones(len(recv), device=p0.device) if w is None else w
    num = torch.zeros_like(p0).index_add_(0, recv, ew[:, None] * p0[send])
    den = torch.zeros(n, device=p0.device).index_add_(0, recv, ew)
    return (num + lam * p0) / (den + lam).clamp_min(1e-12)[:, None]


def resolvent_cg(p0, src, dst, deg, lam=9.0, iters=40):
    """Arm J: solve (I + lam L_rw) p = p0 exactly by CG. L_rw = I - D^-1 A.

    lam = alpha/(1-alpha) matches the fixed point of the power iteration we currently run,
    so this is the same operator without the truncation.
    """
    def A(x):
        agg = torch.zeros_like(x).index_add_(0, src, x[dst])
        return (1 + lam) * x - lam * (agg / deg[:, None])
    x = p0.clone()
    r = p0 - A(x)
    pdir = r.clone()
    rs = (r * r).sum()
    for _ in range(iters):
        Ap = A(pdir)
        denom = (pdir * Ap).sum().clamp_min(1e-20)
        al = rs / denom
        x = x + al * pdir
        r = r - al * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < 1e-7:
            break
        pdir = r + (rs_new / rs.clamp_min(1e-20)) * pdir
        rs = rs_new
    return x.clamp_min(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="how strongly non-conformity raises alpha (arm H)")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    out = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        m = torch.load(f"output/scannet_{scene}_nonfrozen/model.pt",
                       map_location="cpu", weights_only=False)
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = m["points"].shape[0]
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
        keep = src < adjacent
        ei, ej = src[keep], adjacent[keep]

        d = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                       map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy")
        owned = assign >= 0

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            p0 = torch.softmax(1000.0 * (unit @ text.T), dim=-1)
            p0[~vt] = 0.0

            pd = diffuse(p0, src, adjacent, deg)
            # ---- the non-conformity residual, ANoCo's quantity
            r = (pd - p0).abs().sum(-1) / 2.0                 # total-variation distance in [0,1]
            conf = p0.max(-1).values                          # posterior confidence

            alpha_vec = (0.9 + a.beta * (r - r[vt].median())).clamp(0.5, 0.99)
            arms = {
                "diffusion (baseline)": pd,
                "H nonconf adaptive a": diffuse_adaptive(p0, src, adjacent, deg, alpha_vec),
                "I bipartite (conf)":   diffuse_bipartite(p0, ei, ej, conf),
                "I bipartite (1-r)":    diffuse_bipartite(p0, ei, ej, 1.0 - r),
                "J closed form (CG)":   resolvent_cg(p0, src, adjacent, deg),
                "K anchored closed form": anchored_closed_form(p0, ei, ej, conf),
            }
            for tag, pr in arms.items():
                cls = pr.argmax(-1).cpu().numpy() + 1
                live = (pr.sum(-1) > 0).cpu().numpy()
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                out.setdefault((tag, cs), []).append(float(mi) * 100)
            if cs == "opengaussian19":
                print(f"  [{scene}] nonconformity r: p10={r[vt].quantile(.1):.4f} "
                      f"p50={r[vt].median():.4f} p90={r[vt].quantile(.9):.4f}", flush=True)
        print(f"[{scene}] done", flush=True)

    print(f"\n{'arm':<24}" + "".join(f"{c[11:]:>10}" for c in CLASS_SETS) + "   delta vs diffusion")
    base = {c: np.mean(out[("diffusion (baseline)", c)]) for c in CLASS_SETS}
    for tag in ["diffusion (baseline)", "H nonconf adaptive a", "I bipartite (conf)",
                "I bipartite (1-r)", "J closed form (CG)", "K anchored closed form"]:
        if (tag, CLASS_SETS[0]) not in out:
            continue
        row = "".join(f"{np.mean(out[(tag,c)]):10.2f}" for c in CLASS_SETS)
        dl = "  " + " ".join(f"{np.mean(out[(tag,c)])-base[c]:+.2f}" for c in CLASS_SETS)
        print(f"{tag:<24}{row}{dl}")


if __name__ == "__main__":
    main()
