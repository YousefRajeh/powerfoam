"""Four convex posterior operations licensed by the simplex-closure theorem.

WHY THESE AND NOT FEATURE SMOOTHING. simplex-vs-sphere-extension.md proves (Thm 1, explicit
witness) that averaging CLIP directions can produce a class NEITHER contributor supported, and
(Thm 2) that on the simplex q_c <= max_k (p_k)_c makes that impossible. So every convex
operation on POSTERIORS is admissible, while the whole family of feature-space averaging is
retired. Plain diffusion p <- (1-a) p0 + a S p is only the simplest member. These are the rest.

  A  CLASS-SIMILARITY BLUR (convolutional Wasserstein barycenter)
     Plain diffusion treats classes as exchangeable: mass moves between `chair` and `sofa` as
     readily as between `chair` and `ceiling`. A Wasserstein barycenter under a class metric
     moves mass more cheaply between SIMILAR classes. Solomon et al. 2015 show the entropic
     barycenter is alternating blur, so we alternate spatial diffusion with a class-space
     kernel K = normalize(exp(-M/eps)), M_cc' = 1 - cos(t_c, t_c'). K is row-stochastic, so
     pK stays on the simplex -- Thm 2 applies unchanged.
     NOTE: distinct from the already-refuted "Sinkhorn prior matching" (reversal #13), which
     matched class MARGINALS to a prior. This never touches marginals.

  B  SIMPLEX TV (uniform edge weights)
     min_p  1/2 ||p - p0||^2 + lam * sum_ij w_ij ||p_i - p_j||_1  s.t.  p_i in simplex.
     L1 across edges, not L2. TV's proximal operator drives small disagreements to exactly
     zero while leaving large confident jumps intact, so it should preserve boundaries where
     L2 diffusion blurs them. Solved by Chambolle-Pock with simplex projection.

  C  CONFIDENCE-WEIGHTED BARYCENTRE
     Convex weights from the neighbours' own posterior confidence (1 - normalised entropy),
     so a decisive neighbour speaks louder than an ambivalent one. Data-driven but NOT
     learned -- no parameter is fit to the target classes, per the project's no-text-side rule.

  D  FOAM-ONLY: FACET-AREA-WEIGHTED SIMPLEX TV (exact discrete co-area)
     This is the one that needs the geometry. For a bounded disjoint partition, the perimeter
     of a region is EXACTLY the sum of the shared-facet areas on its boundary, so with
     w_ij = facet area, sum_ij w_ij |u_i - u_j| is the exact discrete total variation of the
     indicator u -- the co-area formula holds with equality, not approximation. A Gaussian
     cloud has no facets and no perimeter, so this weighting is not merely worse there, it is
     UNDEFINED. Under Chan-Esedoglu-Nikolova, minimising the convex relaxation and then
     thresholding recovers the global optimum of the binary problem -- a guarantee no method
     in the baseline table can state.

     WHY P2'S NULL DOES NOT KILL THIS. Facet-area edge weights were already tested inside
     DIFFUSION and were null (+0.00 at 19cls). That is expected: the co-area identity is a
     statement about TV (an L1 functional), not about L2 diffusion, where areas carry no such
     meaning. The areas only become exact in the TV objective. If D also comes back null, the
     co-area argument loses its empirical leg and should be dropped from the paper.

FALSIFIER, stated before running: each arm must beat plain diffusion by >= +0.5 mIoU at 19cls
on the pilot to be worth a 10-scene run. Prior is poor -- P2/P4/P5 diffusion reweightings were
all null, and twelve single-scene findings have reversed here.
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

SP = (r"C:\Users\rajehyl\AppData\Local\Temp\claude"
      r"\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad")
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def project_simplex(v):
    """Euclidean projection of each row onto the probability simplex (Duchi et al.)."""
    n, C = v.shape
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, C + 1, device=v.device, dtype=v.dtype)
    cond = u - css / ind > 0
    rho = cond.float().cumsum(-1).argmax(-1)
    theta = css.gather(1, rho[:, None]) / (rho[:, None].to(v.dtype) + 1)
    return (v - theta).clamp_min(0)


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60, K=None):
    """p <- (1-a) p0 + a S p, optionally alternating with a class-space blur K (arm A)."""
    p = p0.clone()
    for t in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
        if K is not None:
            p = p @ K                        # row-stochastic: stays on the simplex
    return p


def tv_chambolle_pock(p0, i, j, w, lam=0.3, iters=200):
    """min 1/2||p-p0||^2 + lam * sum_e w_e ||p_i - p_j||_1  s.t. rows in simplex."""
    n, C = p0.shape
    E = i.shape[0]
    L = float(np.sqrt(8.0))                      # ||K|| for a graph gradient, conservative
    tau = sigma = 1.0 / L
    p = p0.clone()
    pbar = p.clone()
    y = torch.zeros(E, C, device=p0.device)
    cap = (lam * w)[:, None]
    for _ in range(iters):
        y = y + sigma * (pbar[i] - pbar[j])
        y = y.clamp(-cap, cap)                   # prox of the L1 conjugate = box projection
        div = torch.zeros_like(p).index_add_(0, i, y).index_add_(0, j, -y)
        p_new = project_simplex((p - tau * div + tau * p0) / (1 + tau))
        pbar = 2 * p_new - p
        p = p_new
    return p


def tpfa_conservative(p0, i, j, area, dist, V, iters=400, cfl=0.4):
    """Arm E, FOAM-ONLY: mass-conserving finite-volume flow on the partition.

        dp_i/dt = (1/V_i) sum_j T_ij (p_j - p_i),   T_ij = A_ij / d_ij

    Three properties, each requiring the partition:
      * stays on the simplex -- summing over classes gives sum_j T_ij (1-1) = 0, so
        sum_c p_ic is invariant; for small dt the update is a convex combination.
      * CONSERVES sum_i V_i p_ic. Plain diffusion has no conservation law, so mass drains
        from small classes into dominant ones -- our measured pathology (chair/sofa/table win
        0.00% of cells while `picture` wins 22.47%). Conservation is what protects them.
        Needs EXACT cell volumes; for overlapping Gaussians sum_i V_i double-counts, so there
        is no valid measure to conserve against.
      * TPFA is EXACT on a power diagram, because the site-to-site segment is orthogonal to
        the shared facet by construction. On a Gaussian cloud there are no facets, so T_ij is
        undefined rather than merely inaccurate.

    dt is set from a CFL condition so the update stays a convex combination (nonnegative
    coefficients), which is what keeps Theorem 2 applicable.
    """
    n = p0.shape[0]
    T = (area / dist.clamp_min(1e-12))
    outflux = torch.zeros(n, device=p0.device).index_add_(0, i, T).index_add_(0, j, T)
    dt = cfl / (outflux / V.clamp_min(1e-12)).clamp_min(1e-12).max()
    p = p0.clone()
    for _ in range(iters):
        flux = T[:, None] * (p[j] - p[i])
        div = torch.zeros_like(p).index_add_(0, i, flux).index_add_(0, j, -flux)
        p = p + dt * div / V[:, None].clamp_min(1e-12)
        p = p.clamp_min(0)
    return p


def bilateral_tpfa(p0, i, j, area, dist, kappa=0.5, alpha=0.9, iters=60, sharpen=0.0):
    """Arms F/G: BILATERAL conductance in POSTERIOR space, on the foam's exact geometry.

        w_ij = (A_ij / d_ij) * exp(-||p_i - p_j||^2 / kappa^2)

    spatial term  = transmissibility (foam-exact: facet area over site distance)
    range term    = posterior dissimilarity -> similar neighbours mix, dissimilar ones do not

    WHY THIS IS NOT A RERUN OF P4. P4 gated on FEATURE cosine and was null, because the narrow
    CLIP cone makes that gate nearly constant: facet feature cosine is p50 0.9872, p90 0.9995,
    so almost every edge reads as "similar". Posteriors after softmax(1000*sim) are near
    one-hot, so ||p_i - p_j|| is ~0 inside a region and ~sqrt(2) across a boundary. The same
    bilateral construction that was inert on features is sharply discriminative here.

    Recomputing w from the CURRENT p each iteration makes this Perona-Malik style anisotropic
    diffusion: it smooths within regions while refusing to smooth across boundaries, which is
    the "make similar more similar" half.

    sharpen > 0 adds the other half as an unsharp step, p + beta*(p - Sp), followed by simplex
    projection. Note the projection is doing real work: the unsharp term has NEGATIVE
    coefficients, so it leaves the simplex and Theorem 2 does NOT cover it -- the projection is
    what restores validity. Flagged because it means arm G is not protected by the closure
    argument the way A-F are, and it is a decisiveness move, which has lost six times here.
    """
    p = p0.clone()
    for _ in range(iters):
        dij2 = (p[i] - p[j]).pow(2).sum(-1)
        w = (area / dist.clamp_min(1e-12)) * torch.exp(-dij2 / (kappa ** 2))
        num = torch.zeros_like(p).index_add_(0, i, w[:, None] * p[j])
        num = num.index_add_(0, j, w[:, None] * p[i])
        den = torch.zeros(p.shape[0], device=p.device).index_add_(0, i, w).index_add_(0, j, w)
        agg = num / den.clamp_min(1e-12)[:, None]
        p = (1 - alpha) * p0 + alpha * agg
        if sharpen > 0:
            p = project_simplex(p + sharpen * (p - agg))
    return p


def confidence_barycentre(p0, src, dst, alpha=0.9, iters=60):
    """Arm C: neighbour weights from their own confidence (1 - normalised entropy)."""
    C = p0.shape[1]
    p = p0.clone()
    for _ in range(iters):
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1) / np.log(C)
        conf = (1.0 - ent).clamp_min(1e-6)                  # in [0,1]
        wsrc = conf[dst]
        agg = torch.zeros_like(p).index_add_(0, src, p[dst] * wsrc[:, None])
        den = torch.zeros(p.shape[0], device=p.device).index_add_(0, src, wsrc).clamp_min(1e-12)
        p = (1 - alpha) * p0 + alpha * (agg / den[:, None])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--eps", type=float, default=0.15, help="class-metric temperature (arm A)")
    ap.add_argument("--lam", type=float, default=0.3, help="TV strength (arms B, D)")
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

        # facet areas for arm D -- the exact co-area weights
        af = os.path.join(SP, f"area_{scene}_pf_nonfroz.npz")
        area_w = None
        if os.path.exists(af):
            A = np.load(af)
            bounded = ~A["unbounded"].astype(bool)
            ai, aj = A["i"][bounded].astype(np.int64), A["j"][bounded].astype(np.int64)
            aw = A["area"][bounded].astype(np.float64)
            key = {}
            for x, y_, w_ in zip(ai, aj, aw):
                key[(int(x), int(y_))] = float(w_)
            eiv, ejv = ei.cpu().numpy(), ej.cpu().numpy()
            area_w = np.array([key.get((int(x), int(y_)), 0.0) for x, y_ in zip(eiv, ejv)])
            area_w = torch.from_numpy(area_w / max(area_w.mean(), 1e-12)).float().to(dev)

        # cell volumes + site distances for the conservative flow (arm E)
        cg = os.path.join(SP, f"cellgeom_{scene}_pf_nonfroz.npz")
        Vt = (torch.from_numpy(np.load(cg)["V"].astype(np.float32)).to(dev)
              if os.path.exists(cg) else None)
        Pc = m["points"].float().to(dev)
        dist_e = (Pc[ei] - Pc[ej]).norm(dim=-1)

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

            # class-similarity kernel for arm A
            M = 1.0 - (text @ text.T).clamp(-1, 1)
            K = torch.softmax(-M / a.eps, dim=-1)

            arms = {
                "diffusion (baseline)": diffuse(p0, src, adjacent, deg),
                "A class-sim blur":     diffuse(p0, src, adjacent, deg, K=K),
                "B simplex TV":         tv_chambolle_pock(p0, ei, ej,
                                                          torch.ones_like(ei, dtype=torch.float),
                                                          lam=a.lam),
                "C confidence bary":    confidence_barycentre(p0, src, adjacent),
            }
            if area_w is not None:
                arms["D co-area TV (foam)"] = tv_chambolle_pock(p0, ei, ej, area_w, lam=a.lam)
            if area_w is not None:
                arms["F bilateral TPFA (foam)"] = bilateral_tpfa(p0, ei, ej, area_w, dist_e)
                arms["G bilateral+sharpen"] = bilateral_tpfa(p0, ei, ej, area_w, dist_e,
                                                             sharpen=0.5)
            if area_w is not None and Vt is not None:
                arms["E TPFA conserv (foam)"] = tpfa_conservative(
                    p0, ei, ej, area_w, dist_e, Vt)

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
            print(f"  [{scene} {cs[11:]}] done", flush=True)

    print(f"\n{'arm':<24}" + "".join(f"{c[11:]:>10}" for c in CLASS_SETS) + "   delta vs diffusion")
    base = {cs: np.mean(out[("diffusion (baseline)", cs)]) for cs in CLASS_SETS
            if ("diffusion (baseline)", cs) in out}
    for tag in ["diffusion (baseline)", "A class-sim blur", "B simplex TV",
                "C confidence bary", "D co-area TV (foam)", "E TPFA conserv (foam)",
                "F bilateral TPFA (foam)", "G bilateral+sharpen"]:
        if (tag, CLASS_SETS[0]) not in out:
            continue
        row = "".join(f"{np.mean(out[(tag, c)]):10.2f}" for c in CLASS_SETS)
        dl = "  " + " ".join(f"{np.mean(out[(tag,c)])-base[c]:+.2f}" for c in CLASS_SETS)
        print(f"{tag:<24}{row}{dl}")


if __name__ == "__main__":
    main()
