"""Classical feature-space normalisation, with the supervision supplied by foam geometry.

WHY THIS, AFTER 15 FAILED ARMS. The only two components that ever helped are both normalisations of
the FEATURE GEOMETRY: lambda-centering (+2.25, subtract the global mean direction) and CSLS (+1.98,
per-class local density). Everything that failed was a DECISION RULE (priors, capacities, marginals,
mutual-NN, per-cell confidence). lambda-centering is the rank-1, first-moment case of the classical
operation every pre-deep classifier relied on: normalise the within-class scatter so that between-
class differences dominate. We never tested the second-moment version. That is the gap here.

A -- WITHIN-CLASS COVARIANCE NORMALISATION (WCCN / LDA whitening), UNSUPERVISED VIA GEOMETRY.
Classical WCCN needs labels to form S_W. Foam supplies them: cells sharing a true facet are almost
always the same object (adjacent cells agree at median cosine 0.996), so

    S_W = (1/2|E|) sum_{(i,j) in E} (f_i - f_j)(f_i - f_j)^T

is the within-object scatter, estimated from millions of pairs with no labels at all. Whitening by
S_W^{-alpha} suppresses exactly the directions along which one object's cells vary -- illumination,
viewpoint, mask-crop context -- leaving the directions that separate objects. This is foam-native:
it needs an exact, non-overlapping adjacency to guarantee the pair is one object. A Gaussian mixture
has no such predicate.

C -- ALL-BUT-THE-TOP (Mu & Viswanath, ICLR 2018). Remove the top-k principal directions of the cell
cloud rather than just the mean. lambda-centering is the soft k=1 case, so k>1 is the obvious
untested generalisation.

B -- PROCRUSTES REGISTRATION (Conneau et al. 2018 section 2.2, the half of that paper we never
implemented). Build a synthetic dictionary from mutual nearest neighbours between classes and cells,
solve orthogonal Procrustes, iterate. CAVEAT, stated up front: they had 200k anchor words and we have
<=100 classes, so a 512x512 orthogonal map is badly under-determined. Included because it is cheap
and because the failure mode (overfitting to 100 pairs) is diagnosable, not because it is expected to
win.

ALL THREE ARE APPLIED TO BOTH SIDES. The score is a cosine between a cell feature and a text
embedding, so any linear map must be applied to BOTH and both renormalised, or the comparison is
meaningless. All three change per-class scores non-uniformly, so unlike per-cell corrections they
survive rank_encode (see test_csls_paper_ideas.py).
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
from run_overnight import RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


def within_object_scatter(feats, src, dst, max_pairs=4_000_000, seed=0):
    """S_W from difference vectors across true facets. Cells sharing a facet are the same object."""
    E = src.numel()
    if E > max_pairs:
        g = torch.Generator(device=src.device).manual_seed(seed)
        sel = torch.randperm(E, generator=g, device=src.device)[:max_pairs]
        src, dst = src[sel], dst[sel]
    d = feats[src] - feats[dst]
    return (d.T @ d) / (2.0 * d.shape[0])


def inv_power(S, alpha, eps=1e-6):
    """S^{-alpha} via eigendecomposition, ridge-stabilised. alpha=0 is identity (no whitening),
    alpha=0.5 is standard whitening; intermediate values are the partial correction that has beaten
    the full one everywhere else in this project."""
    S = 0.5 * (S + S.T)
    w, V = torch.linalg.eigh(S.double())
    w = w.clamp_min(eps * float(w.max()))
    return (V @ torch.diag(w ** (-alpha)) @ V.T).float()


def all_but_the_top(feats, k):
    """Remove the top-k principal directions of the cell cloud (Mu & Viswanath 2018).
    Returns the projector so the SAME map can be applied to the text embeddings."""
    X = feats - feats.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X[torch.randperm(X.shape[0])[:200_000]], full_matrices=False)
    U = Vh[:k]                                   # (k, D)
    return torch.eye(X.shape[1], device=feats.device) - U.T @ U


def procrustes_align(cells, txt, iters=3, k=10):
    """Conneau et al. section 2.2: mutual-NN dictionary -> orthogonal Procrustes -> repeat.
    Returns W mapping TEXT into the cell-feature frame."""
    D = txt.shape[1]
    W = torch.eye(D, device=txt.device)
    sub = cells[torch.randperm(cells.shape[0])[:100_000]]
    for _ in range(iters):
        t = F.normalize(txt @ W.T, dim=-1)
        sim = t @ sub.T                                   # (C, N)
        best_cell = sim.argmax(1)                         # class -> nearest cell
        back = sim.T.argmax(1)                            # cell  -> nearest class
        cls = torch.arange(txt.shape[0], device=txt.device)
        mutual = back[best_cell] == cls                   # keep only mutual pairs
        if int(mutual.sum()) < 5:
            break
        A, B = txt[mutual], sub[best_cell[mutual]]
        U, _, Vh = torch.linalg.svd(B.T @ A, full_matrices=False)
        W = U @ Vh
    return W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/whitening.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
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
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        cells = cen[vm]
        # src/dst index the FULL P-length arrays; `cells` is only the valid subset, so the edge
        # endpoints must be remapped into subset coordinates before indexing it. (They are already
        # filtered to valid-valid pairs by `ke` above, so every endpoint has an image.)
        sub_idx = torch.full((P,), -1, dtype=torch.long, device=device)
        sub_idx[vm] = torch.arange(int(vm.sum()), device=device)
        src_v, dst_v = sub_idx[src], sub_idx[dst]
        assert int(src_v.min()) >= 0 and int(dst_v.min()) >= 0
        S_W = within_object_scatter(cells, src_v, dst_v)
        log(f"  {scene}: S_W cond {torch.linalg.cond(S_W.double()).item():.3e}")

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

            def finish(cell_f, text_f):
                cv = F.normalize(cell_f, dim=-1) @ F.normalize(text_f, dim=-1).T
                rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
                full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * rK[None, :]
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base": finish(cells, txt)}
            # ---- A: within-object whitening, partial -> full ----
            for al in (0.125, 0.25, 0.5):
                M = inv_power(S_W, al)
                r[f"A_wccn_a{al:g}"] = finish(cells @ M.T, txt @ M.T)
                del M
            # ---- C: all-but-the-top ----
            for k in (1, 2, 4, 8):
                Pk = all_but_the_top(cells, k)
                r[f"C_abtt_k{k}"] = finish(cells @ Pk.T, txt @ Pk.T)
                del Pk
            # ---- B: Procrustes registration of text onto the cell manifold ----
            Wp = procrustes_align(cells, txt)
            r["B_procrustes"] = finish(cells, txt @ Wp.T)
            del Wp
            # ---- A+C stacked: remove the dominant directions THEN whiten the residual ----
            Pk = all_but_the_top(cells, 2)
            cw = cells @ Pk.T
            M = inv_power(within_object_scatter(cw, src_v, dst_v), 0.25)
            r["AC_abtt2_then_wccn"] = finish(cw @ M.T, (txt @ Pk.T) @ M.T)
            del Pk, cw, M
            row[f"top{K}"] = r
            del txt
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " ".join(f"{k}={v:.2f}" for k, v in row.get("top100", {}).items()))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, cells, src, dst, src_v, dst_v, deg, pos, S_W
        torch.cuda.empty_cache()

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([x["base"] for x in ks])
        print(f"\n=== top{K} ({len(ks)} scenes), base {b:.2f} ===")
        rows = sorted(((np.mean([x[k] for x in ks]) - b, k,
                        sum(1 for x in ks if x[k] > x["base"])) for k in ks[0]), reverse=True)
        for d, k, w in rows:
            print(f"  {k:<22}{b+d:7.2f}  {d:+6.2f}  wins {w}/{len(ks)}")


if __name__ == "__main__":
    main()
