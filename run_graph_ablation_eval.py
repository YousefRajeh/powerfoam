"""GRAPH ABLATION: is the "spatial-semantic" thesis actually supported by the graph we build?

WHY THIS IS THESIS-CRITICAL (coauthor Q13.16, E16.19-E16.21). The paper's stated thesis is
"calibrate scene features, calibrate the vocabulary, then perform SPATIAL-SEMANTIC consensus". But
the graph we actually build on 3DGS is `knn_csr_safe(pos, vm, K=30)` -- k-nearest neighbours over
Gaussian MEANS, Euclidean, uniform edge weights. It is **purely spatial**. There is no semantic term
anywhere in it. So either the thesis is overstated, or the graph is under-built and a semantic term
should help. Both are testable and we have never run the test.

ARMS
  `none`            no diffusion (fidelity 1.0) -- isolates what diffusion contributes at all
  `spatial`         current: kNN over positions, uniform weights
  `semantic`        kNN in CENTERED FEATURE space, uniform weights -- no spatial information at all
  `combined_gate`   spatial kNN, edge kept only if feature cosine exceeds a quantile threshold
  `combined_soft`   spatial kNN, edge weight = exp((cos_ij - 1)/T), a soft semantic gate

MATH TO VERIFY FIRST (and the reason two obvious arms are excluded):
  * `diffuse` row-normalises: `w <- w / rowsum[src]`. So multiplying EVERY edge weight by a constant
    is a provable no-op, and any "global strength" arm is vacuous. Only RELATIVE weights matter.
  * an arm that assigns equal weight to all edges of a node is identical to `spatial` regardless of
    the constant, for the same reason.

THE CIRCULARITY RISK, stated because it is the main threat to the semantic arms. Building the graph
from the same features whose class scores are then diffused can reinforce whatever CLIP already
believes, including its mistakes. The purely spatial graph is immune to this by construction, which
is a genuine argument in its favour and should be reported as such if the semantic arms do not win.
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
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign, knn_csr_safe

ART = "artifacts/scannetpp_gs"
SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


def edge_cosine(X, i, j, chunk=2_000_000):
    """Cosine on an edge list, CHUNKED.

    `X[i]` for a 34M-edge list at D=512 asks for 70 GiB; this is the second time that exact
    materialisation has OOMed in this project (see run_fisher_gate.facet_agreement). The accumulation
    is elementwise, so chunking is exact, not an approximation.
    """
    out = torch.empty(i.numel(), device=X.device)
    for s0 in range(0, i.numel(), chunk):
        e0 = min(s0 + chunk, i.numel())
        out[s0:e0] = (X[i[s0:e0]] * X[j[s0:e0]]).sum(-1)
    return out


def knn_edges_in_space(X, vm, K, device, chunk=2048):
    """kNN over an arbitrary embedding X (positions OR features). Returns (src, dst).

    Chunked because X can be (1.2M, 512): a full pairwise distance matrix is impossible, and the
    feature-space arm is exactly where that matters.
    """
    idx = torch.nonzero(vm).squeeze(1)
    Xv = X[idx]
    n = Xv.shape[0]
    srcs, dsts = [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(Xv[s:e], Xv)
        d[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = float("inf")
        nb = d.topk(min(K, n - 1), largest=False).indices          # (b, K)
        srcs.append(idx[s:e].repeat_interleave(nb.shape[1]))
        dsts.append(idx[nb.reshape(-1)])
        del d, nb
    return torch.cat(srcs), torch.cat(dsts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--out", default="artifacts/scannetpp/graph_ablation.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
    for scene in a.scenes.split(","):
        solved = f"{ART}/{scene}/solved_weighted_gs_unfroz_ogl3.pt"
        if not os.path.exists(solved):
            log(f"  [miss] {scene}"); continue
        means, scales, quats = load_gaussians(scene)
        sv = torch.load(solved, map_location="cpu", weights_only=True)
        feats = sv["primitive_features"].float()
        vmn = sv["valid_mask"].numpy()
        P = feats.shape[0]
        if means.shape[0] != P:
            log(f"  [skip] {scene} P mismatch"); continue
        feats = feats.to(device); vm = torch.from_numpy(vmn).to(device)
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        pos = torch.from_numpy(means).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        R = raw.norm(dim=-1) * vm

        # the CURRENT graph, used for feature consensus as well so every arm shares that stage
        adj, off = knn_csr_safe(pos, vm, K=a.K)
        Dm = int((off[1:] - off[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, adj, off, chunk=max(256, 200_000 // max(Dm, 1)))
        s_sp, d_sp, _ = csr_to_edges(adj, off, P, device)
        ke = vm[s_sp] & vm[d_sp]; s_sp, d_sp = s_sp[ke], d_sp[ke]
        del adj, off
        torch.cuda.empty_cache()

        # SEMANTIC graph: kNN in centered feature space, no spatial information whatsoever
        s_se, d_se = knn_edges_in_space(cen, vm, a.K, device)
        log(f"  {scene}: spatial edges {s_sp.numel():,}, semantic edges {s_se.numel():,}")

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
        assigned = np.where(vmn[assigned], assigned, -1)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, means, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        row = {}
        for K in sizes:
            pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
            if not pres: continue
            nm = [top[:K][i] for i in pres]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt = embed_class_names(nm, device); C = len(nm)
            cv = torch.zeros(P, C, device=device); cv[vm] = cen[vm] @ txt.T
            cc = cv.clone()
            cc[vm] = cv[vm] - 0.5 * cv[vm].topk(min(CSLS_K, int(vm.sum())), dim=0).values.mean(0)
            p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0

            def run(src, dst, w=None):
                deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
                    0, src, torch.ones_like(src))
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS, edge_w=w)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"none": score_pred(p0.argmax(-1).cpu().numpy(), assigned, owned,
                                    gt_t, C, gt_pts.shape[0])[0]}
            r["spatial"] = run(s_sp, d_sp)
            r["semantic"] = run(s_se, d_se)
            cos_sp = edge_cosine(cen, s_sp, d_sp)
            q = torch.quantile(cos_sp[torch.randperm(cos_sp.numel(), device=device)[:200_000]]
                               .float(), torch.tensor([0.25, 0.5], device=device))
            for nmq, thr in (("gate25", q[0]), ("gate50", q[1])):
                m = cos_sp >= thr
                r[f"combined_{nmq}"] = run(s_sp[m], d_sp[m])
            for T in (0.05, 0.2):
                r[f"combined_soft_T{T:g}"] = run(s_sp, d_sp,
                                                 torch.exp((cos_sp - 1.0) / T).clamp_min(1e-6))
            row[f"top{K}"] = r
            del txt, cv, cc, p0
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " ".join(f"{k}={v:.2f}" for k, v in row.get("top100", {}).items()))
        json.dump(res, open(a.out, "w"), indent=1)
        del raw, cen, pos, s_sp, d_sp, s_se, d_se, R
        torch.cuda.empty_cache()

    for K in sizes:
        ks = [v[f"top{K}"] for v in res.values() if f"top{K}" in v]
        if not ks: continue
        b = np.mean([x["spatial"] for x in ks])
        print(f"\n=== 3DGS top{K} ({len(ks)} scenes), current spatial graph = {b:.2f} ===")
        for d, k, w_ in sorted(((np.mean([x[k] for x in ks]) - b, k,
                                 sum(1 for x in ks if x[k] > x["spatial"])) for k in ks[0]),
                               reverse=True):
            print(f"  {k:<22}{b+d:7.2f}  {d:+6.2f}  beats spatial on {w_}/{len(ks)}")
        print("  'none' = no diffusion at all, so (spatial - none) is diffusion's true contribution.")


if __name__ == "__main__":
    main()
