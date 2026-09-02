"""WCCN done properly: shrinkage-regularised, and subspace-restricted.

WHY A SECOND ATTEMPT. The first pass (`run_whitening_eval.py`) whitened by the raw sample scatter and
collapsed: alpha=0.5 scored 2.36 against a base of 26.83. That is not evidence against WCCN, it is
the signature of inverting an ill-conditioned matrix -- S_W has condition number ~7e7, so
S_W^{-1/2} amplifies the smallest eigendirections by ~3500x, and those directions are estimation
noise, not signal. Whitening then hands the classifier almost pure noise. The classical remedy is
regularisation, which the first pass omitted (eps was 1e-6 * lambda_max, far too weak).

TWO WELL-CONDITIONED FORMULATIONS.

SHRINKAGE (Ledoit-Wolf form): S_gamma = (1-gamma) S_W + gamma * (tr(S_W)/D) I. gamma=1 is exactly
isotropic, so the transform becomes a scaled identity and the cosine is UNCHANGED -- that arm must
reproduce base exactly, which is a built-in correctness check rather than a wasted run.

SUBSPACE-RESTRICTED: whiten only inside the top-r principal subspace of the DATA (where the signal
demonstrably lives) and leave the orthogonal complement untouched. This never inverts a
near-null direction, so it is conditioned by construction:

    M = V_r diag(w_r^{-alpha}) V_r^T + (I - V_r V_r^T)

with V_r, w_r the leading eigenpairs of S_W restricted to the data subspace.

PRIOR. lambda-centering (+2.25) is the rank-1 first-moment version of this operation and is one of
only two components that ever helped, so the second-moment version deserves a properly conditioned
test before being written off.
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
from run_whitening_eval import within_object_scatter, SCENES


def shrink(S, gamma):
    """(1-gamma) S + gamma (tr S / D) I. gamma=1 -> isotropic -> a no-op for cosine."""
    D = S.shape[0]
    return (1.0 - gamma) * S + gamma * (torch.trace(S) / D) * torch.eye(D, device=S.device)


def whiten_map(S, alpha, gamma=0.0, rank=None):
    """Regularised S^{-alpha}. `rank` restricts the inverse to the leading eigen-subspace and leaves
    the complement as identity, so no near-null direction is ever amplified."""
    S = shrink(0.5 * (S + S.T), gamma).double()
    w, V = torch.linalg.eigh(S)
    if rank is None:
        w = w.clamp_min(1e-12 * float(w.max()))
        return (V @ torch.diag(w ** (-alpha)) @ V.T).float()
    idx = torch.argsort(w, descending=True)[:rank]
    Vr, wr = V[:, idx], w[idx].clamp_min(1e-12 * float(w.max()))
    Mr = Vr @ torch.diag(wr ** (-alpha) - 1.0) @ Vr.T          # (scale-1) inside, +I overall
    return (Mr + torch.eye(S.shape[0], dtype=S.dtype, device=S.device)).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/whitening_shrunk.json")
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
        sub_idx = torch.full((P,), -1, dtype=torch.long, device=device)
        sub_idx[vm] = torch.arange(int(vm.sum()), device=device)
        S_W = within_object_scatter(cells, sub_idx[src], sub_idx[dst])

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

            def finish(M=None):
                cf, tf = (cells, txt) if M is None else (cells @ M.T, txt @ M.T)
                cv = F.normalize(cf, dim=-1) @ F.normalize(tf, dim=-1).T
                rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
                full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * rK[None, :]
                p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
                return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                  gt_t, C, gt_pts.shape[0])[0]

            r = {"base": finish()}
            r["shrink_g1.0_a0.5"] = finish(whiten_map(S_W, 0.5, gamma=1.0))   # must equal base
            for g in (0.99, 0.9, 0.5):
                for al in (0.25, 0.5):
                    r[f"shrink_g{g:g}_a{al:g}"] = finish(whiten_map(S_W, al, gamma=g))
            for rk in (16, 64, 256):
                for al in (0.25, 0.5):
                    r[f"sub_r{rk}_a{al:g}"] = finish(whiten_map(S_W, al, rank=rk))
            row[f"top{K}"] = r
            del txt
            torch.cuda.empty_cache()
        res[scene] = row
        json.dump(res, open(a.out, "w"), indent=1)
        t100 = row.get("top100", {})
        log(f"  {scene}: base={t100.get('base', 0):.2f} "
            f"best={max((v, k) for k, v in t100.items())[1]}={max(t100.values()):.2f}")
        del raw, cen, cells, src, dst, deg, pos, S_W
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
        sanity = [abs(x["shrink_g1.0_a0.5"] - x["base"]) for x in ks]
        print(f"  [sanity] gamma=1 (isotropic) reproduces base exactly: {max(sanity) < 1e-9}")


if __name__ == "__main__":
    main()
