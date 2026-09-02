"""Reproduce the two-way confuser diagnosis on 3DGS, on BOTH datasets.

WHY (coauthor Q15.26, Q15.17). The main contribution -- prototype decorrelation -- rests entirely on
one diagnosis: failing classes score BELOW chance in a two-way contest against their single top
confuser, and those confusers are hypernyms whose prototypes sit at cosine ~0.83. That diagnosis has
only ever been measured on **PowerFoam / ScanNet++**. If it does not reproduce on 3DGS (the reference
representation for this paper) or on ScanNet (the held-out dataset), the framing is dataset- or
representation-specific and must change.

THE DECOMPOSITION. For each class, on the cells whose ground truth IS that class:
  * `acc_N`    accuracy in the real N-way task;
  * `acc_2way` accuracy when only {true class, its top confuser} are available. Chance = 50%.
  * `txtcos`   cosine between the two class prototypes, before any image evidence exists.
An information failure sits at ~50% two-way; a competition failure sits far above it. **Below 50%
means the features actively prefer the wrong class with every other class removed**, which is neither,
and is what points at prototype geometry.

ALSO ANSWERS Q15.17 (why Loewdin gives only +0.17 on the 10-class set while giving +2.01/+2.32 on
19/15): we report the MEAN PAIRWISE PROTOTYPE COSINE per class set. The hypothesis is that a small,
well-separated vocabulary is already near-orthogonal, so there is little redundancy for
decorrelation to remove. That is a directly checkable number, not an opinion.

    python run_degenerate_diag_gs.py --dataset scannetpp
    python run_degenerate_diag_gs.py --dataset scannet
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
from evaluate_point_cloud_miou import (embed_class_names, remap_gt_labels,
                                       load_scannet_pointcept_gt, OPENGAUSSIAN_CLASS_SETS)
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from run_macro_iou_gap import cell_histograms
from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign, knn_csr_safe

SPP_ART = "artifacts/scannetpp_gs"
SN_ART = "artifacts/scannet"
SPP_SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064"]


def prototype_collinearity(T):
    """Mean pairwise cosine among prototypes -- the quantity Loewdin removes."""
    G = T @ T.T
    n = G.shape[0]
    return float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))


def analyse(pred, scores, H, has_gt, names, agg):
    """Per-class N-way accuracy, two-way accuracy against the top confuser, prototype cosine."""
    cell_lab = H.argmax(1)
    m = np.flatnonzero(has_gt)
    for c in range(len(names)):
        sel = m[cell_lab[m] == c]
        if sel.size < 200:
            continue
        pc = pred[sel]
        others, cnt = np.unique(pc[pc != c], return_counts=True)
        if others.size == 0:
            continue                      # no confusions: two-way is undefined, not 0
        conf = int(others[cnt.argmax()])
        e = agg.setdefault(names[c], {"n": 0, "accN": [], "acc2": [], "conf": {}, "tcos": []})
        e["n"] += int(sel.size)
        e["accN"].append(float((pc == c).mean() * 100))
        e["acc2"].append(float((scores[sel, c] > scores[sel, conf]).mean() * 100))
        e["conf"][names[conf]] = e["conf"].get(names[conf], 0) + int((pc == conf).sum())
        e["_confidx"] = conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["scannetpp", "scannet"], default="scannetpp")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    agg, collin = {}, {}

    if a.dataset == "scannetpp":
        top, r2b = benchmark_map()
        jobs = [(s, f"{SPP_ART}/{s}/solved_weighted_gs_unfroz_ogl3.pt", None) for s in SPP_SCENES]
        class_sets = [("spp_top100", None)]
    else:
        from run_cluster_classify_eval import SCENES as SN_SCENES
        jobs = [(s, f"{SN_ART}/{s}/solved_weighted_gs_unfroz_ogl3.pt", split)
                for s, split in SN_SCENES.items()]
        class_sets = [("opengaussian19", "opengaussian19")]

    for scene, solved, split in jobs:
        if not os.path.exists(solved):
            log(f"  [miss] {scene}"); continue
        sv = torch.load(solved, map_location="cpu", weights_only=True)
        feats = sv["primitive_features"].float(); vmn = sv["valid_mask"].numpy()
        P = feats.shape[0]

        if a.dataset == "scannetpp":
            means, scales, quats = load_gaussians(scene)
            if means.shape[0] != P:
                log(f"  [skip] {scene}: P mismatch"); continue
        else:
            from plyfile import PlyData
            ck = f"D:\\Downloads\\gaussians_scannet\\{scene}"
            means = None
            for cand in (f"{ck}/point_cloud/iteration_30000/point_cloud.ply",):
                if os.path.exists(cand):
                    v = PlyData.read(cand)["vertex"]
                    means = np.stack([np.asarray(v[k]) for k in ("x","y","z")],1).astype(np.float64)
                    scales = np.exp(np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)],1))
                    quats = np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)],1)
                    quats /= np.linalg.norm(quats,axis=1,keepdims=True).clip(1e-12)
            if means is None or means.shape[0] != P:
                log(f"  [skip] {scene}: no matching PLY ({'none' if means is None else means.shape[0]} vs {P})")
                continue

        feats = feats.to(device); vm = torch.from_numpy(vmn).to(device)
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        pos = torch.from_numpy(means).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        R = raw.norm(dim=-1) * vm
        adj, off = knn_csr_safe(pos, vm, K=30)
        Dm = int((off[1:] - off[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, adj, off, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(adj, off, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, off, R
        torch.cuda.empty_cache()

        if a.dataset == "scannetpp":
            gt_pts, lab0, _ = load_gt(scene, top, r2b)
            assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
            assigned = np.where(vmn[assigned], assigned, -1)
            keepc, _, _ = coverage_filter(gt_pts, assigned, means, vmn, 20.0)
            lab = np.where(keepc, lab0, -1)
            pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
            names = [top[:100][i] for i in pres]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        else:
            gt_pts, raw_lab, all_names = load_scannet_pointcept_gt(
                f"D:\\Downloads\\scannet_pointcept\\{split}\\{scene}", "segment20")
            n2i = {n: i for i, n in enumerate(all_names)}
            present = set(np.unique(raw_lab).tolist())
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"]
                    if n2i[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
            assigned = np.where(vmn[assigned], assigned, -1)
            gt_t = torch.from_numpy(remap_gt_labels(raw_lab, tids)).long()

        C = len(names)
        if C < 3:
            log(f"  [skip] {scene}: only {C} classes"); continue
        H, _ = cell_histograms(assigned, gt_t, P, C)
        has_gt = (H.sum(1) > 0) & vmn
        txt = embed_class_names(names, device)
        collin.setdefault(class_sets[0][0], []).append(prototype_collinearity(txt))
        cv = torch.zeros(P, C, device=device); cv[vm] = cen[vm] @ txt.T
        cc = cv.clone()
        cc[vm] = cv[vm] - 0.5 * cv[vm].topk(min(CSLS_K, int(vm.sum())), dim=0).values.mean(0)
        p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
        pred = diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy()
        tcos = (txt @ txt.T).cpu().numpy()
        analyse(pred, cc.cpu().numpy(), H, has_gt, names, agg)
        for k, v in agg.items():
            if "_confidx" in v and k in names:
                ci = names.index(k)
                v["tcos"].append(float(tcos[ci, v.pop("_confidx")]))
        log(f"  {scene}: {C} classes, {int(has_gt.sum()):,} labelled primitives")
        del raw, cen, cv, cc, p0, txt, src, dst, deg, pos
        torch.cuda.empty_cache()

    rows = [(float(np.mean(v["accN"])), float(np.mean(v["acc2"])),
             float(np.mean(v["tcos"])) if v["tcos"] else float("nan"),
             v["n"], k, max(v["conf"].items(), key=lambda x: x[1])[0] if v["conf"] else "-")
            for k, v in agg.items() if v["n"] >= 500 and v["accN"]]
    rows.sort()
    print(f"\n=== 3DGS / {a.dataset} : two-way confuser decomposition ===")
    print(f"{'class':<22}{'accN':>7}{'acc2way':>9}{'txtcos':>8}{'cells':>9}  top confuser")
    for aN, a2, tc, n, k, conf in rows:
        print(f"{k:<22}{aN:>7.1f}{a2:>9.1f}{tc:>8.3f}{n:>9,}  {conf}")
    bad = [r for r in rows if r[0] < 40]
    if bad:
        print(f"\n  FAILING CLASSES (accN < 40), n={len(bad)}: mean two-way "
              f"{np.mean([r[1] for r in bad]):.1f}%   (chance = 50)")
        print("  BELOW 50 reproduces the ScanNet++/foam finding; ~50 would mean an information")
        print("  failure instead, and the prototype-geometry framing would not transfer.")
    else:
        print("\n  No class falls below 40% -- the failure mode does NOT reproduce here.")
    for cs, vals in collin.items():
        print(f"\n  mean pairwise PROTOTYPE cosine for {cs}: {np.mean(vals):.4f}")
        print("  (Q15.17: a small, already near-orthogonal vocabulary leaves little for Loewdin.)")
    json.dump({"rows": [dict(zip(("accN","acc2","tcos","n","cls","conf"), r)) for r in rows],
               "collinearity": {k: float(np.mean(v)) for k, v in collin.items()}},
              open(a.out or f"artifacts/degenerate_gs_{a.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
