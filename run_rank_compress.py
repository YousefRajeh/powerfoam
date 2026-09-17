"""Idea 10e (+ its prerequisite, section 6): how many dimensions of the lifted feature field
actually carry signal, and what does a dictionary + per-cell code cost in accuracy?

THE CLAIM UNDER TEST (foam_structure_proposals.md 10e): 1.2M primitives x 512 float32 = 2.4 GB.
If the effective rank is ~R, store a shared 512 x R dictionary plus an R-dim code per cell --
~16x smaller -- and answer text queries with an R-dim dot product after projecting the text
vector once. The residual is bounded by the discarded singular values.

THE PREREQUISITE (section 6): measure the rank on a LIGHTLY regularised solve, never a heavily
ridged one, or you measure the ridge instead of the data. Ours is the streaming geometric median
(solve_geometric_median), which applies no ridge at all -- there is no lambda in this artifact to
contaminate the spectrum. That is why this pair can be measured together in one pass.

WHAT IS AND IS NOT MEASURED. The compression is EXACT in the sense that the rank-R field is the
best rank-R approximation of Phi in Frobenius norm (Eckart-Young), so the reconstruction column is
not an approximation of an approximation. The accuracy column is the real benchmark decision rule:
OpenGaussian's plain normalize->cosine->argmax over the 19-class name bank, per cell, broadcast to
GT points through the exact power-cell assignment -- the same code path every headline number in
this project uses. Nothing here is a proxy metric.

WHY UNCENTERED. The quantity 10e needs is the rank of the stored MATRIX (that is what the
dictionary has to reproduce), not the rank of its variation about the mean. A centered SVD would
need the mean vector stored alongside, which changes the storage accounting; the uncentered
spectrum is the one that matches the proposal's arithmetic. The mean's own weight shows up as the
first singular value, and is reported separately as sigma_1's energy share.

Query-side cost is reported as the dot-product width (R vs 512), not as wall clock: at this scale
the argmax is memory-bound and a timing on one machine would not transfer.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_true_facet_graph import load_points_radii
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, apply_gt_opacity_mask,
                                       calculate_metrics, classify_primitives, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES

RANKS = (4, 8, 16, 32, 64, 128, 256)
POINTCEPT = r"D:\Downloads\scannet_pointcept"


def right_basis(phi, device):
    """Right singular vectors V (512, 512) and singular values (512,) of phi (P, 512).

    Via the 512x512 Gram matrix rather than a P x 512 SVD: G = phi^T phi is symmetric PSD with
    eigenvectors V and eigenvalues sigma^2, so one eigh on a 512x512 matrix replaces an SVD on a
    matrix with up to 1.1M rows. Accumulated in chunks so peak memory is one chunk, not phi.
    """
    G = torch.zeros(phi.shape[1], phi.shape[1], dtype=torch.float64, device=device)
    for s in range(0, phi.shape[0], 200_000):
        b = phi[s:s + 200_000].to(device=device, dtype=torch.float64)
        G += b.T @ b
    evals, evecs = torch.linalg.eigh(G)          # ascending
    evals = evals.flip(0).clamp_min(0)
    evecs = evecs.flip(1)
    return evecs.float(), evals.sqrt().float()


def rank_stats(sv):
    """Spectrum summaries. `erank` is Roy & Vetterli's effective rank: exp of the Shannon entropy
    of the normalised singular values -- a continuous rank that does not need an energy cutoff."""
    s = sv.double()
    e = s ** 2
    tot = float(e.sum())
    csum = torch.cumsum(e, 0) / tot
    p = (s / s.sum()).clamp_min(1e-30)
    return {
        "erank": float(torch.exp(-(p * p.log()).sum())),
        "energy_top1": float(e[0] / tot),
        "r90": int((csum < 0.90).sum()) + 1,
        "r99": int((csum < 0.99).sum()) + 1,
        "r999": int((csum < 0.999).sum()) + 1,
    }


def score(phi, text, assigned, owned, gt_lab, n_cls, device):
    cls = classify_primitives(phi.to(device), text).cpu().numpy()
    pred = np.zeros(gt_lab.shape[0], dtype=np.int64)
    pred[owned] = cls[assigned[owned]] + 1
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pred).long(), n_cls + 1)
    return float(miou) * 100, float(macc) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", default="solved_geometric_median_nonfrozen_ogl3.pt")
    ap.add_argument("--ckpt-tmpl", default="output/scannet_{scene}_nonfrozen")
    ap.add_argument("--out", default="artifacts/rank_compress.json")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opacity-mask", action="store_true",
                    help="apply OpenGaussian's alpha<0.1 GT deletion (eval_scannet.py:127-129), "
                         "with the foam's alpha = 1 - exp(-sigma * mult * radius) proxy")
    ap.add_argument("--opacity-mult", type=float, default=2.0)
    ap.add_argument("--assign", choices=("owner", "valid"), default="owner",
                    help="'owner' (default, and what the reported rows use): the point's TRUE "
                         "power-cell owner from artifacts/ablation_cache/<scene>_<recon>_assign.npy; "
                         "a point whose owner has no feature is left UNPREDICTED and scores as a "
                         "miss. 'valid': re-run the query restricted to cells that have a feature, "
                         "i.e. nearest cell with a feature. The two differ by up to ~1.8 mIoU on a "
                         "single scene, so they are not interchangeable.")
    ap.add_argument("--recon", default="pf_tfroz", help="only used to find the assignment cache")
    args = ap.parse_args()

    enable_determinism()
    dev = args.device
    res = {}

    for scene in args.scenes:
        ck = args.ckpt_tmpl.format(scene=scene)
        sp = f"artifacts/scannet/{scene}/{args.solved}"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            print(f"[skip] {scene}")
            continue
        sd = torch.load(sp, map_location="cpu", weights_only=True)
        phi = sd["primitive_features"].float()
        vm = sd["valid_mask"].numpy()
        c, r = load_points_radii(ck)

        gt, rawl, names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SCENES[scene], scene), "segment20")
        if args.assign == "owner":
            apth = f"artifacts/ablation_cache/{scene}_{args.recon}_assign.npy"
            assigned = (np.load(apth) if os.path.exists(apth)
                        else np.asarray(assign_points_to_power_cells(gt, c, r, valid=None, k=64)))
            owned = (assigned >= 0)
            owned[owned] = vm[assigned[owned]]      # owner without a feature -> no prediction
        else:
            assigned = np.asarray(assign_points_to_power_cells(gt, c, r, valid=vm, k=64))
            owned = assigned >= 0
        n2i = {n: i for i, n in enumerate(names)}
        pres = set(np.unique(rawl).tolist())
        # The text bank must hold EXACTLY the classes the GT was remapped onto, in the same order.
        # Embedding all 19 names while remapping the GT to the present subset silently shifts every
        # class index and produces a meaningless score (measured 10.96 instead of 37.06 on
        # scene0000_00 before this was fixed). This is the same present-classes-only convention
        # backfill_surface_simplex.py and OpenGaussian's own eval use.
        names_kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
        kept = [n2i[n] for n in names_kept]
        gt_lab = remap_gt_labels(rawl, kept)
        dropped_pct = None
        if args.opacity_mask:
            # identical arithmetic to run_percell_masked.primitive_alpha: raw checkpoint values are
            # PRE-activation, so both density and radius go through softplus(beta=100) first.
            raw_ck = torch.load(os.path.join(ck, "model.pt"), map_location="cpu",
                                weights_only=False)
            sigma = torch.nn.functional.softplus(raw_ck["density"].float(), beta=100)
            rad = torch.nn.functional.softplus(raw_ck["radii"].float(), beta=100)
            alpha = (1.0 - torch.exp(-sigma * args.opacity_mult * rad)).numpy().reshape(-1)
            gt_lab, info = apply_gt_opacity_mask(gt_lab, assigned, alpha, 0.1, scene)
            dropped_pct = info["dropped_pct"]
        text = embed_class_names(names_kept, dev)

        # Basis from the OBSERVED cells only. Unobserved rows are ~0 and would otherwise pad the
        # spectrum with a mass of zero rows -- they cost nothing in Frobenius norm but they do
        # dilute every per-row statistic, and they are not what the dictionary has to represent.
        V, sv = right_basis(phi[torch.from_numpy(vm)], dev)
        st = rank_stats(sv)

        base_miou, base_macc = score(phi, text, assigned, owned, gt_lab, len(kept), dev)
        rows = []
        for R in RANKS:
            if R > phi.shape[1]:
                continue
            Vr = V[:, :R]
            code = phi.to(dev) @ Vr                        # (P, R) -- what would be stored
            approx = code @ Vr.T
            rel = float((approx - phi.to(dev)).norm() / phi.to(dev).norm())
            m, a = score(approx, text, assigned, owned, gt_lab, len(kept), dev)
            rows.append({"R": R, "miou": m, "macc": a, "rel_err": rel,
                         "bytes_ratio": (phi.shape[0] * R + 512 * R) / (phi.shape[0] * 512)})
            del code, approx
            torch.cuda.empty_cache()
            print(f"  {scene} R={R:>4}  mIoU {m:6.2f} ({m - base_miou:+5.2f})  "
                  f"relerr {rel:.4f}  {1 / rows[-1]['bytes_ratio']:.1f}x")

        res[scene] = {"cells": int(phi.shape[0]), "valid": int(vm.sum()),
                      "base_miou": base_miou, "base_macc": base_macc,
                      "spectrum": st, "ranks": rows, "dropped_pct": dropped_pct,
                      "sv_head": sv[:64].tolist()}
        print(f"{scene}: base {base_miou:.2f} mIoU | erank {st['erank']:.1f} "
              f"r99={st['r99']} top1energy={st['energy_top1']:.3f}")
        json.dump(res, open(args.out, "w"), indent=1)

    if res:
        print("\n=== mean over scenes ===")
        print(f"base mIoU {np.mean([v['base_miou'] for v in res.values()]):.2f}  "
              f"erank {np.mean([v['spectrum']['erank'] for v in res.values()]):.1f}  "
              f"r99 {np.mean([v['spectrum']['r99'] for v in res.values()]):.1f}")
        for i, R in enumerate(RANKS):
            got = [v["ranks"][i] for v in res.values() if i < len(v["ranks"])]
            if not got:
                continue
            print(f"R={R:>4}  mIoU {np.mean([g['miou'] for g in got]):6.2f}  "
                  f"relerr {np.mean([g['rel_err'] for g in got]):.4f}  "
                  f"{1 / np.mean([g['bytes_ratio'] for g in got]):.1f}x smaller")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
