"""Idea 10b: soft cell assignment for feature queries.

THE PROPOSAL. A hard per-cell feature makes feature(x) = Phi[owner(x)] piecewise constant, so every
segment boundary is a Voronoi face -- quantised to cell size. Replace the hard lookup with a softmax
over the owning cell and its power-adjacency neighbours, weighted by the same power distances the
traversal already computes:

    feature(x) = sum_j softmax_j(-beta * d_j(x)) * Phi_j ,   d_j = ||x - c_j||^2 - r_j^2,
                                                            j in {owner} u N(owner)

beta -> infinity recovers the current hard behaviour exactly, so this is a strict generalisation
with one knob, and beta = 0 is uniform averaging over the neighbourhood.

WHY THIS IS FOAM-SPECIFIC. Both ingredients are stored, not constructed: the power distance is the
same formula the ray-cell traversal kernel uses for cell membership, and N(owner) is the regular
triangulation adjacency saved in model.pt. A Gaussian checkpoint has neither -- no disjoint cell to
own a point and no contact relation between primitives.

BETA IS SCALE-FREE HERE. d has units of length^2, so a bare beta would mean something different in
every scene. beta is therefore reported in units of 1/g, where g is the scene's MEDIAN
runner-up gap d_second - d_owner over GT points: beta_rel = 1 means the runner-up cell is
downweighted by e^-1 at a typical point. Softmax is invariant to a constant shift, so subtracting
d_owner before exponentiating is exact, not an approximation.

WHAT IT IS COMPARED AGAINST. The hard row is the same per-cell argmax protocol every headline
number uses (reproduced to the digit against the stored DB value in run_rank_compress.py), so the
delta isolates the assignment rule and nothing else.

TWO VARIANTS, because the proposal does not say which side to blend on:
  feature  -- blend Phi, then normalise, then cosine-argmax (the literal reading)
  label    -- softmax the per-cell class scores first, then blend on the simplex (what the
              diffusion arms in this project do)

AND A RESULT THAT FELL OUT OF WRITING THEM. Blending features and blending RAW cosine scores are
the same decision: <normalize(sum_j w_j phi_j), t> = (1/Z) sum_j w_j <phi_j, t> with Z > 0 the same
scalar for every class, so the argmax is identical. Measured, before the label arm was changed to
softmax: the two columns agreed to the digit at all nine betas on scene0062_00. So "blend on the
CLIP sphere or on the class scores" is NOT a real choice -- the only thing that makes the two arms
differ is the softmax NONLINEARITY, which is why the label arm below applies one (temperature 1000,
the same value the diffusion arms use) rather than blending raw cosines.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, apply_gt_opacity_mask,
                                       calculate_metrics, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES

POINTCEPT = r"D:\Downloads\scannet_pointcept"
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def neighbourhood_pairs(assign_owned, adjacency, offsets, device):
    """CSR neighbourhood of each owner, flattened to (point_idx, cell_idx) pairs INCLUDING the
    owner itself. Returned as two 1-D tensors so the blend is one index_add per chunk and never
    materialises a (Q, max_degree, 512) tensor."""
    owner = torch.as_tensor(assign_owned, device=device, dtype=torch.long)
    deg = (offsets[owner + 1] - offsets[owner])
    total = int(deg.sum())
    pt = torch.repeat_interleave(torch.arange(owner.numel(), device=device), deg)
    # position within each owner's neighbour run
    run_start = torch.cumsum(deg, 0) - deg
    within = torch.arange(total, device=device) - run_start[pt]
    cell = adjacency[offsets[owner][pt] + within]
    own_pt = torch.arange(owner.numel(), device=device)
    return torch.cat([pt, own_pt]), torch.cat([cell, owner])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", default="solved_geometric_median_nonfrozen_ogl3.pt")
    ap.add_argument("--model-tmpl", default="output/scannet_{scene}_nonfrozen/model.pt")
    ap.add_argument("--out", default="artifacts/soft_assign.json")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opacity-mask", action="store_true")
    ap.add_argument("--opacity-mult", type=float, default=2.0)
    ap.add_argument("--assign", choices=("owner", "valid"), default="owner",
                    help="see run_rank_compress.py; 'owner' is the protocol the reported rows use")
    ap.add_argument("--recon", default="pf_tfroz")
    args = ap.parse_args()

    enable_determinism()
    dev = args.device
    res = {}

    for scene in args.scenes:
        mp = args.model_tmpl.format(scene=scene)
        sp = f"artifacts/scannet/{scene}/{args.solved}"
        if not (os.path.exists(mp) and os.path.exists(sp)):
            print(f"[skip] {scene}")
            continue
        m = torch.load(mp, map_location="cpu", weights_only=False)
        centers = m["points"].float()
        radii = F.softplus(m["radii"].float().squeeze(), beta=100)
        adjacency = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)

        d = torch.load(sp, map_location="cpu", weights_only=True)
        phi = d["primitive_features"].float().to(dev)
        vm = d["valid_mask"].numpy()
        vt = torch.from_numpy(vm).to(dev)
        unit = torch.zeros_like(phi)
        unit[vt] = F.normalize(phi[vt], dim=-1)

        gt, rawl, names_all = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SCENES[scene], scene), "segment20")
        if args.assign == "owner":
            # The point's TRUE owner cell, exactly as run_percell_masked.py reads it. A point whose
            # owner carries no feature stays UNPREDICTED for the hard row -- and for the soft rows
            # too, so the comparison is paired and the delta measures the blend, not extra coverage.
            apth = f"artifacts/ablation_cache/{scene}_{args.recon}_assign.npy"
            assign = (np.load(apth) if os.path.exists(apth)
                      else np.asarray(assign_points_to_power_cells(
                          gt, centers.numpy().astype(np.float64),
                          radii.numpy().astype(np.float64), valid=None, k=64)))
            owned = (assign >= 0)
            owned[owned] = vm[assign[owned]]
        else:
            assign = np.asarray(assign_points_to_power_cells(
                gt, centers.numpy().astype(np.float64),
                radii.numpy().astype(np.float64), valid=vm, k=64))
            owned = assign >= 0
        n2i = {n: i for i, n in enumerate(names_all)}
        pres = set(np.unique(rawl).tolist())
        names_kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
        gt_lab = remap_gt_labels(rawl, [n2i[n] for n in names_kept])
        dropped_pct = None
        if args.opacity_mask:
            # same arithmetic as run_percell_masked.primitive_alpha (pre-activation checkpoint
            # values through softplus(beta=100)); OpenGaussian's 0.1 threshold is untouched.
            sigma = F.softplus(m["density"].float(), beta=100)
            alpha = (1.0 - torch.exp(-sigma * args.opacity_mult
                                     * radii.reshape(-1))).numpy().reshape(-1)
            gt_lab, info = apply_gt_opacity_mask(gt_lab, assign, alpha, 0.1, scene)
            dropped_pct = info["dropped_pct"]
        text = embed_class_names(names_kept, dev)
        nc = len(names_kept) + 1

        x = torch.as_tensor(np.asarray(gt)[owned], device=dev, dtype=torch.float32)
        pt_i, cell_j = neighbourhood_pairs(assign[owned], adjacency, offsets, dev)
        C = centers.to(dev)
        R2 = (radii.to(dev) ** 2)
        dj = ((x[pt_i] - C[cell_j]) ** 2).sum(-1) - R2[cell_j]
        d_own = ((x - C[torch.as_tensor(assign[owned], device=dev)]) ** 2).sum(-1) \
            - R2[torch.as_tensor(assign[owned], device=dev)]
        gap = dj - d_own[pt_i]
        # unobserved cells hold no feature; they must not receive weight
        gap = torch.where(vt[cell_j], gap, torch.full_like(gap, float("inf")))
        pos = gap[gap > 0]
        g_med = float(pos.median()) if pos.numel() else 1.0

        # HARD reference: the owner alone.
        cls_hard = (unit @ text.T).argmax(-1).cpu().numpy() + 1
        pred = np.zeros(gt_lab.shape[0], dtype=np.int64)
        pred[owned] = cls_hard[assign[owned]]
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                            torch.from_numpy(pred).long(), nc)
        base = (float(miou) * 100, float(macc) * 100)
        print(f"{scene}: hard {base[0]:.2f} mIoU  (gap median {g_med:.2e}, "
              f"{pt_i.numel() / x.shape[0]:.1f} cells/point)")

        # Softmax'd per-cell scores: the nonlinearity is the ONLY thing that distinguishes the
        # label-space blend from the feature-space one (see module docstring). Cells with no
        # feature must contribute nothing, so their row is zeroed rather than left at uniform.
        sim_cell = torch.softmax(1000.0 * (unit @ text.T), dim=-1)     # (P, K)
        sim_cell[~vt] = 0.0
        rows = []
        for b in BETAS:
            w = torch.exp(-(b / g_med) * gap)
            w = torch.where(torch.isfinite(gap), w, torch.zeros_like(w))
            denom = torch.zeros(x.shape[0], device=dev).index_add_(0, pt_i, w)
            out = {}
            # feature-space blend
            fb = torch.zeros(x.shape[0], phi.shape[1], device=dev)
            for s in range(0, pt_i.numel(), 4_000_000):
                sl = slice(s, s + 4_000_000)
                fb.index_add_(0, pt_i[sl], w[sl, None] * unit[cell_j[sl]])
            fb = fb / denom[:, None].clamp_min(1e-12)
            c = F.normalize(fb, dim=-1) @ text.T
            for tag, cls in (("feature", c.argmax(-1)),):
                p = np.zeros(gt_lab.shape[0], dtype=np.int64)
                p[owned] = cls.cpu().numpy() + 1
                _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                                 torch.from_numpy(p).long(), nc)
                out[tag] = (float(mi) * 100, float(ma) * 100)
            del fb, c
            # label-space blend (scores on the simplex)
            lb = torch.zeros(x.shape[0], len(names_kept), device=dev)
            for s in range(0, pt_i.numel(), 4_000_000):
                sl = slice(s, s + 4_000_000)
                lb.index_add_(0, pt_i[sl], w[sl, None] * sim_cell[cell_j[sl]])
            p = np.zeros(gt_lab.shape[0], dtype=np.int64)
            p[owned] = (lb / denom[:, None].clamp_min(1e-12)).argmax(-1).cpu().numpy() + 1
            _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                             torch.from_numpy(p).long(), nc)
            out["label"] = (float(mi) * 100, float(ma) * 100)
            del lb
            torch.cuda.empty_cache()
            rows.append({"beta_rel": b, "feature_miou": out["feature"][0],
                         "feature_macc": out["feature"][1],
                         "label_miou": out["label"][0], "label_macc": out["label"][1]})
            print(f"  beta_rel={b:<5} feature {out['feature'][0]:6.2f} "
                  f"({out['feature'][0] - base[0]:+5.2f})   "
                  f"label {out['label'][0]:6.2f} ({out['label'][0] - base[0]:+5.2f})")

        res[scene] = {"hard_miou": base[0], "hard_macc": base[1], "gap_median": g_med,
                      "dropped_pct": dropped_pct,
                      "cells_per_point": float(pt_i.numel() / x.shape[0]), "betas": rows}
        json.dump(res, open(args.out, "w"), indent=1)

    if res:
        print("\n=== mean over scenes ===")
        print(f"hard {np.mean([v['hard_miou'] for v in res.values()]):.2f}")
        for i, b in enumerate(BETAS):
            f = np.mean([v["betas"][i]["feature_miou"] for v in res.values()])
            l = np.mean([v["betas"][i]["label_miou"] for v in res.values()])
            print(f"beta_rel={b:<5} feature {f:6.2f}  label {l:6.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
