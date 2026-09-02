"""Render-derived decontamination: subtract each cell's OWN leak direction, computed from M.

THE IDEA
--------
Global feature centering removes a shared direction mu estimated as the population mean, and gains
+2.25 mIoU. But the contamination in a cell's lifted feature is not a population average -- it is a
specific, computable leak. From f = M g with M = D^-1 A^T W^-1 A row-stochastic:

    f_j = M_jj g_j  +  sum_{l != j} M_jl g_l
          ---------     ----------------------
          own signal    render leak from the cells sharing j's rays

The second term is deterministic, not random: its weights come from the exact ray integral, and it
is available ONLY for a foam, where a ray decomposes into disjoint ordered segments through a
bounded partition (Gaussians have overlapping unbounded support and admit no such decomposition).

Because M is row-stochastic, sum_{l != j} M_jl = 1 - M_jj, so the leak DIRECTION is estimable from
the lifted features themselves with one sparse matmul:

    N_j  = [ (M f)_j - M_jj f_j ] / (1 - M_jj)         the M-weighted average of everything ELSE
    f'_j = normalize( f_j - lam * N_j )

WHY THIS IS THE RIGHT VERSION OF THE EARLIER FAILED TEST
--------------------------------------------------------
An earlier attempt made the centering MAGNITUDE adaptive (lam_j proportional to 1 - M_jj) while
still subtracting the GLOBAL mu. That scaled the right amount of the wrong direction, and it lost:
spearman(cone share, impurity) = +0.012, and the arm scored +1.49 against a global lambda's +2.07.
Here the DIRECTION is per-cell and render-derived, which is the part that was missing.

It also differs from the earlier deconvolution attempt in the space it acts on. That ran the
Richardson step on POSTERIORS and was null across scenes; the subsequent analysis showed operations
belong in the 512-d feature space, where per-view noise is near-orthogonal and the class-relevant
structure survives (projecting first discards ~493 dimensions and cost 1.75 mIoU). This is the same
correction applied where the signal actually lives.

ARMS
  plain                    bare cosine argmax, benchmark rule untouched
  global_lam{lam}          f - lam * mu            (population mean; the +2.25 incumbent)
  leak_lam{lam}            f - lam * N_j           (render-derived per-cell direction)
  leakimp_lam{lam}         f - lam * (1-M_jj) * N_j  (direction AND magnitude from M)
  both_lam{a}_{b}          f - a*mu - b*N_j        (do they remove different things?)
"""
import argparse
import glob
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
from run_simplex_diffusion_eval import HARDEST_FIRST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0590_00,scene0645_00,scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lams", default="0.1,0.2,0.3,0.5")
    p.add_argument("--max-edges", type=int, default=140_000_000)
    p.add_argument("--outdir", default="artifacts/scannet/render_decontam")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    lams = [float(x) for x in a.lams.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"

        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved
        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)

        # ---- M and the per-cell leak direction
        cache = torch.load(sorted(glob.glob(f"{art}/gram_cache_*.pt"))[0],
                           map_location="cpu", weights_only=False)
        keys, vals = cache["S_keys"], cache["S_vals"].float(); del cache
        if keys.numel() > a.max_edges:
            from gram_blocks import prune_edges
            keys, vals, _ = prune_edges(keys, vals, P, a.max_edges, verbose=False)
        idx = torch.stack([keys // P, keys % P]).to(device); del keys
        S = torch.sparse_coo_tensor(idx, vals.to(device), (P, P), device=device).coalesce()
        del idx, vals
        rowsum = torch.sparse.mm(S, torch.ones(P, 1, device=device)).squeeze(1)
        inv = torch.where(rowsum > 0, 1.0 / rowsum.clamp_min(1e-30), torch.zeros_like(rowsum))
        Sd = torch.zeros(P, device=device)
        si = S.indices(); sv = S.values(); dm = si[0] == si[1]
        Sd[si[0][dm]] = sv[dm]
        purity = Sd * inv                                     # M_jj
        Mf = torch.sparse.mm(S, unit) * inv[:, None]          # (M f)_j
        del S
        offw = (1.0 - purity).clamp_min(1e-6)
        leak = (Mf - purity[:, None] * unit) / offw[:, None]  # N_j
        leak = F.normalize(leak, dim=-1)
        leak[~vm] = 0.0
        del Mf

        align = float((leak[vm] * mu).sum(-1).mean())
        selfal = float((leak[vm] * unit[vm]).sum(-1).mean())
        print(f"[{scene}] purity median={float(purity[vm].median()):.4f}", flush=True)
        print(f"  cos(leak_j, global mu)  = {align:+.4f}   <- is the leak just the population mean?",
              flush=True)
        print(f"  cos(leak_j, f_j)        = {selfal:+.4f}   <- how much it already agrees with the cell",
              flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "purity_median": float(purity[vm].median()),
               "cos_leak_mu": align, "cos_leak_self": selfal, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            def run(vec, tag):
                cf = torch.zeros_like(unit)
                cf[vm] = F.normalize(unit[vm] - vec[vm], dim=-1)
                v = score((cf @ text.T).argmax(-1).cpu().numpy(), tag)
                print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
                del cf

            b = score((unit @ text.T).argmax(-1).cpu().numpy(), "plain")
            print(f"  {cs} [plain] mIoU={b:.2f}", flush=True)
            for lam in lams:
                run(lam * mu.expand(P, -1), f"global_lam{lam:g}")
                run(lam * leak, f"leak_lam{lam:g}")
                run(lam * (1 - purity)[:, None] * leak, f"leakimp_lam{lam:g}")
            run(0.3 * mu.expand(P, -1) + 0.2 * leak, "both_0.3mu_0.2leak")
            del text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
