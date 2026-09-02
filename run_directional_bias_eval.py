"""DIRECTIONAL bias correction: centering that depends on the direction a cell was seen from.

WHY BOTH SIGNALS, NEITHER OF WHICH WORKS ALONE
----------------------------------------------
Two separate directional effects exist, and each has already failed on its own:

  * The FORESHORTENING bias is a WEIGHT effect. `A_rj = alpha*T` and a ray at angle theta to the
    dipole normal traverses ~t/cos(theta), so alpha is inflated by 1/cos at grazing incidence and
    the lift gives MOST weight to the views that sample a surface WORST. `weight_transform='cos'`
    (feature_operator.py) cancels exactly that. It corrects how much each view counts.

  * The VIEW-DEPENDENCE bias is a FEATURE effect. A chair seen only from behind yields a different
    CLIP embedding no matter how the views are weighted. No reweighting can fix it, because every
    available observation shares the bias.

  * `R` = ||sum_v w_v d_v|| / sum_v w_v, the mean resultant length of a cell's viewing directions,
    measures how concentrated that sampling was. Alone it is NOT a reliable error predictor: across
    five scenes its decile spread is +0.26, +0.19, +0.06, -0.18, -0.40 -- the sign flips, and it
    only carries signal where the scan has genuine angular diversity (it saturates above 0.9 for
    53-99% of cells depending on scene).

The synthesis: `R` and `mean_dir` say WHERE the view-dependence bias survives and WHICH direction it
points, while the cosine correction removes the weighting bias that would otherwise confound it.

    b(d)  = mean feature over cells whose mean_dir is near d,  minus the global mean
    f'_j  = normalize( f_j - lam * R_j * b(mean_dir_j) )

This generalises feature centering (which gains +2.25 by subtracting ONE global direction mu) to a
direction-CONDITIONED bias field. `R_j` gates it: a cell whose views were spread has already
averaged the direction-dependence away and should be left alone, while a single-viewpoint cell
carries the full bias. Setting b(d) = const recovers plain centering exactly, so plain centering is
the lam-weighted special case and this can only differ where direction actually matters.

b(d) is estimated by binning on the sphere and is label-free; the benchmark's bare cosine argmax is
untouched, so this remains protocol-legal.
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
from run_simplex_diffusion_eval import HARDEST_FIRST


def fibonacci_sphere(n, device):
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi)], -1)


def directional_bias_field(unit, mean_dir, weight, valid, n_bins, device, min_count=50):
    """b(d) per direction bin: the mean feature of cells viewed from near d, minus the global mean.

    Bins are Fibonacci-spaced on the sphere (near-uniform solid angle, deterministic). A bin with
    too few cells falls back to the global mean, i.e. b = 0 there, so sparsely-sampled directions
    are left uncorrected rather than corrected by noise.
    """
    B = fibonacci_sphere(n_bins, device)
    assign = (mean_dir @ B.T).argmax(-1)                 # nearest bin by cosine
    gmean = F.normalize(unit[valid].mean(0, keepdim=True), dim=-1)
    acc = torch.zeros(n_bins, unit.shape[1], device=device)
    cnt = torch.zeros(n_bins, device=device)
    w = (weight * valid.float()).clamp_min(0)
    acc.index_add_(0, assign, unit * w[:, None])
    cnt.index_add_(0, assign, w)
    ok = cnt > 0
    bmean = torch.zeros_like(acc)
    bmean[ok] = F.normalize(acc[ok] / cnt[ok][:, None], dim=-1)
    ncell = torch.zeros(n_bins, device=device).index_add_(
        0, assign, valid.float())
    b = bmean - gmean                                     # the directional offset
    b[ncell < min_count] = 0.0
    return b, assign, int((ncell >= min_count).sum()), gmean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0590_00,scene0645_00,scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lams", default="0.3,0.5,0.8")
    p.add_argument("--bins", default="32,128")
    p.add_argument("--outdir", default="artifacts/scannet/dir_bias")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    lams = [float(x) for x in a.lams.split(",")]
    binlist = [int(x) for x in a.bins.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        vg_path = f"artifacts/scannet/{scene}/view_geometry.pt"
        if not os.path.exists(vg_path):
            print(f"[skip] {scene}: run build_view_geometry.py first", flush=True); continue
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

        vg = torch.load(vg_path, map_location=device, weights_only=True)
        R = vg["R"].to(device).float()
        mean_dir = vg["mean_dir"].to(device).float()
        vw = vg["weight"].to(device).float()
        vmv = vm & (vw > 0)
        mu = F.normalize(unit[vmv].mean(0, keepdim=True), dim=-1)
        print(f"[{scene}] P={P:,}  median R={float(R[vmv].median()):.3f}", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "median_R": float(R[vmv].median()), "arms": {}}

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
                cf[vmv] = F.normalize(unit[vmv] - vec[vmv], dim=-1)
                cf[~vmv] = unit[~vmv]
                v = score((cf @ text.T).argmax(-1).cpu().numpy(), tag)
                print(f"  {cs} [{tag}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
                del cf

            b = score((unit @ text.T).argmax(-1).cpu().numpy(), "plain")
            print(f"  {cs} [plain] mIoU={b:.2f}", flush=True)
            run(0.3 * mu.expand(P, -1), "center_lam0.3")           # the incumbent
            for nb in binlist:
                bf, assign, nfull, _ = directional_bias_field(unit, mean_dir, vw, vmv, nb, device)
                if cs == CLASS_SETS[0]:
                    print(f"  [bins={nb}] {nfull}/{nb} bins populated", flush=True)
                bvec = bf[assign]
                for lam in lams:
                    run(lam * bvec, f"dir_b{nb}_lam{lam:g}")            # ungated
                    run(lam * R[:, None] * bvec, f"dirR_b{nb}_lam{lam:g}")  # gated by R
                del bf, bvec
            del text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
