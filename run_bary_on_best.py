"""Stack barycentric assignment onto our ACTUAL best pipeline: mode-vote + diffusion.

WHY THIS AND NOT THE EARLIER TEST. `run_barycentric_posterior.py` measured barycentric against
plain diffusion and got +0.59 at 19cls over 10 scenes. But plain diffusion is not our best --
mode-vote(true facet) + diffusion is (38.95/41.76/49.54). Since Theorem 1 says
mIoU = Phi(a, S, L) and barycentric moves `a` while mode-vote and diffusion both move `L`, the
three should compose rather than compete. This tests that directly.

FULL LADDER, so each increment is attributable:
    percell                          bare per-primitive argmax
    modevote                         + NormLift reliability-guided mode voting
    modevote+diff                    + posterior simplex diffusion   <- current best
    bary+modevote+diff               + barycentric assignment        <- the question

Also reports bary applied to the weaker bases, so a gain here can be checked against the
+0.59 already measured over plain diffusion.

Surface metrics accompany every arm: barycentric interpolates across cell boundaries, and a
3-scene intermediate of the earlier run showed scd degrading before that reversed at 10 scenes.
Reporting scd/bF1 alongside guards against buying mIoU with boundary placement.

FALSIFIER, stated before running: the barycentric increment on top of modevote+diff must be
>= +0.3 mIoU at 19cls over 10 scenes. Lower than the usual +0.5 bar because the increment is
being asked on top of an already-strong stack, but it must be clearly positive to claim the
three levers compose.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from run_normlift_refine_eval import mode_vote_refine

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--bary", choices=["plain", "regular"], default="regular",
                    help="which triangulation the barycentric coordinates come from. "
                         "`plain` = scipy Delaunay of the sites, which IGNORES the radii. "
                         "`regular` = the weighted Delaunay dual to the POWER diagram, built "
                         "by lifting to 4D with w_i = |x_i|^2 - r_i^2. Measured: 31.3% of GT "
                         "points fall in a DIFFERENT tetrahedron, so this is not cosmetic -- "
                         "and only the regular one is a statement about the partition rather "
                         "than about a point cloud.")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        bz = (f"artifacts/ablation_cache/{scene}_bary_regular.npz" if a.bary == "regular"
              else f"artifacts/ablation_cache/{scene}_bary.npz")
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if not all(os.path.exists(p) for p in (mp, fp, stp, bz, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        n_prim = P.shape[0]
        # TRUE FACET GRAPH -- the exact dual of the power diagram. model.pt's `adjacency` is
        # the renderer's structure and is NOT the same graph; using it understated every arm
        # (measured: modevote+diff read 37.79 here vs 38.95 in the vault).
        tf = f"artifacts/scannet/{scene}/adjacency_true_facet.pt"
        if os.path.exists(tf):
            g = torch.load(tf, map_location="cpu", weights_only=True)
            adjacent = g["adjacent"].long().to(dev)
            offsets = g["offsets"].long().to(dev)
        else:
            print(f"  [warn] {scene}: no true-facet graph, falling back to model.pt", flush=True)
            adjacent = m["adjacency"].long().to(dev)
            offsets = m["adjacency_offsets"].long().to(dev)
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        refined = mode_vote_refine(unit, R, P, adjacent, offsets)

        z = np.load(bz)
        bverts = torch.from_numpy(z["verts"]).to(dev)
        blam = torch.from_numpy(z["lam"]).float().to(dev)
        bins = torch.from_numpy(z["inside"]).to(dev)

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        pts = np.asarray(gt_pts, dtype=np.float64)
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            index = GTSurfaceIndex(pts, gt, nc)

            def emit(tag, pred):
                sm = semantic_surface_metrics(index, pred)
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                res.setdefault((tag, cs), []).append(
                    (float(mi) * 100, sm.get("scd", np.nan) * 100,
                     sm.get("boundary_f1", np.nan)))

            def hard(tag, P_):
                cls = P_.argmax(-1).cpu().numpy() + 1
                live = (P_.sum(-1) > 0).cpu().numpy()
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[sc] = cls[assign[sc]]
                emit(tag, pr)

            def bary(tag, P_):
                mix = (P_[bverts] * blam[..., None]).sum(1)
                cls = mix.argmax(-1).cpu().numpy() + 1
                live = ((mix.sum(-1) > 1e-8) & bins).cpu().numpy()
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[live] = cls[live]
                # outside the hull -> fall back to the hard assignment, never extrapolate
                hv = (P_.sum(-1) > 0).cpu().numpy()
                hc = P_.argmax(-1).cpu().numpy() + 1
                fall = (~live) & owned & hv[assign.clip(0)]
                pr[fall] = hc[assign[fall]]
                emit(tag, pr)

            for base_tag, u in (("percell", unit), ("modevote", refined)):
                simb = u @ text.T
                p0 = torch.softmax(1000.0 * simb, dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, src, adjacent, deg)
                hard(base_tag, p0)
                hard(base_tag + "+diff", pd)
                bary("bary+" + base_tag, p0)
                bary("bary+" + base_tag + "+diff", pd)
        print(f"[{scene}] {(time.time()-t0)/60:.1f} min", flush=True)

    order = ["percell", "bary+percell", "percell+diff", "bary+percell+diff",
             "modevote", "bary+modevote", "modevote+diff", "bary+modevote+diff"]
    n = len(res.get((order[0], CLASS_SETS[0]), []))
    print(f"\n=== {n} scenes ===")
    print(f"{'arm':<22}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS)
          + f"{'scd cm':>9}{'bF1':>7}")
    for tag in order:
        if (tag, CLASS_SETS[0]) not in res:
            continue
        row = "".join(f"{np.mean([r[0] for r in res[(tag,c)]]):9.2f}" for c in CLASS_SETS)
        scd = np.mean([r[1] for r in res[(tag, CLASS_SETS[0])]])
        bf1 = np.mean([r[2] for r in res[(tag, CLASS_SETS[0])]])
        print(f"{tag:<22}{row}{scd:9.2f}{bf1:7.3f}")

    print("\n=== barycentric increment, per base ===")
    for base in ("percell", "percell+diff", "modevote", "modevote+diff"):
        b, bb = ("bary+" + base, base)
        if (b, CLASS_SETS[0]) not in res:
            continue
        dl = " ".join(f"{np.mean([r[0] for r in res[(b,c)]]) - np.mean([r[0] for r in res[(bb,c)]]):+.2f}"
                      for c in CLASS_SETS)
        # per-scene sign count at 19cls
        pa = [r[0] for r in res[(b, CLASS_SETS[0])]]
        pb = [r[0] for r in res[(bb, CLASS_SETS[0])]]
        pos = sum(1 for x, y in zip(pa, pb) if x > y)
        print(f"  on {bb:<16}{dl}   positive on {pos}/{len(pa)} scenes (19cls)")


if __name__ == "__main__":
    main()
