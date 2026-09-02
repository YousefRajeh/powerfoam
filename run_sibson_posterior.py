"""Sibson natural-neighbour posterior interpolation -- the principled form of barycentric.

WHY SIBSON RATHER THAN TETRAHEDRAL BARYCENTRIC. Barycentric coordinates use the 4 vertices of
whichever Delaunay tetrahedron happens to contain x. That tetrahedron is an arbitrary choice
among the simplices meeting near x, so the weights jump discontinuously as x crosses a face,
and the 4 chosen sites need not be the 4 nearest.

Sibson's natural-neighbour coordinates instead insert x into the diagram and ask, for each
existing site j, HOW MUCH VORONOI VOLUME x STEALS FROM j:

    lam_j(x) = vol( V(x) ∩ V_old(j) ) / vol( V(x) )

These are nonnegative, sum to 1 (a convex combination -- admissible under simplex closure,
simplex-vs-sphere-extension.md Thm 2), C^1-continuous away from the sites, and they use exactly
the natural neighbours rather than an arbitrary simplex.

WHY THIS NEEDS FOAM. The weights ARE Voronoi volume ratios. They are definable only on a
bounded space partition. A Gaussian cloud has no cells to steal volume from -- one can
triangulate the means, but there is no partition for the weights to measure.

IMPLEMENTATION. Exact Sibson requires constructing V(x) per query point. We estimate the volume
ratios by Monte Carlo in a local ball around x: sample points, find which site owns each under
the POWER distance (the foam's own membership rule, so this respects radii rather than assuming
an unweighted Voronoi), and count how many would switch to x. That is a direct estimator of the
stolen-volume ratio, and its error is sampling noise rather than a modelling approximation.
The estimator is unbiased; `--samples` controls the variance and is reported.

ARMS
  hard           nearest valid cell, one owner                  (current protocol)
  bary           tetrahedral barycentric                        (previous best: +0.59 w/ diff)
  sibson         natural-neighbour coordinates
  sibson+diff    Sibson interpolation of diffused posteriors

FALSIFIER: sibson+diff must beat bary+diff by >= +0.3 mIoU at 19cls to justify the extra cost.
Sibson is strictly more expensive than barycentric, so matching it is not enough.
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

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
KNN = 24            # candidate natural neighbours per query point


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def sibson_weights(pts, centers, radii, cache, samples=64, chunk=4096, seed=0, dev="cuda"):
    """-> (idx (N,K) int64, w (N,K) float32) natural-neighbour indices and Sibson weights.

    Monte Carlo estimate of vol(V(x) ∩ V_old(j)) / vol(V(x)):
      * sample uniformly in a ball around x whose radius is the distance to the k-th neighbour;
      * a sample s belongs to V(x) iff x wins the POWER distance at s against all candidates;
      * among those, credit the site that owned s BEFORE x was inserted.
    Power distance (|s-c|^2 - r^2) is the foam's own membership rule, so radii are respected.
    """
    if os.path.exists(cache):
        z = np.load(cache)
        return torch.from_numpy(z["idx"]), torch.from_numpy(z["w"])
    from scipy.spatial import cKDTree
    t0 = time.time()
    tree = cKDTree(centers)
    dist, idx = tree.query(pts, k=KNN, workers=-1)
    N = len(pts)
    C = torch.from_numpy(centers).float().to(dev)
    R2 = torch.from_numpy(radii).float().to(dev) ** 2
    X = torch.from_numpy(np.asarray(pts, dtype=np.float32)).to(dev)
    I = torch.from_numpy(idx.astype(np.int64)).to(dev)
    rad = torch.from_numpy(dist[:, -1].astype(np.float32)).to(dev).clamp_min(1e-6)
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.zeros(N, KNN, device=dev)

    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        M = b - a
        u = torch.randn(M, samples, 3, generator=g, device=dev)
        u = u / u.norm(dim=-1, keepdim=True)
        rr = torch.rand(M, samples, 1, generator=g, device=dev) ** (1.0 / 3.0)
        S = X[a:b, None, :] + u * rr * rad[a:b, None, None]        # (M,samples,3)

        cand = C[I[a:b]]                                           # (M,K,3)
        d_cand = ((S[:, :, None, :] - cand[:, None, :, :]) ** 2).sum(-1) - R2[I[a:b]][:, None, :]
        # x is inserted with radius 0 -> its power distance is plain squared distance
        d_x = ((S - X[a:b, None, :]) ** 2).sum(-1)                 # (M,samples)
        best, arg = d_cand.min(-1)
        stolen = d_x < best                                        # sample now belongs to V(x)
        onehot = F.one_hot(arg, KNN).float() * stolen[..., None].float()
        cnt = onehot.sum(1)                                        # (M,K) stolen from each
        W[a:b] = cnt / cnt.sum(-1, keepdim=True).clamp_min(1e-12)

    idx_t, w_t = I.cpu(), W.cpu()
    np.savez_compressed(cache, idx=idx_t.numpy(), w=w_t.numpy())
    print(f"  Sibson weights: {time.time()-t0:.0f}s, "
          f"mean active neighbours {(w_t > 1e-6).float().sum(-1).mean():.2f}", flush=True)
    return idx_t, w_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        m = torch.load(f"output/scannet_{scene}_nonfrozen/model.pt",
                       map_location="cpu", weights_only=False)
        centers = m["points"].float().numpy().astype(np.float64)
        radii = F.softplus(m["radii"].float().squeeze(), beta=100).numpy().astype(np.float64)
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = centers.shape[0]
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()

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
        pts = np.asarray(gt_pts, dtype=np.float64)

        sidx, sw = sibson_weights(pts, centers, radii,
                                  f"artifacts/ablation_cache/{scene}_sibson.npz",
                                  samples=a.samples)
        sidx, sw = sidx.to(dev), sw.to(dev)
        # tetrahedral barycentric, for the paired comparison
        bz = np.load(f"artifacts/ablation_cache/{scene}_bary.npz")
        bverts = torch.from_numpy(bz["verts"]).to(dev)
        blam = torch.from_numpy(bz["lam"]).float().to(dev)
        bins = torch.from_numpy(bz["inside"]).to(dev)

        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            index = GTSurfaceIndex(pts, gt, nc)
            p0 = torch.softmax(1000.0 * (unit @ text.T), dim=-1)
            p0[~vt] = 0.0
            pd = diffuse(p0, src, adjacent, deg)

            def emit(tag, pred):
                sm = semantic_surface_metrics(index, pred)
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                res.setdefault((tag, cs), []).append(
                    (float(mi) * 100, sm.get("scd", np.nan) * 100,
                     sm.get("boundary_f1", np.nan)))

            def mix_score(tag, P, idx_t, w_t, mask):
                mix = (P[idx_t] * w_t[..., None]).sum(1)
                cls = mix.argmax(-1).cpu().numpy() + 1
                live = ((mix.sum(-1) > 1e-8) & mask).cpu().numpy()
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[live] = cls[live]
                hv = (P.sum(-1) > 0).cpu().numpy()
                hard_cls = P.argmax(-1).cpu().numpy() + 1
                fall = (~live) & owned & hv[assign.clip(0)]
                pred[fall] = hard_cls[assign[fall]]
                emit(tag, pred)

            for tag, P in (("hard", p0), ("hard+diff", pd)):
                cls = P.argmax(-1).cpu().numpy() + 1
                live = (P.sum(-1) > 0).cpu().numpy()
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                emit(tag, pred)

            allw = torch.ones(len(pts), dtype=torch.bool, device=dev)
            mix_score("bary", p0, bverts, blam, bins)
            mix_score("bary+diff", pd, bverts, blam, bins)
            mix_score("sibson", p0, sidx, sw, allw)
            mix_score("sibson+diff", pd, sidx, sw, allw)
        print(f"[{scene}] scored", flush=True)

    print(f"\n{'arm':<16}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS)
          + f"{'scd cm':>9}{'bF1':>7}{'d19 vs hard':>13}")
    base = np.mean([r[0] for r in res[("hard", CLASS_SETS[0])]])
    for tag in ("hard", "bary", "sibson", "hard+diff", "bary+diff", "sibson+diff"):
        if (tag, CLASS_SETS[0]) not in res:
            continue
        row = "".join(f"{np.mean([r[0] for r in res[(tag,c)]]):9.2f}" for c in CLASS_SETS)
        scd = np.mean([r[1] for r in res[(tag, CLASS_SETS[0])]])
        bf1 = np.mean([r[2] for r in res[(tag, CLASS_SETS[0])]])
        print(f"{tag:<16}{row}{scd:9.2f}{bf1:7.3f}"
              f"{np.mean([r[0] for r in res[(tag, CLASS_SETS[0])]]) - base:+13.2f}")


if __name__ == "__main__":
    main()
