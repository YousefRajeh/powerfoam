"""Barycentric posterior interpolation on the Delaunay: attack the ASSIGNMENT lever.

THE IDEA (user's). Today every GT point is hard-assigned to exactly ONE cell and its position
inside that cell is discarded: two points in the same cell, one at the centre and one hugging a
facet, get identical predictions. But the Delaunay triangulation -- which is the exact dual of
the power diagram and is maintained by the model itself -- tiles space with simplices whose
vertices are sites. A point x inside a tetrahedron has barycentric coordinates lam_1..lam_4,
nonnegative and summing to 1, so

    p(x) = sum_k lam_k p_k

is a CONVEX COMBINATION of posteriors, admissible under the simplex-closure theorem
(simplex-vs-sphere-extension.md Thm 2: q_c <= max_k (p_k)_c, so no class can appear that no
vertex supported). The result is a continuous posterior field rather than a piecewise-constant
one.

WHY THIS IS THE RIGHT LEVER. mIoU = Phi(a, S, L) exactly (mIoU-levers-proof.md Thm 1), so the
levers are assignment, coverage, labelling. Every convex posterior operation tried so far --
diffusion, class-similarity blur, TV, confidence weighting -- acts on L. This one acts on `a`,
which was measured to be worth ~4 mIoU (nearest-valid 46.25 vs geometric 42.47) and which no
other experiment here has targeted.

WHY IT NEEDS FOAM. The Delaunay is the dual of a space partition. For a Gaussian cloud one can
still triangulate the means, but the result is dual to nothing: measured, at gsplat's own
3-sigma bound the alpha complex over Gaussian means has mean degree 0.05 while every scene
point lies inside ~14-20 splats. There is no partition for the triangulation to be the dual of,
so the barycentric weights would interpolate between primitives that overlap rather than tile.

ARMS
  hard          current protocol: nearest valid cell, one owner        (baseline)
  bary          barycentric interpolation over the containing tetrahedron
  bary+diff     barycentric interpolation of already-diffused posteriors
  bary_clamped  barycentric, but only where all 4 vertices carry features

FALSIFIER, stated before running: bary must beat hard by >= +0.5 mIoU at 19cls. A plausible
failure mode is stated up front: interpolating ACROSS a semantic boundary is exactly what
hard assignment avoids, so this may smear object edges -- the surface metrics (scd, boundary
F1) are reported alongside mIoU to catch that.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay

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


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def barycentric(pts, centers, cache):
    """-> (simplex_vertex_idx (N,4) int64, lam (N,4) float64, inside (N,) bool).

    scipy's Delaunay.find_simplex returns -1 outside the convex hull; those points keep the
    hard assignment rather than being extrapolated (extrapolated barycentric coordinates are
    negative, which would leave the simplex and break Theorem 2).
    """
    if os.path.exists(cache):
        z = np.load(cache)
        return z["verts"], z["lam"], z["inside"]
    t0 = time.time()
    tri = Delaunay(centers)
    print(f"  Delaunay on {len(centers):,} sites: {time.time()-t0:.0f}s, "
          f"{tri.nsimplex:,} simplices", flush=True)
    s = tri.find_simplex(pts)
    inside = s >= 0
    verts = np.zeros((len(pts), 4), dtype=np.int64)
    lam = np.zeros((len(pts), 4), dtype=np.float64)
    si = s[inside]
    T = tri.transform[si]
    d = pts[inside] - T[:, 3]
    b = np.einsum("nij,nj->ni", T[:, :3], d)
    lam[inside] = np.concatenate([b, 1 - b.sum(1, keepdims=True)], axis=1)
    verts[inside] = tri.simplices[si]
    np.savez_compressed(cache, verts=verts, lam=lam, inside=inside)
    return verts, lam, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        m = torch.load(f"output/scannet_{scene}_nonfrozen/model.pt",
                       map_location="cpu", weights_only=False)
        centers = m["points"].float().numpy().astype(np.float64)
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
        verts, lam, inside = barycentric(
            pts, centers, f"artifacts/ablation_cache/{scene}_bary.npz")
        print(f"  [{scene}] {100*inside.mean():.1f}% of GT points inside the convex hull",
              flush=True)
        vt_t = torch.from_numpy(verts).to(dev)
        lam_t = torch.from_numpy(lam).float().to(dev)
        ins_t = torch.from_numpy(inside).to(dev)
        allvalid = torch.from_numpy(valid[verts].all(1) & inside).to(dev)

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

            def score(tag, pred):
                sm = semantic_surface_metrics(index, pred)
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                res.setdefault((tag, cs), []).append(
                    (float(mi) * 100, sm.get("scd", np.nan) * 100,
                     sm.get("boundary_f1", np.nan)))

            # hard assignment (current protocol)
            for tag, P in (("hard", p0), ("hard+diff", pd)):
                cls = P.argmax(-1).cpu().numpy() + 1
                live = (P.sum(-1) > 0).cpu().numpy()
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[sc] = cls[assign[sc]]
                score(tag, pr)

            # barycentric: p(x) = sum_k lam_k p_k  -- convex, stays on the simplex
            for tag, P, mask in (("bary", p0, ins_t), ("bary+diff", pd, ins_t),
                                 ("bary_clamped", p0, allvalid)):
                mix = (P[vt_t] * lam_t[..., None]).sum(1)          # (N, C)
                cls = mix.argmax(-1).cpu().numpy() + 1
                livem = (mix.sum(-1) > 1e-8).cpu().numpy() & mask.cpu().numpy()
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[livem] = cls[livem]
                # points outside the hull keep the hard assignment
                fall = (~livem) & owned
                hard_cls = P.argmax(-1).cpu().numpy() + 1
                hv = (P.sum(-1) > 0).cpu().numpy()
                fall &= hv[assign.clip(0)]
                pr[fall] = hard_cls[assign[fall]]
                score(tag, pr)
        print(f"[{scene}] scored", flush=True)

    print(f"\n{'arm':<16}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS)
          + f"{'scd cm':>9}{'bF1':>7}   delta19")
    base = np.mean([r[0] for r in res[("hard", CLASS_SETS[0])]])
    for tag in ("hard", "bary", "bary_clamped", "hard+diff", "bary+diff"):
        if (tag, CLASS_SETS[0]) not in res:
            continue
        row = "".join(f"{np.mean([r[0] for r in res[(tag,c)]]):9.2f}" for c in CLASS_SETS)
        scd = np.mean([r[1] for r in res[(tag, CLASS_SETS[0])]])
        bf1 = np.mean([r[2] for r in res[(tag, CLASS_SETS[0])]])
        d19 = np.mean([r[0] for r in res[(tag, CLASS_SETS[0])]]) - base
        print(f"{tag:<16}{row}{scd:9.2f}{bf1:7.3f}{d19:+10.2f}")


if __name__ == "__main__":
    main()
