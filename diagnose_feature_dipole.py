"""FEATURE DIPOLE (user's idea): grow segments on dipole macro-geometry AND features jointly.

WHERE THIS COMES FROM. Geometry-only coplanar segments beat the per-cell baseline by
+1.90/+1.86/+0.29 mIoU (19/15/10cls, scene0347_00, best tau). Feature-only k-means pooling
is WORSE than per-cell at 19cls (39.4 vs 42.47). The user's proposal is to use the dipole as
a proxy for macro-scale SEGMENTED geometry and combine it with the features, rather than
choosing one.

THREE PREDICATES, same non-iterative connected-components pass on the true facet graph, so
the only thing that varies is which edges survive:
  geometry  |cos(n_i,n_j)| > tau_n  and  plane-offset < tau_d
  feature   cos(f_i, f_j) > tau_f
  joint     both of the above

WHY JOINT SHOULD BEAT EITHER. The two signals fail in different places. Geometry merges
across a semantic boundary whenever two different objects share a plane (a picture flush on
a wall, a door in a wall, objects resting on a table) -- geometry cannot see the label.
Features merge across a geometric boundary because lifted CLIP over-smooths spatially
(same-label 0.994 vs boundary 0.985 median cosine) and because the CLIP cone is narrow
(median best-centroid cosine 0.9969). Requiring BOTH should cut each other's false merges.

WHY IT MIGHT NOT. Requiring both only ever REMOVES edges, so joint segments are strictly
finer than geometry-only. Finer segmentation approaches the per-cell baseline, so the joint
arm is bounded between geometry-only and per-cell unless the specific merges it forbids are
disproportionately harmful. That is the real question, and it is why per-cell is reported on
every line: an "improvement" that merely walks back toward per-cell is not an improvement.

Note also the SOFTNESS LAW: six independent confirmations that decisiveness loses
monotonically here. tau_f is a hard threshold on a narrow cone, which is exactly the move
that has failed before (k-means cosine floor at tau=0.995 cost -5.5 mIoU). Expect the
feature-only arm to be poor and read the joint arm with that in mind.

FALSIFIER: joint must beat BOTH geometry-only and per-cell at 19cls to be worth a 10-scene
run. Matching geometry-only means the feature term is inert.
"""
import argparse
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0062_00": "train", "scene0000_00": "train", "scene0645_00": "val"}


def quat_normal(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def score(seg, feats, valid, assign, gt, text, nc, dev):
    ns = int(seg.max()) + 1
    st = torch.from_numpy(seg).long().to(dev)
    vt = torch.from_numpy(valid).to(dev)
    pooled = torch.zeros(ns, feats.shape[1], device=dev).index_add_(0, st[vt], feats[vt])
    cnt = torch.zeros(ns, device=dev).index_add_(
        0, st[vt], torch.ones(int(vt.sum()), device=dev))
    cls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1) + 1
    cls[cnt == 0] = 0
    owned = assign >= 0
    pred = np.zeros(len(gt), dtype=np.int64)
    pred[owned] = cls.cpu().numpy()[seg[assign[owned]]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                         torch.from_numpy(pred).long(), nc)
    return float(miou) * 100


def comps(mask, i, j, n_prim):
    ii, jj = i[mask].cpu().numpy(), j[mask].cpu().numpy()
    G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n_prim, n_prim))
    return connected_components(G, directed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--tau-n", type=float, default=0.98)
    ap.add_argument("--tau-d", type=float, default=1.0)
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"

    m = torch.load(f"output/scannet_{a.scene}_nonfrozen/model.pt",
                   map_location="cpu", weights_only=False)
    P = m["points"].float().to(dev)
    radii = F.softplus(m["radii"].float().to(dev), beta=100)
    Nrm = quat_normal(m["quaternions"].float().to(dev))
    adjc, off = m["adjacency"].long().to(dev), m["adjacency_offsets"].long().to(dev)
    n_prim = P.shape[0]
    src = torch.repeat_interleave(torch.arange(n_prim, device=dev), off[1:] - off[:-1])
    k = src < adjc
    i, j = src[k], adjc[k]

    d = torch.load(f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                   map_location=dev, weights_only=True)
    feats = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
    valid = d["valid_mask"].cpu().numpy()
    assign = np.load(f"artifacts/ablation_cache/{a.scene}_pf_nonfroz_assign.npy")

    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    n2i = {n: q for q, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())

    cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
    dp = P[j] - P[i]
    rr = (radii[i] + radii[j]).clamp_min(1e-20)
    offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr
    cos_f = (feats[i] * feats[j]).sum(-1)
    geo = (cos_n > a.tau_n) & (offs < a.tau_d)
    print(f"[{a.scene}] {n_prim:,} cells, {len(i):,} facets, "
          f"geometry predicate keeps {100*geo.float().mean():.1f}% of edges")
    print(f"  facet feature cosine: p10={torch.quantile(cos_f, .10):.4f} "
          f"p50={torch.quantile(cos_f, .50):.4f} p90={torch.quantile(cos_f, .90):.4f}")

    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        gt = remap_gt_labels(raw, [n2i[n] for n in names])
        nc, text = len(names) + 1, embed_class_names(names, dev)
        pc = score(np.arange(n_prim), feats, valid, assign, gt, text, nc, dev)
        ns_g, lab_g = comps(geo, i, j, n_prim)
        g = score(lab_g, feats, valid, assign, gt, text, nc, dev)
        print(f"\n--- {cs[11:]:>3} classes | per-cell {pc:.2f} | "
              f"geometry-only {g:.2f} ({g-pc:+.2f}) segs={ns_g:,} ---")
        print(f"  {'tau_f':>7}{'feat-only':>11}{'segs':>10}{'JOINT':>9}{'segs':>10}"
              f"{'vs geo':>9}{'vs percell':>12}")
        for tf in (0.95, 0.98, 0.99, 0.995, 0.998):
            fm = cos_f > tf
            ns_f, lab_f = comps(fm, i, j, n_prim)
            fo = score(lab_f, feats, valid, assign, gt, text, nc, dev)
            ns_j, lab_j = comps(geo & fm, i, j, n_prim)
            jo = score(lab_j, feats, valid, assign, gt, text, nc, dev)
            print(f"  {tf:>7}{fo:>11.2f}{ns_f:>10,}{jo:>9.2f}{ns_j:>10,}"
                  f"{jo-g:>+9.2f}{jo-pc:>+12.2f}")


if __name__ == "__main__":
    main()
