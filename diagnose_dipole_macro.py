"""Do DIPOLES give macro-scale geometry good enough to grow segments on?

THE USER'S ARGUMENT, and why T3's failure does NOT settle it.
T3 measured dipole normals as a PER-FACET boundary detector and got AUC 0.5674 -- no better
than feature cosine. But a per-edge decision is the noisiest possible use of the signal.
The paper describes the dipole face as "a proxy for MACRO-SCALE geometry". Macro structure
is aggregate: fit a plane across hundreds of cells and per-edge orientation noise averages
out. A signal too weak to classify one edge can be strong across a whole patch.

Second reason to expect structure: indoor rooms are near-Manhattan, so dipole normals should
concentrate into a few dominant orientations rather than spreading over the sphere.

Third, and the reason this matters more than boundary detection: 83% of our errors are
INTERIOR cells of coherent regions -- whole-region misclassification. That is a SEGMENTATION
failure, not a boundary failure. Boundary fixes were already measured to cap at 17%.

WHAT WE GROW, and why it is not the old iterative grower. Previous region growing expanded
by FEATURE similarity, iteratively, with a stopping rule that needed tuning. Here the
predicate is purely GEOMETRIC and the segmentation is a single connected-components pass
over the facet graph -- no iteration, no schedule, no feature input at all:

    edge (i,j) survives  iff  |cos(n_i, n_j)| > tau_n           (parallel)
                         and  plane offset |<p_j-p_i, n_i>| / (r_i+r_j) < tau_d   (coplanar)

Parallel-but-offset planes (the two faces of a door, two shelves) must NOT merge, which is
why the offset term is there and why |cos| alone is insufficient.

WHAT IS MEASURED
  1. Manhattan structure: concentration of dipole normals (is there macro structure at all?)
  2. Segment PURITY vs GT labels -- the quantity that decides whether pooling on these
     segments can work. Compared against the incumbent feature k-means regions.
  3. Segment count / size distribution, and connectivity (the old k-means regions were
     scattered over a median of 15.5 components; these are connected BY CONSTRUCTION).

FALSIFIERS, stated before running:
  * If purity of coplanar segments <= purity of feature k-means at matched segment count,
    the geometry adds nothing and the idea is dead.
  * If the segmentation collapses (one giant component swallowing the room, or ~1 cell per
    segment across the whole tau range), the predicate has no usable operating point.
  * Purity must beat the trivial baseline of "every cell its own segment" (purity 1.0 by
    construction) at a USEFUL granularity -- so purity is always reported WITH segment count.
    A purity number without a segment count is meaningless.
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
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0062_00": "train", "scene0000_00": "train", "scene0645_00": "val"}


def quat_normal(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def purity_of(labels_per_cell, assign, gt, n_seg_ids):
    """Fraction of labelled GT points whose segment-majority label equals their own."""
    ok = (assign >= 0) & (gt > 0)
    seg = labels_per_cell[assign[ok]]
    y = gt[ok]
    good = seg >= 0
    seg, y = seg[good], y[good]
    if len(y) == 0:
        return float("nan"), 0
    nc = int(y.max()) + 1
    H = sp.coo_matrix((np.ones(len(y)), (seg, y)), shape=(n_seg_ids, nc)).tocsr()
    maj = np.asarray(H.argmax(1)).ravel()
    hit = (maj[seg] == y).sum()
    return hit / len(y), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--recon", default="pf_nonfroz")
    ap.add_argument("--class-set", default="opengaussian19")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    m = torch.load(f"output/scannet_{a.scene}_nonfrozen/model.pt",
                   map_location="cpu", weights_only=False)
    P = m["points"].float().to(dev)
    radii = F.softplus(m["radii"].float().to(dev), beta=100)
    Nrm = quat_normal(m["quaternions"].float().to(dev))
    adjc = m["adjacency"].long().to(dev)
    off = m["adjacency_offsets"].long().to(dev)
    n_prim = P.shape[0]

    src = torch.repeat_interleave(torch.arange(n_prim, device=dev), off[1:] - off[:-1])
    k = src < adjc
    i, j = src[k], adjc[k]

    # ---- (1) Manhattan structure
    n_np = Nrm.cpu().numpy()
    absn = np.abs(n_np)
    dom = absn.argmax(1)
    print("=== (1) dipole normal orientation structure ===")
    print(f"  dominant-axis split: x={100*(dom==0).mean():.1f}%  "
          f"y={100*(dom==1).mean():.1f}%  z={100*(dom==2).mean():.1f}%")
    print(f"  max|component| median {np.median(absn.max(1)):.3f}  "
          f"(1.0 = axis-aligned, 0.577 = fully diagonal)")
    axis_aligned = (absn.max(1) > 0.9).mean()
    print(f"  {100*axis_aligned:.1f}% of cells within ~26 deg of a coordinate axis")

    # ---- GT
    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    assign = np.load(f"artifacts/ablation_cache/{a.scene}_{a.recon}_assign.npy")
    n2i = {n: q for q, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt = remap_gt_labels(raw, [n2i[n] for n in names])

    cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
    dp = P[j] - P[i]
    rr = (radii[i] + radii[j]).clamp_min(1e-20)
    offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr

    print("\n=== (2) coplanar connected components: purity vs granularity ===")
    print(f"  {'tau_n':>6}{'tau_d':>7}{'segments':>10}{'largest':>9}"
          f"{'purity':>9}{'pts':>9}")
    results = []
    for tau_n in (0.95, 0.98, 0.995):
        for tau_d in (0.5, 1.0, 3.0):
            keep = (cos_n > tau_n) & (offs < tau_d)
            ii = i[keep].cpu().numpy()
            jj = j[keep].cpu().numpy()
            G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)),
                              shape=(n_prim, n_prim))
            ns, lab = connected_components(G, directed=False)
            sizes = np.bincount(lab)
            pur, npts = purity_of(lab, assign, gt, ns)
            print(f"  {tau_n:>6}{tau_d:>7}{ns:>10,}{sizes.max():>9,}"
                  f"{pur*100:>8.2f}%{npts:>9,}")
            results.append((tau_n, tau_d, ns, pur))

    # ---- (3) incumbent baseline: feature k-means at MATCHED segment counts
    sol = f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt"
    d = torch.load(sol, map_location=dev, weights_only=True)
    feats = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
    valid = d["valid_mask"].cpu().numpy()
    print("\n=== (3) incumbent: feature k-means (spherical) at matched granularity ===")
    print(f"  {'k':>8}{'purity':>9}{'pts':>9}")
    gen = torch.Generator(device=dev).manual_seed(0)
    for K in sorted({min(max(r[2], 2), 20000) for r in results}):
        idx = torch.randperm(n_prim, generator=gen, device=dev)[:K]
        C = feats[idx].clone()
        for _ in range(15):
            lab_t = (feats @ C.T).argmax(1)
            C.zero_().index_add_(0, lab_t, feats)
            C = F.normalize(C, dim=-1)
        lab = lab_t.cpu().numpy()
        lab[~valid] = -1
        pur, npts = purity_of(lab, assign, gt, K)
        print(f"  {K:>8,}{pur*100:>8.2f}%{npts:>9,}")


if __name__ == "__main__":
    main()
