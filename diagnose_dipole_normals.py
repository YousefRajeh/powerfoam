"""Are DIPOLE NORMALS a better semantic-boundary detector than feature cosine?

CONTEXT / WHAT IS ALREADY KNOWN.
  * Facet feature-cosine is a WEAK boundary detector: AUC 0.65 (measured on the Cech graph),
    re-measured 0.67-0.71 on the true facet graph. Lifted CLIP features spatially over-smooth
    (same-label 0.994 vs boundary 0.985 median cosine), so they barely separate.
  * Dipole surface DISTANCE carries no correctness signal (measured: |s| deciles flat,
    61.66% -> 62.47%, gap -0.81 pts). So the dipole's *offset* is inert.
  * But every cell also carries a learned NORMAL n_i (quaternion column 0) -- an oriented
    surface element. That is INDEPENDENT geometric evidence, not derived from CLIP at all,
    and no Gaussian has an analogue: a splat has no normal and no bisecting face.
  * ScanNet's dominant classes (wall, floor, door, window, table, picture) are PLANAR, so
    "same physical surface" is a strong prior for "same label".

HYPOTHESIS (T3). Across a shared facet, disagreement between the two cells' dipole normals
marks a geometric crease, and creases coincide with semantic boundaries. Score:
    disagreement(i,j) = 1 - |cos(n_i, n_j)|
|cos| rather than cos because a surface normal's global sign is not constrained -- two cells
on the same wall may carry opposite-facing normals.

Also tested, since coplanarity is stronger than parallelism: two parallel-but-OFFSET planes
(e.g. the two faces of a thin door) are NOT the same surface. So we additionally score
    offset(i,j) = |<p_j - p_i, n_i>| + |<p_i - p_j, n_j>|   (normalised by radii)
and the combination, which is the standard co-planarity test.

FALSIFIERS, stated before running:
  * If AUC(normal) <= ~0.65, normals are no better than the feature cosine we already have
    and T3 is dead.
  * If AUC(normal) is high but AUC(normal + feature) is no better than AUC(feature), the
    normals are redundant with what CLIP already encodes -- also a kill, because the whole
    point is INDEPENDENT evidence.
  * Prior is genuinely uncertain: the P4 feature-gate on diffusion was NULL, and every
    hard-thresholding move so far has lost to the softness law. A good AUC here is a
    licence to build a SOFT gate, never a hard cut.

GROUND TRUTH. A facet is a "boundary" if the two cells' dominant GT labels differ. Only
cells that own at least one labelled GT point can be scored, so this measures boundary
detection on the labelled subset, which is the subset that matters for mIoU.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

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


def auc(score, label):
    """Rank-based AUC. score high => predict label 1."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


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
    adj = m["adjacency"].long().to(dev)
    off = m["adjacency_offsets"].long().to(dev)
    n_prim = P.shape[0]

    # undirected edge list from CSR, keep i<j once
    src = torch.repeat_interleave(torch.arange(n_prim, device=dev), off[1:] - off[:-1])
    keep = src < adj
    i, j = src[keep], adj[keep]
    print(f"[graph] {n_prim:,} cells, {len(i):,} undirected facets")

    # ---- per-cell dominant GT label
    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    assign = np.load(f"artifacts/ablation_cache/{a.scene}_{a.recon}_assign.npy")
    n2i = {n: k for k, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt = remap_gt_labels(raw, [n2i[n] for n in names])
    nc = len(names) + 1

    hist = np.zeros((n_prim, nc), dtype=np.int32)
    ok = (assign >= 0) & (gt > 0)
    np.add.at(hist, (assign[ok], gt[ok]), 1)
    cell_lbl = hist.argmax(1)
    cell_lbl[hist.sum(1) == 0] = 0                      # unlabelled cells
    lbl = torch.from_numpy(cell_lbl).to(dev)
    print(f"[gt] {(cell_lbl > 0).sum():,} cells carry a label "
          f"({100*(cell_lbl > 0).mean():.1f}%)")

    both = (lbl[i] > 0) & (lbl[j] > 0)
    i, j = i[both], j[both]
    boundary = (lbl[i] != lbl[j]).cpu().numpy().astype(np.int64)
    print(f"[eval] {len(i):,} facets with labels on both sides, "
          f"{boundary.mean()*100:.1f}% are semantic boundaries")

    ni, nj = Nrm[i], Nrm[j]
    cos_n = (ni * nj).sum(-1).abs().clamp(0, 1)
    s_normal = (1 - cos_n).cpu().numpy()                       # parallelism

    dp = P[j] - P[i]
    rr = (radii[i] + radii[j]).clamp_min(1e-20)
    s_offset = (((dp * ni).sum(-1).abs() + (dp * nj).sum(-1).abs()) / rr).cpu().numpy()
    s_coplan = s_normal + np.clip(s_offset, 0, 10) / 10.0       # crude combination

    # ---- feature cosine, the incumbent detector
    sol = f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt"
    d = torch.load(sol, map_location=dev, weights_only=True)
    f_unit = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
    s_feat = (1 - (f_unit[i] * f_unit[j]).sum(-1)).cpu().numpy()

    print(f"\n=== boundary-detection AUC ({a.class_set}, higher = better) ===")
    rows = [("feature cosine (incumbent)", s_feat),
            ("dipole normal  1-|cos|", s_normal),
            ("dipole offset  (planes apart)", s_offset),
            ("coplanarity    (normal+offset)", s_coplan),
            ("normal + feature (sum, z-scored)", None)]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    rows[-1] = ("normal + feature (sum, z-scored)", z(s_normal) + z(s_feat))
    for name, sc in rows:
        print(f"  {name:<34}{auc(sc, boundary):.4f}")

    print(f"\n=== separation: median score, same-label vs boundary facets ===")
    for name, sc in rows:
        same = np.median(sc[boundary == 0])
        bnd = np.median(sc[boundary == 1])
        print(f"  {name:<34}same={same:+8.4f}  boundary={bnd:+8.4f}  gap={bnd-same:+.4f}")


if __name__ == "__main__":
    main()
