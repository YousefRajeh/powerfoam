"""Coplanar dipole segments vs feature k-means: PURITY AT MATCHED GRANULARITY, AND mIoU.

Fixes two defects in diagnose_dipole_macro.py, whose comparison was not decision-grade:
  (a) k-means ran at a single K=20,000 (a clamp collapsed all matched values), so coplanar
      segments at 56k-151k were compared against a 20k baseline. Purity rises trivially with
      segment count, so that comparison was meaningless.
  (b) the two arms scored different point sets (51,310 vs 45,329) because the valid mask was
      applied to k-means labels but not to coplanar labels.
Both arms here use the SAME scorable point set and MATCHED segment counts.

AND the measurement that actually decides it: mIoU from pooling lifted CLIP features over
each segmentation, then classifying the pooled feature. Purity is necessary but nowhere near
sufficient -- a segmentation can be pure and still useless if its segments do not align with
the objects whose labels we must predict. Recall the measured fact that GT instances are NOT
connected components of owner cells (median 96 components per instance).

WHY THIS IS THE RIGHT TEST FOR THE USER'S IDEA. The claim is that dipoles supply macro-scale
geometry, so segments can be grown GEOMETRICALLY in one non-iterative connected-components
pass rather than by iterative feature-similarity growing. If that is true, pooling on
coplanar segments should beat pooling on feature k-means at the same granularity. If mIoU is
equal or worse at matched segment count, the geometry is not adding information the features
lack, and the idea reduces to a differently-shaped clustering with no advantage.

FALSIFIER, stated before running: coplanar must beat matched-K k-means by >= +0.5 mIoU on
19cls to be worth pursuing. Anything less is noise at single-scene scale -- twelve
single-scene findings have already reversed in this project.
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
    """Pool unit features per segment, classify, broadcast to points. Returns (mIoU, purity)."""
    n_prim = feats.shape[0]
    seg = np.asarray(seg)
    ns = int(seg.max()) + 1
    st = torch.from_numpy(seg).long().to(dev)
    vt = torch.from_numpy(valid).to(dev)

    pooled = torch.zeros(ns, feats.shape[1], device=dev)
    pooled.index_add_(0, st[vt], feats[vt])                 # featureless cells contribute nothing
    cnt = torch.zeros(ns, device=dev).index_add_(
        0, st[vt], torch.ones(int(vt.sum()), device=dev))
    has = cnt > 0
    pooled = F.normalize(pooled, dim=-1)
    cls = (pooled @ text.T).argmax(-1) + 1
    cls[~has] = 0                                           # no feature -> no prediction

    owned = assign >= 0
    pred = np.zeros(len(gt), dtype=np.int64)
    pred[owned] = cls.cpu().numpy()[seg[assign[owned]]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                         torch.from_numpy(pred).long(), nc)
    # purity on the SAME scorable set
    ok = owned & (gt > 0)
    s, y = seg[assign[ok]], gt[ok]
    H = sp.coo_matrix((np.ones(len(y)), (s, y)), shape=(ns, nc)).tocsr()
    maj = np.asarray(H.argmax(1)).ravel()
    return float(miou), float(macc), (maj[s] == y).mean(), int(ok.sum())


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

    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[a.scene]}\{a.scene}", "segment20")
    assign = np.load(f"artifacts/ablation_cache/{a.scene}_{a.recon}_assign.npy")
    n2i = {n: q for q, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt = remap_gt_labels(raw, [n2i[n] for n in names])
    nc = len(names) + 1
    text = embed_class_names(names, dev)

    d = torch.load(f"artifacts/scannet/{a.scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                   map_location=dev, weights_only=True)
    feats = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
    valid = d["valid_mask"].cpu().numpy()

    # per-cell baseline (no pooling at all)
    per_cell = np.arange(n_prim)
    mi, ma, pu, npts = score(per_cell, feats, valid, assign, gt, text, nc, dev)
    print(f"[baseline] per-cell (no pooling): mIoU={mi*100:.2f} mAcc={ma*100:.2f} "
          f"purity={pu*100:.2f}% on {npts:,} pts\n")

    cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
    dp = P[j] - P[i]
    rr = (radii[i] + radii[j]).clamp_min(1e-20)
    offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr

    print(f"{'method':<26}{'segs':>9}{'purity':>9}{'mIoU':>8}{'mAcc':>8}")
    gen = torch.Generator(device=dev).manual_seed(0)
    for tau_n, tau_d in ((0.95, 3.0), (0.95, 1.0), (0.95, 0.5), (0.98, 1.0), (0.995, 1.0)):
        keep = (cos_n > tau_n) & (offs < tau_d)
        ii, jj = i[keep].cpu().numpy(), j[keep].cpu().numpy()
        G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n_prim, n_prim))
        ns, lab = connected_components(G, directed=False)
        mi, ma, pu, _ = score(lab, feats, valid, assign, gt, text, nc, dev)
        print(f"{'coplanar t=' + str(tau_n) + '/' + str(tau_d):<26}{ns:>9,}"
              f"{pu*100:>8.2f}%{mi*100:>8.2f}{ma*100:>8.2f}")

        # MATCHED-K feature k-means. Chunked: feats @ C.T at K=65k is a 53 GiB matrix.
        Kt = torch.randperm(n_prim, generator=gen, device=dev)[:ns]
        C = feats[Kt].clone()
        CH = max(1, int(2e8 // max(ns, 1)))          # rows per chunk, ~0.8 GiB of fp32
        for _ in range(15):
            parts = [(feats[b:b + CH] @ C.T).argmax(1) for b in range(0, n_prim, CH)]
            lt = torch.cat(parts)
            C = F.normalize(torch.zeros_like(C).index_add_(0, lt, feats), dim=-1)
        mi2, ma2, pu2, _ = score(lt.cpu().numpy(), feats, valid, assign, gt, text, nc, dev)
        print(f"{'  kmeans matched K':<26}{ns:>9,}{pu2*100:>8.2f}%"
              f"{mi2*100:>8.2f}{ma2*100:>8.2f}   delta={mi*100-mi2*100:+.2f}")


if __name__ == "__main__":
    main()
