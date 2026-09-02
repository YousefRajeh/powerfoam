"""Does the DIPOLE coverage gain stack with mode-voting + posterior diffusion, our real best?

WHY THIS EXISTS. The dipole macro-geometry gain was measured against a per-cell argmax
baseline, which is NOT our pipeline. Our documented best is per-primitive base + NormLift
mode-voting + posterior simplex diffusion on the true facet graph:

    10-scene:  base 36.53/39.26/46.95  ->  mode-vote 37.89/40.57/48.61
               ->  mode-vote + diffusion 38.95/41.76/49.54          <-- the number to beat

THE REASON TO DOUBT STACKING. The dipole gain was decomposed and is ~85-90% a COVERAGE
effect: geometric segments raise the classifiable fraction 88.63% -> 92.02% by letting cells
the lifting never reached borrow a label along a continuous surface. Pooled features scored
on the SAME points gain only +0.21/+0.25/-0.29. But diffusion propagates posteriors on the
SAME facet graph and also reaches featureless cells (p0 is zero there, yet they receive mass
from neighbours). Mode-voting likewise copies a neighbour's direction. So all three may be
rescuing the SAME cells, in which case the gains are redundant and must not be added up in
the paper.

TWO DIPOLE OPERATORS, because the decomposition says the features add nothing:
  dipole_pool  every cell takes its segment's pooled feature      (what was measured: +1.57)
  dipole_fill  cell keeps its OWN prediction where it has one, and takes the segment's only
               where it does not                                  (pure coverage, no risk of
               pooling degrading cells that were already fine)
dipole_fill is the honest operator for the measured mechanism and is what should stack.

FALSIFIER, stated before running: the dipole increment ON TOP of mode-vote + diffusion must
be >= +0.5 mIoU at 19cls. If it collapses to ~0, the mechanisms are redundant, and the
correct paper claim is "diffusion already captures it", not three separate contributions.

Coverage is reported for EVERY arm, because under the coverage law (mIoU ~= 0.53 x
classifiable fraction) coverage is the quantity that predicts the score. If two arms have the
same coverage they are doing the same thing regardless of how differently they are described.
"""
import argparse
import json
import os
import sys
import time

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
from feature_foam_lifting.operator import AccumulatedFeatureStats
from run_normlift_refine_eval import mode_vote_refine

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00"]      # hardest-first pilot
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
TAU_N, TAU_D = 0.98, 1.0        # mid setting from the pilot sweep; NOT the best-scoring one


def quat_normal(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def diffuse(p0, adjacent, offsets, n, alpha=0.9, iters=60):
    dev = p0.device
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    src = torch.repeat_interleave(torch.arange(n, device=dev), offsets[1:] - offsets[:-1])
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[adjacent])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def evaluate(cls, live, assign, gt, nc):
    """cls: (P,) 1-based per-cell class. live: (P,) bool, cell has a prediction."""
    owned = assign >= 0
    sc = owned.copy()
    sc[owned] = live[assign[owned]]
    pred = np.zeros(len(gt), dtype=np.int64)
    pred[sc] = cls[assign[sc]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                         torch.from_numpy(pred).long(), nc)
    return float(miou) * 100, float(macc) * 100, float(sc.mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out", default="artifacts/scannet/dipole_stack.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy"
        if not all(os.path.exists(p) for p in (mp, fp, stp, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        radii = F.softplus(m["radii"].float().to(dev), beta=100)
        Nrm = quat_normal(m["quaternions"].float().to(dev))
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = P.shape[0]

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)

        # torch.load returns the raw dict; the reliability (NormLift Eq. 6-8) lives on the
        # AccumulatedFeatureStats object, so go through its own loader.
        stats = AccumulatedFeatureStats.load(stp)
        R = stats.reliability()["reliability"].to(dev).float() * vt
        del stats

        # --- geometric segments from the dipole macro-geometry
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        k = src < adjacent
        i, j = src[k], adjacent[k]
        cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
        dp = P[j] - P[i]
        rr = (radii[i] + radii[j]).clamp_min(1e-20)
        offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr
        geo = (cos_n > TAU_N) & (offs < TAU_D)
        ii, jj = i[geo].cpu().numpy(), j[geo].cpu().numpy()
        G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n_prim, n_prim))
        ns, lab = connected_components(G, directed=False)
        st = torch.from_numpy(lab).long().to(dev)
        print(f"[{scene}] {n_prim:,} cells, {ns:,} geometric segments, "
              f"{100*valid.mean():.1f}% cells have a feature", flush=True)

        refined = mode_vote_refine(unit, R, P, adjacent, offsets)

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)

            def seg_cls(u):
                """segment-pooled class + whether the segment has any feature at all"""
                pooled = torch.zeros(ns, u.shape[1], device=dev).index_add_(0, st[vt], u[vt])
                cnt = torch.zeros(ns, device=dev).index_add_(
                    0, st[vt], torch.ones(int(vt.sum()), device=dev))
                c = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1).cpu().numpy() + 1
                return c[lab], (cnt > 0).cpu().numpy()[lab]

            arms = {}
            for tag, u in (("base", unit), ("modevote", refined)):
                cls = (u @ text.T).argmax(-1).cpu().numpy() + 1
                arms[tag] = (cls, valid.copy())
                # + diffusion
                p0 = torch.softmax(1000.0 * (u @ text.T), dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, adjacent, offsets, n_prim)
                arms[tag + "+diff"] = (pd.argmax(-1).cpu().numpy() + 1,
                                       (pd.sum(-1) > 0).cpu().numpy())
            # dipole variants on top of every arm
            sc_cls, sc_live = seg_cls(unit)
            for tag in list(arms):
                cls, live = arms[tag]
                fill_cls, fill_live = cls.copy(), live | sc_live
                fill_cls[~live] = sc_cls[~live]                     # borrow only where blind
                arms[tag + "+dipfill"] = (fill_cls, fill_live)
            arms["dipole_pool"] = seg_cls(unit)

            for tag, (cls, live) in arms.items():
                mi, ma, cov = evaluate(cls, live, assign, gt, nc)
                out.setdefault(f"{tag}|{cs}", {})[scene] = {"mIoU": mi, "mAcc": ma, "cov": cov}
                print(f"  {scene} {cs[11:]:>3} {tag:<22} mIoU={mi:6.2f} cov={cov:6.2f}%",
                      flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"[done] {scene} {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n=== MEAN OVER SCENES ===")
    for cs in CLASS_SETS:
        print(f"--- {cs[11:]} classes ---")
        ref = out.get(f"modevote+diff|{cs}", {})
        for tag in ("base", "base+diff", "modevote", "modevote+diff", "dipole_pool",
                    "base+dipfill", "base+diff+dipfill", "modevote+dipfill",
                    "modevote+diff+dipfill"):
            r = out.get(f"{tag}|{cs}", {})
            if not r:
                continue
            common = sorted(set(r) & set(ref)) if ref else sorted(r)
            mi = np.mean([r[s]["mIoU"] for s in common])
            cov = np.mean([r[s]["cov"] for s in common])
            dl = (f"{mi - np.mean([ref[s]['mIoU'] for s in common]):+6.2f}"
                  if ref and common else "     -")
            print(f"  {tag:<24}{mi:7.2f}  cov {cov:6.2f}%   vs best {dl}  (n={len(common)})")


if __name__ == "__main__":
    main()
