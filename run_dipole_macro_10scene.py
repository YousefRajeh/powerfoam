"""10-SCENE validation: do coplanar DIPOLE segments beat matched-K feature k-means?

THE CLAIM UNDER TEST. Pooling lifted CLIP features over segments grown from PowerFoam's
dipole macro-geometry (a single non-iterative connected-components pass on a purely
geometric predicate) beats pooling over feature k-means at the SAME segment count.

Pilot on scene0347_00 (19cls) gave +2.12/+1.82/+0.88/+1.42/+2.43 across five (tau_n, tau_d)
settings -- 5/5 positive, mean +1.73 -- and the best coplanar setting (44.37) beat the
per-cell no-pooling baseline (42.47) by +1.90.

WHY THE DELTA IS THE CLAIM, NOT THE BEST tau. The best tau was read off the pilot scene, so
quoting its absolute mIoU would be selection bias. The delta (coplanar - matched-K kmeans)
was positive at ALL five settings, so it does not depend on choosing tau. This script
therefore reports every setting on every scene and asks whether the delta stays positive.

A SINGLE-SCENE RESULT IS A PILOT, NEVER A CONCLUSION. Twelve single-scene findings have
reversed at 10-scene scale in this project. That is the entire reason this script exists.

FALSIFIER, stated before running: mean delta over 10 scenes must stay >= +0.5 mIoU, and be
positive on a clear majority of scenes. A mean carried by one or two scenes is not a result.
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

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
# hardest-first, as everywhere else in this project
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
          "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]
TAUS = [(0.95, 3.0), (0.95, 1.0), (0.95, 0.5), (0.98, 1.0), (0.995, 1.0)]
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def quat_normal(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def score(seg, feats, valid, assign, gt, text, nc, dev):
    ns = int(seg.max()) + 1
    st = torch.from_numpy(seg).long().to(dev)
    vt = torch.from_numpy(valid).to(dev)
    pooled = torch.zeros(ns, feats.shape[1], device=dev)
    pooled.index_add_(0, st[vt], feats[vt])
    cnt = torch.zeros(ns, device=dev).index_add_(
        0, st[vt], torch.ones(int(vt.sum()), device=dev))
    cls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1) + 1
    cls[cnt == 0] = 0
    owned = assign >= 0
    pred = np.zeros(len(gt), dtype=np.int64)
    pred[owned] = cls.cpu().numpy()[seg[assign[owned]]]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                         torch.from_numpy(pred).long(), nc)
    return float(miou), float(macc)


def kmeans_labels(feats, ns, dev, seed=0):
    n = feats.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    C = feats[torch.randperm(n, generator=gen, device=dev)[:ns]].clone()
    CH = max(1, int(2e8 // max(ns, 1)))
    lt = None
    for _ in range(15):
        lt = torch.cat([(feats[b:b + CH] @ C.T).argmax(1) for b in range(0, n, CH)])
        C = F.normalize(torch.zeros_like(C).index_add_(0, lt, feats), dim=-1)
    return lt.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out", default="artifacts/scannet/dipole_macro_10scene.json")
    ap.add_argument("--skip-kmeans-above", type=int, default=400_000,
                    help="skip the matched-K kmeans arm above this K (cost guard); "
                         "the skip is LOGGED, never silent")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy"
        if not all(os.path.exists(p) for p in (mp, fp, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        radii = F.softplus(m["radii"].float().to(dev), beta=100)
        Nrm = quat_normal(m["quaternions"].float().to(dev))
        adjc = m["adjacency"].long().to(dev)
        off = m["adjacency_offsets"].long().to(dev)
        n_prim = P.shape[0]
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev), off[1:] - off[:-1])
        k = src < adjc
        i, j = src[k], adjc[k]

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = F.normalize(d["primitive_features"].to(dev).float(), dim=-1)
        valid = d["valid_mask"].cpu().numpy()
        assign = np.load(apth)
        if feats.shape[0] != n_prim:
            print(f"[skip] {scene}: feats {feats.shape[0]} vs cells {n_prim}", flush=True)
            continue

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
        dp = P[j] - P[i]
        rr = (radii[i] + radii[j]).clamp_min(1e-20)
        offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr

        segs_cache, km_cache = {}, {}
        for tau_n, tau_d in TAUS:
            keep = (cos_n > tau_n) & (offs < tau_d)
            ii, jj = i[keep].cpu().numpy(), j[keep].cpu().numpy()
            G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)),
                              shape=(n_prim, n_prim))
            ns, lab = connected_components(G, directed=False)
            segs_cache[(tau_n, tau_d)] = (ns, lab)
            if ns <= a.skip_kmeans_above and ns not in km_cache:
                km_cache[ns] = kmeans_labels(feats, ns, dev)
            elif ns > a.skip_kmeans_above:
                print(f"  [cost-guard] {scene} K={ns:,} > {a.skip_kmeans_above:,}: "
                      f"kmeans arm SKIPPED for this setting", flush=True)

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            base, _ = score(np.arange(n_prim), feats, valid, assign, gt, text, nc, dev)
            out.setdefault(f"percell|{cs}", {})[scene] = base * 100
            for tau in TAUS:
                ns, lab = segs_cache[tau]
                cm, _ = score(lab, feats, valid, assign, gt, text, nc, dev)
                key = f"{tau[0]}/{tau[1]}"
                out.setdefault(f"coplanar {key}|{cs}", {})[scene] = cm * 100
                if ns in km_cache:
                    km, _ = score(km_cache[ns], feats, valid, assign, gt, text, nc, dev)
                    out.setdefault(f"kmeans {key}|{cs}", {})[scene] = km * 100
                    print(f"  {scene} {cs[11:]:>3} {key:<10} segs={ns:>8,} "
                          f"copl={cm*100:6.2f} km={km*100:6.2f} "
                          f"delta={cm*100-km*100:+6.2f}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"[done] {scene} {n_prim:,} cells {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n=== MEAN OVER SCENES (delta = coplanar - matched-K kmeans) ===")
    for cs in CLASS_SETS:
        pc = out.get(f"percell|{cs}", {})
        if not pc:
            continue
        print(f"--- {cs[11:]} classes | per-cell baseline "
              f"{np.mean(list(pc.values())):.2f} (n={len(pc)}) ---")
        for tau_n, tau_d in TAUS:
            key = f"{tau_n}/{tau_d}"
            c = out.get(f"coplanar {key}|{cs}", {})
            km = out.get(f"kmeans {key}|{cs}", {})
            common = sorted(set(c) & set(km))
            if not common:
                continue
            dl = [c[s] - km[s] for s in common]
            print(f"  {key:<10} coplanar {np.mean([c[s] for s in common]):6.2f}  "
                  f"kmeans {np.mean([km[s] for s in common]):6.2f}  "
                  f"delta {np.mean(dl):+6.2f}  positive on {sum(x>0 for x in dl)}/{len(dl)}")


if __name__ == "__main__":
    main()
