"""ITEMS 1, 2, 4: the right number of groups, the over-merge/over-split decomposition,
and seed stability as a metric.

Nothing in `results_unified` sweeps K -- every clustering arm in the database uses K=320
because that is OpenGaussian's codebook size. K has never been chosen, only inherited.

--- ITEM 2, the decomposition (never computed) -------------------------------------------
Every misclassified GT point falls into exactly one of three buckets. Let r = region(p),
mstar(r) = GT-majority class of r, c_r = the class the pooled region feature is given,
inst(p) = the GT INSTANCE id (ScanNet `instance.npy`).

  E_MERGE   y_p != mstar(r)
            p is a minority member of a region that spans >1 GT class. NO region-level
            label can save it. This is the cost of over-merging, and it is irreducible
            for the given grouping.
  E_SPLIT   y_p == mstar(r), c_r wrong, but SOME OTHER region overlapping inst(p) is
            classified correctly. The object was fragmented and the fragments disagree --
            fixable by merging fragments / enforcing consistency across the instance.
  E_SOLO    y_p == mstar(r), c_r wrong, and EVERY region of inst(p) is wrong. The object
            is uniformly misread. Not a grouping problem at all.

--- ITEM 1, the K sweep + label-free selectors ---------------------------------------------
mIoU(K), against four statistics computable WITHOUT labels:
  stability   mean pairwise NMI of the partition across 3 reseeds
  distortion  mean cosine of a cell to its own centroid (the k-means objective itself)
  cut         fraction of TRUE-FACET edges whose endpoints are in different regions
  mdl         two-part description length K*D + N*log2(K) vs the residual

If none of them peaks where mIoU peaks, K is unselectable without supervision and the
"choose K principledly" direction is closed.

--- ITEM 4 -------------------------------------------------------------------------------
Per-K mIoU spread across the 3 seeds, and whether stability correlates with mIoU.

FALSIFIERS, pre-registered:
  (1) if mIoU(K) is flat (max-min < 1.0 mIoU at 19cls over 10 scenes across K in
      [20, 20480]), then K is not a lever and no selector is needed.
  (2) if the argmax-K of every label-free statistic differs from the mIoU-optimal K by
      more than one octave on a majority of scenes, unsupervised K selection is closed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       classify_primitives, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
HARD_FIRST = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
              "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def skmeans(x, k, iters=25, seed=0, chunk=16384):
    """Spherical k-means, chunked so K in the thousands fits. x: (N,C) unit."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    cen = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()

    def assign(c):
        out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for i in range(0, x.shape[0], chunk):
            out[i:i + chunk] = (x[i:i + chunk] @ c.T).argmax(1)
        return out

    for _ in range(iters):
        lab = assign(cen)
        new = torch.zeros_like(cen).index_add_(0, lab, x)
        cnt = torch.bincount(lab, minlength=k).clamp_min(1).unsqueeze(1)
        new /= cnt
        nrm = new.norm(dim=1, keepdim=True)
        dead = nrm.squeeze(1) < 1e-8
        if dead.any():
            new[dead] = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:int(dead.sum())]]
            nrm = new.norm(dim=1, keepdim=True)
        cen = new / nrm.clamp_min(1e-8)
    return assign(cen), cen


def nmi(a, b):
    """Normalised mutual information between two hard partitions (torch, on device)."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    n = a.numel()
    joint = torch.zeros(ka * kb, device=a.device)
    joint.index_add_(0, a * kb + b, torch.ones(n, device=a.device))
    joint = (joint / n).reshape(ka, kb)
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * (joint[nz].log() - (pa[:, None] * pb[None, :])[nz].log())).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    return float(2 * mi / (ha + hb).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(HARD_FIRST))
    ap.add_argument("--ks", default="20,40,80,160,320,640,1280,2560,5120,10240,20480")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="artifacts/grouping_K_split.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    KS = [int(k) for k in a.ks.split(",")]

    miou = {}      # (K, cs) -> list over (scene, seed)
    stats = {}     # (K,) -> list of dict per scene
    decomp = {}    # (K, cs) -> list of dict per scene (seed 0 only)

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        tf = f"artifacts/scannet/{scene}/adjacency_true_facet.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        gtd = rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}"
        if not all(os.path.exists(p) for p in (fp, tf, apth)):
            print(f"[skip] {scene}", flush=True)
            continue

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vidx_np = np.where(valid)[0]
        n_prim = feats.shape[0]
        unit = F.normalize(feats[torch.from_numpy(vidx_np).to(dev)], dim=-1)
        Nv, D = unit.shape

        g = torch.load(tf, map_location="cpu", weights_only=True)
        adjacent = g["adjacent"].long().to(dev)
        offsets = g["offsets"].long().to(dev)
        esrc = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                       offsets[1:] - offsets[:-1])
        # map primitive id -> compact valid id (-1 if invalid); cut is measured on valid-valid
        v2c = torch.full((n_prim,), -1, dtype=torch.long, device=dev)
        v2c[torch.from_numpy(vidx_np).to(dev)] = torch.arange(Nv, device=dev)
        ea, eb = v2c[esrc], v2c[adjacent]
        ekeep = (ea >= 0) & (eb >= 0)
        ea, eb = ea[ekeep], eb[ekeep]

        assign = np.load(apth)
        owned = assign >= 0
        gt_pts, raw, names_all = load_scannet_pointcept_gt(gtd, "segment20")
        inst = np.load(os.path.join(gtd, "instance.npy")).astype(np.int64)
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        # per-class-set fixtures
        fx = {}
        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            fx[cs] = (gt, len(names) + 1, embed_class_names(names, dev),
                      torch.from_numpy(gt).long())

        for K in KS:
            if K >= Nv:
                continue
            parts = []
            for s in range(a.seeds):
                lab, _ = skmeans(unit, K, seed=s)
                parts.append(lab)
                # --- mIoU of the standard pool->classify->broadcast arm ---
                pooled = torch.zeros(K, D, device=dev).index_add_(0, lab, unit)
                nrm = pooled.norm(dim=-1, keepdim=True)
                ne = nrm.squeeze(-1) > 1e-8
                pooled = pooled / nrm.clamp_min(1e-8)
                for cs in CLASS_SETS:
                    gt, nc, text, gt_t = fx[cs]
                    rc = torch.zeros(K, dtype=torch.long, device=dev)
                    rc[ne] = classify_primitives(pooled[ne], text) + 1
                    cell = np.zeros(n_prim, dtype=np.int64)
                    cell[vidx_np] = rc[lab].cpu().numpy()
                    pr = np.zeros(len(gt), dtype=np.int64)
                    pr[owned] = cell[assign[owned]]
                    _, mi, _, _ = calculate_metrics(gt_t, torch.from_numpy(pr).long(), nc)
                    miou.setdefault((K, cs), []).append(float(mi) * 100)

                    if s == 0:
                        # ---- ITEM 2: over-merge / over-split / solo decomposition ----
                        sc = owned & (gt != 0)
                        pr_p = np.zeros(len(gt), dtype=np.int64)
                        pr_p[owned] = cell[assign[owned]]
                        # region id per scored point
                        cell_region = np.full(n_prim, -1, dtype=np.int64)
                        cell_region[vidx_np] = lab.cpu().numpy()
                        rp = cell_region[assign[sc]]
                        yp = gt[sc]
                        ip = inst[sc]
                        pp = pr_p[sc]
                        okreg = rp >= 0
                        # region GT majority
                        rr = rp[okreg]
                        cnt = np.zeros((K, nc), dtype=np.int64)
                        np.add.at(cnt, (rr, yp[okreg]), 1)
                        mstar = cnt.argmax(1)
                        ms = np.full(len(yp), 0, dtype=np.int64)
                        ms[okreg] = mstar[rr]
                        wrong = pp != yp
                        merge = wrong & (yp != ms)
                        rest = wrong & (yp == ms)
                        # PLURALITY prediction over the whole GT instance, by point mass.
                        # "any correct point" was too weak a criterion -- a 5000-point
                        # instance with one lucky point would have counted as over-split.
                        ninst = int(inst.max()) + 2
                        ic = np.zeros((ninst, nc), dtype=np.int64)
                        np.add.at(ic, (ip + 1, pp), 1)
                        ipl = ic.argmax(1)[ip + 1]          # instance plurality prediction
                        split = rest & (ipl == yp)          # object mostly right, these dissent
                        solo = rest & (ipl != yp)           # whole object misread
                        tot = len(yp)
                        # THE OVER-SPLIT ORACLE: give every point its instance's plurality.
                        # This is the exact ceiling of any fragment-merging fix.
                        pr_inst = pr_p.copy()
                        pr_inst[sc] = ipl
                        _, mi_i, _, _ = calculate_metrics(
                            gt_t, torch.from_numpy(pr_inst).long(), nc)
                        decomp.setdefault((K, cs), []).append(dict(
                            n=tot, err=int(wrong.sum()), merge=int(merge.sum()),
                            split=int(split.sum()), solo=int(solo.sum()),
                            unowned=int((~okreg).sum()), inst_oracle=float(mi_i) * 100,
                            base=float(mi) * 100))

            # ---- ITEM 1/4: label-free statistics (seed 0 partition + reseed stability) ----
            lab0 = parts[0]
            pooled = torch.zeros(K, D, device=dev).index_add_(0, lab0, unit)
            cen = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dist = float((unit * cen[lab0]).sum(1).mean())
            cut = float((lab0[ea] != lab0[eb]).float().mean())
            st = [nmi(parts[i], parts[j]) for i in range(len(parts))
                  for j in range(i + 1, len(parts))]
            mdl = K * D * 32 + Nv * np.log2(K)          # bits for codebook + assignments
            resid = Nv * D * 32 * (1 - dist)            # crude residual codelength
            stats.setdefault(K, []).append(dict(distortion=dist, cut=cut,
                                                stability=float(np.mean(st)) if st else 1.0,
                                                mdl=float(mdl + resid)))
        print(f"[{scene}] {(time.time()-t0)/60:.1f} min", flush=True)

    ns = len(stats[KS[0]])
    print(f"\n============ {ns} scenes, {a.seeds} seeds ============")
    print("\n--- ITEM 1/4: mIoU(K) and label-free statistics ---")
    print(f"{'K':>7}{'m19':>8}{'m15':>8}{'m10':>8}{'sd19':>7}{'range19':>9}"
          f"{'stabil':>8}{'distort':>9}{'cut':>7}{'mdl(Mb)':>10}")
    for K in KS:
        if (K, CLASS_SETS[0]) not in miou:
            continue
        v = np.array(miou[(K, CLASS_SETS[0])]).reshape(-1, a.seeds)   # (scene, seed)
        row = "".join(f"{np.mean(miou[(K,c)]):8.2f}" for c in CLASS_SETS)
        sd = np.mean(v.std(1))
        rng = np.mean(v.max(1) - v.min(1))
        s = stats[K]
        print(f"{K:>7}{row}{sd:7.2f}{rng:9.2f}"
              f"{np.mean([x['stability'] for x in s]):8.3f}"
              f"{np.mean([x['distortion'] for x in s]):9.4f}"
              f"{np.mean([x['cut'] for x in s]):7.3f}"
              f"{np.mean([x['mdl'] for x in s])/8e6:10.1f}")

    print("\n--- ITEM 2: error decomposition (% of scored GT points), seed 0 ---")
    print(f"{'K':>7}{'cs':>5}{'err%':>8}{'merge%':>9}{'split%':>9}{'solo%':>8}"
          f"{'merge/err':>11}{'split/err':>11}{'solo/err':>10}{'instOracle':>12}{'d_inst':>8}")
    for K in KS:
        for cs in CLASS_SETS:
            if (K, cs) not in decomp:
                continue
            L = decomp[(K, cs)]
            n = sum(x["n"] for x in L)
            e = sum(x["err"] for x in L)
            mg = sum(x["merge"] for x in L)
            sp = sum(x["split"] for x in L)
            so = sum(x["solo"] for x in L)
            io = np.mean([x["inst_oracle"] for x in L])
            bs = np.mean([x["base"] for x in L])
            print(f"{K:>7}{cs[11:]:>5}{100*e/n:8.2f}{100*mg/n:9.2f}{100*sp/n:9.2f}"
                  f"{100*so/n:8.2f}{100*mg/e:11.1f}{100*sp/e:11.1f}{100*so/e:10.1f}"
                  f"{io:12.2f}{io-bs:+8.2f}")

    with open(a.out, "w") as f:
        json.dump({"miou": {f"{k[0]}|{k[1]}": v for k, v in miou.items()},
                   "stats": {str(k): v for k, v in stats.items()},
                   "decomp": {f"{k[0]}|{k[1]}": v for k, v in decomp.items()}}, f, indent=1)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
