"""Instantiate the mIoU-lever theorems with measured quantities.

Companion to ResearchVault/Methods/mIoU-levers-proof.md. The theorems there are exact; this
fills in the numbers that decide whether the bounds BITE on our data.

  Theorem 2 (coverage ceiling)   mIoU <= mean_c cov_c, with
                                 cov_c = |covered GT points of class c| / N_c
                                 -> reports the ceiling and the fraction of it attained
  Theorem 3 (reweight invariance) if margin_i > 4 sin(rho_i) the cell's label cannot be
                                 changed by ANY nonnegative reweighting of its views
                                 -> reports the flippable fraction phi

WHAT phi MEANS, and its known bias. rho_i is the within-cell angular SPREAD of the per-view
features. AccumulatedFeatureStats keeps only aggregates, so rho_i is estimated from `c_intra`
(mean within-cell cosine to the cell mean) as arccos(c_intra). That is a MEAN-based estimate,
not an upper bound: true rho_i is larger, so the true flippable set is LARGER than reported.
phi is therefore a lower bound on flippability and an upper bound on how tightly Theorem 3
constrains us. Stated here rather than buried, because a bound with the wrong sign of error
would be worse than no bound.

Also reports the ORACLE ceiling: mIoU if every covered cell were given its own majority GT
label. That is the maximum any decision rule can reach at fixed coverage and assignment, and
together with the coverage ceiling it brackets where the remaining headroom lives.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from feature_foam_lifting.operator import AccumulatedFeatureStats

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--assignment", default="nearest_valid",
                    choices=["nearest_valid", "geometric"])
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    agg = {cs: [] for cs in CLASS_SETS}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        apth = (f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
                if a.assignment == "nearest_valid"
                else f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy")
        if not all(os.path.exists(p) for p in (fp, stp, apth)):
            print(f"[skip] {scene}", flush=True)
            continue
        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        rel = AccumulatedFeatureStats.load(stp).reliability()
        c_intra = rel["c_intra"].cpu().numpy()
        # rho estimate; see docstring for the direction of the bias
        rho = np.arccos(np.clip(c_intra, -1.0, 1.0))
        thresh = 4.0 * np.sin(np.clip(rho, 0, np.pi / 2))

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            sim = unit @ text.T
            top2 = sim.topk(2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            cls = sim.argmax(-1).cpu().numpy() + 1

            sc = owned.copy()
            sc[owned] = valid[assign[owned]]
            pred = np.zeros(len(gt), dtype=np.int64)
            pred[sc] = cls[assign[sc]]
            _, miou, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                              torch.from_numpy(pred).long(), nc)
            miou = float(miou) * 100

            # --- Theorem 2: coverage ceiling (per-class coverage, averaged over present classes)
            covs = []
            for c in range(1, nc):
                Nc = int((gt == c).sum())
                if Nc == 0:
                    continue
                covs.append(float(((gt == c) & sc).sum()) / Nc)
            ceiling = 100.0 * float(np.mean(covs)) if covs else float("nan")

            # --- oracle at fixed coverage/assignment:each cell gets its own majority GT label
            H = np.zeros((len(valid), nc), dtype=np.int64)
            ok = sc & (gt > 0)
            np.add.at(H, (assign[ok], gt[ok]), 1)
            orc_cls = H.argmax(1)
            pred_o = np.zeros(len(gt), dtype=np.int64)
            pred_o[sc] = orc_cls[assign[sc]]
            _, oracle, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred_o).long(), nc)
            oracle = float(oracle) * 100

            # --- Theorem 3: flippable fraction over SCORED points
            flip_cell = margin <= thresh
            phi = float(flip_cell[assign[sc]].mean()) * 100 if sc.sum() else float("nan")
            agg[cs].append((scene, miou, ceiling, oracle, phi,
                            float(np.median(margin[valid])),
                            float(np.median(thresh[valid]))))

        print(f"[{scene}] done", flush=True)

    print(f"\n=== assignment = {a.assignment} ===")
    for cs in CLASS_SETS:
        rows = agg[cs]
        if not rows:
            continue
        m = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
        print(f"\n--- {cs[11:]} classes ({len(rows)} scenes) ---")
        print(f"  mIoU                              {m[:,0].mean():6.2f}")
        print(f"  Theorem 2 ceiling (mean coverage) {m[:,1].mean():6.2f}"
              f"   -> attained {100*m[:,0].mean()/m[:,1].mean():5.1f}% of ceiling")
        print(f"  oracle @ fixed coverage           {m[:,2].mean():6.2f}"
              f"   -> decision-rule headroom {m[:,2].mean()-m[:,0].mean():+6.2f}")
        print(f"  Theorem 3 flippable phi           {m[:,3].mean():6.2f}%"
              f"  (median margin {m[:,4].mean():.4f} vs 4sin(rho) {m[:,5].mean():.4f})")
        print(f"     => reweighting can alter at most {m[:,3].mean():.1f}% of scored points;"
              f" >= {100-m[:,3].mean():.1f}% are provably invariant")


if __name__ == "__main__":
    main()
