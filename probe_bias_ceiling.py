"""ORACLE DIAGNOSTIC (not a method): how much mIoU is reachable by a per-class ADDITIVE BIAS
on the similarity matrix?

Partial centering is exactly `sim - lam * mean_c`, i.e. one particular point in a C-dimensional
bias space. Coordinate-ascending that bias against the GT metric gives the CEILING of the entire
centering / CSLS / hubness-correction family -- everything that corrects a per-class OFFSET and
nothing else. If the oracle is far above centering, the family is worth more work; if it is
close, the family is saturated and the remaining headroom lives elsewhere.

Also reports the UNSUPERVISED lambda selector: pick lam maximising the entropy of the
point-weighted predicted-label histogram. Uses no labels -- only the prediction itself.
"""
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, remap_gt_labels, embed_class_names,
)
from run_cluster_classify_eval import HARD_FIRST, K_FLAT

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"


def main():
    device = "cuda"
    suffix = "_ogl3"
    cs = os.environ.get("CLASS_SET", "opengaussian19")
    scenes = [s for s in os.environ.get("SCENES", ",".join(HARD_FIRST[:4])).split(",") if s]

    for scene in scenes:
        c = torch.load(os.path.join(CACHE, f"{scene}{suffix}.pt"), map_location="cpu", weights_only=False)
        unit = F.normalize(c["unit"].to(device).float(), dim=-1)
        raw = c["raw_labels"].numpy()
        prow = c["point_row"].numpy()
        owned = prow >= 0
        labels = c["pos_labels"].to(device)
        n2i = {n: i for i, n in enumerate(c["all_names"])}
        present = set(np.unique(raw).tolist())
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        nC = len(kept)
        gt_t = torch.from_numpy(remap_gt_labels(raw, [i for i, _ in kept])).long()
        text = embed_class_names([n for _, n in kept], device)

        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, labels, unit)
        nonempty = pooled.norm(dim=-1) > 1e-8
        pooled = F.normalize(pooled, dim=-1)
        sl = pooled @ text.T
        colmean = (unit @ text.T).mean(0)
        lab_np = labels.cpu().numpy()

        def score(bias):
            s = sl - bias
            cls = torch.full((K_FLAT,), -1, dtype=torch.long, device=device)
            cls[nonempty] = s[nonempty].argmax(-1)
            pc = cls.cpu().numpy()[lab_np]
            pred = np.zeros(raw.shape[0], dtype=np.int64)
            pred[owned] = pc[prow[owned]] + 1
            _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
            return miou, macc, pred

        # --- lam sweep + unsupervised entropy selector
        lams = np.arange(0.0, 1.01, 0.05)
        rows = []
        for lam in lams:
            miou, macc, pred = score(lam * colmean)
            h = np.bincount(pred[pred > 0], minlength=nC + 1)[1:].astype(float)
            h = h / max(h.sum(), 1)
            ent = float(-(h[h > 0] * np.log(h[h > 0])).sum())
            rows.append((lam, miou, macc, ent))
        best_lam = max(rows, key=lambda r: r[1])
        ent_lam = max(rows, key=lambda r: r[3])

        # --- oracle coordinate ascent over the full per-class bias vector
        bias = (best_lam[0] * colmean).clone()
        cur = score(bias)[0]
        grid = torch.linspace(-0.06, 0.06, 25, device=device)
        for _ in range(3):
            for k in range(nC):
                base = bias[k].item()
                best_v, best_s = base, cur
                for d in grid.tolist():
                    bias[k] = base + d
                    s = score(bias)[0]
                    if s > best_s:
                        best_s, best_v = s, base + d
                bias[k] = best_v
                cur = best_s

        print(f"\n{scene} [{cs}]  N={unit.shape[0]}")
        print("  lam   mIoU   mAcc   H(pred)")
        for lam, miou, macc, ent in rows[::2]:
            print(f"  {lam:.2f}  {miou*100:5.2f}  {macc*100:5.2f}  {ent:.3f}")
        print(f"  best lam by mIoU     = {best_lam[0]:.2f} -> {best_lam[1]*100:.2f}")
        print(f"  lam picked by ENTROPY= {ent_lam[0]:.2f} -> {ent_lam[1]*100:.2f}  "
              f"(unsupervised; oracle-lam costs {(best_lam[1]-ent_lam[1])*100:+.2f})")
        print(f"  ORACLE per-class bias   -> {cur*100:.2f}  "
              f"(headroom over best lam: {(cur-best_lam[1])*100:+.2f})")


if __name__ == "__main__":
    main()
