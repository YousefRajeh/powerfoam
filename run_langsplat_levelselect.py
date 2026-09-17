"""LangSplat scored with ITS OWN per-query level selection, not one level at a time.

WHY. LangSplat trains three language fields (subpart / part / whole) and never queries a single one.
`eval/evaluate_iou_loc.py:146-152` picks, per query, the level whose relevancy map has the highest
maximum score and reports THAT level's result:

    for i in range(n_head):  score_lvl[i] = valid_map[i, k].max()
    chosen_lvl = torch.argmax(score_lvl)
    chosen_iou_list.append(iou_lvl[chosen_lvl])

Selection is by the model's own confidence, not by ground truth, so it is legitimate to reproduce.
Reporting l1/l2/l3 as three independent rows -- which is what our table did -- is therefore not
their protocol and understates them.

THE ADAPTATION, stated because it is not a pure port. Their evaluation is per-query 2D localisation
and IoU: each query independently picks a level. A closed-set 3D segmentation must instead emit ONE
label per primitive. So the level choice is made per CLASS exactly as they make it per query --
level(c) = argmax over levels of the maximum relevancy that level assigns to class c anywhere in
the scene -- and each class is then scored from its chosen level:

    score(i, c) = sim_{level(c)}(i, c),      label(i) = argmax_c score(i, c)

This keeps their mechanism (confidence-selected granularity per query) while producing the single
labelling the 3D protocol needs. Everything else -- opacity mask, owner assignment, present-classes
bank, missed-class penalty -- is identical to every other row.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = r"D:\Downloads\scannet_pointcept"
OPACITY_THRESH = 0.1
LEVELS = ["langsplat_frozen_l1", "langsplat_frozen_l2", "langsplat_frozen_l3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/baseline_eval/manifest.json")
    ap.add_argument("--out", default="artifacts/baseline_eval/langsplat_levelsel.json")
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    enable_determinism()
    dev = a.device

    man = json.load(open(a.manifest))
    by = {}
    for e in man:
        if e["tag"] in LEVELS:
            by.setdefault(e["scene"], {})[e["tag"]] = e

    rows = []
    for scene, per in sorted(by.items()):
        if len(per) != 3:
            print(f"[skip] {scene}: {len(per)}/3 levels", flush=True)
            continue
        feats = []
        for t in LEVELS:
            f = torch.load(per[t]["features"], map_location="cpu", weights_only=False).float()
            feats.append(f.to(dev))
        ck = torch.load(per[LEVELS[0]]["ckpt"], map_location="cpu", weights_only=False)["splats"]
        opacity = torch.sigmoid(ck["opacities"].float()).numpy().reshape(-1)

        apth = f"artifacts/ablation_cache/{scene}_gs_froz_assign.npy"
        if not os.path.exists(apth):
            print(f"[skip] {scene}: no assignment cache", flush=True)
            continue
        assign = np.load(apth)
        owned = assign >= 0

        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        pts = np.asarray(gt_pts, dtype=np.float64)
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())

        for cs in a.class_sets.split(","):
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)

            sims = []
            for f in feats:
                z = f.norm(dim=-1) == 0
                s = F.normalize(f, dim=-1) @ text.T          # (P, C)
                s[z] = -np.inf                               # no feature -> cannot win any class
                sims.append(s)
            S = torch.stack(sims, 0)                          # (3, P, C)

            # their rule: per class, the level whose maximum relevancy over the scene is highest
            per_class_max = torch.stack([s.masked_fill(torch.isinf(s), -1e9).max(dim=0).values
                                         for s in sims], 0)   # (3, C)
            chosen = per_class_max.argmax(dim=0)              # (C,)
            sel = S[chosen, :, torch.arange(len(names), device=dev)].T   # (P, C)

            best, cls = sel.max(dim=-1)
            pred_prim = (cls + 1).cpu().numpy()
            pred_prim[torch.isinf(best).cpu().numpy()] = 0    # no level had a feature

            pred = np.zeros(gt.shape[0], dtype=np.int64)
            pred[owned] = pred_prim[assign[owned]]

            gt_m = gt.copy()
            low = np.zeros(gt.shape[0], dtype=bool)
            low[owned] = opacity[assign[owned]] < OPACITY_THRESH
            scored_before = int((gt_m != 0).sum())
            gt_m[low] = 0
            dropped = (scored_before - int((gt_m != 0).sum())) / max(scored_before, 1) * 100

            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_m).long(),
                                                 torch.from_numpy(pred).long(), nc)
            sm = semantic_surface_metrics(GTSurfaceIndex(pts, gt_m, nc), pred)
            rec = {"tag": "langsplat_levelsel", "scene": scene, "class_set": cs,
                   "miou": float(miou) * 100, "macc": float(macc) * 100,
                   "dropped_pct": dropped,
                   "levels_chosen": {names[i]: int(chosen[i]) + 1 for i in range(len(names))}}
            rec.update({k: float(sm[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(sm.get(k), (int, float))})
            rows.append(rec)
            lv = np.bincount(chosen.cpu().numpy(), minlength=3)
            print(f"  {scene} {cs:15s} mIoU={rec['miou']:5.2f} scd={rec['scd']:.4f} "
                  f"hd95={rec['hd95']:.4f} bF1={rec['boundary_f1']:.3f} "
                  f"missed={rec.get('n_missed',0):.0f}  levels l1/l2/l3={lv[0]}/{lv[1]}/{lv[2]}",
                  flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")
    for cs in a.class_sets.split(","):
        v = [r for r in rows if r["class_set"] == cs]
        if v:
            m = lambda k: float(np.mean([x[k] for x in v]))
            print(f"{cs:16s} n={len(v):>2}  mIoU {m('miou'):6.2f}  SCD {m('scd'):.4f}  "
                  f"HD95 {m('hd95'):.4f}  BF1 {m('boundary_f1'):.3f}  "
                  f"missed {m('n_missed'):.2f}")


if __name__ == "__main__":
    main()
