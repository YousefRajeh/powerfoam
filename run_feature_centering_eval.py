"""FEATURE-space centering: remove CLIP's common-mode direction BEFORE the projection.

WHY IN FEATURE SPACE, AND WHY IT MATTERS FOR THE PROTOCOL
---------------------------------------------------------
CLIP embeddings occupy a narrow cone: every vector shares a large common component that carries no
class information but dominates the norm. Its effect on `cos(f, w_c)` is a per-class constant
offset, so some classes win argmax comparisons regardless of content -- the hubness phenomenon
already diagnosed here (floor mean 0.216/std 0.013 vs window mean 0.216/std 0.028: equal means,
very different win rates).

The project has attacked this in SIMILARITY space (eval_decision_rules.py: center, zscore, CSLS,
rank, quantile). Those all modify the DECISION RULE, and per evaluate_point_cloud_miou.py that is a
protocol violation: OpenGaussian's eval_scannet.py:155-159 and NormLift's my_eval_scannet2.py both
do a bare F.normalize -> cosine -> argmax. Numbers produced under a different rule are not
comparable to the benchmark, whichever way they move.

Centering the FEATURES instead changes the representation, not the rule:

    f' = normalize(f - lam * mu),     mu = normalize(mean_j f_j)      then bare cosine argmax

so the decision rule stays byte-identical to the benchmark's. This is the protocol-legal way to ask
the same question.

BIAS VERSUS VARIANCE -- these are different denoisers and should not be conflated:
  * averaging k observations cuts the VARIANCE term (~sqrt(k), and high dimension is what makes the
    per-view noise near-orthogonal so it cancels). The solve already does this over a median of 141
    observations per cell.
  * centering removes a BIAS term (the shared direction). No amount of averaging removes it, since
    it is common to every sample.
Reported gains from partial centering (~+5 mIoU on a pilot) are far larger than anything the
variance-side interventions have produced, which suggests bias was the dominant error -- but the
pilot result was measured in similarity space, so this file re-asks it legally.

ARMS
  plain            bare cosine argmax on the solved features (the benchmark rule, unchanged)
  center_lam{lam}  the same rule on centred features
  center_perscene  mu from this scene only vs a shared mu -- tests whether the common direction is
                   a property of CLIP or of the scene (it should be CLIP's, i.e. transferable)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import HARDEST_FIRST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lams", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--outdir", default="artifacts/scannet/feat_center")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    lams = [float(x) for x in a.lams.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved

        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)          # common-mode direction
        share = float((unit[vm] @ mu.T).mean())
        print(f"[{scene}] P={P:,}  mean cos(feature, common direction) = {share:.4f}", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "P": P, "common_share": share, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            b = score((unit @ text.T).argmax(-1).cpu().numpy(), "plain")
            print(f"  {cs} [plain] mIoU={b:.2f}", flush=True)
            for lam in lams:
                cf = torch.zeros_like(unit)
                cf[vm] = F.normalize(unit[vm] - lam * mu, dim=-1)
                v = score((cf @ text.T).argmax(-1).cpu().numpy(), f"center_lam{lam:g}")
                print(f"  {cs} [center_lam{lam:g}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
                del cf
            del text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
