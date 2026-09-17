"""Multi-class Potts smoothing, scored in the REPORTED protocol, directly against 37.37/40.20/48.14.

CONFIGURATION -- matches the paper exactly:
    recon      truefrozen
    features   solved_geometric_median_truefrozen_ogl3
    grouping   NONE (per-primitive cosine argmax)
    culling    OpenGaussian's low-opacity GT masking at 0.1
    metric     mIoU / mAcc over the 3 OpenGaussian class sets

The only thing that varies is the decision rule:
    lam = 0   plain argmax                -- by our separability proposition this is the exact
                                             optimum of the unary problem, and equals FlashSplat's
                                             weighted majority vote for discrete labels
    lam > 0   + Potts prior on the EXACT facet graph (the power diagram's Delaunay dual)

WHY THIS IS THE ONLY PLACE GEOMETRY CAN HELP. The proposition says every objective linear in the
per-primitive label depends on `A` only through `A^T B` and is separable, so no partition-dependent
correction to the unary problem can improve it. A PAIRWISE term is outside that theorem -- and it
needs an exact adjacency, which a power diagram has (jaccard 1.0000 against radfoam's own CUDA
Delaunay) and 3DGS does not (alpha graph mean degree 0.05 at gsplat's own 3-sigma bound).

NOT SINGLE-QUERY. This variant needs the whole class set, so it is comparable to the reported mIoU
but does not satisfy the one-query-at-a-time constraint. The binary `graphcut.binary_graphcut` is
the single-query counterpart; they are two methods sharing a prior.

ICM is a LOCAL optimiser (see graphcut.py); results are reported as such, not as an optimal solve.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from graphcut import multiclass_potts_icm
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, apply_gt_opacity_mask,
                                       calculate_metrics, embed_class_names, remap_gt_labels)
from diagnose_scannet_miou import (assign_points_to_power_cells, load_foam,
                                   load_scannet_pointcept_gt)

CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
GT_ROOT = r"D:\Downloads\scannet_pointcept"
PAPER = {"opengaussian19": (37.60, 59.38), "opengaussian15": (40.48, 62.43),
         "opengaussian10": (48.16, 68.32)}


def main():
    enable_determinism()
    device = "cuda"
    lams = [float(x) for x in os.environ.get("LAMS", "0,0.002,0.005,0.01,0.02,0.05").split(",")]
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = [s for s in SCENES if (s in only if only else True)]
    recon = os.environ.get("RECON", "truefrozen")
    feat_file = os.environ.get("FEAT_FILE", f"solved_geometric_median_{recon}_ogl3")
    cull = os.environ.get("GT_OPACITY_MASK", "1") == "1"
    thr = float(os.environ.get("OPACITY_THRESHOLD", "0.1"))
    print(f"[config] recon={recon} feat={feat_file} culling={cull}@{thr} lams={lams} "
          f"scenes={len(scenes)}", flush=True)

    res = {l: {cs: {} for cs in CLASS_SETS} for l in lams}

    for scene in scenes:
        cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        if cull:
            centers, radii, density = load_foam(f"output/scannet_{scene}_{recon}", device,
                                                return_density=True)
            alpha = 1.0 - np.exp(-density * (radii.reshape(-1) * 2.0))
        else:
            centers, radii = load_foam(f"output/scannet_{scene}_{recon}", device)
            alpha = None
        d = torch.load(f"artifacts/scannet/{scene}/{feat_file}.pt", map_location=device,
                       weights_only=True)
        feats = d["primitive_features"].to(device).float()
        valid = d["valid_mask"].cpu().numpy()
        assert feats.shape[0] == centers.shape[0], (feats.shape[0], centers.shape[0])

        g = torch.load(f"artifacts/ablation_cache/{scene}_pf_"
                       f"{'tfroz' if recon == 'truefrozen' else 'nonfroz'}_delaunay.pt",
                       map_location="cpu", weights_only=False)
        indptr = g["offsets"].numpy().astype(np.int64)
        indices = g["adjacent"].numpy().astype(np.int64)

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
        owned = assigned >= 0
        unit = F.normalize(feats, dim=-1)
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        print(f"\n===== {scene} =====", flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            nC = len(tids)
            gt_np = remap_gt_labels(raw_labels, tids)
            if cull:
                gt_np, _ = apply_gt_opacity_mask(gt_np, assigned, alpha, thr, f"{scene}/{cs}")
            gt_t = torch.from_numpy(gt_np).long()
            text = embed_class_names([n for _, n in kept], device)
            sim = (unit @ text.T).cpu().numpy()

            for lam in lams:
                lab = (sim.argmax(1) if lam == 0 else
                       multiclass_potts_icm(sim, indptr, indices, lam=lam,
                                            live=valid.astype(bool)))
                pred = np.zeros(raw_labels.shape[0], dtype=np.int64)
                pred[owned] = lab[assigned[owned]] + 1
                _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
                res[lam][cs][scene] = {"mIoU": miou, "mAcc": macc}
            print(f"  {cs}: " + "  ".join(
                f"lam{l}={res[l][cs][scene]['mIoU']*100:.2f}" for l in lams), flush=True)

    print(f"\n\n=== REPORTED PROTOCOL: per-primitive argmax, {recon}, gm, culling={cull} ===")
    print(f"{'lam':>8}" + "".join(f"{cs[13:] + ' mIoU':>14}{'mAcc':>8}" for cs in CLASS_SETS))
    base = {}
    for lam in lams:
        row = []
        for cs in CLASS_SETS:
            mi = st.mean([v["mIoU"] for v in res[lam][cs].values()]) * 100
            ma = st.mean([v["mAcc"] for v in res[lam][cs].values()]) * 100
            row += [mi, ma]
            if lam == lams[0]:
                base[cs] = mi
        print(f"{lam:>8}" + "".join(f"{row[2*i]:>14.2f}{row[2*i+1]:>8.2f}"
                                    for i in range(len(CLASS_SETS))))
    print(f"\n{'lam':>8}" + "".join(f"{cs[13:] + ' d':>14}" for cs in CLASS_SETS) +
          "   (delta vs lam=0)")
    for lam in lams:
        print(f"{lam:>8}" + "".join(
            f"{st.mean([v['mIoU'] for v in res[lam][cs].values()])*100 - base[cs]:>+14.2f}"
            for cs in CLASS_SETS))
    print("\npaper reference: " + "  ".join(
        f"{cs[13:]} {PAPER[cs][0]:.2f}/{PAPER[cs][1]:.2f}" for cs in CLASS_SETS))
    out = f"artifacts/scannet/icm_paperconfig_{len(scenes)}scene.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in res.items()}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
