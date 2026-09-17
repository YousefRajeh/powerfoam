"""DECISION RULES on the REPORTED protocol: per-primitive argmax, truefrozen, opacity mask.

WHY A NEW SCRIPT. `eval_decision_rules.py` explores this space thoroughly but ONLY through
clustered pooling (`pos_aware_64x5` / `feat_kmeans320`), and its cache is built on the `_ogl3`
(nonfrozen) arm. The reported protocol is per-primitive cosine argmax on `truefrozen`, so the
existing decision-rule results -- including the +3.3 shrinkage gain and the +8..+11 oracle
per-class-bias headroom -- are measured on a protocol the paper does not use. This runs the same
family per-primitive on the right arm.

THE NEW IDEA: CONFIDENCE-CONDITIONED REFERENCE POPULATION.
Every rule in that family corrects a per-class baseline `g_c` estimated from the scene's own
primitive similarity distribution. That estimate is polluted: at ~37-48 mIoU, most primitives are
misclassified, so `g_c` describes a population dominated by unreliable features. The unsupervised
selectors fail for exactly this reason -- the entropy-picked lambda is worse than no correction at
all on some scenes.

`||x_raw,j||` (the norm of the RAW, un-normalised lift) predicts per-primitive correctness at
AUC 0.738 / 0.643 / 0.697, beating agreement alone, and survives stratification by agreement
(AUC > 0.5 in 13 of 15 quintile strata). So restricting the reference population to the top-q
fraction by that confidence should give a cleaner per-class baseline -- while staying FULLY
UNSUPERVISED (no labels, no class-name information, no prompts).

This also sidesteps the session's main negative result: confidence used as a continuous WEIGHT
loses at every site tried (accumulator weight -0.23/-0.46/-0.53, tangent step -0.14/-0.15/-0.19,
pooling weight -0.13/-0.20/-0.01). Here it is used as a SELECTOR for estimating a statistic, which
is the same category as the one intervention that worked (hard exclusion, +0.86 clustered).

NO LABELS ARE USED BY ANY RULE. GT enters only to score.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, remap_gt_labels)
from diagnose_scannet_miou import (assign_points_to_power_cells, load_foam,
                                   load_scannet_pointcept_gt)

CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
GT_ROOT = r"D:\Downloads\scannet_pointcept"


# ---------------------------------------------------------------- rules
# Every rule takes the (N, C) similarity matrix and a reference population MASK, and returns a
# score matrix whose row-argmax is the prediction. `ref` selects which primitives define the
# per-class baseline; that is the only thing the confidence variants change.

def r_plain(sim, ref):
    return sim


def r_center(sim, ref, lam=1.0):
    """sim - lam * per-class MEAN over the reference population."""
    return sim - lam * sim[ref].mean(0, keepdim=True)


def r_quant(sim, ref, lam=1.0, q=0.5):
    """sim - lam * per-class q-QUANTILE over the reference population.

    Spans the family: q=0.5 is a robust location, q->1 approaches CSLS's upper-tail term
    (CSLS's row term is constant along a row and cannot change an argmax, so it reduces to this).
    """
    return sim - lam * torch.quantile(sim[ref].float(), q, dim=0, keepdim=True)


def r_zscore(sim, ref):
    r = sim[ref]
    return (sim - r.mean(0, keepdim=True)) / r.std(0, keepdim=True).clamp_min(1e-6)


def conf_mask(conf, q):
    """Top-q fraction by confidence. q=1.0 -> all primitives (the existing behaviour)."""
    if q >= 1.0:
        return torch.ones_like(conf, dtype=torch.bool)
    thr = torch.quantile(conf.float(), 1.0 - q)
    m = conf >= thr
    if int(m.sum()) < 10:                      # never let the reference set collapse
        return torch.ones_like(conf, dtype=torch.bool)
    return m


def build_rules():
    R = {"plain": (r_plain, 1.0)}
    for lam in (0.25, 0.5, 0.75, 1.0):
        R[f"cAll{lam}"] = ((lambda l: lambda s, r: r_center(s, r, l))(lam), 1.0)
    R["zAll"] = (r_zscore, 1.0)
    for q in (0.5, 0.9):
        R[f"qAll{q}"] = ((lambda qq: lambda s, r: r_quant(s, r, 1.0, qq))(q), 1.0)
    # --- the new family: same corrections, reference population restricted by CONFIDENCE ---
    for cq in (0.5, 0.25, 0.1):
        for lam in (0.5, 1.0):
            R[f"cConf{lam}@{cq}"] = ((lambda l: lambda s, r: r_center(s, r, l))(lam), cq)
        R[f"zConf@{cq}"] = (r_zscore, cq)
        R[f"qConf0.5@{cq}"] = ((lambda: lambda s, r: r_quant(s, r, 1.0, 0.5))(), cq)
    return R


def main():
    enable_determinism()
    device = "cuda"
    rules = build_rules()
    want = [r for r in os.environ.get("RULES", "").split(",") if r] or list(rules)
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = [s for s in SCENES if (s in only if only else True)]
    feat_file = os.environ.get("FEAT_FILE", "solved_geometric_median_truefrozen_ogl3")
    conf_file = os.environ.get("CONF_FILE", "solved_rawlift_nonorm_tf")
    recon = os.environ.get("RECON", "truefrozen")

    print(f"[config] RECON={recon} FEAT_FILE={feat_file} CONF_FILE={conf_file} "
          f"rules={len(want)} scenes={len(scenes)}", flush=True)

    res = {r: {cs: {} for cs in CLASS_SETS} for r in want}
    text_cache = {}
    import glob

    for scene in scenes:
        cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        centers, radii = load_foam(f"output/scannet_{scene}_{recon}", device)
        d = torch.load(f"artifacts/scannet/{scene}/{feat_file}.pt", map_location=device,
                       weights_only=True)
        feats = d["primitive_features"].to(device).float()
        valid = d["valid_mask"].cpu().numpy()
        cd_ = torch.load(f"artifacts/scannet/{scene}/{conf_file}.pt", map_location=device,
                         weights_only=True)
        conf_all = cd_["primitive_features"].to(device).float().norm(dim=-1)

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
        owned = assigned >= 0
        vidx = np.where(valid)[0]
        vt = torch.from_numpy(vidx).to(device)
        unit = F.normalize(feats[vt], dim=-1)
        conf = conf_all[vt]
        # map primitive id -> row in the valid array, so predictions can be scattered back
        row_of = np.full(centers.shape[0], -1, dtype=np.int64)
        row_of[vidx] = np.arange(vidx.size)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        print(f"\n===== {scene} (valid={vidx.size:,}) =====", flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]
            tnames = [n for _, n in kept]
            nC = len(tids)
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            key = tuple(tnames)
            if key not in text_cache:
                text_cache[key] = embed_class_names(tnames, device)
            sim = unit @ text_cache[key].T                         # (Nvalid, C)

            for rname in want:
                fn, cq = rules[rname]
                ref = conf_mask(conf, cq)
                score = fn(sim, ref)
                cls = score.argmax(dim=-1).cpu().numpy()
                pred = np.zeros(raw_labels.shape[0], dtype=np.int64)
                rows = row_of[assigned[owned]]
                ok = rows >= 0
                sel = np.where(owned)[0][ok]
                pred[sel] = cls[rows[ok]] + 1
                _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
                res[rname][cs][scene] = {"mIoU": miou, "mAcc": macc}
            print(f"  {cs}: " + "  ".join(
                f"{r}={res[r][cs][scene]['mIoU']*100:.2f}" for r in want[:6]), flush=True)

    # ------------------------------------------------------------ report
    print("\n\n=== per-primitive decision rules, "
          f"{len(scenes)} scenes, {recon} ===")
    base = "plain" if "plain" in res else want[0]
    hdr = f"{'rule':<16}" + "".join(f"{cs[13:]:>9}" for cs in CLASS_SETS) + "   delta vs plain"
    print(hdr)
    import statistics as st
    rows = []
    for r in want:
        m = [st.mean([v["mIoU"] for v in res[r][cs].values()]) * 100 for cs in CLASS_SETS]
        b = [st.mean([v["mIoU"] for v in res[base][cs].values()]) * 100 for cs in CLASS_SETS]
        wins = [sum(1 for s in res[r][cs] if res[r][cs][s]["mIoU"] > res[base][cs][s]["mIoU"])
                for cs in CLASS_SETS]
        rows.append((st.mean([x - y for x, y in zip(m, b)]), r, m, b, wins))
        print(f"{r:<16}" + "".join(f"{x:>9.2f}" for x in m) + "   " +
              " ".join(f"{x-y:+.2f}" for x, y in zip(m, b)) +
              "   wins " + "/".join(str(w) for w in wins))
    rows.sort(reverse=True)
    print(f"\nBEST: {rows[0][1]}  mean delta {rows[0][0]:+.3f}")
    out = f"artifacts/scannet/decision_rules_pp_{len(scenes)}scene.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
