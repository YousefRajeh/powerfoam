"""SINGLE-QUERY (open-vocabulary) evaluation: one class at a time, no argmax over a class set.

WHY THIS PROTOCOL. A method that needs the full class set present is not usable for a single
open-vocabulary query. Two consequences that kill most of what this project has measured:

  * Any PER-CLASS offset/scale correction is a MONOTONE transform of that class's scores, so it
    cannot change the ranking of primitives for a single query. Verified: per-class centering
    (the +2.00/+1.19/+1.47 winner in A14), per-class z-score, and feature-centering WITHOUT
    renormalisation all leave the single-query ranking bit-identical. They only ever helped by
    changing how class c compares with class c' inside an argmax.

  * Conversely, any PER-PRIMITIVE factor is EXACTLY INERT under argmax
    (`argmax_c conf_j * sim_jc == argmax_c sim_jc`) but reorders single-query ranking completely.
    Verified: 0 % of argmax decisions change, 100 % of ranking positions move.

That second point retroactively explains this session's confidence nulls. `||x_raw,j||` predicts
per-primitive correctness at AUC 0.738/0.643/0.697, yet every attempt to spend it failed --
accumulator weight, tangent step, soft pooling weight -- and the ONE success (A8's pooling gate)
worked only because pooling MIXES primitives, which is the sole channel by which a per-primitive
quantity can reach an argmax. The signal was never weak; the multi-class protocol is structurally
blind to it. Single-query is not.

METRIC. Threshold-free: per-class Average Precision over GT POINTS, so no threshold has to be
tuned or transferred between scenes. Also reports IoU at the oracle threshold (an upper bound, NOT
a method) and IoU at a fixed prevalence-matched cut, so the numbers can be compared with the
argmax protocol's IoU at least in spirit.

ARMS. Everything here is query-INDEPENDENT or per-primitive, hence single-query legal:
  plain          rank by sim_jc
  conf{p}        rank by (conf_j ** p) * sim_jc          -- per-primitive confidence as a factor
  gate{q}        exclude the bottom (1-q) by confidence, then rank by sim_jc
  centre         rank by normalize(x_j - mu) . t_c       -- mu is the scene's mean primitive
                                                            feature; query-independent
  centre+conf1   both
No labels are used by any arm. GT enters only to score.
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
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_gaussian_means_opacities)
from diagnose_scannet_miou import (assign_points_to_power_cells, load_foam,
                                   load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_nearest_center
from readout import embed_negatives


def load_backend(recon, scene, device):
    """-> (n_primitives, assign_fn(gt_points, valid) -> assigned) for foam OR 3DGS.

    3DGS has no power cells: a Gaussian's territory is its nearest-centre Voronoi region, which is
    what `evaluate_point_cloud_miou.evaluate_splat_feature_solver` uses (line 278). Opacity gates
    the CANDIDATE SET there, matching OpenGaussian's own `--frozen_init_pts` protocol; we do the
    same so the 3DGS arm is scored the way that repo scores it, not the way foam is.
    """
    if recon.startswith("gs_"):
        ck = f"recon_remote/{recon}/{scene}/ckpt.pt"
        means, opac = load_gaussian_means_opacities(ck, device)
        keep = opac >= float(os.environ.get("OPACITY_THRESHOLD", "0.1"))
        print(f"  [3DGS] {means.shape[0]:,} gaussians, {int(keep.sum()):,} with opacity>=0.1",
              flush=True)

        def assign(gt_points, valid):
            return assign_points_to_nearest_center(gt_points, means, valid=(valid & keep))
        return means.shape[0], assign

    centers, radii = load_foam(f"output/scannet_{scene}_{recon}", device)

    def assign(gt_points, valid):
        return assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
    return centers.shape[0], assign

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
GT_ROOT = r"D:\Downloads\scannet_pointcept"

# Shared, scene-independent cuts for the DEPLOYABLE IoU columns. Quantiles are of each scene's own
# score distribution (the rule is shared, the cut adapts); absolute cosines are shared outright.
# CLIP cosines to class text span only ~0.13-0.22 in this data, hence the tight absolute grid.
QUANTS = (0.90, 0.95, 0.99)
THRESHES = (0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28)
# The negative-query readout lives on a different scale from a raw cosine: it is a probability
# against a fixed reference, so its natural operating point is 0.5 ("closer to the query than to
# any generic negative") rather than a hand-picked point in the narrow 0.13-0.23 cosine band.
# Both sets are evaluated for every arm; only the arm's own natural point is interpretable, and
# `NATURAL` records which that is.
RELEV_THRESHES = (0.4, 0.5, 0.6)
NATURAL = {"relev": 0.5}
NATURAL_DEFAULT = 0.21


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """AP = area under the precision-recall curve, computed exactly by rank (no interpolation).

    Ties matter here: many primitives share a score after gating (all excluded ones get -inf), so
    ranking must be stable and the positives among tied scores must not be counted optimistically.
    Sorting by (-score) with a stable sort and using cumulative counts handles it.
    """
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], y[order]
    tp = np.cumsum(y)
    k = np.arange(1, y.size + 1)
    # TIES MUST BE GROUPED. Accumulating item-by-item lets array order break ties, which is
    # optimistic: with all scores equal and the positives listed first this returns 1.0 instead of
    # the prevalence. That would have inflated every gated arm, since gating puts all excluded
    # primitives in one -inf tie block. Evaluate precision/recall only at DISTINCT score
    # boundaries, so every item sharing a score is counted together (sklearn's convention).
    last = np.r_[s[1:] != s[:-1], True]
    p = tp[last] / k[last]
    r = tp[last] / n_pos
    return float((p * np.diff(np.r_[0.0, r])).sum())


def iou_at_oracle(scores: np.ndarray, y: np.ndarray) -> float:
    """Best IoU over all thresholds -- an UPPER BOUND, not a method (it uses labels)."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, ys.size + 1)
    return float((tp / (k + n_pos - tp)).max())


def iou_at_prevalence(scores: np.ndarray, y: np.ndarray) -> float:
    """IoU when the top-|P| scored items are predicted positive, |P| = true positive count.

    Uses the class PREVALENCE only (a count, not which items) -- reported as a calibration-free
    companion to AP, and flagged as still using one number derived from labels.
    """
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sel = np.zeros(y.size, dtype=bool)
    sel[order[:n_pos]] = True
    inter = int((sel & (y > 0)).sum())
    return float(inter / (sel.sum() + n_pos - inter))


def iou_at_score(scores: np.ndarray, y: np.ndarray, t: float) -> float:
    """IoU when everything scoring >= t is predicted positive.

    A FIXED, SCENE-INDEPENDENT, LABEL-FREE threshold -- so unlike `iou_at_oracle` (best over
    thresholds) and `iou_at_prevalence` (uses the true positive count), this one is actually
    DEPLOYABLE: choose t once, apply it to every scene and every query.
    """
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    sel = scores >= t
    inter = int((sel & (y > 0)).sum())
    union = int(sel.sum()) + n_pos - inter
    return float(inter / union) if union else 0.0


def iou_at_quantile(scores: np.ndarray, y: np.ndarray, q: float) -> float:
    """IoU when the top (1-q) fraction BY SCORE is predicted positive.

    Also label-free and deployable: the RULE (a quantile) is shared across scenes, even though the
    resulting cut adapts to each scene's own score distribution. Reported alongside the absolute
    thresholds because CLIP cosines drift between scenes, so a shared absolute t may be a worse
    rule than a shared quantile.
    """
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    fin = scores[np.isfinite(scores)]
    if fin.size == 0:
        return float("nan")
    return iou_at_score(scores, y, float(np.quantile(fin, q)))


def build_arms():
    """arm -> (feature_transform, score_transform). Both query-independent or per-primitive."""
    def ident(unit, mu):
        return unit

    def centre(unit, mu):
        return F.normalize(unit - mu, dim=-1)

    arms = {"plain": (ident, None)}
    for p in (0.5, 1.0, 2.0):
        arms[f"conf{p}"] = (ident, ("mul", p))
    # NOTE ON THE RANGE. The deployable-IoU optimum sits at the LEAST aggressive setting tested,
    # and the trend over 0.25 -> 0.5 -> 0.75 is monotonically improving, so the sweep must extend
    # toward 1.0 or it truncates before the optimum. mAP peaks at 0.5 and deployable IoU does not;
    # trusting the threshold-free metric would stop the sweep in the wrong place.
    for q in (0.95, 0.9, 0.85, 0.75, 0.5, 0.25):
        arms[f"gate{q}"] = (ident, ("gate", q))
    # MANDATORY CONTROLS. Gating pushes excluded primitives to -inf, which shrinks the
    # retrievable set; `rand{q}` excludes the SAME FRACTION at random, so any gain that survives
    # is attributable to WHICH primitives were dropped rather than to how many. `anti{q}` keeps
    # the LOWEST-confidence fraction -- if the signal is real this must be much worse than random,
    # which is a stronger test than the random control alone.
    # Controls exist at EVERY gate setting, so the control always matches the operating point
    # being reported. Asking for a control that was not defined is a silent KeyError mid-run.
    for q in (0.95, 0.9, 0.85, 0.75, 0.5, 0.25):
        arms[f"rand{q}"] = (ident, ("rand", q))
        arms[f"anti{q}"] = (ident, ("anti", q))
    # Negative-query relevancy (LERF / LangSplat / Splat Feature Solver), with the four canonical
    # negatives. Provably INERT under closed-set argmax -- see readout.py -- so it is evaluated
    # here, in the single-query protocol, which is the only place it can act.
    arms["relev"] = (ident, ("relev", None))
    arms["centre"] = (centre, None)
    arms["centre+conf1"] = (centre, ("mul", 1.0))
    arms["centre+gate0.5"] = (centre, ("gate", 0.5))
    return arms


def main():
    enable_determinism()
    device = "cuda"
    arms = build_arms()
    want = [a for a in os.environ.get("ARMS", "").split(",") if a] or list(arms)
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = [s for s in SCENES if (s in only if only else True)]
    cs_name = os.environ.get("CLASS_SET", "opengaussian19")
    feat_file = os.environ.get("FEAT_FILE", "solved_geometric_median_truefrozen_ogl3")
    conf_file = os.environ.get("CONF_FILE", "solved_rawlift_nonorm_tf")
    recon = os.environ.get("RECON", "truefrozen")
    print(f"[config] RECON={recon} FEAT={feat_file} CONF={conf_file} CLASS_SET={cs_name} "
          f"arms={len(want)} scenes={len(scenes)}", flush=True)

    res = {a: {} for a in want}          # arm -> scene -> class -> metrics

    for scene in scenes:
        cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        n_prim, assign_fn = load_backend(recon, scene, device)
        d = torch.load(f"artifacts/scannet/{scene}/{feat_file}.pt", map_location=device,
                       weights_only=True)
        feats = d["primitive_features"].to(device).float()
        valid = d["valid_mask"].cpu().numpy()
        assert feats.shape[0] == n_prim, (
            f"{scene}: features have {feats.shape[0]} rows but {recon} has {n_prim} primitives -- "
            f"wrong arm/feature pairing (this exact mismatch produced mIoU 0.05 earlier)")
        cd_ = torch.load(f"artifacts/scannet/{scene}/{conf_file}.pt", map_location=device,
                         weights_only=True)
        conf_all = cd_["primitive_features"].to(device).float().norm(dim=-1)
        assert conf_all.shape[0] == n_prim, (conf_all.shape[0], n_prim)

        assigned = assign_fn(gt_points, valid)
        owned = assigned >= 0
        vidx = np.where(valid)[0]
        vt = torch.from_numpy(vidx).to(device)
        unit = F.normalize(feats[vt], dim=-1)
        conf = conf_all[vt]
        conf = conf / conf.median().clamp_min(1e-12)          # scale-free
        mu = unit.mean(0, keepdim=True)
        row_of = np.full(n_prim, -1, dtype=np.int64)
        row_of[vidx] = np.arange(vidx.size)
        prow = np.full(raw_labels.shape[0], -1, dtype=np.int64)
        prow[owned] = row_of[assigned[owned]]
        scored = prow >= 0                                     # points we can score at all

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        cls_list = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs_name]
                    if name_to_id[n] in present]
        text = embed_class_names([n for _, n in cls_list], device)
        neg_text, _, _ = embed_negatives(device)
        print(f"\n===== {scene} (valid={vidx.size:,}, classes={len(cls_list)}) =====", flush=True)

        for a in want:
            ftr, sctr = arms[a]
            u = ftr(unit, mu)
            sim = u @ text.T                                   # (Nvalid, K)
            if sctr is not None:
                kind, prm = sctr
                if kind == "mul":
                    sim = sim * (conf[:, None] ** prm)
                elif kind == "gate":
                    thr = torch.quantile(conf.float(), 1.0 - prm)
                    keep = conf >= thr
                    sim = sim.masked_fill(~keep[:, None], float("-inf"))
                elif kind == "rand":
                    g = torch.Generator(device="cpu").manual_seed(abs(hash(scene)) % (2 ** 31))
                    r = torch.rand(conf.numel(), generator=g).to(conf.device)
                    keep = r < prm                      # same expected fraction as gate{prm}
                    sim = sim.masked_fill(~keep[:, None], float("-inf"))
                elif kind == "relev":
                    # r = min_i sigmoid(sim_c - sim_negative_i), computed in the pairwise form
                    sn = u @ neg_text.T                        # (Nvalid, 4)
                    sim = torch.stack(
                        [torch.sigmoid(sim - sn[:, i:i + 1]) for i in range(sn.shape[1])],
                        -1).min(-1).values
                elif kind == "anti":
                    thr = torch.quantile(conf.float(), prm)
                    keep = conf <= thr                  # keep the LEAST confident
                    sim = sim.masked_fill(~keep[:, None], float("-inf"))
            simn = sim.cpu().numpy()
            per_cls = {}
            for k, (tid, nm) in enumerate(cls_list):
                y = (raw_labels[scored] == tid).astype(np.int64)
                sc = simn[prow[scored], k]
                fin = np.isfinite(sc)
                if fin.sum() < 10 or y[fin].sum() == 0:
                    continue
                rec = {"ap": average_precision(sc[fin], y[fin]),
                       "iou_oracle": iou_at_oracle(sc[fin], y[fin]),
                       "iou_prev": iou_at_prevalence(sc[fin], y[fin])}
                for q in QUANTS:
                    rec[f"iou_q{q}"] = iou_at_quantile(sc[fin], y[fin], q)
                for t in THRESHES + RELEV_THRESHES:
                    rec[f"iou_t{t}"] = iou_at_score(sc[fin], y[fin], t)
                # A19 found the absolute-cosine rule over-selects, with the per-class selected
                # fractions summing to 410%. Record the selection rate at each arm's OWN natural
                # operating point so that failure mode is visible directly rather than inferred.
                tnat = NATURAL.get(a, NATURAL_DEFAULT)
                rec["t_nat"] = tnat
                rec["sel_nat"] = float((sc[fin] >= tnat).mean())
                rec["prev"] = float(y[fin].mean())
                rec["iou_nat"] = iou_at_score(sc[fin], y[fin], tnat)
                per_cls[nm] = rec
            res[a][scene] = per_cls
            aps = [v["ap"] for v in per_cls.values() if v["ap"] == v["ap"]]
            print(f"  {a:<16} mAP {st.mean(aps)*100:6.2f}  ({len(aps)} classes)", flush=True)

    # ------------------------------------------------------------------ report
    print(f"\n\n=== SINGLE-QUERY, {len(scenes)} scenes, {recon}, {cs_name} ===")
    print(f"{'arm':<16}{'mAP':>9}{'IoU@prev':>11}{'IoU@oracle':>12}   delta mAP   wins")
    base = "plain"

    def agg(a, key):
        v = [st.mean([c[key] for c in res[a][s].values() if c[key] == c[key]])
             for s in res[a] if res[a][s]]
        return st.mean(v) * 100, v

    b_map, b_per = agg(base, "ap")
    rows = []
    for a in want:
        m, per = agg(a, "ap")
        ip, _ = agg(a, "iou_prev")
        io_, _ = agg(a, "iou_oracle")
        wins = sum(1 for x, y in zip(per, b_per) if x > y)
        rows.append((m - b_map, a))
        print(f"{a:<16}{m:>9.2f}{ip:>11.2f}{io_:>12.2f}   {m-b_map:>+9.3f}   {wins}/{len(per)}")
    rows.sort(reverse=True)
    print(f"\nBEST: {rows[0][1]}  delta mAP {rows[0][0]:+.3f}")

    # ---- DEPLOYABLE IoU: label-free, shared threshold rule ----
    dep = [f"iou_q{q}" for q in QUANTS] + [f"iou_t{t}" for t in THRESHES]

    def col(a, key):
        v = [st.mean([c[key] for c in res[a][s].values() if c[key] == c[key]])
             for s in res[a] if res[a][s]]
        return st.mean(v) * 100 if v else float("nan")

    print("\n=== DEPLOYABLE IoU (no labels used to pick the cut) ===")
    print(f"{'arm':<16}" + "".join(f"{c[4:]:>9}" for c in dep))
    for a in want:
        print(f"{a:<16}" + "".join(f"{col(a, c):>9.2f}" for c in dep))
    print(f"\n=== DEPLOYABLE IoU, delta vs {base} ===")
    bv = [col(base, c) for c in dep]
    print(f"{'arm':<16}" + "".join(f"{c[4:]:>9}" for c in dep))
    for a in want:
        print(f"{a:<16}" + "".join(
            f"{col(a, c) - b:>+9.2f}" for c, b in zip(dep, bv)))
    # ---- CALIBRATION at each arm's own natural operating point (the A19 failure, directly) ----
    # A19 showed the absolute-cosine rule selects far more primitives than the class occupies,
    # with the per-class selected fractions summing to 410%. `over` below is selected/prevalence:
    # 1.0 is perfect calibration, and the sum column is the 410% statistic reproduced per arm.
    print("\n=== CALIBRATION at the natural threshold (sel = selected fraction, "
          "prev = true prevalence) ===")
    print(f"{'arm':<16}{'t_nat':>7}{'sel%':>8}{'prev%':>8}{'over':>8}{'sum sel%':>10}{'IoU@nat':>9}")
    for a in want:
        sel, prv, sums = [], [], []
        for sc_ in res[a]:
            cs_ = [c for c in res[a][sc_].values() if c.get("sel_nat") == c.get("sel_nat")]
            if not cs_:
                continue
            sel += [c["sel_nat"] for c in cs_]
            prv += [c["prev"] for c in cs_]
            sums.append(sum(c["sel_nat"] for c in cs_))
        if not sel:
            continue
        tn = NATURAL.get(a, NATURAL_DEFAULT)
        ov = st.mean([s_ / p_ for s_, p_ in zip(sel, prv) if p_ > 0])
        print(f"{a:<16}{tn:>7.2f}{st.mean(sel)*100:>8.2f}{st.mean(prv)*100:>8.2f}"
              f"{ov:>8.2f}{st.mean(sums)*100:>10.1f}{col(a, 'iou_nat'):>9.2f}")

    out = f"artifacts/scannet/single_query_{len(scenes)}scene_{cs_name}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
