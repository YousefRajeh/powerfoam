"""Is the accuracy-vs-contamination gradient causal, or is it selection?

THE CLAIM UNDER TEST. Cells sorted by mean LOO residual show accuracy 0.743 (cleanest quartile) down
to 0.393 (dirtiest), against 0.544 overall -- which was quoted as ~20 points of headroom available to
a contamination filter. That reading is observational. The cleanest cells may simply be EASY cells --
large, well-observed, on unambiguous classes -- in which case the gradient measures difficulty and no
filter can collect it. The direct evidence already leans that way: trimming on a 0.79 within-cell
detector moved mIoU by ~0.

THE TEST. Recompute the same gradient INSIDE strata of the plausible confounders -- projected size,
number of views, and ground-truth class -- by ranking each cell against only its own stratum. If the
gradient survives, contamination is doing real work. If it flattens, the headroom is selection and
the whole contamination line (mask-level pooling, exact-cell PLA, multi-level agreement) is not worth
building.

Reported three ways so the failure mode is visible rather than averaged away:
  * raw quartiles (the original claim)
  * within-stratum quartiles (the corrected claim)
  * a logistic fit, coefficient on residual before and after adjustment
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from pla_multiscene import SPLIT, POINTCEPT  # noqa: E402


def quartile_table(score, correct, label):
    q = np.quantile(score, [0, .25, .5, .75, 1.0])
    q[-1] += 1e-9
    out = []
    for i in range(4):
        m = (score >= q[i]) & (score < q[i + 1])
        if m.sum() < 20:
            continue
        out.append((int(m.sum()), float(correct[m].mean())))
    if len(out) >= 2:
        print(f"  {label:<34} " + "  ".join(f"{a:.3f}" for _, a in out) +
              f"   spread {out[0][1]-out[-1][1]:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from oracle_labels import oracle_labels

    enable_determinism()
    z = np.load(f"artifacts/residuals_{a.scene}.npz")
    cell, thl, cont = z["cell"], z["theta_loo"].astype(float), z["cont"]
    ok = ~np.isnan(thl)
    cell, thl, cont = cell[ok], thl[ok], cont[ok]

    da = np.load("artifacts/detector_auc.npz")
    acell, aarea = da["cell"].astype(np.int64), da["area"].astype(float)

    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt = remap_gt_labels(raw, [n2i[n] for n in names])
    K = len(names)
    cc, rr = load_points_radii(f"output/scannet_{a.scene}_{a.variant}")
    oracle, _ = oracle_labels(np.asarray(cc, float), np.asarray(rr, float), pts, gt, K + 1)
    d = torch.load(f"artifacts/scannet/{a.scene}/{a.solved}", map_location="cpu",
                   weights_only=True)
    X = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1)
    T = embed_class_names(names, "cuda")
    pred = classify_primitives(X.cuda(), T).cpu().numpy() + 1
    correct_all = pred == oracle

    P = len(oracle)
    o = np.argsort(cell, kind="stable")
    cell_s, thl_s, cont_s = cell[o], thl[o], cont[o]
    uc, start = np.unique(cell_s, return_index=True)
    cnt = np.diff(np.r_[start, len(cell_s)])
    res = np.add.reduceat(thl_s, start) / cnt
    con = np.add.reduceat(cont_s.astype(float), start) / cnt

    area = np.zeros(P); an = np.zeros(P)
    np.add.at(area, acell, aarea); np.add.at(an, acell, 1.0)
    area_c = np.where(an > 0, area / np.maximum(an, 1), 0.0)[uc]

    keep = oracle[uc] > 0
    uc, res, con, cnt, area_c = uc[keep], res[keep], con[keep], cnt[keep], area_c[keep]
    corr = correct_all[uc]
    cls = oracle[uc]
    print(f"{len(uc):,} cells with GT; overall accuracy {corr.mean():.4f}, "
          f"mean contamination {con.mean():.4f}\n")

    print("RAW (the original claim) -- accuracy by residual quartile, cleanest first")
    quartile_table(res, corr, "residual (unadjusted)")

    print("\nCONFOUNDERS: do they also predict accuracy?")
    quartile_table(-area_c, corr, "projected area (small->large)")
    quartile_table(-cnt.astype(float), corr, "view count (few->many)")

    print("\nWITHIN-STRATUM (residual ranked only against cells of the same stratum)")
    for nm, strat in [("class", cls),
                      ("class x view-count tercile", None),
                      ("class x view x size tercile", None)]:
        if nm == "class":
            key = cls
        elif nm == "class x view-count tercile":
            vt = np.digitize(cnt, np.quantile(cnt, [1 / 3, 2 / 3]))
            key = cls * 10 + vt
        else:
            vt = np.digitize(cnt, np.quantile(cnt, [1 / 3, 2 / 3]))
            at = np.digitize(area_c, np.quantile(area_c, [1 / 3, 2 / 3]))
            key = cls * 100 + vt * 10 + at
        rank = np.zeros(len(res))
        for k in np.unique(key):
            m = key == k
            if m.sum() < 40:
                rank[m] = np.nan
                continue
            r = np.argsort(np.argsort(res[m])) / max(m.sum() - 1, 1)
            rank[m] = r
        v = ~np.isnan(rank)
        quartile_table(rank[v], corr[v], nm)

    # logistic fit: coefficient on residual, before and after adjustment
    def fit(Xd, y, iters=300, lr=0.5):
        Xd = np.c_[np.ones(len(Xd)), (Xd - Xd.mean(0)) / np.maximum(Xd.std(0), 1e-9)]
        w = np.zeros(Xd.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xd @ w))
            g = Xd.T @ (y - p) / len(y)
            w += lr * g
        return w

    y = corr.astype(float)
    w1 = fit(res[:, None], y)
    w2 = fit(np.c_[res, np.log1p(area_c), np.log1p(cnt)], y)
    print(f"\nLOGISTIC coefficient on residual (standardised):")
    print(f"  unadjusted                      {w1[1]:+.4f}")
    print(f"  adjusted for log(area), log(n)  {w2[1]:+.4f}   "
          f"({100*(w2[1]/w1[1]-1) if w1[1] else 0:+.1f}% change)")
    print(f"  (area coef {w2[2]:+.4f}, view-count coef {w2[3]:+.4f})")


if __name__ == "__main__":
    main()
