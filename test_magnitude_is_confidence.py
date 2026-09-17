"""Does CLIP embedding MAGNITUDE predict CORRECTNESS? The decisive test of the confidence claim.

THE CLAIM UNDER TEST. `||f||` of a per-mask CLIP embedding is a proxy for the model's confidence,
so feeding raw (un-normalised) vectors into the lift/accumulation gives a "vote of confidence"
from the data itself -- most pointedly in the geometric-median accumulation, which minimises
`sum_i w_i ||x - v_i||` and is therefore NOT scale-invariant in its inputs.

WHY THE EARLIER EVIDENCE DOES NOT SETTLE IT.
  * CLIP's loss is exactly scale-invariant (verified: rescaling rows changes the loss by 0.0,
    radial gradient 2.8e-17). That shows magnitude is never TRAINED to mean confidence. It does
    NOT show magnitude is UNCORRELATED with correctness -- an untrained quantity can still carry
    incidental signal.
  * `spearman(||f||, mask area) = +0.114` (1.9 % of variance) tests against a NUISANCE variable,
    not against correctness. Wrong target.
  * `run_raw_lift.py` compares raw vs normalised for the WEIGHTED MEAN only. It never touches
    `gm_z`, so its null says nothing about the geometric-median accumulation.

WHAT THIS MEASURES. Per primitive j:
    m_j = ||x_raw,j|| / ||x_norm,j||
which is exactly the support-weighted mean CLIP magnitude of the masks that j observed, since both
numerators are accumulated from the SAME rays and weights and divided by the same support. Then:
    correct_j = (argmax_c cos(x_norm,j, text_c) == GT class of j)
and we ask whether m_j separates correct from incorrect -- AUC, and the correctness rate by
magnitude decile.

INTERPRETATION. AUC ~ 0.5 means magnitude carries no information about correctness, and the
confidence story is dead regardless of where in the pipeline it is applied -- no accumulator
change can extract signal that is not there. AUC meaningfully above 0.5 means the signal exists
and the accumulation site is worth modifying.

This is deliberately run BEFORE implementing the magnitude-aware accumulator: a re-accumulation
over 10 scenes is expensive, and this test costs seconds.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from diagnose_scannet_miou import (OPENGAUSSIAN_CLASS_SETS, assign_points_to_power_cells,
                                   embed_class_names, load_foam, load_scannet_pointcept_gt)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney). labels: 1 = correct, 0 = incorrect."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over ties so a constant score gives exactly 0.5
    s = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--raw", default="solved_rawlift_nonorm_tf")
    ap.add_argument("--norm", default="solved_normlift_nonorm_tf")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    a = ap.parse_args()

    dev = "cuda"
    import glob, os
    cand = [p for p in glob.glob(os.path.join(a.gt_root, "*", a.scene)) if os.path.isdir(p)]
    assert cand, f"no GT for {a.scene} under {a.gt_root}"
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")

    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    centers, radii = load_foam(ckpt, dev)

    dr = torch.load(f"artifacts/scannet/{a.scene}/{a.raw}.pt", map_location=dev, weights_only=True)
    dn = torch.load(f"artifacts/scannet/{a.scene}/{a.norm}.pt", map_location=dev, weights_only=True)
    xr, xn = dr["primitive_features"].float(), dn["primitive_features"].float()
    valid = dn["valid_mask"].cpu().numpy()

    # m_j = support-weighted mean CLIP magnitude seen by primitive j
    nr, nn = xr.norm(dim=-1), xn.norm(dim=-1)
    m = (nr / nn.clamp_min(1e-12)).cpu().numpy()

    # GT class per primitive: majority label among the GT points it owns
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
    owned = assigned >= 0
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if name_to_id[n] in present]
    target_ids = [i for i, _ in kept]
    target_names = [n for _, n in kept]
    id_to_slot = {tid: k for k, tid in enumerate(target_ids)}

    P = centers.shape[0]
    votes = np.zeros((P, len(target_ids)), dtype=np.int64)
    lab = raw_labels[owned]
    prim = assigned[owned]
    for tid, slot in id_to_slot.items():
        sel = lab == tid
        if sel.any():
            np.add.at(votes[:, slot], prim[sel], 1)
    has_gt = votes.sum(1) > 0
    gt_slot = votes.argmax(1)

    # prediction: plain cosine argmax of the NORMALISED lifted feature
    text = embed_class_names(target_names, dev)
    pred = (F.normalize(xn, dim=-1) @ text.T).argmax(-1).cpu().numpy()

    sel = has_gt & valid & np.isfinite(m) & (m > 0)
    correct = (pred[sel] == gt_slot[sel]).astype(np.int64)
    mm = m[sel]
    a_ = auc(mm, correct)
    print(f"\n=== {a.scene} / {a.variant} / {a.class_set} ===")
    print(f"primitives with GT and features: {sel.sum():,}   accuracy {correct.mean():.4f}")
    print(f"mean CLIP magnitude: {mm.mean():.4f}  sd {mm.std():.4f}  CV {mm.std()/mm.mean():.4f}")
    print(f"\nAUC( ||f|| predicts correctness ) = {a_:.4f}     (0.5 = no information)")
    q = np.quantile(mm, np.linspace(0, 1, 11))
    print(f"\n{'decile':>8}{'mag range':>26}{'n':>9}{'accuracy':>10}")
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        b = (mm >= lo) & (mm <= hi if i == 9 else mm < hi)
        if b.sum():
            print(f"{i+1:>8}{f'[{lo:.3f}, {hi:.3f})':>26}{b.sum():>9,}{correct[b].mean():>10.4f}")
    lo_acc = correct[mm < np.median(mm)].mean()
    hi_acc = correct[mm >= np.median(mm)].mean()
    print(f"\nlow-half accuracy {lo_acc:.4f}   high-half accuracy {hi_acc:.4f}   "
          f"gap {hi_acc-lo_acc:+.4f}")


if __name__ == "__main__":
    main()


def control(scene, variant="truefrozen", raw="solved_rawlift_nonorm_tf",
            norm="solved_normlift_nonorm_tf", class_set="opengaussian19",
            gt_root=r"D:\Downloads\scannet_pointcept"):
    """CONFOUND CONTROL. m = ||x_raw||/||x_norm|| has the AGREEMENT signal ||x_norm|| in its
    denominator, and ||x_norm|| independently predicts correctness (that is the threshold gate).
    So a high m can mean "confident masks" OR "disagreeing views". This reports AUC for both
    signals separately, and for m WITHIN strata of ||x_norm||, so magnitude's contribution is
    measured at (approximately) fixed agreement."""
    import glob, os
    dev = "cuda"
    cand = [p for p in glob.glob(os.path.join(gt_root, "*", scene)) if os.path.isdir(p)]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", dev)
    dr = torch.load(f"artifacts/scannet/{scene}/{raw}.pt", map_location=dev, weights_only=True)
    dn = torch.load(f"artifacts/scannet/{scene}/{norm}.pt", map_location=dev, weights_only=True)
    xr, xn = dr["primitive_features"].float(), dn["primitive_features"].float()
    valid = dn["valid_mask"].cpu().numpy()
    nr, nn = xr.norm(dim=-1), xn.norm(dim=-1)
    m = (nr / nn.clamp_min(1e-12)).cpu().numpy()
    agree = nn.cpu().numpy()

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
    owned = assigned >= 0
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[class_set] if name_to_id[n] in present]
    tids = [i for i, _ in kept]; tnames = [n for _, n in kept]
    P = centers.shape[0]
    votes = np.zeros((P, len(tids)), dtype=np.int64)
    lab, prim = raw_labels[owned], assigned[owned]
    for slot, tid in enumerate(tids):
        s = lab == tid
        if s.any(): np.add.at(votes[:, slot], prim[s], 1)
    has_gt = votes.sum(1) > 0; gt_slot = votes.argmax(1)
    text = embed_class_names(tnames, dev)
    pred = (F.normalize(xn, dim=-1) @ text.T).argmax(-1).cpu().numpy()

    sel = has_gt & valid & np.isfinite(m) & (m > 0)
    c = (pred[sel] == gt_slot[sel]).astype(np.int64)
    mm, aa = m[sel], agree[sel]
    print(f"\n=== {scene} CONFOUND CONTROL (n={sel.sum():,}, acc {c.mean():.4f}) ===")
    print(f"  AUC( m = ||x_raw||/||x_norm||  ) = {auc(mm, c):.4f}   <- 'magnitude'")
    print(f"  AUC( ||x_norm||  (agreement)   ) = {auc(aa, c):.4f}   <- the GATE signal")
    print(f"  AUC( ||x_raw||                 ) = {auc(nr.cpu().numpy()[sel], c):.4f}")
    print(f"  spearman(m, ||x_norm||) = ", end="")
    try:
        from scipy.stats import spearmanr
        print(f"{spearmanr(mm, aa).statistic:+.4f}")
    except Exception:
        print("scipy unavailable")
    # magnitude's AUC WITHIN agreement quintiles -> does it add anything at fixed agreement?
    q = np.quantile(aa, np.linspace(0, 1, 6))
    print(f"  {'agreement quintile':>20}{'n':>8}{'acc':>8}{'AUC(m | stratum)':>20}")
    for i in range(5):
        lo, hi = q[i], q[i+1]
        b = (aa >= lo) & (aa <= hi if i == 4 else aa < hi)
        if b.sum() > 50:
            print(f"  {f'{lo:.3f}-{hi:.3f}':>20}{b.sum():>8,}{c[b].mean():>8.3f}{auc(mm[b], c[b]):>20.4f}")
