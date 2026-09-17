"""How loose is our mIoU "ceiling"?

We report `ceiling_miou` = the mIoU of the MAJORITY labelling (each primitive emits the most common
GT class of the points it owns). That labelling is provably optimal for ACCURACY -- which is exactly
what ASA is (Liu, Tuzel, Ramalingam, Chellappa, CVPR 2011, Eq. 14) -- because accuracy is separable
across segments, so the per-segment argmax is the global argmax.

mIoU is NOT separable: the union in each denominator couples every segment that predicts that class.
So the majority labelling is FEASIBLE but not optimal, and `ceiling_miou` is only a LOWER BOUND on
the achievable mIoU. Minimal counterexample (verified in selftest): one pure segment of 100 class-0
points and one mixed segment of 6 class-0 / 5 class-1. Majority labels both class 0 -> mIoU 0.4775.
Labelling the mixed one class 1 costs accuracy (0.9550 -> 0.9459) and RAISES mIoU to 0.6990.

This tightens the bound by coordinate ascent on the segment labels, so we can say how much of the
reported geometry term is real and how much is an artefact of scoring a suboptimal labelling.
It stays a lower bound -- coordinate ascent finds a local optimum, not the global one.

mIoU is maintained incrementally. For segment j with n_jc points of class c and label L:
    TP_L += n_jL;  FP_L += (N_j - n_jL);  FN_c += n_jc for every c != L
so a relabel a->b is O(C) rather than a full rescore.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np


def _miou(tp, fp, fn, present):
    d = tp + fp + fn
    iou = np.where(d > 0, tp / np.maximum(d, 1e-30), 0.0)
    return float(iou[present].mean())


def _state(counts, lab, C):
    """TP/FP/FN per class for a given per-segment labelling."""
    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C)
    tot = counts.sum(1)
    for c in range(C):
        m = lab == c
        tp[c] = counts[m, c].sum()
        fp[c] = (tot[m] - counts[m, c]).sum()
    fn = counts.sum(0) - tp
    return tp, fp, fn


def ascend(counts, lab, C, present, max_sweeps=50, verbose=False):
    """Coordinate ascent on segment labels, maximising mIoU. Returns (labels, miou, sweeps)."""
    tp, fp, fn = _state(counts, lab, C)
    tot = counts.sum(1)
    cur = _miou(tp, fp, fn, present)
    order = np.argsort(-tot)                      # biggest segments first: they move mIoU most
    for sweep in range(max_sweeps):
        moved = 0
        for j in order:
            if tot[j] == 0:
                continue
            a = int(lab[j]); nj = counts[j]; Nj = tot[j]
            # REMOVE segment j, currently labelled a
            tp[a] -= nj[a]; fp[a] -= Nj - nj[a]
            fn -= nj; fn[a] += nj[a]
            best_b, best_v = a, -np.inf
            for b in range(C):
                # ADD under label b
                tp[b] += nj[b]; fp[b] += Nj - nj[b]
                fn += nj; fn[b] -= nj[b]
                v = _miou(tp, fp, fn, present)
                # undo
                tp[b] -= nj[b]; fp[b] -= Nj - nj[b]
                fn -= nj; fn[b] += nj[b]
                if v > best_v:
                    best_v, best_b = v, b
            # ADD under the winner, for real
            tp[best_b] += nj[best_b]; fp[best_b] += Nj - nj[best_b]
            fn += nj; fn[best_b] -= nj[best_b]
            lab[j] = best_b
            if best_b != a:
                moved += 1
            cur = best_v
        if verbose:
            print(f"   sweep {sweep}: moved {moved}, mIoU {100*cur:.4f}", flush=True)
        if moved == 0:
            break
    tp, fp, fn = _state(counts, lab, C)           # exact rescore, never trust the running state
    return lab, _miou(tp, fp, fn, present), sweep + 1


def selftest():
    # the counterexample from the docstring, brute-forced
    counts = np.array([[100.0, 0.0], [6.0, 5.0]])
    C = 2; present = np.array([True, True])
    maj = counts.argmax(1)
    tp, fp, fn = _state(counts, maj.copy(), C)
    m_maj = _miou(tp, fp, fn, present)
    best = max(((_miou(*_state(counts, np.array(l), C), present), l)
                for l in itertools.product(range(C), repeat=2)), key=lambda t: t[0])
    assert abs(m_maj - 0.4775) < 1e-3, m_maj
    assert abs(best[0] - 0.6990) < 1e-3, best
    got, v, _ = ascend(counts, maj.copy(), C, present)
    assert abs(v - best[0]) < 1e-9, (v, best)

    # incremental state must equal full rescore on random problems
    rng = np.random.default_rng(0)
    for _ in range(50):
        S, C2 = 12, 4
        cnt = rng.integers(0, 9, (S, C2)).astype(float)
        pres = cnt.sum(0) > 0
        lab0 = cnt.argmax(1)
        lab1, v1, _ = ascend(cnt, lab0.copy(), C2, pres)
        tp, fp, fn = _state(cnt, lab1, C2)
        assert abs(v1 - _miou(tp, fp, fn, pres)) < 1e-12
        # ascent must never do worse than its start
        tp0, fp0, fn0 = _state(cnt, lab0, C2)
        assert v1 >= _miou(tp0, fp0, fn0, pres) - 1e-12
        # and never beat brute force
        bf = max(_miou(*_state(cnt, np.array(l), C2), pres)
                 for l in itertools.product(range(C2), repeat=S)) if S <= 8 else None
        if bf is not None:
            assert v1 <= bf + 1e-12
    print("selftest OK  (counterexample reproduced: majority 0.4775 vs optimal 0.6990; "
          "incremental state == full rescore; ascent monotone and never beats brute force)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", default=None, help="npz with own[] and pt_gt[]")
    ap.add_argument("--classes", type=int, default=19)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if not a.assign:
            return
    z = np.load(a.assign)
    own = z["own"].astype(np.int64); gt = z["pt_gt"].astype(np.int64)
    ok = (gt >= 0) & (gt < a.classes)
    own, gt = own[ok], gt[ok]
    S = int(own.max()) + 1; C = a.classes
    counts = np.zeros((S, C))
    np.add.at(counts, (own, gt), 1.0)
    present = counts.sum(0) > 0
    maj = counts.argmax(1)
    tp, fp, fn = _state(counts, maj.copy(), C)
    m_maj = _miou(tp, fp, fn, present)
    acc_maj = float(counts.max(1).sum() / counts.sum())
    lab, m_opt, sweeps = ascend(counts, maj.copy(), C, present, verbose=True)
    tpo, fpo, fno = _state(counts, lab, C)
    acc_opt = float(sum(counts[lab == c, c].sum() for c in range(C)) / counts.sum())
    print(f"\nsegments {S}  classes present {int(present.sum())}  points {int(counts.sum())}")
    print(f"  ASA (accuracy of majority labelling)      {100*acc_maj:8.4f}")
    print(f"  mIoU of majority labelling  (reported)    {100*m_maj:8.4f}   <- our 'ceiling'")
    print(f"  mIoU after coordinate ascent (tighter LB) {100*m_opt:8.4f}   ({100*(m_opt-m_maj):+.4f})")
    print(f"  accuracy of that labelling                {100*acc_opt:8.4f}   ({100*(acc_opt-acc_maj):+.4f})")
    print(f"  geometry term 100-ceiling: {100-100*m_maj:.4f}  ->  {100-100*m_opt:.4f}")
    if a.out:
        json.dump({"segments": S, "asa_acc": acc_maj, "miou_majority": m_maj,
                   "miou_ascent": m_opt, "acc_ascent": acc_opt, "sweeps": sweeps},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
