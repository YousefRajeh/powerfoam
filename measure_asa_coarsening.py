"""Does ASA actually fall when Cut Pursuit coarsens the partition?

The CUTPURSUIT brief asserted a paradox: "coarsening strictly lowers the achievable ceiling, yet
mIoU rises". The first half was never measured. This measures it.

ASA (Achievable Segmentation Accuracy, Liu et al. CVPR 2011 Eq. 14) for a partition {S} of
primitives, where each GT point p is owned by primitive own[p] with true class pt_gt[p]:

    ASA(S) = (1/N) * sum_S max_c |{p : seg(own[p]) = S, pt_gt[p] = c}|

We report BOTH ASA (accuracy of the majority labelling, the real ASA) and the mIoU of that same
majority labelling -- they are different quantities and conflating them was error R4.

Self-test (--selftest) proves, on hand-computable inputs:
  1. ASA against a brute-force enumeration of all label assignments (it must be the max).
  2. ASA is monotone NON-INCREASING under a genuine nested merge, with equality when merged cells
     share a maximising class (the FINDINGS2 correction: non-strict, not strict).
  3. The trivial partition (one segment) gives ASA = the majority-class frequency.
  4. The finest partition (one primitive each) gives ASA = per-primitive purity.
"""
import argparse, itertools, json, os, sys
import numpy as np
import torch


def asa(seg_of_point: np.ndarray, gt: np.ndarray, C: int):
    """ASA and the majority labelling, for an arbitrary grouping of POINTS."""
    K = int(seg_of_point.max()) + 1 if seg_of_point.size else 0
    cnt = np.zeros((K, C), dtype=np.int64)
    np.add.at(cnt, (seg_of_point, gt), 1)
    maj = cnt.argmax(1)                       # best single label per segment
    correct = cnt.max(1).sum()
    return float(correct / max(gt.size, 1)), maj


def miou_of(pred: np.ndarray, gt: np.ndarray, C: int):
    ious = []
    for c in range(C):
        p, g = pred == c, gt == c
        u = (p | g).sum()
        if g.sum() == 0:                      # class absent from GT -> not scored
            continue
        ious.append((p & g).sum() / u if u else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


def selftest():
    rng = np.random.default_rng(0)
    # --- 1. brute force on a tiny case
    for trial in range(200):
        N, K, C = 9, 3, 3
        seg = rng.integers(0, K, N); gt = rng.integers(0, C, N)
        a, maj = asa(seg, gt, C)
        best = 0
        for lab in itertools.product(range(C), repeat=K):
            best = max(best, sum(1 for i in range(N) if lab[seg[i]] == gt[i]))
        assert abs(a - best / N) < 1e-12, (trial, a, best / N)
        assert sum(1 for i in range(N) if maj[seg[i]] == gt[i]) == best
    # --- 2. monotone NON-increasing under a genuine nested merge, and equality is reachable
    strict_drops = equalities = 0
    for trial in range(500):
        N, K, C = 40, 8, 4
        seg = rng.integers(0, K, N); gt = rng.integers(0, C, N)
        fine, _ = asa(seg, gt, C)
        merge = rng.integers(0, K // 2, K)            # nested: each fine seg -> one coarse seg
        coarse, _ = asa(merge[seg], gt, C)
        assert coarse <= fine + 1e-12, (fine, coarse)
        if coarse < fine - 1e-12: strict_drops += 1
        else: equalities += 1
    assert equalities > 0, "never saw equality -- the non-strict claim needs a witness"
    # explicit witness: merging two PURE same-class cells cannot lower ASA
    seg = np.array([0, 0, 1, 1]); gt = np.array([2, 2, 2, 2])
    assert asa(seg, gt, 3)[0] == 1.0 and asa(np.array([0, 0, 0, 0]), gt, 3)[0] == 1.0
    # --- 3. trivial partition = majority-class frequency
    gt = rng.integers(0, 5, 100)
    assert abs(asa(np.zeros(100, dtype=np.int64), gt, 5)[0]
               - np.bincount(gt, minlength=5).max() / 100) < 1e-12
    # --- 4. finest partition = per-point purity = 1.0
    assert asa(np.arange(100), gt, 5)[0] == 1.0
    print(f"selftest OK  (nested merges: {strict_drops} strict drops, {equalities} equalities "
          f"-- confirms NON-strict monotonicity)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", required=True, help="npz with own[] and pt_gt[]")
    ap.add_argument("--labels", nargs="+", default=[], help="CP .pt files carrying labels[]")
    ap.add_argument("--classes", type=int, default=19)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if not a.assign: return
    z = np.load(a.assign)
    own = z["own"].astype(np.int64); gt = z["pt_gt"].astype(np.int64)
    ok = (gt >= 0) & (gt < a.classes)
    own, gt = own[ok], gt[ok]
    rows = []
    base_asa, base_maj = asa(own, gt, a.classes)
    rows.append({"partition": "primitives", "segments": int(own.max()) + 1, "asa": base_asa,
                 "miou_of_majority": miou_of(base_maj[own], gt, a.classes)})
    for f in a.labels:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lab = d["labels"].numpy().astype(np.int64)
        s, maj = asa(lab[own], gt, a.classes)
        rows.append({"partition": os.path.basename(f), "segments": int(d.get("num_segments", lab.max() + 1)),
                     "asa": s, "miou_of_majority": miou_of(maj[lab[own]], gt, a.classes)})
    print(f"{'partition':34s} {'segments':>9s} {'ASA':>8s} {'dASA':>8s} {'mIoU(maj)':>10s}")
    for r in rows:
        print(f"{r['partition']:34s} {r['segments']:9d} {100*r['asa']:8.2f} "
              f"{100*(r['asa']-base_asa):+8.2f} {100*r['miou_of_majority']:10.2f}")
    if a.out: json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
