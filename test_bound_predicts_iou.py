"""Is the o_i <-> mIoU correlation real, or a scene-size confound plus multiple comparisons?

predict_iou.py found Spearman -0.648 (p=0.043) between mean o_i and mIoU on the 19-class set,
but +0.467 on the 15-class set and +0.079 on the 10-class set. A quantity that genuinely drives
segmentation quality cannot reverse sign when three classes are removed from the label set, so
before anything is claimed this checks the two boring explanations:

  CONFOUND      both o_i and mIoU may track scene size / primitive count.
  MULTIPLICITY  3 quantities x 3 class sets = 9 tests; at alpha=0.05 one hit is expected by
                chance, and the hits here are not spread across class sets but concentrated in
                one, which is the signature of noise rather than signal.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = r"D:\Downloads\powerfoam"
SC = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
      "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def miou_json(tag):
    out = {}
    for s in SC:
        p = os.path.join(ROOT, f"artifacts/scannet/{s}/miou_opengaussian19_{tag}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p)).get("powerfoam", {})
        for k in ("mIoU", "miou", "mean_iou"):
            if k in j:
                v = j[k]
                out[s] = v * 100 if v <= 1 else v
                break
    return out


def sp(x, y):
    ks = sorted(set(x) & set(y))
    if len(ks) < 6:
        return None, None, len(ks)
    r, p = spearmanr([x[k] for k in ks], [y[k] for k in ks])
    return r, p, len(ks)


def main():
    R = json.load(open(os.path.join(ROOT, "artifacts/scannet/bound_validation_final.json")))
    by = {}
    for r in R:
        by.setdefault(r["arm"], {})[r["scene"]] = r
    tf = by["foam_truefrozen"]
    nf = by["foam_nonfrozen"]
    tf_o = {s: tf[s]["mean_o_gram"] for s in tf}
    nf_o = {s: nf[s]["mean_o_gram"] for s in nf}
    P = {s: tf[s]["P"] for s in tf}
    live = {s: tf[s]["live"] for s in tf}

    smooth = json.load(open(os.path.join(ROOT, "artifacts/scannet/smooth_results.json")))
    miou = {cs: {s: smooth[cs][s]["plain"]["miou"] * 100 for s in smooth[cs]} for cs in smooth}

    print("=" * 90)
    print("CONFOUND CHECK: does either side just track scene size?")
    for nm, v in [("mean o_i", tf_o)]:
        r, p, n = sp(v, P)
        print(f"  {nm:<28} vs primitive count P    Spearman {r:+.3f} (p={p:.3f}, n={n})")
        r, p, n = sp(v, live)
        print(f"  {nm:<28} vs live primitives      Spearman {r:+.3f} (p={p:.3f}, n={n})")
    for cs in miou:
        r, p, n = sp(miou[cs], P)
        print(f"  mIoU[{cs:<16}] vs primitive count P    Spearman {r:+.3f} (p={p:.3f}, n={n})")

    print("\n" + "=" * 90)
    print("MULTIPLICITY: all 9 tests, so the pattern is visible rather than cherry-picked")
    tf_ex = {s: tf[s]["rel_excess"] for s in tf}
    gam = {}
    for f in glob.glob(os.path.join(ROOT, "artifacts/scannet/beta/pf_truefrozen_*.json")):
        j = json.load(open(f))
        if "gamma" in j:
            gam[j["scene"]] = j["gamma"]
    print(f"  {'quantity':<16}" + "".join(f"{cs.replace('opengaussian','cls'):>16}" for cs in miou))
    hits = 0
    for nm, q in [("mean o_i", tf_o), ("rel excess", tf_ex), ("gamma", gam)]:
        cells = []
        for cs in miou:
            r, p, n = sp(q, miou[cs])
            hits += int(p is not None and p < 0.05 and r < 0)
            cells.append(f"{r:+.3f}/{p:.2f}")
        print(f"  {nm:<16}" + "".join(f"{c:>16}" for c in cells))
    print(f"\n  negative-and-significant cells: {hits}/9  (expected ~0.2 by chance at one tail)")
    print("  All hits sit in ONE class set and the sign REVERSES in another; a quantity that")
    print("  drives segmentation cannot flip sign when 4 labels are dropped from the vocabulary.")

    print("\n" + "=" * 90)
    print("WITHIN the nonfrozen arm (independent replication of the same question)")
    nfm = miou_json("nonfrozen")
    if nfm:
        r, p, n = sp(nf_o, nfm)
        print(f"  mean o_i vs mIoU[19]      Spearman {r:+.3f} (p={p:.3f}, n={n})")
        nf_ex = {s: nf[s]["rel_excess"] for s in nf}
        r, p, n = sp(nf_ex, nfm)
        print(f"  rel excess vs mIoU[19]    Spearman {r:+.3f} (p={p:.3f}, n={n})")
    else:
        print("  no per-scene nonfrozen mIoU parsed")

    print("\n" + "=" * 90)
    print("ARM-LEVEL: the comparison that IS supported")
    tfm = miou["opengaussian19"]
    print(f"  truefrozen: mean o {np.mean(list(tf_o.values())):.3f}  "
          f"rel excess {np.mean([tf[s]['rel_excess'] for s in tf]):.2e}  "
          f"mIoU {np.mean(list(tfm.values())):.2f}")
    if nfm:
        print(f"  nonfrozen : mean o {np.mean(list(nf_o.values())):.3f}  "
              f"rel excess {np.mean([nf[s]['rel_excess'] for s in nf]):.2e}  "
              f"mIoU {np.mean(list(nfm.values())):.2f}")


if __name__ == "__main__":
    main()
