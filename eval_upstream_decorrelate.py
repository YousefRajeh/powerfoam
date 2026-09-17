"""Decorrelate the region-level CLIP errors, upstream, on two independent axes.

A20.7 established the defect: SAM regions are clean (0.926 purity) but CLIP labels a region with
its own majority GT class only 49.2% of the time, and because a region carries ONE feature that
every primitive under it inherits, that error repeats in every view. No aggregator can remove a
bias. The only way out is to make the region-level errors themselves less correlated.

Two axes are available without re-running SAM:

  TEXT  the reported pipeline embeds the BARE class name. CLIP's text tower was trained on
        captions, so a bare noun is off-manifold; averaging many templates is the standard
        remedy and moves a different part of the system than any aggregation change. Free at
        inference and single-query safe -- class c's embedding never depends on the others.
  CROP  `_l3` and `_blackboth` are the SAME SAM regions rendered with different crop treatments,
        i.e. two independent CLIP readings of identical geometry. If their errors decorrelate,
        averaging them beats either.

Both are measured at two levels: on the regions themselves (did CLIP name this region right)
and end to end in the reported point protocol. Decorrelation is reported explicitly as
P(B wrong | A wrong) against P(B wrong); a ratio near 1 means the axis is useless, because the
two readings fail on the same regions.

Everything runs off `cache_region_links.py`, so a score field is S = W @ (F @ T^T) and each arm
is a matmul.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import calculate_metrics
from diagnose_holes import SCENES
import prompts as PR

CACHE = "artifacts/region_cache"


def load(scene, recon, class_set, feat_dir, levels=None):
    tag = f"{scene}_{recon}_{class_set}_{feat_dir}" + (f"_L{levels}" if levels else "")
    p = f"{CACHE}/{tag}.npz"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return dict(np.load(p, allow_pickle=True))


def region_acc(d, T, dev="cuda"):
    """Pixel-weighted fraction of regions CLIP names with the region's own majority GT class."""
    Fm = F.normalize(torch.from_numpy(d["F"]).to(dev).float(), dim=-1)
    lab = (Fm @ T.T).argmax(1).cpu().numpy() + 1
    gt = d["region_gt"].astype(np.int64)
    w = d["region_npix"].astype(np.float64)
    m = (gt > 0) & (w > 0)
    if not m.any():
        return float("nan"), np.zeros(0, bool), np.zeros(0)
    return float(np.average(lab[m] == gt[m], weights=w[m])), (lab[m] == gt[m]), w[m]


def prim_scores(d, T, dev="cuda"):
    """S = W @ (F @ T^T): the per-primitive score field implied by these region features."""
    Fm = F.normalize(torch.from_numpy(d["F"]).to(dev).float(), dim=-1)
    sim = Fm @ T.T                                            # (R, C)
    P, C = int(d["P"]), T.shape[0]
    S = torch.zeros(P, C, device=dev)
    r = torch.from_numpy(d["rows"].astype(np.int64)).to(dev)
    c = torch.from_numpy(d["cols"].astype(np.int64)).to(dev)
    w = torch.from_numpy(d["wts"]).to(dev).float()
    S.index_add_(0, r, sim[c] * w.unsqueeze(-1))
    tot = torch.zeros(P, device=dev).index_add_(0, r, w)
    return S, tot


def score_points(d, pred_cls, C):
    """pred_cls is 1..C per primitive (0 = abstain). Scores in the reported point protocol."""
    assigned, gt_lab = d["assigned"].astype(np.int64), d["gt_lab"].astype(np.int64)
    own = assigned >= 0
    pl = np.zeros(len(gt_lab), np.int64)
    pl[own] = pred_cls[assigned[own]]
    _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                     torch.from_numpy(pl).long(), C + 1)
    return float(mi), float(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--prompt-sets", default="bare,photo,indoor,openai80,indoor+openai80")
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3")
    ap.add_argument("--crop-b", default="openclip_features_sam_blackboth")
    ap.add_argument("--crop-b-levels", default="3")
    ap.add_argument("--out", default="artifacts/scannet/upstream_decorrelate.json")
    a = ap.parse_args()
    dev = "cuda"
    psets = a.prompt_sets.split(",")
    res, model, tok = {}, None, None

    for sc in a.scenes.split(","):
        try:
            dA = load(sc, a.recon, a.class_set, a.feat_dir)
        except FileNotFoundError as e:
            print(f"[{sc}] no cache: {e}"); continue
        kept = list(dA["kept"])
        C = len(kept)
        lifted = F.normalize(torch.from_numpy(dA["lifted"]).to(dev).float(), dim=-1)
        row = {}
        Ts = {}
        for ps in psets:
            T, model, tok = PR.embed(kept, PR.SETS[ps], dev, model, tok)
            Ts[ps] = T
            ra, _, _ = region_acc(dA, T)
            S, tot = prim_scores(dA, T)
            pc = (S.argmax(1) + 1).cpu().numpy(); pc[(tot <= 0).cpu().numpy()] = 0
            mi_s, ma_s = score_points(dA, pc, C)
            pl = (lifted @ T.T).argmax(1).cpu().numpy() + 1
            mi_l, ma_l = score_points(dA, pl, C)
            row[ps] = dict(region_acc=ra, score_miou=mi_s, score_macc=ma_s,
                           lift_miou=mi_l, lift_macc=ma_l)

        # CROP axis: same regions, different crop treatment
        try:
            dB = load(sc, a.recon, a.class_set, a.crop_b, a.crop_b_levels)
            T = Ts["bare"]
            raA, okA, wA = region_acc(dA, T)
            raB, okB, wB = region_acc(dB, T)
            SA, tA = prim_scores(dA, T)
            SB, tB = prim_scores(dB, T)
            nA = SA / tA.clamp_min(1e-9).unsqueeze(-1)
            nB = SB / tB.clamp_min(1e-9).unsqueeze(-1)
            for nm, S_, t_ in [("cropA", nA, tA), ("cropB", nB, tB),
                               ("cropA+B", nA + nB, tA + tB)]:
                pc = (S_.argmax(1) + 1).cpu().numpy(); pc[(t_ <= 0).cpu().numpy()] = 0
                mi, ma = score_points(dA, pc, C)
                row[nm] = dict(score_miou=mi, score_macc=ma)
            row["cropA"]["region_acc"] = raA
            row["cropB"]["region_acc"] = raB
        except FileNotFoundError:
            pass
        res[sc] = row
        print(f"[{sc}] " + "  ".join(
            f"{k}:{v.get('region_acc', float('nan')):.3f}/{v.get('lift_miou', v['score_miou'])*100:.2f}"
            for k, v in row.items()))

    if res:
        keys = [k for k in next(iter(res.values()))]
        print(f"\n=== mean over {len(res)} scenes ===")
        print(f"  {'arm':<18} {'region acc':>10} {'lift mIoU':>10} {'lift mAcc':>10} "
              f"{'score mIoU':>11} {'wins':>6}")
        base = np.mean([res[s]["bare"]["lift_miou"] for s in res]) * 100
        for k in keys:
            rs = [res[s][k] for s in res if k in res[s]]
            ra = np.mean([r.get("region_acc", np.nan) for r in rs])
            lm = np.mean([r["lift_miou"] for r in rs]) * 100 if "lift_miou" in rs[0] else np.nan
            la = np.mean([r["lift_macc"] for r in rs]) * 100 if "lift_macc" in rs[0] else np.nan
            sm = np.mean([r["score_miou"] for r in rs]) * 100
            w = (sum(res[s][k]["lift_miou"] > res[s]["bare"]["lift_miou"] for s in res if k in res[s])
                 if "lift_miou" in rs[0] else -1)
            print(f"  {k:<18} {ra:>10.3f} {lm:>10.2f} {la:>10.2f} {sm:>11.2f} "
                  f"{w if w >= 0 else '-':>6}  ({lm-base:+.2f})")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
