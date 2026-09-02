"""A/B many DECISION RULES on the cached partition + clustering.

Everything downstream of `pooled_unit @ text.T` only. No text-side changes: every statistic
used to correct the similarity matrix is estimated from the PRIMITIVE distribution of the
scene itself (or from the lifting's own per-primitive confidence), never from class names,
prompts, or GT.

Usage:  RULES=plain,zprim,cprim0.5 python eval_decision_rules.py
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, calculate_metrics, remap_gt_labels, embed_class_names,
)
from run_cluster_classify_eval import SCENES, HARD_FIRST, K_FLAT

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


# ---------------------------------------------------------------- decision rules
# Each rule maps (sim_leaf (L,C), sim_prim (N,C), gt-free context) -> (L,C) score.
# `sim_prim` is the similarity of EVERY valid primitive to the text bank: the empirical
# distribution the correction is estimated from.

def rule_plain(sl, sp, **kw):
    return sl


def rule_center(sl, sp, lam=1.0, pop="prim", **kw):
    """sim - lam * column-mean. Column mean is a per-CLASS statistic estimated over the
    scene's own primitives -- no class-name information enters."""
    m = (sp if pop == "prim" else sl).mean(dim=0, keepdim=True)
    return sl - lam * m


def rule_zscore(sl, sp, pop="prim", **kw):
    ref = sp if pop == "prim" else sl
    return (sl - ref.mean(0, keepdim=True)) / ref.std(0, keepdim=True).clamp_min(1e-6)


def rule_csls(sl, sp, k=10, pop="prim", **kw):
    """CSLS (Lample et al. 2018). r_T(c) = mean similarity of class c to its k nearest
    PRIMITIVES -- a local-density correction estimated from the image side only.
    r_S(l) = mean similarity of leaf l to its k nearest classes."""
    ref = sp if pop == "prim" else sl
    kk = min(k, ref.shape[0])
    r_t = ref.topk(kk, dim=0).values.mean(dim=0, keepdim=True)      # (1, C)
    kc = min(k, sl.shape[1])
    r_s = sl.topk(kc, dim=1).values.mean(dim=1, keepdim=True)       # (L, 1)
    return 2 * sl - r_t - r_s


def rule_rank(sl, sp, pop="prim", **kw):
    """Fully non-parametric hubness correction: replace each class column's score by the
    leaf's QUANTILE within that class column's primitive distribution. Monotone per column,
    so it cannot sharpen anything -- it only removes per-class scale/offset."""
    ref = sp if pop == "prim" else sl
    ref_sorted, _ = ref.sort(dim=0)
    out = torch.empty_like(sl)
    for c in range(sl.shape[1]):
        out[:, c] = torch.searchsorted(ref_sorted[:, c].contiguous(), sl[:, c].contiguous()).float()
    return out / ref.shape[0]


def rule_quant(sl, sp, lam=0.5, q=0.5, **kw):
    """sim - lam * (per-class q-quantile of the PRIMITIVE similarity distribution).

    This one family contains everything worth testing here. CSLS's r_S(l) term is
    constant along a row, so it cannot change an argmax: CSLS reduces EXACTLY to
    `sim - 0.5 * (mean of the class's top-k primitive similarities)`, i.e. this rule with
    lam=0.5 and an upper-tail location statistic. Mean-centering is this rule with the
    mean in place of the quantile. So `lam` (how much to correct) and `q` (which part of
    the class's own primitive distribution defines its baseline) span the whole design
    space, and both are estimated from primitives alone.
    """
    g = sp.quantile(q, dim=0, keepdim=True) if q is not None else sp.mean(0, keepdim=True)
    return sl - lam * g


RULES = {
    "plain": rule_plain,
    "zleaf": lambda sl, sp, **kw: rule_zscore(sl, sp, pop="leaf"),
    "zprim": lambda sl, sp, **kw: rule_zscore(sl, sp, pop="prim"),
    "cleaf1.0": lambda sl, sp, **kw: rule_center(sl, sp, 1.0, "leaf"),
    "csls10": lambda sl, sp, **kw: rule_csls(sl, sp, 10, "prim"),
    "csls100": lambda sl, sp, **kw: rule_csls(sl, sp, 100, "prim"),
    "rankprim": lambda sl, sp, **kw: rule_rank(sl, sp, pop="prim"),
}
for _lam in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
    RULES[f"cprim{_lam}"] = (lambda l: lambda sl, sp, **kw: rule_center(sl, sp, l, "prim"))(_lam)
for _lam in (0.25, 0.5, 0.75, 1.0):
    for _q in (0.5, 0.9, 0.99):
        RULES[f"q{_q}_{_lam}"] = (lambda l, q: lambda sl, sp, **kw: rule_quant(sl, sp, l, q))(_lam, _q)


def pool(labels, unit, num_labels, weights=None):
    pooled = torch.zeros(num_labels, unit.shape[1], device=unit.device)
    pooled.index_add_(0, labels, unit * weights[:, None] if weights is not None else unit)
    return F.normalize(pooled, dim=-1), pooled.norm(dim=-1) > 1e-8


def main():
    enable_determinism()
    device = "cuda"
    suffix = os.environ.get("FEAT_SUFFIX", "_ogl3")
    want = [r for r in os.environ.get("RULES", "").split(",") if r] or list(RULES)
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = [s for s in HARD_FIRST if (s in only if only else True)]
    methods = os.environ.get("METHODS", "pos_aware_64x5,feat_kmeans320").split(",")

    # POOLING WEIGHTS: per-primitive confidence from the lifting itself, used to WEIGHT the
    # leaf pool (never to threshold). `norm_f` is the magnitude term the geometric-median
    # solver discards; see probe_lifting_confidence.py for its monotonicity with correctness.
    wants_w = [w for w in os.environ.get("WEIGHTS", "none").split(",") if w]
    res = {f"{r}|{w}": {m: {cs: {} for cs in CLASS_SETS} for m in methods}
           for r in want for w in wants_w}
    text_cache = {}

    for scene in scenes:
        c = torch.load(os.path.join(CACHE, f"{scene}{suffix}.pt"), map_location="cpu", weights_only=False)
        unit = F.normalize(c["unit"].to(device).float(), dim=-1)
        raw_labels = c["raw_labels"].numpy()
        point_row = c["point_row"].numpy()
        owned = point_row >= 0
        all_names = c["all_names"]
        clus = {"pos_aware_64x5": c["pos_labels"].to(device), "feat_kmeans320": c["flat_labels"].to(device)}
        wmap = {"none": None}
        if wants_w != ["none"]:
            cf = torch.load(os.path.join(CACHE, f"conf_{scene}{suffix}.pt"),
                            map_location=device, weights_only=False)
            for k, v in cf.items():
                wmap[k] = v.float()
            wmap["norm_f2"] = cf["norm_f"].float() ** 2
            wmap["logn_eff"] = cf["n_eff"].float().clamp_min(1.0).log()
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        print(f"\n===== {scene} (N={unit.shape[0]}) =====", flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
            target_ids = [i for i, _ in kept]
            target_names = [n for _, n in kept]
            nC = len(target_ids)
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
            key = tuple(target_names)
            if key not in text_cache:
                text_cache[key] = embed_class_names(target_names, device)
            text = text_cache[key]
            sim_prim = unit @ text.T                                    # (N, C)

            for m in methods:
                labels = clus[m]
                for w in wants_w:
                    pooled, nonempty = pool(labels, unit, K_FLAT, wmap[w])
                    sim_leaf = pooled @ text.T                          # (L, C)
                    for r in want:
                        score = RULES[r](sim_leaf, sim_prim)
                        cls = torch.full((K_FLAT,), -1, dtype=torch.long, device=device)
                        cls[nonempty] = score[nonempty].argmax(dim=-1)
                        prim_cls = cls[labels].cpu().numpy()
                        pred = np.zeros(raw_labels.shape[0], dtype=np.int64)
                        pred[owned] = prim_cls[point_row[owned]] + 1
                        _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
                        res[f"{r}|{w}"][m][cs][scene] = {"mIoU": miou, "mAcc": macc}
            print(f"  {cs}: " + "  ".join(
                f"{k}={res[k][methods[0]][cs][scene]['mIoU']*100:.2f}" for k in res), flush=True)

    n = len(scenes)
    print(f"\n\n=== {n}-scene mean{'' if n > 1 else ' (PILOT -- not a conclusion)'}, feats={suffix} ===")
    for m in methods:
        print(f"\n-- {m} --")
        print(f"{'rule':<20} " + "  ".join(f"{cs[13:]:>14}" for cs in CLASS_SETS))
        for r in res:
            cells = []
            for cs in CLASS_SETS:
                mi = [v["mIoU"] for v in res[r][m][cs].values()]
                ma = [v["mAcc"] for v in res[r][m][cs].values()]
                cells.append(f"{np.mean(mi)*100:6.2f}/{np.mean(ma)*100:6.2f}")
            print(f"{r:<20} " + "  ".join(cells))

    out = os.path.join(CACHE, f"rules_{n}scene{suffix}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
