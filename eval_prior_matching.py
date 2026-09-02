"""Generalise partial centering from ONE scalar to the full per-class bias vector, without
labels and without touching the text side.

Partial centering is `sim - lam * colmean`: a 1-parameter path through a C-dimensional bias
space whose ORACLE is worth +10 mIoU more (probe_bias_ceiling.py). The text-free way to pick
all C biases is MARGINAL MATCHING (iterative proportional fitting / logit adjustment): choose
b so the point-weighted predicted class distribution hits a target prior. `floor` winning
everything and chair/sofa/table winning nothing is a statement about the predicted MARGINAL,
so correcting the marginal is the direct attack on it -- and mIoU is class-averaged, which is
exactly what a flatter marginal serves.

Target prior: p0^(1-a) * uniform^a, renormalised. a=0 is a no-op, a=1 forces uniform mass.
p0 is the method's OWN predicted marginal at lam=0, so nothing external enters.

Softness stance: this is a per-class OFFSET on the score, applied before a single argmax. It
changes WHICH class wins, never how much evidence is retained -- no observation is discarded,
no weight is sharpened. It is not a decisiveness move.
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
from run_cluster_classify_eval import HARD_FIRST, K_FLAT

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def marginal_match(sl, nonempty, leaf_mass, alpha, iters=100, eta=0.02, temp=None):
    """Iterative proportional fitting. leaf_mass: (L,) point counts.

    temp=None uses the HARD argmax marginal, which is a step function of b over only 320
    atoms carrying very unequal point mass -- IPF on it has no stable fixed point (measured:
    it collapses to 2-30 mIoU). temp>0 replaces it with a softmax marginal, making p(b)
    smooth and the iteration a proper Sinkhorn scaling; the final decision is still a hard
    argmax of the bias-adjusted score.
    """
    C = sl.shape[1]
    b = torch.zeros(C, device=sl.device)
    m = leaf_mass.clamp_min(0.0)

    def marginal(bias):
        if temp is None:
            cls = (sl - bias).argmax(-1)
            h = torch.zeros(C, device=sl.device)
            h.index_add_(0, cls[nonempty], m[nonempty])
        else:
            r = torch.softmax((sl - bias) / temp, dim=-1)
            h = (r[nonempty] * m[nonempty, None]).sum(0)
        return h / h.sum().clamp_min(1e-9)

    p0 = marginal(b).clamp_min(1e-6)
    target = (p0 ** (1 - alpha)) * ((1.0 / C) ** alpha)
    target = target / target.sum()
    for _ in range(iters):
        p = marginal(b).clamp_min(1e-6)
        b = b + eta * (torch.log(p) - torch.log(target))
    return (sl - b).argmax(-1)


def main():
    enable_determinism()
    device = "cuda"
    suffix = "_ogl3"
    alphas = [float(a) for a in os.environ.get("ALPHAS", "0.25,0.5,0.75,1.0").split(",")]
    lams = [float(l) for l in os.environ.get("LAMS", "0,0.3").split(",")]
    scenes = [s for s in os.environ.get("SCENES", ",".join(HARD_FIRST)).split(",") if s]

    temps = [float(t) for t in os.environ.get("TEMPS", "0.01,0.02,0.05").split(",")]
    keys = ([f"lam{l}" for l in lams]
            + [f"a{a}T{t}" for t in temps for a in alphas]
            + [f"lam0.3+a{a}T{t}" for t in temps for a in alphas])
    res = {k: {cs: {} for cs in CLASS_SETS} for k in keys}
    text_cache = {}

    for scene in scenes:
        c = torch.load(os.path.join(CACHE, f"{scene}{suffix}.pt"), map_location="cpu", weights_only=False)
        unit = F.normalize(c["unit"].to(device).float(), dim=-1)
        raw = c["raw_labels"].numpy()
        prow = c["point_row"].numpy()
        owned = prow >= 0
        labels = c["pos_labels"].to(device)
        lab_np = labels.cpu().numpy()
        n2i = {n: i for i, n in enumerate(c["all_names"])}
        present = set(np.unique(raw).tolist())
        # point mass per leaf -- uses GT point POSITIONS (given at eval time), never labels
        leaf_mass = torch.zeros(K_FLAT, device=device)
        lm = np.bincount(lab_np[prow[owned]], minlength=K_FLAT)
        leaf_mass += torch.from_numpy(lm).to(device).float()
        print(f"\n===== {scene} =====", flush=True)

        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, labels, unit)
        nonempty = pooled.norm(dim=-1) > 1e-8
        pooled = F.normalize(pooled, dim=-1)

        for cs in CLASS_SETS:
            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            nC = len(kept)
            gt_t = torch.from_numpy(remap_gt_labels(raw, [i for i, _ in kept])).long()
            key = tuple(n for _, n in kept)
            if key not in text_cache:
                text_cache[key] = embed_class_names(list(key), device)
            text = text_cache[key]
            sl = pooled @ text.T
            colmean = (unit @ text.T).mean(0)

            def run(cls):
                cl = torch.full((K_FLAT,), -1, dtype=torch.long, device=device)
                cl[nonempty] = cls[nonempty]
                pc = cl.cpu().numpy()[lab_np]
                pred = np.zeros(raw.shape[0], dtype=np.int64)
                pred[owned] = pc[prow[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
                return miou, macc

            for l in lams:
                res[f"lam{l}"][cs][scene] = run((sl - l * colmean).argmax(-1))
            for t in temps:
                for a in alphas:
                    res[f"a{a}T{t}"][cs][scene] = run(
                        marginal_match(sl, nonempty, leaf_mass, a, temp=t))
                    res[f"lam0.3+a{a}T{t}"][cs][scene] = run(
                        marginal_match(sl - 0.3 * colmean, nonempty, leaf_mass, a, temp=t))
            print(f"  {cs}: " + "  ".join(f"{k}={res[k][cs][scene][0]*100:.2f}" for k in keys), flush=True)

    n = len(scenes)
    print(f"\n=== {n}-scene mean{'' if n > 1 else ' (PILOT)'} ===")
    print(f"{'rule':<16} " + "  ".join(f"{cs[13:]:>14}" for cs in CLASS_SETS))
    for k in keys:
        cells = []
        for cs in CLASS_SETS:
            v = list(res[k][cs].values())
            cells.append(f"{np.mean([x[0] for x in v])*100:6.2f}/{np.mean([x[1] for x in v])*100:6.2f}")
        print(f"{k:<16} " + "  ".join(cells))
    with open(os.path.join(CACHE, f"prior_{n}scene.json"), "w") as f:
        json.dump({k: {cs: {s: list(v) for s, v in d.items()} for cs, d in vv.items()}
                   for k, vv in res.items()}, f, indent=1)


if __name__ == "__main__":
    main()
