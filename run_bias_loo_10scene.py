"""Is there ONE general per-class bias that works on every scene? Strict leave-one-scene-out.

THE QUESTION. The oracle per-class additive bias is worth +10.2 to +12.5 mIoU over the best
hand-tuned lambda on every scene measured. If that bias were a property of the SCENE it would
be unusable (we would need labels to fit it). But the evidence says it is a property of the
CLASS NAME'S CLIP EMBEDDING: the deviation from centering, eps = b - 0.5*colmean, is tiny
(|eps| <= 0.035), sign-consistent across scenes (wall and floor negative everywhere, i.e.
centering OVER-suppresses the dominant classes), and correlates with NO unsupervised scene
statistic (all |spearman| <= 0.33 against column mean, sd, q90, argmax win-share, GT share).

If it is a class-name property, one vector should transfer to unseen scenes.

THE PROTOCOL, and why it is honest. For each held-out scene, eps is fit by coordinate ascent on
the mean mIoU of the OTHER NINE scenes, then applied to the held-out scene, which contributed
nothing to the fit. Every reported number is therefore out-of-sample. This is standard
cross-validated calibration, not oracle selection: the alternative -- fitting on all ten and
reporting on all ten -- would be meaningless and is not computed here.

ADMISSIBILITY, stated plainly rather than assumed. eps is fit against ground-truth labels for
the target class names, on disjoint scenes. ScanNet's train/val split licenses cross-scene
calibration, and this is the same discipline any hyperparameter tuning uses (our lambda=0.5 was
itself chosen against GT on these scenes, with LESS rigour, since it was not held out). But it
does sit against the project rule "no learned projection trained on target classes". The rule's
purpose is to forbid learning a MAPPING from features to the target classes; eps is 19 scalars
of per-class offset with no feature dependence. That is a judgement call for the user; the
result is reported either way, and if ruled inadmissible as a METHOD it stands as an ANALYSIS
result bounding the family.

ANCHORING. All fits are anchored at lambda = 0.5 (the validated global value) and only the
DEVIATION eps is transferred. An earlier attempt transferred the raw b across scenes fitted at
different lambda anchors and read -24.19 -- mixing two scales. Recorded so it is not re-derived.

FALSIFIER, pre-registered: LOO must beat partial centering at the validated lambda by >= +0.5
mIoU at 19cls AND be positive on >= 8/10 scenes. If it fails, the +12.5 oracle is per-scene
overfitting and the entire per-class-bias direction closes -- which is itself worth having,
since it retires the question that has already consumed capacity matching, perimeter
minimisation and max-entropy.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, remap_gt_labels)
from run_cluster_classify_eval import K_FLAT

CACHE = (r"C:\Users\rajehyl\AppData\Local\Temp\claude"
         r"\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache")
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
          "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]
LAM = 0.5


def load_scene(scene, cs, device):
    """-> a closure scoring any bias vector, plus the scene's colmean and class names."""
    c = torch.load(os.path.join(CACHE, f"{scene}_ogl3.pt"), map_location="cpu",
                   weights_only=False)
    unit = F.normalize(c["unit"].to(device).float(), dim=-1)
    raw = c["raw_labels"].numpy()
    prow = c["point_row"].numpy()
    owned = prow >= 0
    labels = c["pos_labels"].to(device)
    n2i = {n: i for i, n in enumerate(c["all_names"])}
    present = set(np.unique(raw).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
    names = [n for _, n in kept]
    gt_t = torch.from_numpy(remap_gt_labels(raw, [i for i, _ in kept])).long()
    text = embed_class_names(names, device)

    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
    pooled.index_add_(0, labels, unit)
    nonempty = pooled.norm(dim=-1) > 1e-8
    sl = F.normalize(pooled, dim=-1) @ text.T
    colmean = (unit @ text.T).mean(0)
    lab_np = labels.cpu().numpy()
    nC = len(kept)

    def score(bias):
        s = sl - bias
        cls = torch.full((K_FLAT,), -1, dtype=torch.long, device=device)
        cls[nonempty] = s[nonempty].argmax(-1)
        pc = cls.cpu().numpy()[lab_np]
        pred = np.zeros(raw.shape[0], dtype=np.int64)
        pred[owned] = pc[prow[owned]] + 1
        _, miou, _, _ = calculate_metrics(gt_t, torch.from_numpy(pred).long(), nC + 1)
        return float(miou) * 100

    return {"score": score, "colmean": colmean, "names": names, "device": device}


def fit_eps(scenes_data, name_index, n_names, device, span=0.06, steps=25, sweeps=3):
    """Coordinate-ascend eps on the MEAN mIoU of the given scenes, anchored at lam=0.5.

    eps is indexed by a GLOBAL class-name table so scenes with different present-class subsets
    contribute to the same coordinates. A scene that lacks a class simply does not constrain it.
    """
    eps = torch.zeros(n_names, device=device)

    def mean_miou(e):
        tot = 0.0
        for d in scenes_data:
            idx = torch.tensor([name_index[n] for n in d["names"]], device=device)
            tot += d["score"](LAM * d["colmean"] + e[idx])
        return tot / len(scenes_data)

    best = mean_miou(eps)
    grid = torch.linspace(-span, span, steps, device=device)
    for _ in range(sweeps):
        for k in range(n_names):
            keep, cur = eps[k].item(), best
            for g in grid:
                eps[k] = g
                v = mean_miou(eps)
                if v > cur:
                    cur, keep = v, g.item()
            eps[k] = keep
            best = cur
    return eps, best


def main():
    enable_determinism()
    device = "cuda"
    out = {}
    for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
        data = {}
        for s in SCENES:
            p = os.path.join(CACHE, f"{s}_ogl3.pt")
            if os.path.exists(p):
                data[s] = load_scene(s, cs, device)
        if len(data) < 3:
            print(f"[skip] {cs}: only {len(data)} scenes cached", flush=True)
            continue
        all_names = sorted({n for d in data.values() for n in d["names"]})
        nidx = {n: i for i, n in enumerate(all_names)}

        rows = []
        for held in data:
            fit_on = [d for s, d in data.items() if s != held]
            eps, fit_score = fit_eps(fit_on, nidx, len(all_names), device)
            d = data[held]
            idx = torch.tensor([nidx[n] for n in d["names"]], device=device)
            base = d["score"](LAM * d["colmean"])                 # partial centering
            plain = d["score"](torch.zeros_like(d["colmean"]))    # plain argmax
            loo = d["score"](LAM * d["colmean"] + eps[idx])
            rows.append((held, plain, base, loo, loo - base))
            print(f"  [{cs[11:]}] {held}: plain {plain:6.2f} | centering {base:6.2f} | "
                  f"LOO {loo:6.2f}  ({loo-base:+.2f})   [fit mIoU on 9 = {fit_score:.2f}]",
                  flush=True)
        out[cs] = rows
        d19 = [r[4] for r in rows]
        print(f"  [{cs[11:]}] MEAN: plain {np.mean([r[1] for r in rows]):.2f} | "
              f"centering {np.mean([r[2] for r in rows]):.2f} | "
              f"LOO {np.mean([r[3] for r in rows]):.2f}  "
              f"({np.mean(d19):+.2f}, positive on {sum(x>0 for x in d19)}/{len(d19)})\n",
              flush=True)

    os.makedirs("artifacts/scannet", exist_ok=True)
    json.dump({k: [list(r) for r in v] for k, v in out.items()},
              open("artifacts/scannet/bias_loo_10scene.json", "w"), indent=1)
    print("=== SUMMARY (all numbers OUT OF SAMPLE) ===")
    print(f"{'class set':<12}{'plain':>8}{'centering':>11}{'LOO':>8}{'delta':>8}{'pos':>7}")
    for cs, rows in out.items():
        d = [r[4] for r in rows]
        print(f"{cs[11:]:<12}{np.mean([r[1] for r in rows]):8.2f}"
              f"{np.mean([r[2] for r in rows]):11.2f}{np.mean([r[3] for r in rows]):8.2f}"
              f"{np.mean(d):+8.2f}{sum(x>0 for x in d):>4}/{len(d)}")


if __name__ == "__main__":
    main()
