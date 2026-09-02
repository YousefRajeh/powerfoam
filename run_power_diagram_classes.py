"""Semi-discrete OT on the CLIP sphere: solve for the optimal class POWER weights.

THE STRUCTURE (user's observation, formalised).

Both ends of our pipeline are Voronoi diagrams:
  * the foam partitions R^3   -- power diagram, sites c_i, weights r_i^2
  * the class embeddings partition the CLIP sphere -- because f and t_c are both unit,
        ||f - t_c||^2 = 2 - 2 <f, t_c>,   so   argmax_c <f,t_c> = argmin_c ||f - t_c||^2
    i.e. `argmax` IS the nearest-site Voronoi membership query with sites {t_c}.

Lifting is a map from the first diagram to the second. The natural generalisation of the second
is a POWER diagram, argmin_c ||f - t_c||^2 - w_c, equivalently

        argmax_c [ <f, t_c> + w_c / 2 ].

PARTIAL CENTERING IS ALREADY THIS. Our validated text-side trick is `sim - lam * colmean`,
which is exactly w_c = -2 lam * colmean_c. So the +2.75/+1.67/+1.38 that partial centering buys
was a hand-tuned weight vector for a spherical power diagram. This script asks what the OPTIMAL
weight vector is.

THE THEOREM. Aurenhammer, Hoffmann and Aronov (1998): given sites {t_c} and target capacities
{mu_c}, there is a weight vector w -- unique up to an additive constant -- whose power cells
carry exactly those capacities, and it maximises a CONCAVE functional with gradient

        dPhi/dw_c = mu_c - rho(cell_c).

So plain gradient ascent converges globally: raise the weight of under-represented classes.
These are the Kantorovich potentials of a semi-discrete optimal transport problem. No
hyperparameter, no line search needed at this scale.

WHY THIS IS NOT THE REFUTED PRIOR MATCHING. Sinkhorn prior matching (reversal #13) and the
mass-conserving TPFA flow both matched MARGINALS OF SOFT POSTERIORS, and were refuted -- and
independently we measured that diffusion moves the class-mass distribution AWAY from GT (TV
0.274 -> 0.343) while scoring BETTER, so soft marginals are not what mIoU rewards. This instead
moves the HARD decision boundaries, which is the same thing partial centering does, and that
direction is already validated.

TARGET CAPACITIES, and the honesty constraint. mu must not come from GT -- that would be
cheating. Two admissible choices, both class-agnostic:
    uniform      mu_c = 1/C            each class owns an equal share of cells
    volume       mu_c = 1/C of VOLUME  equal share of space (foam-exact cell volumes)
The GT-derived target is computed too, but ONLY as an oracle upper bound, clearly labelled and
never proposed as a method.

FALSIFIER: the solved weights must beat partial centering at its validated lambda by >= +0.3
mIoU at 19cls. Matching it is not enough -- the claim is that solving beats tuning.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SP = (r"C:\Users\rajehyl\AppData\Local\Temp\claude"
      r"\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad")
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
# validated per-class-set lambda for partial centering (Experiment-F)
LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def solve_power_weights(sim, mu, mass=None, iters=300, lr=1.0, verbose=False):
    """Maximise the AHA/semi-discrete-OT functional by gradient ascent on w.

        assignment(f) = argmax_c [ sim(f,c) + w_c/2 ]
        dPhi/dw_c     = mu_c - rho(cell_c)

    `mass` weights each cell's contribution to rho (e.g. cell volume); uniform if None.
    Concave in w, so this converges globally; w is fixed up to an additive constant, which is
    removed by centring at each step since only differences affect the argmax.
    """
    dev = sim.device
    C = sim.shape[1]
    w = torch.zeros(C, device=dev)
    m = torch.ones(sim.shape[0], device=dev) if mass is None else mass
    m = m / m.sum().clamp_min(1e-12)
    for t in range(iters):
        cls = (sim + w[None, :] / 2).argmax(-1)
        rho = torch.zeros(C, device=dev).index_add_(0, cls, m)
        g = mu - rho
        w = w + lr * g
        w = w - w.mean()
        if verbose and t % 100 == 0:
            print(f"      it{t:>4} |mu-rho|_1={g.abs().sum():.4f}", flush=True)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if not all(os.path.exists(p) for p in (fp, apth)):
            print(f"[skip] {scene}", flush=True)
            continue
        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        cg = os.path.join(SP, f"cellgeom_{scene}_pf_nonfroz.npz")
        V = (torch.from_numpy(np.load(cg)["V"].astype(np.float32)).to(dev)
             if os.path.exists(cg) else None)

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            C = len(names)
            text = embed_class_names(names, dev)
            sim = unit @ text.T

            def score(tag, cls_vec):
                cls = cls_vec.cpu().numpy() + 1
                sc = owned.copy()
                sc[owned] = valid[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                res.setdefault((tag, cs), []).append(float(mi) * 100)

            # --- baselines
            score("argmax (Voronoi)", sim.argmax(-1))
            lam = LAMBDA[cs]
            colmean = sim[vt].mean(0)
            score(f"partial centering (lam={lam})", (sim - lam * colmean[None, :]).argmax(-1))

            # --- solved power weights, uniform capacity over CELLS
            mu = torch.full((C,), 1.0 / C, device=dev)
            w = solve_power_weights(sim[vt], mu, iters=a.iters)
            score("power w (uniform cells)", (sim + w[None, :] / 2).argmax(-1))

            # --- solved power weights, uniform capacity over VOLUME (foam-exact)
            if V is not None:
                wv = solve_power_weights(sim[vt], mu, mass=V[vt], iters=a.iters)
                score("power w (uniform volume)", (sim + wv[None, :] / 2).argmax(-1))

            # --- ORACLE ONLY: capacity from the GT class distribution. Not a method.
            Gc = torch.tensor([float((gt == c + 1).sum()) for c in range(C)], device=dev)
            if Gc.sum() > 0:
                Gc = Gc / Gc.sum()
                wg = solve_power_weights(sim[vt], Gc, iters=a.iters)
                score("[oracle] power w (GT capacity)", (sim + wg[None, :] / 2).argmax(-1))
        print(f"[{scene}] done", flush=True)

    n = len(next(iter(res.values())))
    print(f"\n=== {n} scenes ===")
    print(f"{'arm':<34}" + "".join(f"{c[11:]:>10}" for c in CLASS_SETS) + "   delta vs centering")
    base = {c: np.mean(res[(f"partial centering (lam={LAMBDA[c]})", c)]) for c in CLASS_SETS}
    for tag in ["argmax (Voronoi)"] + [f"partial centering (lam={LAMBDA[c]})"
                                       for c in CLASS_SETS[:1]] + \
               ["power w (uniform cells)", "power w (uniform volume)",
                "[oracle] power w (GT capacity)"]:
        keys = [(tag, c) for c in CLASS_SETS]
        if not all(k in res for k in keys):
            # partial centering has a per-class-set name; handle it separately
            if tag.startswith("partial centering"):
                row = "".join(f"{base[c]:10.2f}" for c in CLASS_SETS)
                print(f"{'partial centering (validated lam)':<34}{row}"
                      + "   " + " ".join("+0.00" for _ in CLASS_SETS))
            continue
        row = "".join(f"{np.mean(res[(tag,c)]):10.2f}" for c in CLASS_SETS)
        dl = "  " + " ".join(f"{np.mean(res[(tag,c)])-base[c]:+.2f}" for c in CLASS_SETS)
        print(f"{tag:<34}{row}{dl}")


if __name__ == "__main__":
    main()
