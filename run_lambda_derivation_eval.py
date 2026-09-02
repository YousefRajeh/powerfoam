"""Can lambda be DERIVED instead of swept on the test set?

WHAT LAMBDA IS
--------------
Centering is written `f' = normalize(f - lam*mu_hat)`, which reads like a denoising strength. It is
not. The prediction is `argmax_c <f', t_c>`, and the normalisation multiplies every class score by
the same positive per-cell scalar, so it cannot change the argmax. Dropping it:

    score_c(j) = <f_j, t_c> - lam * <mu_hat, t_c>

**lam is a per-class constant offset, identical for every cell in the scene.** It subtracts each
class's alignment with the scene's mean feature direction. At lam=0 the generic classes whose text
embedding points along the scene mean carry a scene-wide head start; at large lam the offset
dominates and every cell collapses onto the LEAST mu-aligned class. Hence the interior optimum. It is
logit adjustment against a scene prior estimated from features rather than from label counts.

THE DERIVATION
--------------
Split each feature along the mean direction: f_j = a_j*mu_hat + r_j, with r_j orthogonal to mu_hat.

    <f_j, t_c> = a_j*<mu_hat, t_c>  +  <r_j, t_c>
                 \___ nuisance ___/    \_ signal _/

The nuisance term ranks classes identically for every cell and merely scales by the cell's cone
share a_j, so it can only pull every cell toward the same few classes. Two label-free rules cancel it:

  RULE P (per-cell, exact)  lam_j = a_j = <f_j, mu_hat>
      Cancels the nuisance term cell by cell -- i.e. project onto mu_hat's orthogonal complement.
      Zero free parameters. Predicted to beat any single global lam if the split is the true story.

  RULE G (global, decorrelation)  lam = a_bar = || mean_j f_hat_j ||
      The scene-average score is (a_bar - lam)*<mu_hat, t_c>, so lam = a_bar makes every class's
      scene-wide mean score exactly zero: no class gets a head start. Zero free parameters.

Both are computable from the features alone. If either lands on the swept optimum, lam stops being a
tuned hyperparameter.

THE FOAM-SPECIFIC PART
----------------------
mu_hat is currently an unweighted mean over CELLS, which is an arbitrary weighting: it counts a
sliver and a room-sized cell equally. The power diagram partitions space DISJOINTLY, so the
geometrically correct scene mean is VOLUME-weighted -- the spatial average feature of the scene, an
integral a Gaussian mixture cannot even define (overlapping, unbounded support, no partition of
unity). Volumes are estimated by Monte-Carlo: uniform samples in the bounding box assigned through
the same exact `argmin ||x-c||^2 - r^2` membership the renderer uses, so cell volume is proportional
to the sample count it captures. Also compared: render-support weighting (what the lift actually saw).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii


def mc_cell_volumes(centers, radii, valid, n_samples, device, seed=0):
    """Volume of each power cell by Monte-Carlo over the scene bounding box.

    Uses the SAME membership rule as the renderer (argmin of the power distance), so this is the
    exact disjoint partition, not a proxy. Returns counts; only ratios matter downstream.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo = torch.as_tensor(centers, dtype=torch.float64).min(0).values
    hi = torch.as_tensor(centers, dtype=torch.float64).max(0).values
    vol = torch.zeros(centers.shape[0], dtype=torch.float64)
    done = 0
    while done < n_samples:
        n = min(500_000, n_samples - done)
        pts = (torch.rand(n, 3, generator=g, dtype=torch.float64) * (hi - lo) + lo).numpy()
        a = assign_points_to_power_cells(pts, centers, radii, valid=valid, k=64)
        a = a[a >= 0]
        vol.index_add_(0, torch.from_numpy(a).long(), torch.ones(a.shape[0], dtype=torch.float64))
        done += n
    return vol.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lam-max", type=float, default=1.5)
    p.add_argument("--lam-step", type=float, default=0.05)
    p.add_argument("--mc-samples", type=int, default=1_000_000)
    p.add_argument("--outdir", default="artifacts/scannet/lambda_derive")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    lams = [round(x, 4) for x in np.arange(0.0, a.lam_max + 1e-9, a.lam_step)]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"
        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved

        # ---- three weightings of the scene mean ------------------------------------------
        w_uni = vm.float()
        try:
            sup = torch.load(f"{art}/stats_{a.variant}{a.suffix}.pt", map_location=device,
                             weights_only=True)["support"].to(device).float()
            w_sup = sup.clamp_min(0) * vm.float()
        except Exception as e:
            print(f"  [warn] no support stats ({e}); skipping support weighting", flush=True)
            w_sup = None
        tv = time.time()
        vol = mc_cell_volumes(centers, radii, valid_mask, a.mc_samples, device).to(device)
        w_vol = vol * vm.float()
        print(f"[{scene}] P={P:,}  MC volumes {time.time()-tv:.0f}s  "
              f"cells hit={int((vol>0).sum()):,}", flush=True)

        weightings = {"uniform": w_uni, "volume": w_vol}
        if w_sup is not None:
            weightings["support"] = w_sup

        mus, abars = {}, {}
        for k, w in weightings.items():
            wn = w / w.sum().clamp_min(1e-30)
            mu = F.normalize((unit * wn[:, None]).sum(0, keepdim=True), dim=-1)
            mus[k] = mu
            # a_bar under the SAME weighting: the mean cone share = || weighted mean of unit feats ||
            abars[k] = float((wn[:, None] * unit).sum(0).norm())
        print(f"[{scene}] a_bar: " + "  ".join(f"{k}={v:.4f}" for k, v in abars.items()), flush=True)
        print(f"[{scene}] mu agreement: uni.vol={float(mus['uniform']@mus['volume'].T):.4f}"
              + (f"  uni.sup={float(mus['uniform']@mus['support'].T):.4f}" if w_sup is not None else ""),
              flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "a_bar": abars, "P": P,
               "mu_uniform": mus["uniform"].squeeze(0).cpu().tolist(),
               "mu_cos": {k: float(mus["uniform"] @ mus[k].T) for k in mus},
               "sweep": {}, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            cos = torch.zeros(P, len(names), device=device); cos[vm] = unit[vm] @ text.T

            def score(cls_np):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                return float(miou) * 100, float(macc) * 100

            # the offset form: score_c = <f,t_c> - lam*<mu,t_c>.  No renormalisation needed --
            # it is a per-cell positive scale and cannot move an argmax.
            for wk, mu in mus.items():
                off = (mu @ text.T).squeeze(0)                       # (C,)
                curve, ent, nact = [], [], []
                wv = (w_vol / w_vol.sum().clamp_min(1e-30))
                for lam in lams:
                    cls = (cos - lam * off).argmax(-1)
                    m, _ = score(cls.cpu().numpy())
                    curve.append(m)
                    # LABEL-FREE criterion: entropy of the volume-weighted predicted class
                    # histogram. At lam=0 mass concentrates on mu-aligned classes; at large lam it
                    # collapses onto the single least-aligned class. So it has an interior maximum,
                    # unlike the confidence-style criteria tried before (all monotone in lam).
                    h = torch.zeros(len(names), device=device).index_add_(0, cls, wv)
                    h = h / h.sum().clamp_min(1e-30)
                    ent.append(float(-(h * (h + 1e-30).log()).sum()))
                    nact.append(int((h > 1e-4).sum()))
                res["sweep"].setdefault(cs, {})[wk] = {"lams": lams, "miou": curve,
                                                       "entropy": ent, "n_active": nact}
                best = int(np.argmax(curve))
                print(f"  {cs} [{wk}] lam*={lams[best]:.2f} -> {curve[best]:.2f} "
                      f"(lam=0: {curve[0]:.2f}, a_bar={abars[wk]:.3f} -> "
                      f"{curve[int(round(abars[wk]/a.lam_step))]:.2f})", flush=True)

                # RULE G: lam = a_bar, no free parameter
                m, mc = score((cos - abars[wk] * off).argmax(-1).cpu().numpy())
                res["arms"].setdefault(f"ruleG_{wk}", {})[cs] = {"mIoU": m, "mAcc": mc}

                # RULE P: per-cell lam_j = <f_j, mu>, i.e. remove the mu component entirely
                aj = (unit @ mu.T)                                    # (P,1)
                m, mc = score((cos - aj * off[None, :]).argmax(-1).cpu().numpy())
                res["arms"].setdefault(f"ruleP_{wk}", {})[cs] = {"mIoU": m, "mAcc": mc}

                # RULE P shrunk: lam_j = beta*a_j -- does the per-cell rule want damping?
                for beta in (0.25, 0.5, 0.75):
                    m, mc = score((cos - beta * aj * off[None, :]).argmax(-1).cpu().numpy())
                    res["arms"].setdefault(f"ruleP{beta:g}_{wk}", {})[cs] = {"mIoU": m, "mAcc": mc}

            m, mc = score(cos.argmax(-1).cpu().numpy())
            res["arms"].setdefault("base", {})[cs] = {"mIoU": m, "mAcc": mc}
            del cos, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
