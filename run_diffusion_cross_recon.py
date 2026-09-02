"""Posterior simplex diffusion across ALL SIX reconstructions — is the gain foam-specific?

THE QUESTION THIS SETTLES. Posterior diffusion on the true facet graph is worth +1.07/+1.19/+0.93
mIoU on PowerFoam, on top of NormLift-style mode-voting, 9/10 scenes. That is our best result. But
the claim we want to make is that it works BECAUSE the graph is the exact dual of a disjoint bounded
partition. If Gaussians gain the same amount on a Delaunay-of-means graph, the mechanism is generic
and the foam story is dead.

PREDICTIONS, stated before running (this is the falsifier):
  pf_*        largest gain — graph is the exact facet dual of a disjoint partition
  rf_unfroz   smaller — 37% vacuum cells own territory but carry no feature, so they relay
              nothing and fragment the graph's connectivity
  gs_*        least — "adjacency" of Gaussian MEANS is neither a partition boundary nor an
              overlap relation. Measured: at gsplat's own 3-sigma bound the alpha complex over
              Gaussian means has mean degree 0.05, i.e. splats do not even reach their Delaunay
              neighbours, while every scene point lies inside ~14-20 splats simultaneously.
If gs gains match pf gains, the representation claim is refuted and diffusion is simply a good
generic post-process. Report that outcome plainly if it happens.

The diffusion itself: p0 = softmax(s * cos(f, text)); p <- (1-a) p0 + a S p; argmax once at the end.
S is row-stochastic on the cached Delaunay graph. The simplex is closed under convex combination, so
no mixture can point at a third class -- this is why it is safe where smoothing CLIP FEATURES is not
(that was tried and correctly rejected: NormLift's own ablation has linear feature averaging at -0.6).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SCENES = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
          "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
          "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
          "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

FEATURES = {
    "pf_nonfroz": "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt",
    "pf_tfroz":   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt",
    "rf_froz":    "artifacts/scannet/{s}/solved_geometric_median_rf_froz_ogl3.pt",
    "rf_unfroz":  "artifacts/scannet/{s}/solved_geometric_median_rf_unfroz_ogl3.pt",
    "gs_froz":    "artifacts/scannet/{s}/solved_weighted_gs_froz_ogl3.pt",
    "gs_unfroz":  "artifacts/scannet/{s}/solved_weighted_gs_unfroz_ogl3.pt",
}


def load_graph(scene, recon, device):
    p = f"artifacts/ablation_cache/{scene}_{recon}_delaunay.pt"
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["adjacent"].to(device).long(), d["offsets"].to(device).long()


def diffuse(p0, adjacent, offsets, n, alpha, iters, device):
    """p <- (1-a) p0 + a * S p, S row-stochastic over the cached CSR."""
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    src = torch.repeat_interleave(torch.arange(n, device=device),
                                  (offsets[1:] - offsets[:-1]))
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p)
        agg.index_add_(0, src, p[adjacent])
        agg /= deg[:, None]
        p = (1 - alpha) * p0 + alpha * agg
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(FEATURES))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--scale", type=float, default=1000.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="artifacts/scannet/diffusion_cross_recon.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"

    recons = [r for r in a.recons.split(",") if r]
    scenes = [s for s in a.scenes.split(",") if s in SCENES]
    out = {}

    for recon in recons:
        for scene in scenes:
            fp = FEATURES[recon].format(s=scene)
            ap_ = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            if not (os.path.exists(fp) and os.path.exists(ap_)):
                print(f"[skip] {recon}/{scene}: missing artifact", flush=True)
                continue
            g = load_graph(scene, recon, dev)
            if g is None:
                print(f"[skip] {recon}/{scene}: no delaunay graph", flush=True)
                continue
            adjacent, offsets = g

            gt_pts, raw, names_all = load_scannet_pointcept_gt(
                rf"D:\Downloads\scannet_pointcept\{SCENES[scene]}\{scene}", "segment20")
            n2i = {n: i for i, n in enumerate(names_all)}
            present = set(np.unique(raw).tolist())

            d = torch.load(fp, map_location=dev, weights_only=True)
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].cpu().numpy()
            assigned = np.load(ap_)
            n_prim = feats.shape[0]
            if len(offsets) - 1 != n_prim:
                print(f"[skip] {recon}/{scene}: graph {len(offsets)-1} vs feats {n_prim}", flush=True)
                continue
            unit = F.normalize(feats, dim=-1)
            owned = assigned >= 0
            pv = torch.from_numpy(valid).to(dev)

            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                sim = unit @ text.T                                  # (P, C)
                p0 = torch.softmax(a.scale * sim, dim=-1)
                p0[~pv] = 0.0                                        # featureless cells relay nothing
                pd = diffuse(p0, adjacent, offsets, n_prim, a.alpha, a.iters, dev)

                res = {}
                for tag, prob in (("base", p0), ("diffused", pd)):
                    cls = prob.argmax(dim=-1).cpu().numpy()
                    scorable = owned.copy()
                    scorable[owned] = valid[assigned[owned]] if tag == "base" else \
                        (prob.sum(-1) > 0).cpu().numpy()[assigned[owned]]
                    pred = np.zeros(len(gt), dtype=np.int64)
                    pred[scorable] = cls[assigned[scorable]] + 1
                    _, miou, acc, macc = calculate_metrics(
                        torch.from_numpy(gt).long(), torch.from_numpy(pred).long(), nc)
                    res[tag] = {"mIoU": float(miou), "mAcc": float(macc)}
                out.setdefault(f"{recon}|{cs}", {})[scene] = res
                print(f"  {recon:<12}{scene} {cs[11:]:>3}: "
                      f"base {res['base']['mIoU']*100:6.2f} -> diff {res['diffused']['mIoU']*100:6.2f} "
                      f"({(res['diffused']['mIoU']-res['base']['mIoU'])*100:+.2f})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print(f"\n{'recon':<14}{'cls':>4}{'base':>9}{'diffused':>10}{'delta':>8}{'n':>4}")
    for k, per in sorted(out.items()):
        r, cs = k.split("|")
        b = np.mean([v["base"]["mIoU"] for v in per.values()]) * 100
        f_ = np.mean([v["diffused"]["mIoU"] for v in per.values()]) * 100
        print(f"{r:<14}{cs[11:]:>4}{b:>9.2f}{f_:>10.2f}{f_-b:>+8.2f}{len(per):>4}")


if __name__ == "__main__":
    main()
