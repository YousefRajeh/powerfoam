"""Why does the simplex work? Isolate projection vs softmax vs confidence weighting.

BACKGROUND
----------
Corollary 2.1 of [[simplex-vs-sphere-extension]] claimed posterior mixing cannot produce a class
neither contributor supported, and that this was why posterior diffusion beats feature smoothing.
That corollary was RETRACTED 2026-08-29: it does not follow from Theorem 2 (the bound constrains
each coordinate, not the argmax), and measurement shows third classes DO occur on the simplex --
2.85% at s=50. Worse, the effect is far too small to matter (0.37% of all adjacent pairs) and it
points the wrong way: the third-class rate FALLS as posteriors sharpen, while the softness law says
sharper loses. So the real mechanism must PREFER soft posteriors, and closure does not.

THE ASYMMETRY TO EXPLAIN
------------------------
Simplex representations WIN for cross-cell aggregation (diffusion +1.22, region growing +1.11,
9/10 scenes) and LOSE for within-cell aggregation (posterior consensus -1.75, simplex geometric
median -3.99, single observation -6.93). Same representation, opposite signs -- so the benefit is
not a property of the simplex itself but of what is being combined.

THE ISOLATION
-------------
W^T is linear and the diffusion operator N acts on the CELL index, so they COMMUTE:

    W^T (N f) = N (W^T f)

Diffusing raw logits is therefore exactly diffusing features, absent renormalisation. That splits
the question cleanly:

  A  simplex     softmax(s*cos) -> diffuse -> argmax         19-d, nonlinear
  B  logit       cos -> diffuse -> argmax                    19-d, LINEAR  (== feature diffusion)
  C  feature     f -> diffuse (renormalise each step) -> cos 512-d, nonlinear only via renorm
  D  logitconf   cos -> diffuse with per-cell confidence weights -> argmax
  E  simplexflat softmax then STRIP confidence (renormalise each row to uniform peak) -> diffuse

Readings:
  B ~ A            the projection carries the benefit; the softmax is incidental
  B ~ C << A       the softmax is the mechanism
  D ~ A            specifically CONFIDENCE WEIGHTING; the simplex is just a convenient way to get it
  E << A           confirms it, by removing only the confidence information from the simplex

D uses the top1-top2 margin as the confidence, which is exactly the quantity a peaked posterior
encodes implicitly: an ambiguous cell has a flat posterior that barely moves an argmax, while a
confident one has a peaked posterior that moves it a lot. Feature and logit diffusion give every
cell equal pull regardless of how ambiguous it is.
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
from run_simplex_diffusion_eval import csr_to_edges, HARDEST_FIRST


def diffuse(x0, src, dst, deg, alpha, iters, node_w=None, renorm=None, chunk=None):
    """x <- (1-a) x0 + a * rowstochastic(S) x, optionally with per-node emission weights.

    node_w scales what each cell CONTRIBUTES to its neighbours (not what it keeps), which is the
    confidence-weighting hypothesis stated explicitly. renorm='l2' reproduces feature diffusion's
    projection back onto the sphere at every step.
    """
    P, D = x0.shape
    # chunk on ELEMENTS not edges: the gather is (chunk, D), so a fixed edge count that is fine at
    # D=19 allocates 16 GiB at D=512 and OOMs. Cap the temporary at ~1.5e8 floats (~600 MB).
    if chunk is None:
        chunk = max(65_536, int(1.5e8 // max(D, 1)))
    w = torch.ones(src.numel(), device=x0.device)
    if node_w is not None:
        w = w * node_w[dst]                       # weight by the SENDER's confidence
    rowsum = torch.zeros(P, device=x0.device).index_add_(0, src, w)
    w = w / rowsum.clamp_min(1e-30)[src]
    a = torch.full((P, 1), alpha, device=x0.device)
    a = torch.where((deg > 0).unsqueeze(1), a, torch.zeros_like(a))
    x = x0.clone()
    for _ in range(iters):
        acc = torch.zeros_like(x)
        for s in range(0, src.numel(), chunk):
            e = min(s + chunk, src.numel())
            acc.index_add_(0, src[s:e], x[dst[s:e]] * w[s:e, None])
        x = (1 - a) * x0 + a * acc
        if renorm == "l2":
            x = F.normalize(x, dim=-1)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0590_00,scene0645_00,scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--scales", default="50,200")
    p.add_argument("--alphas", default="0.9,0.95")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--outdir", default="artifacts/scannet/mechanism")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    scales = [float(x) for x in a.scales.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]

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

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        src, dst, _ = csr_to_edges(adj["adjacent"].to(device).long(),
                                   adj["offsets"].to(device).long(), P, device)
        keep = vm[src] & vm[dst]; src, dst = src[keep], dst[keep]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
            0, src, torch.ones_like(src))
        print(f"[{scene}] P={P:,} E={src.numel():,}", flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            cos = torch.zeros(P, len(names), device=device); cos[vm] = unit[vm] @ text.T

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            b = score(cos.argmax(-1).cpu().numpy(), "base")
            print(f"  {cs} [base] mIoU={b:.2f}", flush=True)

            for al in alphas:
                # C: FEATURE diffusion (512-d, renormalised each step) -- historically rejected
                xf = diffuse(unit, src, dst, deg, al, a.iters, renorm="l2")
                v = score((xf @ text.T).argmax(-1).cpu().numpy(), f"C_feature_a{al:g}")
                print(f"  {cs} [C_feature_a{al:g}] {v:.2f} ({v-b:+.2f})", flush=True); del xf

                # B: LOGIT diffusion (19-d, linear) -- equals feature diffusion without renorm
                xl = diffuse(cos, src, dst, deg, al, a.iters)
                v = score(xl.argmax(-1).cpu().numpy(), f"B_logit_a{al:g}")
                print(f"  {cs} [B_logit_a{al:g}] {v:.2f} ({v-b:+.2f})", flush=True); del xl

                for s in scales:
                    # A: SIMPLEX diffusion (the incumbent)
                    p0 = torch.softmax(s * cos, dim=-1); p0[~vm] = 0.0
                    xs = diffuse(p0, src, dst, deg, al, a.iters)
                    v = score(xs.argmax(-1).cpu().numpy(), f"A_simplex_s{s:g}_a{al:g}")
                    print(f"  {cs} [A_simplex_s{s:g}_a{al:g}] {v:.2f} ({v-b:+.2f})", flush=True)
                    del xs

                    # D: LOGIT diffusion with per-cell confidence emission weights
                    t2 = p0.topk(2, dim=-1).values
                    conf = (t2[:, 0] - t2[:, 1]).clamp_min(0)
                    conf = conf / conf.max().clamp_min(1e-12)
                    xd = diffuse(cos, src, dst, deg, al, a.iters, node_w=conf * vm.float())
                    v = score(xd.argmax(-1).cpu().numpy(), f"D_logitconf_s{s:g}_a{al:g}")
                    print(f"  {cs} [D_logitconf_s{s:g}_a{al:g}] {v:.2f} ({v-b:+.2f})", flush=True)
                    del xd

                    # E: SIMPLEX with confidence STRIPPED. Every cell gets the SAME distribution
                    # shape, mapped onto its own class RANKING -- so ordering is preserved and
                    # per-cell confidence is destroyed. (The previous version divided by the row max
                    # and renormalised, which is algebraically the identity since sum(p/max)=1/max;
                    # it returned p unchanged and matched arm A exactly, measuring nothing.)
                    K = p0.shape[1]
                    tmpl = torch.softmax(s * torch.linspace(1.0, -1.0, K, device=device), 0)
                    order = p0.argsort(dim=-1, descending=True)
                    pf = torch.zeros_like(p0)
                    pf.scatter_(1, order, tmpl.expand(p0.shape[0], -1))
                    pf[~vm] = 0.0
                    xe = diffuse(pf, src, dst, deg, al, a.iters)
                    v = score(xe.argmax(-1).cpu().numpy(), f"E_simplexflat_s{s:g}_a{al:g}")
                    print(f"  {cs} [E_simplexflat_s{s:g}_a{al:g}] {v:.2f} ({v-b:+.2f})", flush=True)
                    del xe, pf, p0
            del cos, text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
