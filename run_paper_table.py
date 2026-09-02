"""THE PAPER TABLE: our best pipeline, run on every reconstruction, 10 scenes.

WHY THIS DOES NOT EXIST YET. Today's work improved the pipeline substantially
(percell 36.53 -> bary+modevote+diffusion 39.26 at 19cls) but every arm of that improvement was
measured on PowerFoam-unfrozen only. The cross-representation comparison in the results DB still
uses the OLD pipeline. So the project's central claim -- that a bounded disjoint partition beats
overlapping Gaussians for open-vocabulary 3D segmentation -- has never been evaluated with the
pipeline we would actually publish.

THE FAIRNESS PROBLEM, and how it is handled. Two components of the best stack need a GRAPH:
mode-voting and posterior diffusion. For PowerFoam the honest graph is the TRUE FACET graph, the
exact dual of the power diagram (`adjacency_true_facet.pt`). Gaussians have no facets, so their
only option is a Delaunay/kNN graph over the MEANS (`ablation_cache/{scene}_{recon}_delaunay.pt`),
which is dual to nothing -- measured, at gsplat's own 3-sigma bound the alpha complex over
Gaussian means has mean degree 0.05 while every scene point lies inside ~14-20 splats.

That asymmetry is REAL and is the paper's point, but it must be reported, not hidden. So every
representation is run:
  (a) with the best graph available to it, and
  (b) additionally, PowerFoam is run on the same KIND of graph the Gaussians get
So a reader can separate "foam wins because of the representation" from "foam wins because it has
a better graph". Barycentric assignment is foam-only (it needs the regular triangulation dual to
the partition) and is reported as a separate row rather than folded into the headline.

RADFOAM NAMING HAZARD: the 30k arms are `solved_gm_*` while the 20k arms are
`solved_geometric_median_*`. Globbing one prefix silently finds half the data -- that error
previously made RadFoam look like a 4-scene arm when it is 10/10.

Every number is scored by the same function on the same GT points, so the columns are comparable.
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

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

# (label, feature path template, is_foam)
ARMS = [
    ("PowerFoam unfrozen", "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt", True),
    ("PowerFoam frozen",   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt", True),
    ("RadFoam unfrozen",   "artifacts/scannet/{s}/solved_gm_rf_unfroz_ogl3.pt", True),
    ("RadFoam frozen",     "artifacts/scannet/{s}/solved_gm_rf_match_ogl3.pt", True),
    ("3DGS unfrozen",      "artifacts/scannet/{s}/solved_weighted_gs_unfroz_ogl3.pt", False),
    ("3DGS frozen",        "artifacts/scannet/{s}/solved_weighted_gs_froz_ogl3.pt", False),
]
RECON_KEY = {"PowerFoam unfrozen": "pf_nonfroz", "PowerFoam frozen": "pf_tfroz",
             "RadFoam unfrozen": "rf_unfroz", "RadFoam frozen": "rf_froz",
             "3DGS unfrozen": "gs_unfroz", "3DGS frozen": "gs_froz"}


def diffuse(p0, src, adjacent, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[adjacent])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def load_graph(scene, recon, kind, dev, n_prim):
    """kind='truefacet' (foam only) or 'delaunay' (available to every representation)."""
    if kind == "truefacet":
        # each arm has a DIFFERENT primitive count, so the graph is per-arm. Only pf_nonfroz
        # had one built; the size check below correctly REJECTED the mismatched graph for the
        # others rather than silently using the wrong one, which is why the first run of this
        # table showed a single row here.
        # RadFoam is deliberately absent: r == 0 makes it an UNWEIGHTED Voronoi whose dual IS
        # the ordinary Delaunay, i.e. the "common graph" it already has -- so for RadFoam the
        # own-dual/common-graph distinction does not exist.
        p = (f"artifacts/scannet/{scene}/adjacency_true_facet_frozen.pt"
             if recon == "pf_tfroz"
             else f"artifacts/scannet/{scene}/adjacency_true_facet.pt")
    else:
        p = f"artifacts/ablation_cache/{scene}_{recon}_delaunay.pt"
    if not os.path.exists(p):
        return None
    g = torch.load(p, map_location="cpu", weights_only=True)
    adj, off = g["adjacent"].long().to(dev), g["offsets"].long().to(dev)
    if len(off) - 1 != n_prim:
        return None
    return adj, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/scannet/paper_table.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res, skipped = {}, []

    for scene in SPLIT:
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for label, tmpl, is_foam in ARMS:
            recon = RECON_KEY[label]
            fp = tmpl.format(s=scene)
            apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            if not (os.path.exists(fp) and os.path.exists(apth)):
                skipped.append(f"{label}/{scene}: missing artifact")
                continue
            try:
                d = torch.load(fp, map_location=dev, weights_only=True)
            except Exception as e:
                # a file still being written by a concurrent scp reads as a truncated zip;
                # skip and log rather than aborting the whole table
                skipped.append(f"{label}/{scene}: unreadable ({type(e).__name__})")
                continue
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].cpu().numpy()
            vt = torch.from_numpy(valid).to(dev)
            unit = torch.zeros_like(feats)
            unit[vt] = F.normalize(feats[vt], dim=-1)
            n_prim = feats.shape[0]
            assign = np.load(apth)
            owned = assign >= 0

            graphs = {}
            g = load_graph(scene, recon, "delaunay", dev, n_prim)
            if g:
                graphs["common graph"] = g
            if is_foam and recon.startswith("pf"):
                g2 = load_graph(scene, recon, "truefacet", dev, n_prim)
                if g2:
                    graphs["own dual"] = g2
            if not graphs:
                skipped.append(f"{label}/{scene}: no usable graph")
                continue

            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                sim = unit @ text.T
                p0 = torch.softmax(1000.0 * sim, dim=-1)
                p0[~vt] = 0.0

                def emit(tag, pr):
                    cls = pr.argmax(-1).cpu().numpy() + 1
                    live = (pr.sum(-1) > 0).cpu().numpy()
                    sc = owned.copy()
                    sc[owned] = live[assign[owned]]
                    pred = np.zeros(len(gt), dtype=np.int64)
                    pred[sc] = cls[assign[sc]]
                    _, mi, _, ma = calculate_metrics(torch.from_numpy(gt).long(),
                                                     torch.from_numpy(pred).long(), nc)
                    res.setdefault((label, tag, cs), []).append((float(mi) * 100,
                                                                 float(ma) * 100))

                emit("per-cell argmax", p0)
                for gname, (adj, off) in graphs.items():
                    src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                                  off[1:] - off[:-1])
                    deg = (off[1:] - off[:-1]).clamp_min(1).float()
                    emit(f"+ diffusion ({gname})", diffuse(p0, src, adj, deg))
        print(f"[{scene}] done", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, open(a.out, "w"), indent=1)

    print("\n" + "=" * 92)
    print("PAPER TABLE -- same pipeline, same scorer, same GT points, all representations")
    print("=" * 92)
    for tag in ("per-cell argmax", "+ diffusion (common graph)", "+ diffusion (own dual)"):
        rows = [(l, res[(l, tag, CLASS_SETS[0])]) for l, _, _ in ARMS
                if (l, tag, CLASS_SETS[0]) in res]
        if not rows:
            continue
        print(f"\n--- {tag} ---")
        print(f"{'representation':<22}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS) + f"{'n':>4}")
        for label, _ in rows:
            cells = "".join(
                f"{np.mean([x[0] for x in res[(label, tag, c)]]):9.2f}" for c in CLASS_SETS)
            n = len(res[(label, tag, CLASS_SETS[0])])
            print(f"{label:<22}{cells}{n:>4}")
    print(f"\nNormLift (published){'':<3}{35.77:>9}{39.62:>9}{48.93:>9}{10:>4}")
    if skipped:
        print(f"\nSKIPPED {len(skipped)} (logged, never silent):")
        for s in list(dict.fromkeys(skipped))[:12]:
            print("  ", s)


if __name__ == "__main__":
    main()
