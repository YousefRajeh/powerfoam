"""tab:adjacency's mIoU column: what each neighbourhood construction is worth, on SFS features.

THE TABLE'S CLAIM. A bounded partition supplies *the* adjacency graph as the dual of the diagram.
Gaussians require a choice, and the seven published constructions span a 23x range in mean degree
(1.99 to 46.57). The degree column already shows the choice is consequential; the mIoU column shows
whether it *matters*.

WHY THIS ONE TABLE USES DIFFUSION, WHEN NOTHING ELSE IN THE PAPER DOES. Everywhere else the
classifier is plain per-cell argmax, deliberately. But under plain argmax nothing reads the graph,
so every row of this table would be identical and the column would be meaningless. The only way the
question "does the neighbourhood construction matter" has an answer is to run something that
consumes the neighbourhood. Diffusion is that something, and it is held FIXED across rows
(alpha=0.95, 100 iters, rank-encoded on the simplex) so the only thing varying is the graph.

That makes this table a statement about GRAPHS, not about the pipeline, and it should be read that
way: the argmax number is the paper's result, and these rows say how much a graph-consuming method
would gain or lose depending on a choice Gaussians have to make and a foam does not.

ON TOP OF SFS. Features are Splat Feature Solver's lifted field on the Gaussian arm -- the baseline
everyone shares -- so a difference between rows cannot come from our lifting. PowerFoam's row uses
its own shared-facet dual, which is not a choice: it is the dual of the power diagram, and it is the
whole point of the comparison.
"""
import argparse
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
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from graph_variants import BUILDERS
from run_percell_masked import OPACITY_THRESH, SPLIT, primitive_alpha

ART = "artifacts/scannet"
# held fixed across every row -- the graph is the only thing that varies
ALPHA, ITERS, RANK_S = 0.95, 100, 200.0
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def rank_encode(scores, s=RANK_S):
    K = scores.shape[1]
    tmpl = torch.softmax(s * torch.linspace(1.0, -1.0, K, device=scores.device), 0)
    order = scores.argsort(dim=-1, descending=True)
    return torch.zeros_like(scores).scatter_(1, order, tmpl.expand(scores.shape[0], -1))


def diffuse(p0, src, dst, n, alpha=ALPHA, iters=ITERS):
    deg = torch.zeros(n, device=p0.device).index_add_(
        0, src, torch.ones(src.numel(), device=p0.device))
    w = 1.0 / deg.clamp_min(1.0)[src]
    a = torch.where(deg > 0, torch.full((n,), alpha, device=p0.device),
                    torch.zeros(n, device=p0.device))[:, None]
    p = p0.clone()
    for _ in range(iters):
        acc = torch.zeros_like(p).index_add_(0, src, p[dst] * w[:, None])
        p = (1 - a) * p0 + a * acc
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default="scene0347_00,scene0000_00,scene0062_00")
    ap.add_argument("--recon", default="gs_unfroz")
    ap.add_argument("--graphs", default=",".join(BUILDERS))
    ap.add_argument("--graph-k", type=int, default=30)
    ap.add_argument("--out", default="artifacts/scannet/adjacency_miou.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        fp = f"{ART}/{scene}/solved_weighted_{a.recon}_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_{a.recon}_assign.npy"
        ck = f"recon_remote/{a.recon}/{scene}/ckpt.pt"
        if not all(os.path.exists(p) for p in (fp, apth, ck)):
            print(f"  [miss] {scene}", flush=True)
            continue
        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        vm = d["valid_mask"].to(dev)
        P = feats.shape[0]
        unit = torch.zeros_like(feats)
        unit[vm] = F.normalize(feats[vm], dim=-1)

        c = torch.load(ck, map_location="cpu", weights_only=False)
        s = c["splats"] if isinstance(c, dict) and "splats" in c else c
        pos = s["means"].float().to(dev)
        sc = torch.exp(s["scales"].float()).to(dev)
        qt = s["quats"].float().to(dev)
        alpha_g = primitive_alpha(a.recon, scene, 2.0)

        assigned = np.load(apth)
        owned = assigned >= 0
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())
        low = np.zeros(len(gt_pts), dtype=bool)
        low[owned] = alpha_g[assigned[owned]] < OPACITY_THRESH
        vmn = vm.cpu().numpy()

        for gname in a.graphs.split(","):
            if gname not in BUILDERS:
                continue
            try:
                src, dst, _ = BUILDERS[gname](pos=pos, vm=vm, feat=unit, scales=sc, quats=qt,
                                              K=a.graph_k, device=dev)
            except Exception as exc:                       # noqa: BLE001
                print(f"  [{scene}/{gname}] builder failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            keep = vm[src] & vm[dst]
            src, dst = src[keep], dst[keep]
            deg = torch.zeros(P, device=dev).index_add_(
                0, src, torch.ones(src.numel(), device=dev))
            rec = {"scene": scene, "recon": a.recon, "graph": gname,
                   "mean_degree": float(deg[vm].mean()), "n_edges": int(src.numel())}
            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                gt[low] = 0
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                cv = torch.zeros(P, len(names), device=dev)
                cv[vm] = unit[vm] @ text.T
                p0 = rank_encode(cv)
                p0[~vm] = 0.0
                lab = diffuse(p0, src, dst, P).argmax(-1).cpu().numpy() + 1
                live = vmn
                shown = owned.copy()
                shown[owned] = live[assigned[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[shown] = lab[assigned[shown]]
                _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                                     torch.from_numpy(pred).long(), nc)
                rec[cs] = {"mIoU": float(miou) * 100, "mAcc": float(macc) * 100}
            rows.append(rec)
            print(f"  {scene} {gname:10s} deg={rec['mean_degree']:6.2f}  "
                  f"19cls mIoU={rec['opengaussian19']['mIoU']:6.2f}", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1, default=float)

    if rows:
        print("\n=== mean over scenes, per construction ===")
        for g in dict.fromkeys(r["graph"] for r in rows):
            sel = [r for r in rows if r["graph"] == g]
            print(f"  {g:10s} deg={np.mean([r['mean_degree'] for r in sel]):6.2f}  "
                  f"19cls mIoU={np.mean([r['opengaussian19']['mIoU'] for r in sel]):6.2f}  "
                  f"(n={len(sel)})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
