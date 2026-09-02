"""Audit the last two unverified knobs in the shipping stack: `alpha` and `iters`.

`s` (rank-encode) and CSLS `k` are inert, `lam` is leave-one-out cross-validated at <=0.38 leakage,
and the graph choice predates this work. That leaves the diffusion's retention `alpha`, selected on
the evaluation scenes, and `iters`, which was inherited from the earlier stack and never questioned
at all. Both are swept here on all 10 scenes and all three class sets, and `alpha` is then chosen by
leave-one-out so its leakage cost is measured rather than assumed.

Stack under audit (the plug-in was dropped as redundant, see
[[Prior-correction-derived-2026-08-29]] section 6):

    centre (lam=0.3) -> mode-vote prerefine -> CSLS -> rank-encode -> diffuse(alpha, iters) -> argmax
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
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--alphas", default="0.80,0.85,0.90,0.95,0.99")
    p.add_argument("--iters", default="5,10,30,100")
    p.add_argument("--rank-s", type=float, default=200.0)
    p.add_argument("--csls-k", type=int, default=1000)
    p.add_argument("--center-lam", type=float, default=0.3)
    p.add_argument("--outdir", default="artifacts/scannet/alpha_iters")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    alphas = [float(x) for x in a.alphas.split(",")]
    iterl = [int(x) for x in a.iters.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
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
        positions = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)
        unit[vm] = F.normalize(unit[vm] - a.center_lam * mu, dim=-1)

        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0 = adj["adjacent"].to(device).long(); of0 = adj["offsets"].to(device).long()
        sp = next(c for c in [f"{art}/train_stats_sam_{a.variant}{a.suffix}.pt",
                              f"{art}/stats_{a.variant}{a.suffix}.pt"] if os.path.exists(c))
        Rr = AccumulatedFeatureStats.load(sp).reliability()["reliability"].to(device).float() * vm
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        from run_normlift_refine_eval import mode_vote_refine
        unit = mode_vote_refine(unit, Rr, positions, ad0, of0,
                                chunk=max(256, 200_000 // max(Dm, 1)))
        del Rr
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        keep = vm[src] & vm[dst]; src, dst = src[keep], dst[keep]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
            0, src, torch.ones_like(src))
        del adj, ad0, of0
        torch.cuda.empty_cache()

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{SCENES[scene]}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "arms": {}}
        print(f"[{scene}] P={P:,} E={src.numel():,}", flush=True)

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            C = len(names)
            cos = torch.zeros(P, C, device=device); cos[vm] = unit[vm] @ text.T
            kk = min(a.csls_k, int(vm.sum()))
            r_t = cos[vm].topk(kk, dim=0).values.mean(0)
            cos[vm] = cos[vm] - 0.5 * r_t[None, :]              # CSLS
            p0 = rank_encode(cos, a.rank_s, device); p0[~vm] = 0.0

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}

            score(p0.argmax(-1).cpu().numpy(), "no_diffusion")
            for al in alphas:
                for it in iterl:
                    x = diffuse(p0, src, dst, deg, al, it)
                    score(x.argmax(-1).cpu().numpy(), f"a{al:g}_i{it}")
                    del x
            print(f"  {cs} done", flush=True)
            del cos, p0, text
            torch.cuda.empty_cache()

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
