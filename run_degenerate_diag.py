"""WHY do `kitchen counter` (2.3%), `refrigerator` (20.6%) and `kitchen cabinet` (23.2%) fail while
`microwave`, `shelf`, `ceiling` and `floor` sit at 95-99%?

THE DECOMPOSITION THAT MATTERS. Two very different things produce a 2% class:

  (i) CLIP genuinely cannot tell the class from its confuser. Then even a 2-way decision between
      just those two classes is near chance, and no lifting, decision rule or normalisation can help
      -- the information is absent.
  (ii) CLIP CAN tell them apart, but in the 100-way argmax the class loses to a HUB that beats it
      everywhere. Then the information is present and the failure is competitive, i.e. fixable.

These are indistinguishable in the 100-way accuracy we have been reporting, and everything closed so
far (13 hubness arms, macro-IoU, attribution, whitening, Fisher) was aimed at (ii) without ever
establishing that (ii) is what is happening. This script separates them:

  * `acc100`     -- accuracy in the real 100-way task
  * `acc_2way`   -- accuracy restricted to {true class, its top confuser}, cells of the true class
                    only. Chance is 50%.
  * `text_cos`   -- cosine between the two class TEXT embeddings, i.e. how close the words are
                    before any image evidence is involved.

A class with acc100 ~2% but acc_2way ~90% is a competition failure. A class with acc_2way ~50% is an
information failure. The mix decides whether there is anything left to win.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from run_macro_iou_gap import cell_histograms
from run_overnight import RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log
from run_spp_eval import benchmark_map, load_gt, coverage_filter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="f9f95681fd,c50d2d1d42,3864514494,578511c8a9")
    p.add_argument("--text-white", type=float, default=0.0,
                   help="alpha for text-prototype whitening; 0 = off (the original diagnosis)")
    p.add_argument("--out", default="artifacts/scannetpp/degenerate_diag.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    agg = {}
    for scene in a.scenes.split(","):
        art = f"artifacts/scannetpp/{scene}"
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"{art}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.exists(sp) and os.path.isdir(ck)):
            continue
        centers, radii = load_points_radii(ck)
        sv = torch.load(sp, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
             .reliability()["reliability"].to(device).float() * vm)
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
        nm = [top[:100][i] for i in pres]
        gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        C = len(nm)
        H, _ = cell_histograms(assigned, gt_t, len(centers), C)
        has_gt = (H.sum(1) > 0) & vmn
        cell_lab = H.argmax(1)

        txt = embed_class_names(nm, device)
        if a.text_white > 0:
            from run_text_and_pseudo_eval import text_whiten
            txt = text_whiten(txt, a.text_white)[0]
        cv = cen[vm] @ txt.T
        rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
        full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * rK[None, :]
        p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
        pred = diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy()

        scores = full.cpu().numpy()                     # post-CSLS class scores per cell
        tcos = (txt @ txt.T).cpu().numpy()
        m = np.flatnonzero(has_gt)
        for c in range(C):
            sel = m[cell_lab[m] == c]
            if sel.size < 200:
                continue
            pc = pred[sel]
            acc100 = float((pc == c).mean() * 100)
            others, cnt = np.unique(pc[pc != c], return_counts=True)
            conf = int(others[cnt.argmax()]) if others.size else c
            # 2-way: on the SAME cells, is the true class beaten by its confuser one-on-one?
            two = float((scores[sel, c] > scores[sel, conf]).mean() * 100)
            e = agg.setdefault(nm[c], {"n": 0, "acc100": [], "acc2": [], "conf": {},
                                       "tcos": [], "scenes": 0})
            e["n"] += int(sel.size); e["acc100"].append(acc100); e["acc2"].append(two)
            e["tcos"].append(float(tcos[c, conf]))
            e["conf"][nm[conf]] = e["conf"].get(nm[conf], 0) + int((pc == conf).sum())
            e["scenes"] += 1
        log(f"  {scene}: {C} classes, {len(m):,} labelled cells")
        del raw, cen, cv, full, txt, src, dst, deg, pos
        torch.cuda.empty_cache()

    rows = []
    for k, v in agg.items():
        if v["n"] < 500:
            continue
        conf = max(v["conf"].items(), key=lambda x: x[1])[0] if v["conf"] else "-"
        rows.append((float(np.mean(v["acc100"])), float(np.mean(v["acc2"])),
                     float(np.mean(v["tcos"])), v["n"], k, conf))
    rows.sort()
    print(f"\n{'class':<20}{'acc100':>8}{'acc2way':>9}{'txtcos':>8}{'cells':>9}  top confuser")
    for a100, a2, tc, n, k, conf in rows:
        print(f"{k:<20}{a100:>8.1f}{a2:>9.1f}{tc:>8.3f}{n:>9,}  {conf}")
    bad = [r for r in rows if r[0] < 40]
    if bad:
        print(f"\n  FAILING CLASSES (acc100 < 40): mean 2-way accuracy "
              f"{np.mean([r[1] for r in bad]):.1f}%  (chance = 50)")
        print("  >50 means the evidence IS present and the class loses a COMPETITION;")
        print("  ~50 means CLIP cannot separate it from its confuser at all.")
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in agg.items()},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
