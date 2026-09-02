"""Split NormLift's reliability into its two factors and use each where it belongs.

NormLift scores every primitive with R = ||f|| * N_eff/(N_eff+beta), and their own factorisation
(paper contribution 2) is ||f|| = c_intra * c_inter, where

    c_intra(j) = sum_v W^(v) ||f^(v)|| / sum_v W^(v)     within-view concentration
    c_inter(j) = || sum_v W^(v) f^(v) || / sum_v W^(v)||f^(v)||   cross-view agreement

Measured on ScanNet++, these two factors do COMPLETELY DIFFERENT jobs. Predicting whether a cell
straddles a GT semantic boundary:

    scene         straddling   AUC c_intra   AUC c_inter   AUC ||f||
    f9f95681fd       5.7%         0.722         0.434        0.702
    c50d2d1d42       4.5%         0.746         0.446        0.719
    0d2ee665be       3.2%         0.664         0.535        0.659

c_intra is a geometric signal (this cell spans a boundary); c_inter carries none (chance). Their
score MULTIPLIES them, diluting the geometric signal with an unrelated photometric one.

WHY THIS IS FOAM-SPECIFIC. For a Gaussian, low within-view concentration is confounded: a Gaussian
is spread AND overlaps its neighbours, so disagreement among the pixels it touches may be neighbour
contamination rather than a property of the primitive. A foam cell's footprint in a view is
EXCLUSIVE -- the partition is disjoint, no other cell claims those samples -- so low c_intra can
only mean the cell itself spans a semantic boundary. The separation is available only here.

TWO ARMS, both falsifiable against the current full stack (26.59 top100):
  A  diffusion edges gated by the SENDER's c_intra: a boundary-straddling cell should EMIT less,
     because what it emits is a blend of two classes.
  B  mode_vote_refine driven by c_inter*gate instead of R: the refinement asks "is this feature
     trustworthy", which is the photometric question, so the geometric factor does not belong in it.
  C  both.
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
from run_overnight import SPP, RECON, LAM, RANK_S, ALPHA, ITERS, log, csls, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-sizes", default="100,20")
    p.add_argument("--out", default="artifacts/scannetpp/intra_inter.json")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    res = {}
    for scene in SPP:
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
        st = AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
        rel = st.reliability()
        R = rel["reliability"].to(device).float() * vm
        c_intra = rel["c_intra"].to(device).float().clamp(0, 1)
        c_inter = rel["c_inter"].to(device).float().clamp(0, 1)
        n_eff = rel["n_eff"].to(device).float()
        gate = n_eff / (n_eff + 1.0)
        R_photo = (c_inter * gate) * vm            # arm B: photometric factor only
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        chunk = max(256, 200_000 // max(Dm, 1))
        cen_R = mode_vote_refine(cen, R, pos, ad0, of0, chunk=chunk)         # baseline refine
        cen_P = mode_vote_refine(cen, R_photo, pos, ad0, of0, chunk=chunk)   # arm B refine
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        # arm A: the SENDER's within-view concentration weights what it emits
        ew = c_intra[dst].clamp_min(1e-6)
        del adj, ad0, of0
        torch.cuda.empty_cache()

        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        lab = np.where(keepc, lab0, -1)
        row = {}
        for K in sizes:
            pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
            if not pres: continue
            nm = [top[:K][i] for i in pres]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt = embed_class_names(nm, device); C = len(nm)

            def stack(u, edge_w):
                c = torch.zeros(P, C, device=device); c[vm] = u[vm] @ txt.T
                cc = csls(c, vm)
                p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, ALPHA, ITERS, edge_w=edge_w)
                out = score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                                 gt_t, C, gt_pts.shape[0])[0]
                del c, cc, p0, x
                return out

            row[f"top{K}"] = {
                "base": stack(cen_R, None),                 # current full stack
                "A_intra_edges": stack(cen_R, ew),          # gate edges by sender c_intra
                "B_photo_refine": stack(cen_P, None),       # refine with c_inter*gate
                "C_both": stack(cen_P, ew),
            }
            del txt
            torch.cuda.empty_cache()
        res[scene] = row
        log(f"  {scene}: " + " | ".join(
            f"top{K} " + " ".join(f"{k}={v:.2f}" for k, v in row[f'top{K}'].items())
            for K in sizes if f"top{K}" in row))
        del raw, cen, cen_R, cen_P, src, dst, deg, pos, ew
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    for K in sizes:
        ks = [f"top{K}" in v for v in res.values()]
        if not any(ks): continue
        print(f"\n=== top{K} ({sum(ks)} scenes) ===")
        for arm in ("base", "A_intra_edges", "B_photo_refine", "C_both"):
            m = np.mean([v[f"top{K}"][arm] for v in res.values() if f"top{K}" in v])
            b = np.mean([v[f"top{K}"]["base"] for v in res.values() if f"top{K}" in v])
            print(f"  {arm:<16}{m:7.2f}  {m-b:+6.2f}")


if __name__ == "__main__":
    main()
