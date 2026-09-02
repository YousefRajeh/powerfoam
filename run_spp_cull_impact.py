"""Does the opacity cull recover ScanNet++ mIoU the way it does on ScanNet?

MOTIVATION, measured not assumed. On ScanNet, NormLift's/OpenGaussian's `sigmoid(opacity)<0.1` cull
is worth **+4.84 / +5.11 / +5.17 mIoU** (19/15/10cls) on gs_froz -- isolated by running their exact
protocol with cull on and off. Identity-vs-Mahalanobis assignment contributed exactly 0.00, so the
whole protocol gap was the cull. ScanNet++ applies no cull on any arm, so its numbers carry the
full weight of GT points whose owning primitive is essentially transparent.

FOAM ANALOGUE. Foam has no opacity attribute -- only unbounded density sigma (softplus), units 1/m.
The renderer's alpha for a cell is `1 - exp(-sigma*l)` with `l` the distance the ray travels through
it, so the cell's characteristic alpha uses its MEAN CHORD, the expected traversal length of a random
ray through a convex body: `l = 4V/S`, which for the effective sphere of the cell's own volume is
`(4/3)(3V/4pi)^(1/3)`. Cull when that alpha < 0.1, i.e. the same physical question OpenGaussian asks.

Volume is the right invariant here. Earlier attempts used a scene-global spacing and then a
nearest-neighbour distance; both are spacings BETWEEN centres rather than traversal lengths, and both
made the arms incomparable (RadFoam's 4xSfM packs many more, much smaller cells into the same space,
so it kept only 29-40% against PowerFoam's 79-95%). A cell that occupies little volume cannot absorb
much light no matter how its neighbours are arranged.

Both ways are always reported: culling REMOVES GT points from the metric, so it can only be read
against the unculled number.
"""
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
from run_lambda_derivation_eval import mc_cell_volumes
from run_overnight import SPP, RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, csls, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_cull_eval import mean_chord_length, SIGMA_NUM


def foam_alpha(ckpt_dir, centers, valid, n_mc=1_000_000):
    """Characteristic alpha per cell: 1 - exp(-sigma * mean_chord)."""
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=False)
    raw = sd["density"].float().reshape(-1)
    sigma = torch.nn.functional.softplus(raw, beta=10).numpy().astype(np.float64)
    vol_counts = mc_cell_volumes(centers, np.zeros(len(centers)) if centers is None else
                                 load_points_radii(ckpt_dir)[1], valid, n_mc, "cpu").numpy()
    lo = np.asarray(centers).min(0); hi = np.asarray(centers).max(0)
    bbox_vol = float(np.prod(hi - lo))
    vol = vol_counts / max(vol_counts.sum(), 1.0) * bbox_vol      # counts -> m^3
    ell = mean_chord_length(vol)
    return 1.0 - np.exp(-sigma * ell), sigma, ell, vol


def main():
    enable_determinism()
    device = "cuda"
    top, raw2bench = benchmark_map()
    out = {}
    print(f"{'scene':<13}{'kept':>7}{'medA':>8}{'base':>8}{'baseC':>8}{'stack':>8}{'stackC':>8}")
    for scene in SPP:
        art = f"artifacts/scannetpp/{scene}"
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        solved = f"{art}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not os.path.exists(solved):
            continue
        centers, radii = load_points_radii(ck)
        sv = torch.load(solved, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        alpha, sigma, ell, vol = foam_alpha(ck, centers, vmn)
        keep_prim = alpha >= 0.1

        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
             .reliability()["reliability"].to(device).float() * vm)
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        cen_r = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        gt_pts, gt_lab0, _ = load_gt(scene, top, raw2bench)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
        gt_lab = np.where(keepc, gt_lab0, -1)
        # the cull: a GT point whose owning cell is effectively transparent leaves the metric,
        # exactly as `gt_tensor[low_opacity] = 0` does in their evaluator
        gt_lab_c = np.where(keep_prim[np.where(owned, assigned, 0)], gt_lab, -1)

        row = {}
        present = sorted(set(np.unique(gt_lab).tolist()) & set(range(100)))
        nm = [top[:100][i] for i in present]
        txt = embed_class_names(nm, device); C = len(nm)
        cosr = torch.zeros(P, C, device=device); cosr[vm] = raw[vm] @ txt.T
        cosc = torch.zeros(P, C, device=device); cosc[vm] = cen_r[vm] @ txt.T
        cc = csls(cosc, vm)
        p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
        stack = diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy()
        base = cosr.argmax(-1).cpu().numpy()
        for tag, lab in (("", gt_lab), ("C", gt_lab_c)):
            pres2 = sorted(set(np.unique(lab).tolist()) & set(range(100)))
            if not pres2: continue
            nm2 = [top[:100][i] for i in pres2]
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres2)).long()
            t2 = embed_class_names(nm2, device); C2 = len(nm2)
            cr = torch.zeros(P, C2, device=device); cr[vm] = raw[vm] @ t2.T
            cs2 = torch.zeros(P, C2, device=device); cs2[vm] = cen_r[vm] @ t2.T
            cc2 = csls(cs2, vm); p2 = rank_encode(cc2, RANK_S, device); p2[~vm] = 0.0
            st2 = diffuse(p2, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy()
            row["base" + tag] = score_pred(cr.argmax(-1).cpu().numpy(), assigned, owned,
                                           gt_t, C2, gt_pts.shape[0])[0]
            row["stack" + tag] = score_pred(st2, assigned, owned, gt_t, C2, gt_pts.shape[0])[0]
            del cr, cs2, cc2, p2, t2
        row["kept"] = float(keep_prim[vmn].mean()); row["med_alpha"] = float(np.median(alpha[vmn]))
        out[scene] = row
        print(f"{scene:<13}{row['kept']:7.3f}{row['med_alpha']:8.3f}"
              f"{row.get('base',0):8.2f}{row.get('baseC',0):8.2f}"
              f"{row.get('stack',0):8.2f}{row.get('stackC',0):8.2f}", flush=True)
        del raw, cen, cen_r, src, dst, deg, pos, cosr, cosc, cc, p0, txt
        torch.cuda.empty_cache()
    if out:
        k = lambda f: np.mean([v[f] for v in out.values() if f in v])
        print(f"\nMEAN   kept={k('kept'):.3f}  base {k('base'):.2f} -> culled {k('baseC'):.2f} "
              f"({k('baseC')-k('base'):+.2f}) | stack {k('stack'):.2f} -> culled {k('stackC'):.2f} "
              f"({k('stackC')-k('stack'):+.2f})")
    json.dump(out, open("artifacts/scannetpp/cull_impact.json", "w"), indent=1)


if __name__ == "__main__":
    main()
