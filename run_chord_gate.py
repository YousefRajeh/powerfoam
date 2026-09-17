"""GATE for the semantic-triangulation idea: is there anything there to triangulate?

THE IDEA BEING GATED. A power cell is a bounded convex polytope, so every ray through it has a
definite chord. For a cell whose views split into two semantic clusters, the chords say WHY:
  * clusters occupy different sub-regions -> the cell straddles a boundary -> SPLIT it, and the
    separating plane estimates where the boundary lies inside the cell;
  * chords interleave -> some view's mask is simply wrong -> robust accumulation, do not split.
`B_j` alone cannot tell these apart. Gaussians cannot do this at all: unbounded support means no
entry, no exit, no chord.

WHY GATE FIRST. Two things kill it, and both are cheap to check:
  1. FREQUENCY. If genuinely bimodal cells are 2% of the scene, the ceiling on any fix is 2%.
  2. BASELINE. Triangulation needs parallax. If a cell's views all look from nearly the same
     direction, its chords are near-parallel and nothing can be localised ALONG the ray. With
     cameras on a scanning trajectory this is a live risk, not a theoretical one.

WHAT THIS COMPUTES. Two passes over the operator, then free geometry:
  pass 1  M_j = sum_v M_jv, per-view weights w_jv = ||M_jv||   (also reproduces W, B)
  pass 2  c_jv = <u_jv, u_bar_j>, the alignment of each view's direction to the consensus
Regimes are then read off the weight-fraction of low-c views, with no third pass:
  (i)   f_low ~ 0                 unanimous
  (ii)  0 < f_low < BIMODAL_F     majority + outliers   -> robust estimator should win here
  (iii) f_low >= BIMODAL_F        two comparable clusters -> the triangulation candidates
Baseline uses the camera-to-cell-centre direction rather than per-ray directions: a cell is small
relative to its distance from the camera, so the rays through it are near-parallel and the centre
direction is an accurate stand-in. It costs nothing and needs no operator.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401

import configargparse
import numpy as np
import torch

OUT = "artifacts/scannet/chord_gate"
BIMODAL_F = 0.25          # weight share of dissenting views above which we call it two clusters
TAU = 0.5                 # cos below this counts as dissenting from the consensus direction


def load_view_features(feat_dir, stem, H, W):
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[-1] if s.shape[0] <= 4 else s[..., -1]
    t = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(f)).float(), dim=-1)
    seg = torch.from_numpy(np.ascontiguousarray(s)).long()
    if seg.shape != (H, W):
        seg = torch.nn.functional.interpolate(
            seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    from configs import Params, add_group
    from data_loader import DataHandler
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    cfg = f"output/scannet_{a.scene}_truefrozen/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
    stems = sorted(os.path.splitext(f)[0][:-2] for f in os.listdir(feat_dir) if f.endswith("_f.npy"))

    wp.init()
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"output/scannet_{a.scene}_truefrozen/model.pt")
    centres = model.points.detach().float().to(dev)
    P, V = int(centres.shape[0]), len(sel)

    def view_resultant(vi):
        """M_jv for one view, plus its support S_jv."""
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                       max_intersections=4096)
        ri = op.row_indices.to(torch.int64)
        ci = op.col_indices.to(torch.int64)
        vv = op.values.float()
        del op
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W)
        seg, tab = seg.to(dev), tab.to(dev)
        M = torch.zeros((P, tab.shape[1]), device=dev)
        S = torch.zeros(P, device=dev)
        CH = 4_000_000
        for s0 in range(0, vv.numel(), CH):
            e0 = min(s0 + CH, vv.numel())
            b = tab[seg[ri[s0:e0]].clamp(0, tab.shape[0] - 1)]
            M.index_add_(0, ci[s0:e0], vv[s0:e0, None] * b)
        S.index_add_(0, ci, vv)
        del ri, ci, vv, seg, tab
        return M, S

    # ---- pass 1: consensus direction and per-view weights -----------------------------------
    M_tot, S_tot, D = None, torch.zeros(P, device=dev), None
    w = torch.zeros((P, V), device=dev)
    for n, vi in enumerate(sel):
        M, S = view_resultant(vi)
        if M_tot is None:
            D = M.shape[1]
            M_tot = torch.zeros((P, D), device=dev)
        M_tot += M
        S_tot += S
        w[:, n] = M.norm(dim=1)
        del M, S
    live = S_tot > 1e-8
    u_bar = torch.nn.functional.normalize(M_tot, dim=1)

    # ---- pass 2: per-view alignment AND per-view predicted class -----------------------------
    # A cosine threshold cannot express "these two views disagree semantically". CLIP puts indoor
    # classes in a very tight cone -- for the 19 ScanNet class names the TEXT embeddings never fall
    # below cosine 0.53 and sit at median 0.67, and image features are tighter still. So an absolute
    # tau of 0.5 or 0.7 is below anything the label geometry can distinguish, and reports "no
    # disagreement" whatever the data does. The semantically exact test needs no threshold: classify
    # each view's own direction and ask whether the ARGMAX CLASS differs.
    from evaluate_point_cloud_miou import embed_class_names, OPENGAUSSIAN_CLASS_SETS, \
        SCANNET20_CLASS_NAMES
    ids = OPENGAUSSIAN_CLASS_SETS["opengaussian19"]
    names = [SCANNET20_CLASS_NAMES[i] for i in ids] if isinstance(ids[0], int) else list(ids)
    T = torch.nn.functional.normalize(embed_class_names(names, dev).float(), dim=-1)  # (C,D)

    c = torch.zeros((P, V), device=dev)
    lab = torch.full((P, V), -1, dtype=torch.int16, device=dev)
    # WITHIN-VIEW LABEL HOMOGENEITY. W ~ 1 says a cell's rays return identical FEATURES in a view,
    # but a cell spanning two SAM segments with similar features would hide inside that. The direct
    # test is whether the cell's own rays, in a SINGLE view, classify to more than one label. If they
    # never do, cells never straddle a boundary within a view and sub-cell subdivision has nothing to
    # resolve. If some do, those are the real split candidates.
    n_pairs = torch.zeros((), device=dev)
    n_impure = torch.zeros((), device=dev)
    purity_sum = torch.zeros((), device=dev)
    cell_ever_impure = torch.zeros(P, dtype=torch.bool, device=dev)
    C_ = T.shape[0]
    for n, vi in enumerate(sel):
        cam = dh.cameras[vi]
        H, W_ = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                       max_intersections=4096)
        ri = op.row_indices.to(torch.int64)
        ci = op.col_indices.to(torch.int64)
        vv = op.values.float()
        del op
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W_)
        seg, tab = seg.to(dev), tab.to(dev)
        seg_lab = (tab @ T.T).argmax(1)                    # label of each SAM segment in this view
        M = torch.zeros((P, tab.shape[1]), device=dev)
        hist = torch.zeros((P, C_), device=dev)
        CH = 4_000_000
        for s0 in range(0, vv.numel(), CH):
            e0 = min(s0 + CH, vv.numel())
            sid = seg[ri[s0:e0]].clamp(0, tab.shape[0] - 1)
            M.index_add_(0, ci[s0:e0], vv[s0:e0, None] * tab[sid])
            hist.index_put_((ci[s0:e0], seg_lab[sid]), vv[s0:e0], accumulate=True)
        u = torch.nn.functional.normalize(M, dim=1)
        c[:, n] = (u * u_bar).sum(1)
        lab[:, n] = (u @ T.T).argmax(1).to(torch.int16)
        tot_v = hist.sum(1)
        m = tot_v > 1e-8
        pur = hist.max(1).values[m] / tot_v[m]
        n_pairs += m.sum()
        n_impure += (pur < 1 - 1e-6).sum()
        purity_sum += pur.sum()
        idx = torch.nonzero(m, as_tuple=True)[0]
        cell_ever_impure[idx[pur < 1 - 1e-6]] = True
        del M, u, hist, ri, ci, vv, seg, tab, seg_lab
    seen = w > 1e-8                                     # cell actually observed in that view

    # CALIBRATION. TAU is an absolute cosine, but CLIP features are not spread over the sphere --
    # even unrelated classes sit at cosine 0.5-0.8 -- so a fixed 0.5 may never fire and would report
    # "no bimodal cells" whatever the data looked like. Print the observed distribution so the
    # threshold can be read off it rather than assumed.
    cs = c[seen]
    cq = torch.quantile(cs.double(), torch.tensor([.001, .01, .05, .25, .5], device=dev,
                                                  dtype=torch.float64))
    print(f"  [calib] c=<u_jv,u_bar> over observed (cell,view): "
          f"p0.1%={cq[0]:.3f} p1%={cq[1]:.3f} p5%={cq[2]:.3f} p25%={cq[3]:.3f} p50%={cq[4]:.3f}",
          flush=True)
    permin = torch.where(seen, c, torch.ones_like(c)).amin(1)
    mq = torch.quantile(permin[live].double(), torch.tensor([.01, .05, .25, .5], device=dev,
                                                            dtype=torch.float64))
    print(f"  [calib] per-cell MIN over views:  p1%={mq[0]:.3f} p5%={mq[1]:.3f} "
          f"p25%={mq[2]:.3f} p50%={mq[3]:.3f}", flush=True)

    # ---- regimes -----------------------------------------------------------------------------
    wsum = (w * seen).sum(1).clamp_min(1e-12)
    dissent = seen & (c < TAU)
    f_low = (w * dissent).sum(1) / wsum
    n_seen = seen.sum(1)
    ok = live & (n_seen >= 3)                           # need >=3 views to speak of clusters
    reg_i = ok & (f_low < 1e-3)
    reg_iii = ok & (f_low >= BIMODAL_F)
    reg_ii = ok & ~reg_i & ~reg_iii

    # ---- baseline: parallax available to a triangulation -------------------------------------
    cam_o = torch.stack([dh.c2ws[vi][:3, 3].float() for vi in sel]).to(dev)     # (V,3)
    dirs = torch.nn.functional.normalize(centres[:, None, :] - cam_o[None, :, :], dim=2)  # (P,V,3)

    def spread_deg(mask):
        """Max pairwise angle between viewing directions of the masked views, per cell."""
        d = dirs * (mask.float()[:, :, None])
        cos = torch.einsum("pvd,pwd->pvw", d, d).clamp(-1, 1)
        big = mask[:, :, None] & mask[:, None, :]
        cos = torch.where(big, cos, torch.ones_like(cos))
        return torch.rad2deg(torch.arccos(cos.amin(dim=(1, 2)).clamp(-1, 1)))

    base_all = spread_deg(seen)
    # For the two-cluster cells, the angle BETWEEN the clusters' mean directions is what a split
    # would have to resolve; near 0 means the two readings come from the same viewpoint and no
    # chord geometry can separate them.
    hi = seen & (c >= TAU)
    lo = dissent
    mh = torch.nn.functional.normalize((dirs * (hi.float() * w)[:, :, None]).sum(1), dim=1)
    ml = torch.nn.functional.normalize((dirs * (lo.float() * w)[:, :, None]).sum(1), dim=1)
    inter = torch.rad2deg(torch.arccos((mh * ml).sum(1).clamp(-1, 1)))

    def pct(t):
        return float(t.sum()) / max(float(ok.sum()), 1) * 100

    def qs(t, m):
        if int(m.sum()) < 10:
            return None
        return [round(float(x), 2) for x in torch.quantile(
            t[m].double(), torch.tensor([.1, .5, .9], device=dev, dtype=torch.float64))]

    # THRESHOLD ROBUSTNESS. "0% bimodal" must not be an artifact of one arbitrary TAU, so sweep it
    # across the whole plausible range -- from well below the 1st percentile of c up to its median,
    # beyond which "dissent" would just be relabelling ordinary spread.
    sweep = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        dis = seen & (c < tau)
        fl = (w * dis).sum(1) / wsum
        sweep.append({"tau": tau,
                      "pct_ii": round(pct(ok & (fl > 1e-3) & (fl < BIMODAL_F)), 2),
                      "pct_iii": round(pct(ok & (fl >= BIMODAL_F)), 2)})
    print("  [tau sweep]  " + "  ".join(
        f"t={s['tau']}: ii={s['pct_ii']:.1f}% iii={s['pct_iii']:.1f}%" for s in sweep), flush=True)

    # ---- LABEL-BASED regimes: the threshold-free version -------------------------------------
    C_ = T.shape[0]
    onehot = torch.zeros((P, C_), device=dev)
    for n in range(V):
        m = seen[:, n]
        if not bool(m.any()):
            continue
        onehot[m] = onehot[m].scatter_add(
            1, lab[m, n].long()[:, None], w[m, n][:, None])       # weight-vote per class
    tot = onehot.sum(1).clamp_min(1e-12)
    top2 = onehot.topk(2, dim=1).values
    share1 = top2[:, 0] / tot
    share2 = top2[:, 1] / tot
    L_i = ok & (share1 > 1 - 1e-6)                                # every view agrees on one label
    L_iii = ok & (share2 >= BIMODAL_F)                            # runner-up carries >= 25% weight
    L_ii = ok & ~L_i & ~L_iii
    res_lab = {"pct_label_i_unanimous": pct(L_i),
               "pct_label_ii_minority": pct(L_ii),
               "pct_label_iii_bimodal": pct(L_iii),
               "n_distinct_labels_median": float(
                   (onehot > 0).sum(1)[ok].float().median())}
    print(f"  [LABEL] unanimous {res_lab['pct_label_i_unanimous']:5.1f}%   "
          f"minority-dissent {res_lab['pct_label_ii_minority']:5.1f}%   "
          f"BIMODAL {res_lab['pct_label_iii_bimodal']:5.1f}%   "
          f"(median distinct labels/cell {res_lab['n_distinct_labels_median']:.0f})", flush=True)
    base_lab = qs(base_all, L_iii)
    mh2 = torch.nn.functional.normalize(
        (dirs * (seen & (lab == onehot.argmax(1, keepdim=True).to(torch.int16))).float()[:, :, None]
         * w[:, :, None]).sum(1), dim=1)
    lo2 = seen & (lab != onehot.argmax(1, keepdim=True).to(torch.int16))
    ml2 = torch.nn.functional.normalize((dirs * lo2.float()[:, :, None] * w[:, :, None]).sum(1),
                                        dim=1)
    inter_lab = torch.rad2deg(torch.arccos((mh2 * ml2).sum(1).clamp(-1, 1)))
    res_lab["pct_cellview_impure"] = float(n_impure / n_pairs.clamp_min(1)) * 100
    res_lab["mean_within_view_purity"] = float(purity_sum / n_pairs.clamp_min(1))
    res_lab["pct_cells_ever_impure"] = pct(ok & cell_ever_impure)
    print(f"  [WITHIN-VIEW] (cell,view) pairs spanning >1 label: "
          f"{res_lab['pct_cellview_impure']:.2f}%   mean purity "
          f"{res_lab['mean_within_view_purity']:.4f}   cells ever impure "
          f"{res_lab['pct_cells_ever_impure']:.1f}%", flush=True)
    res_lab["baseline_deg_label_bimodal_q"] = base_lab
    res_lab["intercluster_angle_label_q"] = qs(inter_lab, L_iii)
    print(f"  [LABEL] parallax on bimodal cells {base_lab} deg   "
          f"inter-cluster angle {res_lab['intercluster_angle_label_q']} deg", flush=True)

    res = {"scene": a.scene, "views": a.views, "P": P, "n_ok": int(ok.sum()),
           "tau": TAU, "bimodal_f": BIMODAL_F,
           "pct_regime_i_unanimous": pct(reg_i),
           "pct_regime_ii_outliers": pct(reg_ii),
           "pct_regime_iii_bimodal": pct(reg_iii),
           "baseline_deg_all_q": qs(base_all, ok),
           "baseline_deg_bimodal_q": qs(base_all, reg_iii),
           "intercluster_angle_deg_q": qs(inter, reg_iii),
           "median_views_seen": float(n_seen[ok].float().median()),
           "tau_sweep": sweep, **res_lab}
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)
    print(f"[{a.scene}] cells scored {res['n_ok']:,}  median views/cell "
          f"{res['median_views_seen']:.0f}", flush=True)
    print(f"  (i)   unanimous        {res['pct_regime_i_unanimous']:5.1f}%", flush=True)
    print(f"  (ii)  majority+outlier {res['pct_regime_ii_outliers']:5.1f}%   <- robust estimator",
          flush=True)
    print(f"  (iii) two clusters     {res['pct_regime_iii_bimodal']:5.1f}%   <- triangulation "
          f"candidates", flush=True)
    print(f"  parallax all cells   (10/50/90%) {res['baseline_deg_all_q']} deg", flush=True)
    print(f"  parallax bimodal     (10/50/90%) {res['baseline_deg_bimodal_q']} deg", flush=True)
    print(f"  inter-cluster angle  (10/50/90%) {res['intercluster_angle_deg_q']} deg", flush=True)


if __name__ == "__main__":
    main()
