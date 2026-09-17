"""Use the gradient to CLEAN the lifted feature instead of to split the cell.

WHY THE INVERSION. run_partition_ceiling.py shows an oracle labelling every cell by the majority GT
label of its own points reaches 99.5%, against ~41% actually achieved. So the partition already
resolves the ground truth: the 58-point gap is entirely in the FEATURES, and sub-cell splitting has
at most ~0.5% to win. A strong feature gradient across a footprint therefore does NOT mean the cell
straddles two objects -- the GT says it almost never does. It means the cell is COLLECTING
CONTAMINATED OBSERVATIONS, and its lifted feature is a mixture of something that should be one
thing. The gradient's job is to say which observations to stop averaging in.

WHERE THE CONTAMINATION IS. 18% of (cell,view) pairs span more than one predicted label: the
footprint crosses a SAM mask seam and the weighted mean blends both segments. The natural repair is
a mode vote INSIDE a view -- take the dominant segment's feature rather than the mean over all the
cell's pixels. That is only definable on a bounded footprint; a Gaussian has no such thing.

ARMS, all sharing one operator pass so the comparison is exact:
  mean       current behaviour: sum_i A_ij B_i over every pixel, every view.
  mode       per (cell, view) take the SINGLE segment carrying the most weight, contribute its
             feature with the cell's total weight in that view. Across views, sum as usual.
  gated      mode ONLY where the gradient magnitude says the footprint is contaminated; mean
             elsewhere. This is the version the magnitude signal actually buys: mode is a blunt
             instrument and discards real averaging where nothing is wrong.
  topfrac    softer than mode: keep segments until `--keep` of the weight is covered, drop the tail.

Scored against ScanNet GT through exact power-cell membership, same protocol as the eval.
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

OUT = "artifacts/scannet/feature_cleanup"


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
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    ap.add_argument("--gate-pct", type=float, default=0.25,
                    help="top fraction by gradient magnitude that the gated arm cleans")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    from configs import Params, add_group
    from data_loader import DataHandler
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    from point_cloud_query import assign_points_to_power_cells
    from evaluate_point_cloud_miou import (embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels, OPENGAUSSIAN_CLASS_SETS,
                                           SCANNET20_CLASS_NAMES)

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
    P = int(model.points.shape[0])
    centres = model.points.detach().float()
    radii = model.get_radii().detach().float().reshape(-1)

    names = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    T = torch.nn.functional.normalize(embed_class_names(names, dev).float(), dim=-1)
    D, C_ = T.shape[1], T.shape[0]

    for sub in ("train", "val", "test"):
        gt_dir = os.path.join(a.gt_root, sub, a.scene)
        if os.path.isdir(gt_dir):
            break
    gt_pts, gt_raw, _ = load_scannet_pointcept_gt(gt_dir)
    target_ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]
    gl_np = remap_gt_labels(gt_raw, target_ids) - 1
    owner_np = assign_points_to_power_cells(gt_pts.astype(np.float64),
                                            centres.cpu().numpy(), radii.cpu().numpy())
    keep = (owner_np >= 0) & (gl_np >= 0)
    own = torch.from_numpy(owner_np[keep]).long().to(dev)
    gl = torch.from_numpy(gl_np[keep]).long().to(dev)
    votes = torch.zeros((P, C_), device=dev)
    votes.index_put_((own, gl), torch.ones(own.numel(), device=dev), accumulate=True)
    cell_gt = votes.argmax(1)
    has_gt = votes.sum(1) > 0

    M_mean = torch.zeros((P, D), device=dev)
    M_mode = torch.zeros((P, D), device=dev)
    M_top = torch.zeros((P, D), device=dev)
    S_tot = torch.zeros(P, device=dev)
    mag_acc = torch.zeros(P, device=dev)      # max gradient magnitude seen over views
    KEEP = 0.75

    for vi in sel:
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
        sid = seg[ri].clamp(0, tab.shape[0] - 1)
        nseg = tab.shape[0]

        M_mean.index_add_(0, ci, vv[:, None] * tab[sid])
        Wc = torch.zeros(P, device=dev).index_add_(0, ci, vv)
        S_tot += Wc

        # per (cell, segment) weight -- sparse via a flat (cell*nseg) key, then dense per-cell argmax
        # over the segments that actually appear (nseg is a few hundred, so a P x nseg dense table is
        # too large; use scatter on a compacted key instead).
        key = ci * nseg + sid
        uk, inv = torch.unique(key, return_inverse=True)
        wseg = torch.zeros(uk.numel(), device=dev).index_add_(0, inv, vv)
        kc = (uk // nseg)
        ks = (uk % nseg)
        # dominant segment per cell = argmax of wseg within each cell group
        best_w = torch.zeros(P, device=dev).index_reduce_(0, kc, wseg, "amax", include_self=False)
        is_best = wseg >= best_w[kc] - 1e-9
        # ties: keep the first
        M_mode.index_add_(0, kc[is_best], (Wc[kc[is_best]])[:, None] * tab[ks[is_best]])

        # topfrac: keep segments covering KEEP of the cell's weight, renormalised
        order = torch.argsort(wseg, descending=True)
        kc_o, ks_o, w_o = kc[order], ks[order], wseg[order]
        # cumulative share within each cell, computed by sorting cells then a segmented cumsum
        srt = torch.argsort(kc_o, stable=True)
        kc_s, ks_s, w_s = kc_o[srt], ks_o[srt], w_o[srt]
        csum = torch.cumsum(w_s, 0)
        starts = torch.zeros_like(csum)
        first = torch.ones_like(kc_s, dtype=torch.bool)
        first[1:] = kc_s[1:] != kc_s[:-1]
        idx0 = torch.nonzero(first, as_tuple=True)[0]
        base = torch.zeros_like(csum)
        base[idx0] = csum[idx0] - w_s[idx0]
        base = torch.cummax(base, 0).values
        share = (csum - base) / Wc[kc_s].clamp_min(1e-8)
        prev = share - w_s / Wc[kc_s].clamp_min(1e-8)
        keepm = prev < KEEP
        M_top.index_add_(0, kc_s[keepm], w_s[keepm][:, None] * tab[ks_s[keepm]])

        # gradient magnitude for the gate
        py = (ri // W_).float()
        px = (ri % W_).float()
        Wn = Wc.clamp_min(1e-8)
        mx = (torch.zeros(P, device=dev).index_add_(0, ci, vv * px)) / Wn
        my = (torch.zeros(P, device=dev).index_add_(0, ci, vv * py)) / Wn
        dx, dy = px - mx[ci], py - my[ci]
        Sxx = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dx)
        Syy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dy * dy)
        Sxy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dy)
        Gx = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dx)[:, None] * tab[sid])
        Gy = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dy)[:, None] * tab[sid])
        det = (Sxx * Syy - Sxy * Sxy).clamp_min(1e-12)
        bx = (Syy[:, None] * Gx - Sxy[:, None] * Gy) / det[:, None]
        by = (Sxx[:, None] * Gy - Sxy[:, None] * Gx) / det[:, None]
        aa = (bx * bx).sum(1); bb = (bx * by).sum(1); cc = (by * by).sum(1)
        lam1 = 0.5 * ((aa + cc) + ((aa + cc) ** 2 - 4 * (aa * cc - bb * bb).clamp_min(0)).clamp_min(0).sqrt())
        rms = ((Sxx + Syy) / Wn).clamp_min(1e-12).sqrt()
        mag_acc = torch.maximum(mag_acc, lam1.clamp_min(0).sqrt() * rms)
        del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by

    live = (S_tot > 1e-8) & has_gt

    def acc_of(M):
        pred = (torch.nn.functional.normalize(M, dim=1) @ T.T).argmax(1)
        return float((pred[live] == cell_gt[live]).float().mean())

    thr = torch.quantile(mag_acc[live].double(), 1.0 - a.gate_pct).float()
    gate = (mag_acc >= thr)[:, None]
    M_gated = torch.where(gate, M_mode, M_mean)
    M_gated_top = torch.where(gate, M_top, M_mean)

    res = {"scene": a.scene, "views": a.views, "n_live": int(live.sum()),
           "gate_pct": a.gate_pct, "keep": KEEP,
           "acc_mean": acc_of(M_mean), "acc_mode": acc_of(M_mode),
           "acc_topfrac": acc_of(M_top),
           "acc_gated_mode": acc_of(M_gated), "acc_gated_topfrac": acc_of(M_gated_top)}
    res["delta_mode"] = res["acc_mode"] - res["acc_mean"]
    res["delta_topfrac"] = res["acc_topfrac"] - res["acc_mean"]
    res["delta_gated_mode"] = res["acc_gated_mode"] - res["acc_mean"]
    res["delta_gated_topfrac"] = res["acc_gated_topfrac"] - res["acc_mean"]
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)
    print(f"[{a.scene}] live {res['n_live']:,}  gate top {a.gate_pct*100:.0f}% by magnitude",
          flush=True)
    print(f"  mean (current)      {res['acc_mean']*100:6.2f}%", flush=True)
    print(f"  mode  (all cells)   {res['acc_mode']*100:6.2f}%  ({res['delta_mode']*100:+.2f} pp)",
          flush=True)
    print(f"  topfrac{KEEP} (all) {res['acc_topfrac']*100:6.2f}%  "
          f"({res['delta_topfrac']*100:+.2f} pp)", flush=True)
    print(f"  GATED mode          {res['acc_gated_mode']*100:6.2f}%  "
          f"({res['delta_gated_mode']*100:+.2f} pp)", flush=True)
    print(f"  GATED topfrac       {res['acc_gated_topfrac']*100:6.2f}%  "
          f"({res['delta_gated_topfrac']*100:+.2f} pp)", flush=True)


if __name__ == "__main__":
    main()
