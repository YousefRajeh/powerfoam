"""How well do foam's geometric factors detect a CONTAMINATED observation? Measured, on real data.

WHY THIS NUMBER DECIDES EVERYTHING. Post-Lifting Aggregation, our geometric weighting, and any
trimming scheme are all the same operation -- lower the contaminated weight fraction eps in

    d(x^M, x0)  <=  2 r0 + (eps/(1-eps)) * pi

and the simulation in pla_vs_geometric_weights.py showed the choice of OPERATOR (reject vs
down-weight) matters far less than the quality of the DETECTOR: at every operator, better detector
AUC gave lower eps. So the question that settles whether foam can beat PLA's pseudo-mask IoU is
simply how well foam's own geometry separates good observations from bad ones.

GROUND TRUTH, from GT alone. Observation (cell j, view v) is CONTAMINATED iff the SAM mask it
predominantly reads is itself dominated by ground-truth points of a different class than cell j's
own ground-truth label. Both sides come from ScanNet labels: the cell's label by exact power-cell
ownership, the mask's label by the visible GT points landing inside it. No CLIP is involved, so this
measures mask/geometry contamination and not the text encoder's mistakes -- which matters, because
those are the errors the confuser diagnosis showed are NOT fixable here.

DETECTORS COMPARED (all computable in one streaming pass, no clustering, no pseudo-mask projection):
  share    fraction of the cell's rendering weight in this view falling in its dominant mask.
           Exact ownership makes "this cell's pixels" a well-defined set; semantic bleed at a mask
           boundary shows up directly.
  weight   the cell's total rendering weight in this view -- a visibility/occlusion proxy: a barely
           visible cell is reading mostly other things.
  nmask    how many distinct masks the cell touches in this view (fragmentation).
  area     the cell's projected footprint in pixels (a size control, to show the detectors are not
           just measuring "big cell").
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def auc(score, label):
    """P(score of a clean obs > score of a contaminated obs). 0.5 = useless."""
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0            # ranks, ties handled crudely
    n_pos, n_neg = int((~label).sum()), int(label.sum())
    return float((r[~label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--vis-tol", type=float, default=0.05)
    ap.add_argument("--min-gt-per-mask", type=int, default=20)
    ap.add_argument("--out", default="artifacts/detector_auc.npz")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from accumulate_hard_mask import load_masks
    from build_true_facet_graph import load_points_radii
    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from eval_surface_chamfer import cos_map
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from oracle_labels import oracle_labels
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.rasterize import VisOptions
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    model.update_vis_cache()
    P = model.points.shape[0]

    vis = VisOptions()
    vis.transmittance_threshold = 1e-3
    vis.max_intersections = 1024
    vis.depth_quantile = 0.5
    vis.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    K = len(names)
    cc, rr = load_points_radii(ck)
    centers = np.asarray(cc, np.float64); radii = np.asarray(rr, np.float64)
    cell_lab, _ = oracle_labels(centers, radii, pts, gt_lab, K + 1)
    print(f"{P:,} cells, {int((cell_lab>0).sum()):,} with a GT label; {K} classes")

    feat_dir = a.features.format(scene=a.scene)
    from pathlib import Path
    stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())
    assert len(stems) == len(dh.cameras)

    S, W, NM, AR, CONT, CID, VID = [], [], [], [], [], [], []
    P3 = np.asarray(pts, np.float64)
    for vi, cam in enumerate(dh.cameras):
        H, W_ = int(cam.height), int(cam.width)
        _, seg = load_masks(feat_dir, stems[vi], a.level, H, W_)
        seg = seg.reshape(-1).numpy()
        M = int(seg.max()) + 1
        if M <= 0:
            continue

        # ---- which GT class does each MASK belong to? project GT points, depth-test, vote -------
        with torch.no_grad():
            out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis)
        dep = out[1].detach().float().cpu().numpy()
        alp = out[3].detach().float().cpu().numpy()
        dep = dep[..., 0] if dep.ndim == 3 else dep
        alp = alp[..., 0] if alp.ndim == 3 else alp
        z_img = dep * cos_map(cam)
        prm = cam.to_open3d()
        extr = np.asarray(prm.extrinsic, np.float64)
        Kk = np.asarray(prm.intrinsic.intrinsic_matrix, np.float64)
        pc = P3 @ extr[:3, :3].T + extr[:3, 3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = Kk[0, 0] * pc[:, 0] / z + Kk[0, 2]
            v = Kk[1, 1] * pc[:, 1] / z + Kk[1, 2]
        ui, vj = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        okp = (z > 1e-3) & (ui >= 0) & (ui < W_) & (vj >= 0) & (vj < H) & (gt_lab > 0)
        idx = np.where(okp)[0]
        pix = vj[idx] * W_ + ui[idx]
        seen = (alp.reshape(-1)[pix] >= 0.5) & (np.abs(z_img.reshape(-1)[pix] - z[idx]) <= a.vis_tol)
        idx, pix = idx[seen], pix[seen]
        mid_pt = seg[pix]
        good = mid_pt >= 0
        hist = np.zeros((M, K + 1), np.int64)
        np.add.at(hist, (mid_pt[good], gt_lab[idx][good]), 1)
        mask_lab = hist.argmax(1)
        mask_lab[hist.sum(1) < a.min_gt_per_mask] = -1        # too little GT to judge

        # ---- per (cell, view): dominant mask, share, weight, fragmentation, area ---------------
        op = export_operator_for_views(model, [cam], [vi])
        rows = op.row_indices.cpu().numpy()
        cols = op.col_indices.cpu().numpy()
        vals = op.values.cpu().numpy().astype(np.float64)
        mm = seg[rows]
        keep = mm >= 0
        cols, vals, mm, rows = cols[keep], vals[keep], mm[keep], rows[keep]
        if len(cols) == 0:
            continue
        Wv = np.bincount(cols, weights=vals, minlength=P)
        key = cols * M + mm
        h = np.bincount(key, weights=vals, minlength=P * M).reshape(P, M)
        best_m = h.argmax(1)
        best_w = h.max(1)
        nmask = (h > 1e-9).sum(1)
        area = np.bincount(cols, minlength=P).astype(np.float64)
        pres = (Wv > 1e-9) & (cell_lab > 0) & (mask_lab[best_m] >= 0)
        if not pres.any():
            continue
        S.append(best_w[pres] / Wv[pres])
        W.append(Wv[pres])
        NM.append(nmask[pres])
        AR.append(area[pres])
        CONT.append(mask_lab[best_m[pres]] != cell_lab[pres])
        CID.append(np.where(pres)[0])
        VID.append(np.full(int(pres.sum()), vi))
        if vi % 10 == 0:
            print(f"  view {vi}: {int(pres.sum()):,} judgeable observations, "
                  f"{CONT[-1].mean():.1%} contaminated", flush=True)

    S = np.concatenate(S); W = np.concatenate(W); NM = np.concatenate(NM)
    AR = np.concatenate(AR); CONT = np.concatenate(CONT)
    CID = np.concatenate(CID); VID = np.concatenate(VID)
    print(f"\n{len(S):,} judgeable (cell, view) observations")
    print(f"contaminated: {CONT.mean():.2%}   <- this is eps, measured")

    print(f"\n{'detector':>28} {'AUC':>7}   (0.5 = useless, >0.5 = higher score means cleaner)")
    for nm, sc in [("mask share  s_jv", S), ("rendering weight  W_jv", W),
                   ("-(masks touched)  -n_jv", -NM.astype(float)),
                   ("-(projected area)", -AR),
                   ("share x weight", S * W),
                   ("share x weight / masks", S * W / np.maximum(NM, 1))]:
        print(f"{nm:>28} {auc(sc, CONT):7.4f}")

    # what a filter at the natural operating point would actually do
    print(f"\nif we trim the worst 25% by 'mask share':")
    thr = np.quantile(S, 0.25)
    keep = S > thr
    print(f"   eps {CONT.mean():.4f} -> {CONT[keep].mean():.4f}   "
          f"keep_clean {(~CONT & keep).sum() / max((~CONT).sum(),1):.3f}  "
          f"keep_dirty {(CONT & keep).sum() / max(CONT.sum(),1):.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, share=S.astype(np.float32), weight=W.astype(np.float32),
                        nmask=NM.astype(np.int32), area=AR.astype(np.float32), cont=CONT,
                        cell=CID.astype(np.int32), view=VID.astype(np.int32))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
