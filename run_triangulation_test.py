"""Two questions, one pass: does view disagreement predict ERROR, and is within-view disagreement
SPATIALLY organised (i.e. is there anything for sub-cell subdivision to resolve)?

(A) THE JOIN. The gate found ~47% of cells bimodal across views and ~18% of (cell,view) pairs
    spanning more than one label. That is a population, not a payoff. If those cells are
    misclassified at the base rate, fixing the accumulator buys nothing; if they carry a
    disproportionate share of the errors, the ceiling is real. Joins the per-cell argmax against
    ScanNet's own point labels through exact power-cell membership.

(B) SEMANTIC TRIANGULATION, done in the regime the data actually supports. The earlier attempt
    looked for BETWEEN-view clusters separated by viewpoint and found the inter-cluster angle small.
    But within-view impurity is real (18% of pairs), and there the spatial signal needs no parallax
    at all: inside ONE view the cell has a compact image footprint, and if it straddles a semantic
    boundary the two label groups occupy DIFFERENT PARTS of that footprint. Transverse position in
    the image is transverse position in the cell.

    So for each impure (cell, view): take the top two labels, compute each group's pixel centroid,
    and compare their separation to the footprint's own radius:

        sep_norm = ||centroid_A - centroid_B|| / rms_radius_of_footprint

    sep_norm near 0  -> the two labels are interleaved over the same pixels: mask noise at a segment
                        edge. Subdivision cannot help; a robust accumulator can.
    sep_norm >~ 1    -> the groups occupy distinct parts of the footprint: the cell genuinely
                        straddles a boundary, and the separating line localises it. THIS is the
                        case where splitting the cell is the right fix, and it is available only on
                        a bounded partition -- a Gaussian has no footprint to divide.

    A null is needed or any number looks impressive: two random halves of the SAME label's pixels
    are also compared, giving the separation expected from sampling alone.
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

OUT = "artifacts/scannet/triangulation"


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

    ids = OPENGAUSSIAN_CLASS_SETS["opengaussian19"]
    names = [SCANNET20_CLASS_NAMES[i] for i in ids] if isinstance(ids[0], int) else list(ids)
    T = torch.nn.functional.normalize(embed_class_names(names, dev).float(), dim=-1)
    C_ = T.shape[0]

    # ---- ground truth per cell, via exact power-cell membership ------------------------------
    for sub in ("train", "val", "test"):
        gt_dir = os.path.join(a.gt_root, sub, a.scene)
        if os.path.isdir(gt_dir):
            break
    gt_pts, gt_raw, _ = load_scannet_pointcept_gt(gt_dir)
    # remap_gt_labels uses OpenGaussian's convention: 1..K for the kept classes, 0 = ignore. Shift
    # to 0..K-1 to index the prototype rows, and drop the ignore points.
    # OPENGAUSSIAN_CLASS_SETS holds NAMES; remap_gt_labels needs the raw integer ids that
    # segment20.npy actually stores (0..19 indexing SCANNET20_CLASS_NAMES, -1 = ignore). Passing the
    # names straight through silently matched nothing and zeroed every label.
    target_ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]
    gt_lab = remap_gt_labels(gt_raw, target_ids)
    gp = torch.from_numpy(gt_pts).float().to(dev)
    gl = torch.from_numpy(np.asarray(gt_lab)).long().to(dev) - 1
    centres = model.points.detach().float().to(dev)
    # model.get_radii(), NOT model.radius -- a hasattr fallback to zeros silently degrades the power
    # diagram to a plain Voronoi and the membership no longer matches the eval protocol.
    radii = model.get_radii().detach().float().to(dev).reshape(-1)
    # assign_points_to_power_cells is a numpy/cKDTree routine, not a torch op -- feed it host arrays.
    owner_np = assign_points_to_power_cells(gt_pts.astype(np.float64),
                                            centres.detach().cpu().numpy(),
                                            radii.detach().cpu().numpy())
    owner = torch.from_numpy(np.asarray(owner_np)).long().to(dev)
    votes = torch.zeros((P, C_ + 1), device=dev)
    ok_pt = (owner >= 0) & (gl >= 0)
    votes.index_put_((owner[ok_pt], gl[ok_pt]), torch.ones(int(ok_pt.sum()), device=dev),
                     accumulate=True)
    cell_gt = votes[:, :C_].argmax(1)
    has_gt = votes[:, :C_].sum(1) > 0
    print(f"  [gt] points {gp.shape[0]:,}  owned {int((owner >= 0).sum()):,}  "
          f"labelled {int((gl >= 0).sum()):,}  usable {int(ok_pt.sum()):,}  "
          f"cells with GT {int(has_gt.sum()):,}/{P:,}", flush=True)

    # ---- lift + per-view label maps + footprint geometry --------------------------------------
    M_tot = torch.zeros((P, T.shape[1]), device=dev)
    S_tot = torch.zeros(P, device=dev)
    lab_w = torch.zeros((P, C_), device=dev)          # between-view weight per label
    seen_cnt = torch.zeros(P, device=dev)

    sep_num, sep_den, sep_cnt = [], [], []
    null_num = []
    n_impure_pairs = 0
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
        seg_lab = (tab @ T.T).argmax(1)
        sid = seg[ri].clamp(0, tab.shape[0] - 1)
        rl = seg_lab[sid]                                  # per-nonzero predicted label
        M_tot.index_add_(0, ci, vv[:, None] * tab[sid])
        S_tot.index_add_(0, ci, vv)

        # per-(cell,label) pixel moments in THIS view
        py = (ri // W_).float()
        px = (ri % W_).float()
        wsum = torch.zeros((P, C_), device=dev)
        sx = torch.zeros((P, C_), device=dev)
        sy = torch.zeros((P, C_), device=dev)
        wsum.index_put_((ci, rl), vv, accumulate=True)
        sx.index_put_((ci, rl), vv * px, accumulate=True)
        sy.index_put_((ci, rl), vv * py, accumulate=True)
        tot = wsum.sum(1)
        m = tot > 1e-8
        lab_w += wsum
        seen_cnt += m.float()

        # footprint rms radius from all of the cell's pixels this view
        cx = torch.zeros(P, device=dev).index_add_(0, ci, vv * px)
        cy = torch.zeros(P, device=dev).index_add_(0, ci, vv * py)
        tt = tot.clamp_min(1e-8)
        cx, cy = cx / tt, cy / tt
        r2 = torch.zeros(P, device=dev).index_add_(
            0, ci, vv * ((px - cx[ci]) ** 2 + (py - cy[ci]) ** 2))
        rms = (r2 / tt).clamp_min(1e-12).sqrt()

        n_impure_pairs += int((m & (wsum.topk(2, dim=1).values[:, 1] > 1e-8)).sum())

        # ---- FEATURE-DRIVEN spatial coherence, no labels, no GT --------------------------------
        # Grouping rays by argmax label quantises a 512-d feature into 19 buckets and discards the
        # geometry triangulation needs. Instead regress the FEATURE ITSELF on pixel position inside
        # the footprint: if the cell straddles a boundary, the feature varies systematically ACROSS
        # the footprint, and a linear model in (x, y) explains a large share of its variance.
        #     R^2 = explained / total,  total = sum_i w_i ||f_i - f_bar||^2 = W (1 - ||f_bar||^2)
        # (the last equality holds because the f_i are unit vectors).
        # R^2 -> 1 : feature varies systematically with position -> genuine straddle, and the fitted
        #            gradient direction IS the split line, localised inside the cell.
        # R^2 -> 0 : variation is spatially unstructured -> mask noise; splitting cannot help.
        # The gradient needs no parallax: within one view, transverse image position is transverse
        # position in the cell. This is what a bounded footprint buys and a Gaussian cannot supply.
        Wc = torch.zeros(P, device=dev).index_add_(0, ci, vv)
        Wn = Wc.clamp_min(1e-8)
        mx = (torch.zeros(P, device=dev).index_add_(0, ci, vv * px)) / Wn
        my = (torch.zeros(P, device=dev).index_add_(0, ci, vv * py)) / Wn
        dx, dy = px - mx[ci], py - my[ci]
        Sxx = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dx)
        Syy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dy * dy)
        Sxy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dy)
        Fb = torch.zeros((P, T.shape[1]), device=dev).index_add_(0, ci, vv[:, None] * tab[sid])
        Fb = Fb / Wn[:, None]
        Gx = torch.zeros((P, T.shape[1]), device=dev).index_add_(
            0, ci, (vv * dx)[:, None] * tab[sid])
        Gy = torch.zeros((P, T.shape[1]), device=dev).index_add_(
            0, ci, (vv * dy)[:, None] * tab[sid])
        det = (Sxx * Syy - Sxy * Sxy)
        good = m & (det > 1e-6) & (Wc > 1e-6)
        # Solve the 2x2 normal equations per cell, per feature dimension, in closed form.
        bx = (Syy[:, None] * Gx - Sxy[:, None] * Gy) / det.clamp_min(1e-12)[:, None]
        by = (Sxx[:, None] * Gy - Sxy[:, None] * Gx) / det.clamp_min(1e-12)[:, None]
        expl = (bx * Gx).sum(1) + (by * Gy).sum(1)
        total = Wc * (1.0 - (Fb * Fb).sum(1)).clamp_min(0)
        r2 = torch.where(total > 1e-9, (expl / total.clamp_min(1e-12)).clamp(0, 1),
                         torch.zeros_like(total))
        sep_num.append(r2[good])

        # NULL: identical regression with pixel coordinates SHUFFLED within each cell. Same weights,
        # same features, same footprint size -- only the position/feature pairing is destroyed. Any
        # R^2 above this is real spatial structure rather than fitting 2 parameters to few samples.
        perm = torch.randperm(ci.numel(), device=dev)
        pxs, pys = px[perm], py[perm]
        mxs = (torch.zeros(P, device=dev).index_add_(0, ci, vv * pxs)) / Wn
        mys = (torch.zeros(P, device=dev).index_add_(0, ci, vv * pys)) / Wn
        dxs, dys = pxs - mxs[ci], pys - mys[ci]
        Sxx2 = torch.zeros(P, device=dev).index_add_(0, ci, vv * dxs * dxs)
        Syy2 = torch.zeros(P, device=dev).index_add_(0, ci, vv * dys * dys)
        Sxy2 = torch.zeros(P, device=dev).index_add_(0, ci, vv * dxs * dys)
        Gx2 = torch.zeros((P, T.shape[1]), device=dev).index_add_(
            0, ci, (vv * dxs)[:, None] * tab[sid])
        Gy2 = torch.zeros((P, T.shape[1]), device=dev).index_add_(
            0, ci, (vv * dys)[:, None] * tab[sid])
        det2 = (Sxx2 * Syy2 - Sxy2 * Sxy2)
        bx2 = (Syy2[:, None] * Gx2 - Sxy2[:, None] * Gy2) / det2.clamp_min(1e-12)[:, None]
        by2 = (Sxx2[:, None] * Gy2 - Sxy2[:, None] * Gx2) / det2.clamp_min(1e-12)[:, None]
        expl2 = (bx2 * Gx2).sum(1) + (by2 * Gy2).sum(1)
        r2n = torch.where(total > 1e-9, (expl2 / total.clamp_min(1e-12)).clamp(0, 1),
                          torch.zeros_like(total))
        null_num.append(r2n[good & (det2 > 1e-6)])
        del (ri, ci, vv, seg, tab, sid, rl, wsum, sx, sy, py, px, Gx, Gy, Gx2, Gy2, Fb,
             bx, by, bx2, by2)

    # ---- (A) accuracy join --------------------------------------------------------------------
    u = torch.nn.functional.normalize(M_tot, dim=1)
    pred = (u @ T.T).argmax(1)
    live = (S_tot > 1e-8) & has_gt
    correct = (pred == cell_gt) & live
    tot_lab = lab_w.sum(1).clamp_min(1e-12)
    top2b = lab_w.topk(2, dim=1).values
    share2 = top2b[:, 1] / tot_lab
    bimodal = live & (share2 >= 0.25)
    unan = live & (top2b[:, 1] <= 1e-8)

    def acc(m):
        n = int(m.sum())
        return (float(correct[m].float().mean()) if n else float("nan"), n)

    a_all, n_all = acc(live)
    a_bi, n_bi = acc(bimodal)
    a_un, n_un = acc(unan)
    err_all = (1 - a_all) * n_all
    err_bi = (1 - a_bi) * n_bi

    # ---- (B) triangulation: is within-view impurity spatially organised? ---------------------
    ratio = torch.cat(sep_num) if sep_num else torch.zeros(0, device=dev)
    null = torch.cat(null_num) if null_num else torch.zeros(0, device=dev)

    def qq(t):
        if t.numel() < 10:
            return None
        return [round(float(x), 3) for x in torch.quantile(
            t.double(), torch.tensor([.1, .25, .5, .75, .9], device=dev, dtype=torch.float64))]

    res = {"scene": a.scene, "views": a.views, "P": P, "n_live": n_all,
           "acc_all": a_all, "acc_bimodal": a_bi, "n_bimodal": n_bi,
           "acc_unanimous": a_un, "n_unanimous": n_un,
           "err_share_of_bimodal": float(err_bi / max(err_all, 1e-9)),
           "cell_share_of_bimodal": float(n_bi / max(n_all, 1)),
           "n_impure_pairs": n_impure_pairs,
           "r2_q": qq(ratio), "null_r2_q": qq(null),
           "r2_median": float(ratio.median()) if ratio.numel() else None,
           "null_r2_median": float(null.median()) if null.numel() else None,
           "frac_r2_gt_0p5": float((ratio > 0.5).float().mean()) if ratio.numel() else None,
           "frac_null_gt_0p5": float((null > 0.5).float().mean()) if null.numel() else None}
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)

    print(f"[{a.scene}] live cells {n_all:,}", flush=True)
    print(f"  (A) accuracy   all {a_all*100:.2f}%   unanimous {a_un*100:.2f}% (n={n_un:,})   "
          f"BIMODAL {a_bi*100:.2f}% (n={n_bi:,})", flush=True)
    print(f"      bimodal cells are {res['cell_share_of_bimodal']*100:.1f}% of cells but "
          f"{res['err_share_of_bimodal']*100:.1f}% of ERRORS", flush=True)
    print(f"  (B) within-view impure pairs {n_impure_pairs:,}", flush=True)
    print(f"      feature-vs-position R2 (10/25/50/75/90) {res['r2_q']}   median {res['r2_median']}",
          flush=True)
    print(f"      NULL (shuffled pixels)  (10/25/50/75/90) {res['null_r2_q']}   median "
          f"{res['null_r2_median']}", flush=True)
    print(f"      frac R2>0.5: {res['frac_r2_gt_0p5']}   null: {res['frac_null_gt_0p5']}",
          flush=True)


if __name__ == "__main__":
    main()
