"""Does the feature gradient localise the TRUE semantic boundary inside a cell?

R^2 > 0.5 says the CLIP feature varies systematically across a cell's footprint. That is necessary
for sub-cell splitting to make sense but not sufficient: the gradient could track shading, texture
or mask-edge artefacts rather than the object boundary. This is the sufficiency test, and it is the
last thing standing between the idea and actually placing new sites.

METHOD. For each (cell, view) with a strong gradient:
  1. The direction of maximum feature change is the top left-singular vector of the 2xD matrix
     [bx; by] from the spatial regression -- equivalently the top eigenvector of the 2x2 Gram
     [[bx.bx, bx.by],[bx.by, by.by]], available in closed form as theta = atan2(2b, a-c)/2.
  2. Project the cell's OWN ScanNet GT points into that view and score each by
     s = (u - u_bar) cos(theta) + (v - v_bar) sin(theta).
  3. Ask how well s separates the cell's two dominant GT classes, by AUC -- rank-based, so no
     threshold is chosen and class imbalance does not flatter it.

AUC ~ 0.5  -> the gradient is unrelated to the semantic boundary. Splitting along it is arbitrary,
              and the R^2 signal was tracking something else.
AUC >> 0.5 -> the gradient IS the boundary. The split plane is then determined, not searched for,
              and a new site can be placed on it directly.

NULL: the identical AUC computed along a RANDOM image direction for the same points. Any competent
direction beats 0.5 slightly by chance on small samples; the null says how much.

Only cells whose GT genuinely spans two classes are scored -- on a cell that is entirely one class
there is no boundary to find and the AUC would be undefined.
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

OUT = "artifacts/scannet/split_validation"
R2_MIN = 0.5
MIN_PTS = 12          # per cell, total GT points
MIN_PER_CLASS = 4     # per each of the two dominant classes


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


def auc(scores, pos):
    """Rank-based AUC. scores/pos are 1-D tensors; pos is a boolean membership of class A."""
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = torch.argsort(torch.argsort(scores)).float() + 1.0
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    from camera_bridge import K_from_ray_dirs
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
    centres = model.points.detach().float().to(dev)
    radii = model.get_radii().detach().float().to(dev).reshape(-1)

    names = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    T = torch.nn.functional.normalize(embed_class_names(names, dev).float(), dim=-1)
    D = T.shape[1]

    for sub in ("train", "val", "test"):
        gt_dir = os.path.join(a.gt_root, sub, a.scene)
        if os.path.isdir(gt_dir):
            break
    gt_pts, gt_raw, _ = load_scannet_pointcept_gt(gt_dir)
    target_ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]
    gl_np = remap_gt_labels(gt_raw, target_ids) - 1
    owner_np = assign_points_to_power_cells(gt_pts.astype(np.float64),
                                            centres.detach().cpu().numpy(),
                                            radii.detach().cpu().numpy())
    keep = (owner_np >= 0) & (gl_np >= 0)
    gp = torch.from_numpy(gt_pts[keep]).float().to(dev)
    gl = torch.from_numpy(gl_np[keep]).long().to(dev)
    own = torch.from_numpy(owner_np[keep]).long().to(dev)
    print(f"  [gt] usable points {gp.shape[0]:,} over {int(own.unique().numel()):,} cells",
          flush=True)

    aucs, nulls, n_cand, n_scored = [], [], 0, 0
    rng = torch.Generator(device=dev).manual_seed(0)

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
        py = (ri // W_).float()
        px = (ri % W_).float()
        Wc = torch.zeros(P, device=dev).index_add_(0, ci, vv)
        Wn = Wc.clamp_min(1e-8)
        mx = (torch.zeros(P, device=dev).index_add_(0, ci, vv * px)) / Wn
        my = (torch.zeros(P, device=dev).index_add_(0, ci, vv * py)) / Wn
        dx, dy = px - mx[ci], py - my[ci]
        Sxx = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dx)
        Syy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dy * dy)
        Sxy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dy)
        Fb = (torch.zeros((P, D), device=dev).index_add_(0, ci, vv[:, None] * tab[sid])) / Wn[:, None]
        Gx = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dx)[:, None] * tab[sid])
        Gy = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dy)[:, None] * tab[sid])
        det = (Sxx * Syy - Sxy * Sxy)
        bx = (Syy[:, None] * Gx - Sxy[:, None] * Gy) / det.clamp_min(1e-12)[:, None]
        by = (Sxx[:, None] * Gy - Sxy[:, None] * Gx) / det.clamp_min(1e-12)[:, None]
        expl = (bx * Gx).sum(1) + (by * Gy).sum(1)
        total = Wc * (1.0 - (Fb * Fb).sum(1)).clamp_min(0)
        r2 = torch.where(total > 1e-9, (expl / total.clamp_min(1e-12)).clamp(0, 1),
                         torch.zeros_like(total))
        strong = (Wc > 1e-6) & (det > 1e-6) & (r2 > R2_MIN)
        if not bool(strong.any()):
            del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by, Fb
            continue

        # principal direction of feature change: top eigenvector of the 2x2 Gram of [bx; by]
        aa = (bx * bx).sum(1)
        bb = (bx * by).sum(1)
        cc = (by * by).sum(1)
        th = 0.5 * torch.atan2(2 * bb, (aa - cc))
        gxv, gyv = torch.cos(th), torch.sin(th)

        # project GT points into this view once
        K, _ = K_from_ray_dirs(cam)
        K = K.to(dev).float()
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :4] = dh.c2ws[vi].double()
        w2c = torch.linalg.inv(c2w).float().to(dev)
        Xc = (w2c[:3, :3] @ gp.T + w2c[:3, 3:4]).T
        z = Xc[:, 2]
        front = z > 1e-4
        uu = torch.full_like(z, float("nan"))
        vvp = torch.full_like(z, float("nan"))
        uu[front] = K[0, 0] * Xc[front, 0] / z[front] + K[0, 2]
        vvp[front] = K[1, 1] * Xc[front, 1] / z[front] + K[1, 2]
        inimg = front & (uu >= 0) & (uu < W_) & (vvp >= 0) & (vvp < H_ if False else vvp < H)

        cand = torch.nonzero(strong, as_tuple=True)[0]
        n_cand += int(cand.numel())
        cand_set = torch.zeros(P, dtype=torch.bool, device=dev)
        cand_set[cand] = True
        sel_pt = cand_set[own] & inimg
        if not bool(sel_pt.any()):
            del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by, Fb
            continue
        o_s = own[sel_pt]
        l_s = gl[sel_pt]
        u_s = uu[sel_pt]
        v_s = vvp[sel_pt]
        s_s = (u_s - mx[o_s]) * gxv[o_s] + (v_s - my[o_s]) * gyv[o_s]
        ang = torch.rand(1, generator=rng, device=dev) * 3.14159265
        s_n = (u_s - mx[o_s]) * torch.cos(ang) + (v_s - my[o_s]) * torch.sin(ang)

        for j in torch.unique(o_s).tolist():
            m = o_s == j
            if int(m.sum()) < MIN_PTS:
                continue
            lj = l_s[m]
            uq, cn = torch.unique(lj, return_counts=True)
            if uq.numel() < 2:
                continue
            top = torch.argsort(cn, descending=True)[:2]
            if int(cn[top[1]]) < MIN_PER_CLASS or int(cn[top[0]]) < MIN_PER_CLASS:
                continue
            keep2 = (lj == uq[top[0]]) | (lj == uq[top[1]])
            pos = lj[keep2] == uq[top[0]]
            A1 = auc(s_s[m][keep2], pos)
            A0 = auc(s_n[m][keep2], pos)
            if A1 is None or A0 is None:
                continue
            aucs.append(max(A1, 1 - A1))        # direction sign is arbitrary; fold to >= 0.5
            nulls.append(max(A0, 1 - A0))
            n_scored += 1
        del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by, Fb

    A = torch.tensor(aucs)
    N = torch.tensor(nulls)
    res = {"scene": a.scene, "views": a.views, "r2_min": R2_MIN,
           "n_strong_cellviews": n_cand, "n_scored": n_scored,
           "auc_mean": float(A.mean()) if A.numel() else None,
           "auc_median": float(A.median()) if A.numel() else None,
           "null_mean": float(N.mean()) if N.numel() else None,
           "null_median": float(N.median()) if N.numel() else None,
           "frac_auc_gt_0p7": float((A > 0.7).float().mean()) if A.numel() else None,
           "frac_null_gt_0p7": float((N > 0.7).float().mean()) if N.numel() else None}
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)
    print(f"[{a.scene}] strong (cell,view) {n_cand:,}   scored against GT {n_scored:,}", flush=True)
    if A.numel():
        print(f"  AUC  gradient  mean {res['auc_mean']:.4f}  median {res['auc_median']:.4f}  "
              f"frac>0.7 {res['frac_auc_gt_0p7']*100:.1f}%", flush=True)
        print(f"  AUC  RANDOM    mean {res['null_mean']:.4f}  median {res['null_median']:.4f}  "
              f"frac>0.7 {res['frac_null_gt_0p7']*100:.1f}%", flush=True)
    else:
        print("  no cells met the GT criteria", flush=True)


if __name__ == "__main__":
    main()
