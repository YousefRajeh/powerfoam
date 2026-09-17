"""Partition the 58-point gap into buckets that map onto DIFFERENT fixes.

The ceiling says the partition supports 99.1% and we achieve ~41.5%. That gap is not one problem.
Each wrong cell falls into exactly one bucket, and each bucket is addressable by a different stage --
or by nothing we control:

  A. BIMODAL & WRONG        the cell's views disagree about its label.
                            -> a better ACCUMULATOR can arbitrate. This is the only bucket a
                               per-cell estimator can touch.

  B. UNANIMOUS & WRONG, neighbourhood right
                            every view agrees and is wrong, but the majority PREDICTION among the
                            cell's facet neighbours matches its GT.
                            -> no accumulator can help (there is no disagreement to resolve), but
                               POST-SOLVE SPATIAL REFINEMENT can: the correct answer is present in
                               the neighbourhood and only needs propagating.

  C. UNANIMOUS & WRONG, neighbourhood also wrong
                            the cell and its neighbours all predict the same wrong thing.
                            -> neither accumulator nor graph refinement can recover it. The 2D
                               features are wrong over a whole region. Only better 2D features (or
                               a different foundation model / prompt set) can move this bucket, and
                               it caps EVERY lifting method on this benchmark, not just ours.

Bucket B is measured as an ORACLE: it asks whether the information exists in the neighbourhood at
all, not whether any particular diffusion would find it. So B is an upper bound on what refinement
can win, and C is a hard floor on the error until the 2D features change.
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

OUT = "artifacts/scannet/error_budget"
BIMODAL_F = 0.25


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

    M_tot = torch.zeros((P, D), device=dev)
    S_tot = torch.zeros(P, device=dev)
    lab_w = torch.zeros((P, C_), device=dev)
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
        M_v = torch.zeros((P, D), device=dev).index_add_(0, ci, vv[:, None] * tab[sid])
        S_v = torch.zeros(P, device=dev).index_add_(0, ci, vv)
        M_tot += M_v
        S_tot += S_v
        seen = S_v > 1e-8
        lv = (torch.nn.functional.normalize(M_v, dim=1) @ T.T).argmax(1)
        lab_w[seen] = lab_w[seen].scatter_add(1, lv[seen][:, None], S_v[seen][:, None])
        del ri, ci, vv, seg, tab, sid, M_v, S_v

    live = (S_tot > 1e-8) & has_gt
    pred = (torch.nn.functional.normalize(M_tot, dim=1) @ T.T).argmax(1)
    wrong = live & (pred != cell_gt)
    tot_lab = lab_w.sum(1).clamp_min(1e-12)
    top2 = lab_w.topk(2, dim=1).values
    bimodal = live & ((top2[:, 1] / tot_lab) >= BIMODAL_F)
    unanimous = live & (top2[:, 1] <= 1e-8)

    # neighbourhood majority PREDICTION (excluding the cell itself)
    model.aabb_tree.update(model.points.detach(), model.get_radii().detach())
    adjacent, offsets = model.aabb_tree.build_cech_complex()
    adjacent = adjacent.long().to(dev)
    offsets = offsets.long().to(dev)
    src = torch.repeat_interleave(torch.arange(P, device=dev), (offsets[1:] - offsets[:-1]))
    dst = adjacent
    ok_e = live[src] & live[dst]
    nb_votes = torch.zeros((P, C_), device=dev)
    nb_votes.index_put_((src[ok_e], pred[dst[ok_e]]),
                        torch.ones(int(ok_e.sum()), device=dev), accumulate=True)
    has_nb = nb_votes.sum(1) > 0
    nb_pred = nb_votes.argmax(1)
    nb_right = has_nb & (nb_pred == cell_gt)

    n_live = int(live.sum())
    n_wrong = int(wrong.sum())
    A_ = wrong & bimodal
    rest = wrong & ~bimodal
    B_ = rest & nb_right
    C_bucket = rest & ~nb_right

    def sh(m):
        return int(m.sum()), float(m.sum()) / max(n_wrong, 1) * 100

    nA, pA = sh(A_)
    nB, pB = sh(B_)
    nC, pC = sh(C_bucket)
    res = {"scene": a.scene, "views": a.views, "n_live": n_live, "n_wrong": n_wrong,
           "accuracy": 1 - n_wrong / max(n_live, 1),
           "A_bimodal_wrong": nA, "A_pct_of_errors": pA,
           "B_unanimousish_wrong_nbr_right": nB, "B_pct_of_errors": pB,
           "C_unanimousish_wrong_nbr_wrong": nC, "C_pct_of_errors": pC,
           "acc_if_A_fixed": (n_live - n_wrong + nA) / max(n_live, 1),
           "acc_if_B_fixed": (n_live - n_wrong + nB) / max(n_live, 1),
           "acc_if_AB_fixed": (n_live - n_wrong + nA + nB) / max(n_live, 1),
           "unanimous_cells": int(unanimous.sum()),
           "unanimous_acc": float((live & unanimous & ~wrong).sum()) / max(int(unanimous.sum()), 1)}
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)
    print(f"[{a.scene}] live {n_live:,}  accuracy {res['accuracy']*100:.2f}%  "
          f"errors {n_wrong:,}", flush=True)
    print(f"  A  bimodal & wrong          {nA:7,d}  {pA:5.1f}% of errors  -> ACCUMULATOR",
          flush=True)
    print(f"  B  agreeing & wrong, nbr OK {nB:7,d}  {pB:5.1f}% of errors  -> SPATIAL REFINEMENT",
          flush=True)
    print(f"  C  agreeing & wrong, nbr XX {nC:7,d}  {pC:5.1f}% of errors  -> NEEDS BETTER 2D "
          f"FEATURES", flush=True)
    print(f"  oracle accuracy if A fixed {res['acc_if_A_fixed']*100:.2f}%   "
          f"if B fixed {res['acc_if_B_fixed']*100:.2f}%   "
          f"if both {res['acc_if_AB_fixed']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
