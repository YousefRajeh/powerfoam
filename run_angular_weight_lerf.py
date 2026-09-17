"""Angular-diversity weighting on LERF-OVS, where the capture actually provides angular diversity.

THE IDEA. A cell's observations are not independent votes. Measured: ScanNet gives an angular
effective count of 1.01 from 6.6 views (7.4 deg mean pairwise baseline), LERF on a full orbit gives
2.51 from 96.6 views (48.9 deg). So on ScanNet there is literally nothing to reweight -- every view
is the same viewpoint -- while on LERF a cell has roughly 2-3 genuinely distinct viewpoints hiding
inside ~100 near-duplicates. Naive weighting lets a dense cluster of near-identical views outvote a
distinct one, which matters because contamination is angularly CLUSTERED (occluders block a range of
angles, measured: contaminated pairs sit 1.53 deg closer together than mixed pairs).

THE WEIGHT. For cell j with unit viewing directions u_v, define the local angular density with a
von Mises-Fisher kernel and divide it out:

    rho_v = sum_{v'} exp( kappa (u_v . u_v' - 1) )        kappa ~ 20 halves the kernel at ~15 deg
    w'_v  = w_v / rho_v

A view sitting alone on the far side keeps its full weight; one of twenty near-duplicates keeps
about a twentieth. This is inverse-density (Horvitz-Thompson style) reweighting, not a heuristic:
it estimates the same quantity while removing the sampling bias of where the camera happened to
dwell. It composes directly with the L1 theory because that bound's eps is a WEIGHT fraction.

SCORING. 3D mIoU against the back-projected LERF GT (unproject_lerf_gt.py). That GT is derived from
foam's own depth, so it is not an independent accuracy measurement -- but every arm here is scored
against the identical GT with the identical ownership, so the COMPARISON between estimators is
sound, which is what this run is for.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from pla_multiscene import group_dominant  # noqa: E402
from run_loo_reweight_eval import seg_median, seg_sum  # noqa: E402


def angular_density(DIR, start, cnt, kappa, chunk=4000):
    """Per-observation von Mises-Fisher density among its own cell's viewing directions.

    Chunked over cells. The unchunked version allocates sum(cnt^2) index pairs at once -- on
    teatime that is ~116M pairs of int64 on top of a 7 GB embedding array, which killed the run
    silently. Memory here is bounded by chunk * max(cnt)^2 regardless of scene size.
    """
    dens = np.zeros(len(DIR))
    for lo in range(0, len(cnt), chunk):
        hi = min(lo + chunk, len(cnt))
        c = cnt[lo:hi]
        st = start[lo:hi]
        npair = c * c
        tot = int(npair.sum())
        if tot == 0:
            continue
        pstart = np.r_[0, np.cumsum(npair)[:-1]]
        off = np.arange(tot) - np.repeat(pstart, npair)
        n_rep = np.repeat(c, npair)
        i = np.repeat(st, npair) + off // n_rep
        j = np.repeat(st, npair) + off % n_rep
        k = np.exp(kappa * (np.einsum("ij,ij->i", DIR[i], DIR[j]) - 1.0))
        np.add.at(dens, i, k)
    return dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="teatime")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--max-views", type=int, default=60)
    ap.add_argument("--kappas", nargs="*", type=float, default=[10.0, 20.0, 40.0])
    ap.add_argument("--features", default="artifacts/lerf_ovs/{scene}/openclip_features_sam")
    ap.add_argument("--out", default="artifacts/angular_weight_lerf.json")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from accumulate_hard_mask import load_masks
    from build_true_facet_graph import load_points_radii
    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import embed_class_names
    from oracle_labels import oracle_labels
    from point_cloud_query import assign_points_to_power_cells
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/lerf_ovs_{a.scene}"
    feat_dir = a.features.format(scene=a.scene)
    gtz = np.load(f"artifacts/lerf_gt3d/{a.scene}_gt3d.npz", allow_pickle=True)
    id2name = json.loads(str(gtz["class_id_to_name"].item()))
    names = [id2name[str(i)] for i in range(1, len(id2name) + 1)]
    gpts = gtz["points"].astype(np.float64)
    glab = gtz["labels"].astype(np.int64)
    K = len(names)
    print(f"{a.scene}: {len(gpts):,} GT points, {K} classes", flush=True)

    wp.init()
    pr = configargparse.ArgParser(); add_group(pr, Params)
    pr.add_argument("-c", "--config", is_config_file=True)
    cargs = pr.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs); dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]
    cc, rr = load_points_radii(ck)
    centres = np.asarray(cc, np.float64); radii = np.asarray(rr, np.float64)

    cell_lab, st = oracle_labels(centres, radii, gpts, glab, K + 1)
    keep_cell = cell_lab > 0
    print(f"{P:,} cells, {int(keep_cell.sum()):,} own GT ({st['vote_purity']:.3f} purity)",
          flush=True)
    owner = np.asarray(assign_points_to_power_cells(gpts, centres, radii, valid=None, k=8))

    from featurefoam_lerf_bridge import load_manifest_index
    midx = load_manifest_index(a.scene)
    inv = {v: k for k, v in midx.items()}
    ids = list(range(0, len(dh.cameras),
                     max(1, len(dh.cameras) // a.max_views)))[:a.max_views]

    CID, EMB, WT, DIR = [], [], [], []
    for k in ids:
        cam = dh.cameras[k]
        stem = Path(inv[k]).stem if k in inv else None
        if stem is None:
            continue
        fp = Path(feat_dir) / f"{stem}_f.npy"
        if not fp.exists():
            continue
        H, W = int(cam.height), int(cam.width)
        fmask, seg = load_masks(feat_dir, stem, a.level, H, W)
        fmask = fmask.numpy().astype(np.float32); seg = seg.reshape(-1).numpy()
        M = int(seg.max()) + 1
        if M <= 0:
            continue
        op = export_operator_for_views(model, [cam], [k])
        rws = op.row_indices.cpu().numpy(); cls = op.col_indices.cpu().numpy()
        vls = op.values.cpu().numpy().astype(np.float64)
        mm = seg[rws]; ok = (mm >= 0) & keep_cell[cls]
        c_, v_, m_ = cls[ok], vls[ok], mm[ok]
        if len(c_) == 0:
            continue
        dom, best, tot, nmk = group_dominant(c_, m_, v_, P)
        pres = (tot > 1e-9) & keep_cell
        ci = np.where(pres)[0]
        eye = cam.eye.detach().cpu().numpy().astype(np.float64).reshape(3)
        d = eye[None, :] - centres[ci]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        CID.append(ci); EMB.append(fmask[dom[pres]]); WT.append(tot[pres]); DIR.append(d)
        if k % 20 == 0:
            print(f"  view {k}: {len(ci):,} obs", flush=True)

    CID = np.concatenate(CID); EMB = np.concatenate(EMB).astype(np.float32)
    WT = np.concatenate(WT); DIR = np.concatenate(DIR)
    o = np.argsort(CID, kind="stable")
    CID, EMB, WT, DIR = CID[o], EMB[o], WT[o], DIR[o]
    uc, start = np.unique(CID, return_index=True)
    cnt = np.diff(np.r_[start, len(CID)])
    gi = np.repeat(np.arange(len(uc)), cnt)
    print(f"\n{len(CID):,} observations over {len(uc):,} cells "
          f"({cnt.mean():.1f} views/cell)", flush=True)

    # angular effective count, and the inverse-density weights
    S3 = seg_sum(DIR, start)
    neff = cnt.astype(float) ** 2 / np.maximum((S3 ** 2).sum(1), 1e-12)
    print(f"angular n_eff: mean {neff.mean():.2f}  median {np.median(neff):.2f}", flush=True)

    T = Fn.normalize(embed_class_names(names, "cuda").float(), dim=-1).cpu().numpy()

    def score(feat):
        X = np.zeros((P, feat.shape[1]), np.float32)
        X[uc] = feat
        n = np.linalg.norm(X, axis=1)
        pc = np.zeros(P, np.int64)
        live = n > 1e-8
        pc[live] = (X[live] / n[live][:, None] @ T.T).argmax(1) + 1
        pred = np.where(owner >= 0, pc[np.clip(owner, 0, P - 1)], 0)
        ious, accs = [], []
        for c in range(1, K + 1):
            g = glab == c
            if not g.any():
                continue
            p = pred == c
            i_ = float((g & p).sum()); u_ = float((g | p).sum())
            ious.append(i_ / u_ if u_ else 0.0); accs.append(i_ / float(g.sum()))
        return float(np.mean(ious) * 100), float(np.mean(accs) * 100)

    rec = {"scene": a.scene, "n_obs": int(len(CID)), "n_cells": int(len(uc)),
           "views_per_cell": float(cnt.mean()), "neff": float(neff.mean())}
    mean_feat = seg_sum(EMB * WT[:, None].astype(np.float32), start) / \
        np.maximum(seg_sum(WT, start), 1e-12)[:, None]
    rec["mean_mIoU"], rec["mean_mAcc"] = score(mean_feat)
    rec["median_mIoU"], rec["median_mAcc"] = score(seg_median(EMB, WT, start, gi, len(uc)))
    for kap in a.kappas:
        dv = angular_density(DIR, start, cnt, kap)
        w_ang = WT / np.maximum(dv, 1e-9)
        rec[f"ang{kap:g}_mIoU"], rec[f"ang{kap:g}_mAcc"] = score(
            seg_median(EMB, w_ang, start, gi, len(uc)))
        rec[f"ang{kap:g}_wratio"] = float((w_ang / np.maximum(WT, 1e-12)).std())
    print("\n" + "  ".join(f"{k[:-5]} {rec[k]:.2f}" for k in rec if k.endswith("_mIoU")))

    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    rows = [r for r in rows if r.get("scene") != a.scene] + [rec]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
