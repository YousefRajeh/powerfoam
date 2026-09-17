"""Reweight each cell's median by its leave-one-out cross-view residual, and score mIoU.

THE MECHANISM, and why this one is worth running. Every detector tried before described HOW a cell
is seen (mask share, rendering weight, masks touched) -- properties of the cell, not of one of its
views. They reached pooled AUC 0.73 but only 0.58 WITHIN a cell, and within-cell is the only axis a
per-cell estimator responds to, because between-cell differences cancel in the normalisation. The
leave-one-out residual asks instead whether THIS view agrees with the cell's other views, which is
within-cell by construction, and it measured 0.79 within-cell -- the first detector strong enough to
expect an effect on the estimate.

It is also the estimator's own influence function: down-weighting by a function of the residual is
exactly IRLS, so this is not a bolt-on filter but one step of the robust solve the L1 theory calls
for.

ESTIMATORS COMPARED, all on the identical observation set so nothing but the weighting differs:
  mean          weighted mean, then normalise (the extrinsic L2 estimator = Eq. 9 + normalisation)
  median        weighted geometric median (our current solver's target)
  trim-q        median after dropping each cell's worst q fraction by LOO residual
  soft-k        median with w *= exp(-(theta_loo / (k * s_j))^2), s_j the cell's median residual
                -- scale-adaptive, so a coherent cell is barely touched and a split one is cut hard

SCORING is the published protocol: classes present in the scene, bare arg-max, opacity mask, GT
points assigned by exact power-cell ownership. Only the per-cell feature changes between arms.
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

from pla_multiscene import SPLIT, POINTCEPT, group_dominant  # noqa: E402


def seg_sum(arr, start):
    """Segment sums over data already sorted by segment. reduceat is ~50x faster than add.at."""
    return np.add.reduceat(arr, start, axis=0)


def seg_median(emb, w, start, gi, ncell, iters=15, eps=1e-9):
    """Weighted geometric median per segment, Weiszfeld with a spherical renormalisation.

    Uses reduceat on cell-sorted data throughout; the previous np.add.at version dominated runtime
    (one scene took ~11 min, almost all of it inside add.at and per-cell Python loops).
    """
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    w = w.astype(np.float32)
    buf = np.empty_like(emb)                       # reused; avoids a 480 MB temporary per iteration
    np.multiply(emb, w[:, None], out=buf)
    num = seg_sum(buf, start)
    x = (num / np.maximum(np.linalg.norm(num, axis=1, keepdims=True), eps)).astype(np.float32)
    for _ in range(iters):
        # both emb rows and x rows are unit, so ||e - x||^2 = 2 - 2<e,x> -- no difference array
        dot = np.einsum("ij,ij->i", emb, x[gi], optimize=True)
        dist = np.sqrt(np.maximum(2.0 - 2.0 * dot, 0.0))
        u = (w / np.maximum(dist, 1e-6)).astype(np.float32)
        np.multiply(emb, u[:, None], out=buf)
        y = seg_sum(buf, start) / np.maximum(seg_sum(u, start), eps)[:, None]
        x = (y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), eps)).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--trims", nargs="*", type=float, default=[0.10, 0.25])
    ap.add_argument("--softs", nargs="*", type=float, default=[1.0, 2.0])
    ap.add_argument("--out", default="artifacts/loo_reweight_eval.json")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from accumulate_hard_mask import load_masks
    from build_true_facet_graph import load_points_radii
    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from point_cloud_query import assign_points_to_power_cells
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    # RESUME. Each scene's record is appended to the json as it completes, so a run that is killed
    # (as this one was, by GPU contention) can pick up where it stopped instead of redoing hours.
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["scene"] for r in rows}
    if done:
        print(f"resuming: {len(done)} scenes already done ({', '.join(sorted(done))})", flush=True)
    for scene in a.scenes:
        if scene in done:
            continue
        ck = f"output/scannet_{scene}_{a.variant}"
        feat_dir = a.features.format(scene=scene)
        if not (os.path.isdir(ck) and os.path.isdir(feat_dir)):
            print(f"[miss] {scene}"); continue
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
        centers = np.asarray(cc, np.float64); radii = np.asarray(rr, np.float64)
        sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        sigma = Fn.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
        alpha = 1.0 - np.exp(-np.maximum(sigma, 0.0) * 2.0 * radii)

        pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names]); nc = len(names) + 1
        T = Fn.normalize(embed_class_names(names, "cuda").float(), dim=-1).cpu().numpy()
        owner = np.asarray(assign_points_to_power_cells(pts, centers, radii, valid=None, k=8))
        stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())

        CID, EMB, WT = [], [], []
        for k, cam in enumerate(dh.cameras):
            H, Wd = int(cam.height), int(cam.width)
            fmask, seg = load_masks(feat_dir, stems[k], a.level, H, Wd)
            fmask = fmask.numpy(); seg = seg.reshape(-1).numpy()
            M = int(seg.max()) + 1
            if M <= 0:
                continue
            op = export_operator_for_views(model, [cam], [k])
            rws = op.row_indices.cpu().numpy(); cls = op.col_indices.cpu().numpy()
            vls = op.values.cpu().numpy().astype(np.float64)
            mm = seg[rws]; keep = mm >= 0
            c_, v_, m_ = cls[keep], vls[keep], mm[keep]
            if len(c_) == 0:
                continue
            dom, best, tot, nmk = group_dominant(c_, m_, v_, P)
            pres = tot > 1e-9
            ci = np.where(pres)[0]
            CID.append(ci); EMB.append(fmask[dom[pres]].astype(np.float32)); WT.append(tot[pres])
        if not CID:
            print(f"[skip] {scene}"); continue
        CID = np.concatenate(CID); EMB = np.concatenate(EMB); WT = np.concatenate(WT)
        o = np.argsort(CID, kind="stable")
        CID, EMB, WT = CID[o], EMB[o], WT[o]
        uc, start = np.unique(CID, return_index=True)
        cnt = np.diff(np.r_[start, len(CID)])
        gi = np.repeat(np.arange(len(uc)), cnt)

        # leave-one-out residual against the cell's other views
        S = np.add.reduceat(EMB.astype(np.float64) * WT[:, None], start, axis=0)
        Wsum = np.add.reduceat(WT, start)
        loo = (S[gi] - EMB * WT[:, None]) / np.maximum(Wsum[gi] - WT, 1e-12)[:, None]
        nrm = np.linalg.norm(loo, axis=1)
        th = np.full(len(CID), 0.0)
        good = (cnt[gi] > 1) & (nrm > 1e-8)
        th[good] = np.arccos(np.clip((EMB[good] * (loo[good] / nrm[good][:, None])).sum(1), -1, 1))

        def score(feat_cells):
            X = np.zeros((P, EMB.shape[1]), np.float32)
            X[uc] = feat_cells
            n = np.linalg.norm(X, axis=1)
            pc = np.zeros(P, np.int64)
            live = n > 1e-8
            pc[live] = (X[live] / n[live][:, None] @ T.T).argmax(1) + 1
            pc[alpha < a.alpha] = 0
            pred = np.where(owner >= 0, pc[np.clip(owner, 0, P - 1)], 0)
            ious, accs = [], []
            for c in range(1, nc):
                g = gt_lab == c
                if not g.any():
                    continue
                p = pred == c
                i_ = float((g & p).sum()); u_ = float((g | p).sum())
                ious.append(i_ / u_ if u_ else 0.0); accs.append(i_ / float(g.sum()))
            return float(np.mean(ious) * 100), float(np.mean(accs) * 100)

        rec = {"scene": scene, "n_cells": int(len(uc)), "n_obs": int(len(CID))}
        mean_feat = S / np.maximum(Wsum, 1e-12)[:, None]
        rec["mean_mIoU"], rec["mean_mAcc"] = score(mean_feat)
        rec["median_mIoU"], rec["median_mAcc"] = score(seg_median(EMB, WT, start, gi, len(uc)))
        # per-segment order by residual, computed once and reused by the scale and the trims
        pos_in_seg = np.arange(len(CID)) - start[gi]
        seg_order = np.lexsort((th, CID))                    # within each cell, ascending residual
        rank = np.empty(len(CID), np.int64)
        rank[seg_order] = pos_in_seg                          # rank of each obs inside its cell
        mid = start + cnt // 2
        s_j = np.maximum(th[seg_order][mid], 1e-3)            # per-cell median residual
        for q in a.trims:
            k_drop = np.floor(q * cnt).astype(np.int64)
            k_drop[cnt < 3] = 0
            keep_mask = rank < (cnt[gi] - k_drop[gi])          # drop the worst k by residual
            w2 = np.where(keep_mask, WT, 0.0)
            rec[f"trim{q}_mIoU"], rec[f"trim{q}_mAcc"] = score(
                seg_median(EMB, w2, start, gi, len(uc)))
        for kk in a.softs:
            w3 = WT * np.exp(-(th / (kk * s_j[gi])) ** 2)
            rec[f"soft{kk}_mIoU"], rec[f"soft{kk}_mAcc"] = score(
                seg_median(EMB, w3, start, gi, len(uc)))
        rows.append(rec)
        keys = [k for k in rec if k.endswith("_mIoU")]
        print(f"  {scene}: " + "  ".join(f"{k[:-5]} {rec[k]:.2f}" for k in keys), flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    if rows:
        keys = [k for k in rows[0] if k.endswith("_mIoU")]
        print(f"\n=== {len(rows)} scenes ===")
        base = np.array([r["median_mIoU"] for r in rows])
        print(f"{'arm':>12} {'mIoU':>7} {'d vs median':>12} {'SE':>6} {'win':>6}")
        for k in keys:
            v = np.array([r[k] for r in rows]); d = v - base
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            print(f"{k[:-5]:>12} {v.mean():7.2f} {d.mean():+12.2f} {se:6.2f} "
                  f"{int((d>0).sum()):>3d}/{len(d)}")


if __name__ == "__main__":
    main()
