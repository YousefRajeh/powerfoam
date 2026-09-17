"""Judge the MASK, not the observation: pooling contamination evidence across the cells a mask covers.

WHY THIS IS A DIFFERENT STATISTICAL REGIME. Every detector so far scores one (cell, view) observation
in isolation, and a cell has a median of ~5 views, so each judgement rests on ~5 samples. But a bad
SAM mask corrupts EVERY cell it covers at once -- hundreds to thousands of them. Scoring reliability
at the mask level pools that evidence, taking the per-judgement sample size from ~5 to ~10^3. That is
not a better weighting scheme; it is a better-conditioned estimator of the same latent quantity.

THE STATISTIC, which the LOO machinery already gives us. For observation (j, v) reading mask m, the
leave-one-out residual theta_loo is the angle between the mask's embedding f_m and the consensus of
cell j's OTHER views. So cos(theta_loo) is exactly "does this mask agree with what the rest of the
scene says about this cell". Averaging that over all cells the mask covers:

    trust(v, m)  =  sum_{j in C_m} w_jv * cos(theta_loo(j,v))  /  sum_{j in C_m} w_jv

A mask covering a coherent object agrees with all its cells' consensus and scores high; a mask that
straddles two objects, or that a pose error has misplaced, disagrees with most of them and scores
low -- and it does so on evidence from every cell it touches, not from one.

WHAT IT IS COMPARED AGAINST. The per-observation LOO residual (within-cell AUC 0.7931, the best so
far), rendering weight (0.5797), and PLA's cluster pseudo-mask IoU (0.5273). PLA is the closest
relative: it also judges a mask by cross-view geometric consistency, but through HDBSCAN clusters
that leave 60-78% of primitives as noise. This uses exact power-cell ownership instead, so there is
no clustering step to fail.

A CAVEAT BUILT INTO THE MEASUREMENT. Since every observation of a mask inherits one trust value, this
detector cannot distinguish cells WITHIN a mask. Its within-cell AUC comes only from a cell's
different views reading different masks. That is a real limit and is reported rather than hidden: the
`n_cells_per_mask` column shows how much pooling each judgement actually gets.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from pla_multiscene import SPLIT, POINTCEPT, auc, group_dominant, within_auc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--vis-tol", type=float, default=0.05)
    ap.add_argument("--min-gt-per-mask", type=int, default=20)
    ap.add_argument("--out", default="artifacts/mask_level_detector.json")
    a = ap.parse_args()

    import configargparse
    import json
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
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["scene"] for r in rows}

    for scene in a.scenes:
        if scene in done:
            print(f"[skip] {scene}"); continue
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
        model.load_pt(f"{ck}/model.pt"); model.update_vis_cache()
        P = model.points.shape[0]
        vis = VisOptions(); vis.transmittance_threshold = 1e-3
        vis.max_intersections = 1024; vis.depth_quantile = 0.5
        vis.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

        pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names]); K = len(names)
        cc, rr = load_points_radii(ck)
        cell_lab, _ = oracle_labels(np.asarray(cc, np.float64), np.asarray(rr, np.float64),
                                    pts, gt_lab, K + 1)
        d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu",
                       weights_only=True)
        vmask = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(P, bool)
        stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())
        P3 = np.asarray(pts, np.float64)

        CID, VID, MID, EMB, WT, CONT = [], [], [], [], [], []
        for k, cam in enumerate(dh.cameras):
            H, Wd = int(cam.height), int(cam.width)
            fmask, seg = load_masks(feat_dir, stems[k], a.level, H, Wd)
            fmask = fmask.numpy().astype(np.float32); seg = seg.reshape(-1).numpy()
            M = int(seg.max()) + 1
            if M <= 0:
                continue
            with torch.no_grad():
                out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis)
            dep = out[1].detach().float().cpu().numpy(); alp = out[3].detach().float().cpu().numpy()
            dep = dep[..., 0] if dep.ndim == 3 else dep
            alp = alp[..., 0] if alp.ndim == 3 else alp
            z_img = dep * cos_map(cam)
            prm = cam.to_open3d()
            extr = np.asarray(prm.extrinsic, np.float64)
            Kk = np.asarray(prm.intrinsic.intrinsic_matrix, np.float64)
            pc = P3 @ extr[:3, :3].T + extr[:3, 3]; z = pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = Kk[0, 0] * pc[:, 0] / z + Kk[0, 2]
                v = Kk[1, 1] * pc[:, 1] / z + Kk[1, 2]
            ui, vj = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            okp = (z > 1e-3) & (ui >= 0) & (ui < Wd) & (vj >= 0) & (vj < H) & (gt_lab > 0)
            idx = np.where(okp)[0]; pix = vj[idx] * Wd + ui[idx]
            seen = (alp.reshape(-1)[pix] >= 0.5) & \
                   (np.abs(z_img.reshape(-1)[pix] - z[idx]) <= a.vis_tol)
            idx, pix = idx[seen], pix[seen]
            mid = seg[pix]; g = mid >= 0
            hist = np.zeros((M, K + 1), np.int64)
            np.add.at(hist, (mid[g], gt_lab[idx][g]), 1)
            mask_lab = hist.argmax(1); mask_lab[hist.sum(1) < a.min_gt_per_mask] = -1

            op = export_operator_for_views(model, [cam], [k])
            rws = op.row_indices.cpu().numpy(); cls = op.col_indices.cpu().numpy()
            vls = op.values.cpu().numpy().astype(np.float64)
            mm = seg[rws]; keep = mm >= 0
            c_, v_, m_ = cls[keep], vls[keep], mm[keep]
            if len(c_) == 0:
                continue
            dom, best, tot, nmk = group_dominant(c_, m_, v_, P)
            pres = (tot > 1e-9) & vmask & (cell_lab > 0) & (mask_lab[dom] >= 0)
            if not pres.any():
                continue
            ci = np.where(pres)[0]
            CID.append(ci); VID.append(np.full(len(ci), k)); MID.append(dom[pres])
            EMB.append(fmask[dom[pres]]); WT.append(tot[pres])
            CONT.append(mask_lab[dom[pres]] != cell_lab[pres])
            if k % 10 == 0:
                print(f"  view {k}: {len(ci):,} obs, {M} masks", flush=True)

        CID = np.concatenate(CID); VID = np.concatenate(VID); MID = np.concatenate(MID)
        EMB = np.concatenate(EMB); WT = np.concatenate(WT); CONT = np.concatenate(CONT)
        o = np.argsort(CID, kind="stable")
        CID, VID, MID, EMB, WT, CONT = (x[o] for x in (CID, VID, MID, EMB, WT, CONT))
        uc, start = np.unique(CID, return_index=True)
        cnt = np.diff(np.r_[start, len(CID)])
        gi = np.repeat(np.arange(len(uc)), cnt)

        # leave-one-out consensus per cell, and the per-observation agreement cos(theta_loo)
        S = np.add.reduceat(EMB.astype(np.float64) * WT[:, None], start, axis=0)
        Wsum = np.add.reduceat(WT, start)
        loo = (S[gi] - EMB * WT[:, None]) / np.maximum(Wsum[gi] - WT, 1e-12)[:, None]
        nrm = np.linalg.norm(loo, axis=1)
        good = (cnt[gi] > 1) & (nrm > 1e-8)
        cosl = np.zeros(len(CID))
        cosl[good] = np.clip((EMB[good] * (loo[good] / nrm[good][:, None])).sum(1), -1, 1)
        theta_loo = np.where(good, np.arccos(cosl), np.nan)

        # ---- the mask-level statistic: pool agreement over every cell the mask covers -----------
        key = VID.astype(np.int64) * 100000 + MID.astype(np.int64)
        uk, inv = np.unique(key[good], return_inverse=True)
        num = np.bincount(inv, weights=(WT[good] * cosl[good]))
        den = np.bincount(inv, weights=WT[good])
        ncell = np.bincount(inv)
        trust_of_mask = num / np.maximum(den, 1e-12)
        lut = {int(k_): i for i, k_ in enumerate(uk)}
        sel = np.array([lut.get(int(k_), -1) for k_ in key])
        has = sel >= 0
        mask_trust = np.where(has, trust_of_mask[np.clip(sel, 0, len(uk) - 1)], np.nan)
        mask_n = np.where(has, ncell[np.clip(sel, 0, len(uk) - 1)], 0)

        print(f"\n{scene}: {len(CID):,} obs, eps {CONT.mean():.4f}, "
              f"{len(uk):,} (view,mask) pairs")
        print(f"  cells pooled per mask: median {np.median(ncell):.0f}  "
              f"mean {ncell.mean():.1f}  p90 {np.percentile(ncell, 90):.0f}")
        v = ~np.isnan(theta_loo)
        rec = {"scene": scene, "n_obs": int(len(CID)), "eps": float(CONT.mean()),
               "n_masks": int(len(uk)), "cells_per_mask": float(ncell.mean())}
        print(f"\n{'detector':>34} {'pooled':>8} {'within':>8}")
        for nm, sc, m_ in [("LOO residual  -theta_loo", -theta_loo, v),
                           ("MASK-LEVEL trust", mask_trust, v & has),
                           ("mask trust x per-obs cos", mask_trust * cosl, v & has),
                           ("rendering weight", WT, np.ones(len(WT), bool))]:
            p_ = auc(sc[m_], CONT[m_]); w_ = within_auc(sc[m_], CID[m_], CONT[m_])
            rec[nm] = {"pooled": p_, "within": w_}
            print(f"{nm:>34} {p_:8.4f} {w_:8.4f}")
        rows.append(rec)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    if len(rows) > 1:
        print(f"\n=== {len(rows)} scenes ===")
        keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        for k in keys:
            print(f"{k:>34} {np.mean([r[k]['pooled'] for r in rows]):8.4f} "
                  f"{np.mean([r[k]['within'] for r in rows]):8.4f}")


if __name__ == "__main__":
    main()
