"""Per-observation residuals: robust r0 for the certificate, and cross-view consistency as a detector.

ONE PASS, TWO RESULTS. Both open questions need the same quantity -- the geodesic residual of each
observation against its cell's own estimate, theta_jv = d(x_j, B_jv):

  1. ROBUST r0. The certificate's feature bound uses r0 = the radius of the CLEAN observations. We
     previously fed it the RMS angular spread (sqrt of the conflict statistic), which is computed
     over ALL observations including the contamination the bound is supposed to tolerate -- so the
     radius was inflated by exactly the thing it is meant to exclude, and the certificate came out
     vacuous. The right estimator is the (1-eps)-quantile of theta, which by construction covers the
     clean mass and ignores the eps tail.

  2. CROSS-VIEW CONSISTENCY. Every detector tried so far (mask share, rendering weight, masks
     touched) describes HOW a cell is seen, which is a property of the cell, not of one of its views
     -- hence pooled AUC 0.73 but within-cell only 0.59, and within-cell is what a per-cell
     estimator responds to. A residual is within-cell by construction: it asks whether THIS view
     agrees with the cell's other views. It is also not ad hoc -- it is the influence function of the
     estimator itself, i.e. what IRLS reweights on.

LEAVE-ONE-OUT, because it matters. Scoring an observation against an estimate it helped produce is
optimistic, and with a median of ~5 views per cell the self-influence is large. The residual here is
therefore computed against the mean of the cell's OTHER views, which is a genuine LOO statistic and
costs one vectorised subtraction.
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
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.1355)
    ap.add_argument("--vis-tol", type=float, default=0.05)
    ap.add_argument("--min-gt-per-mask", type=int, default=20)
    ap.add_argument("--out", default="artifacts/residuals_{scene}.npz")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from accumulate_hard_mask import load_masks
    from build_true_facet_graph import load_points_radii
    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from eval_surface_chamfer import cos_map
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from label_certificate import certified, delta_bound, normalised_margin
    from oracle_labels import oracle_labels
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.rasterize import VisOptions
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
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
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names]); K = len(names)
    cc, rr = load_points_radii(ck)
    cell_lab, _ = oracle_labels(np.asarray(cc, np.float64), np.asarray(rr, np.float64),
                                pts, gt_lab, K + 1)
    d = torch.load(f"artifacts/scannet/{a.scene}/{a.solved}", map_location="cpu",
                   weights_only=True)
    Xh = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1).numpy()
    vmask = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(P, bool)

    feat_dir = a.features.format(scene=a.scene)
    stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())
    P3 = np.asarray(pts, np.float64)

    CID, VID, TH, EMB, WT, CONT = [], [], [], [], [], []
    for k, cam in enumerate(dh.cameras):
        H, Wd = int(cam.height), int(cam.width)
        fmask, seg = load_masks(feat_dir, stems[k], a.level, H, Wd)
        fmask = fmask.numpy(); seg = seg.reshape(-1).numpy()
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
        rows = op.row_indices.cpu().numpy(); cols = op.col_indices.cpu().numpy()
        vals = op.values.cpu().numpy().astype(np.float64)
        mm = seg[rows]; keep = mm >= 0
        c_, v_, m_ = cols[keep], vals[keep], mm[keep]
        if len(c_) == 0:
            continue
        dom, best, tot, nmk = group_dominant(c_, m_, v_, P)
        pres = (tot > 1e-9) & vmask & (cell_lab > 0) & (mask_lab[dom] >= 0)
        if not pres.any():
            continue
        ci = np.where(pres)[0]
        f = fmask[dom[pres]]                                   # the observation this cell reads
        CID.append(ci); VID.append(np.full(len(ci), k)); EMB.append(f.astype(np.float32))
        WT.append(tot[pres]); CONT.append(mask_lab[dom[pres]] != cell_lab[pres])
        TH.append(np.arccos(np.clip((f * Xh[ci]).sum(1), -1, 1)))   # residual vs the estimate
        if k % 10 == 0:
            print(f"  view {k}: {len(ci):,} obs", flush=True)

    CID = np.concatenate(CID); VID = np.concatenate(VID)
    EMB = np.concatenate(EMB); WT = np.concatenate(WT)
    CONT = np.concatenate(CONT); TH = np.concatenate(TH)
    print(f"\n{len(CID):,} observations; eps = {CONT.mean():.4f}")

    # ---- leave-one-out residual: agreement with the cell's OTHER views ---------------------------
    order = np.argsort(CID, kind="stable")
    CID, VID, EMB, WT, CONT, TH = (x[order] for x in (CID, VID, EMB, WT, CONT, TH))
    uc, start = np.unique(CID, return_index=True)
    cnt = np.diff(np.r_[start, len(CID)])
    S = np.zeros((len(uc), EMB.shape[1]), np.float64)
    np.add.at(S, np.repeat(np.arange(len(uc)), cnt), EMB * WT[:, None])
    Wsum = np.zeros(len(uc)); np.add.at(Wsum, np.repeat(np.arange(len(uc)), cnt), WT)
    gi = np.repeat(np.arange(len(uc)), cnt)
    loo = S[gi] - EMB * WT[:, None]
    den = np.maximum(Wsum[gi] - WT, 1e-12)[:, None]
    loo = loo / den
    nrm = np.linalg.norm(loo, axis=1)
    ok_loo = (cnt[gi] > 1) & (nrm > 1e-8)
    theta_loo = np.full(len(CID), np.nan)
    theta_loo[ok_loo] = np.arccos(np.clip(
        (EMB[ok_loo] * (loo[ok_loo] / nrm[ok_loo][:, None])).sum(1), -1, 1))

    print(f"\n=== cross-view consistency as a detector ({int(ok_loo.sum()):,} obs with >1 view)")
    print(f"{'detector':>34} {'pooled':>8} {'within':>8}")
    print(f"{'residual vs own estimate  -theta':>34} {auc(-TH, CONT):8.4f} "
          f"{within_auc(-TH, CID, CONT):8.4f}")
    print(f"{'LEAVE-ONE-OUT residual  -theta_loo':>34} "
          f"{auc(-theta_loo[ok_loo], CONT[ok_loo]):8.4f} "
          f"{within_auc(-theta_loo[ok_loo], CID[ok_loo], CONT[ok_loo]):8.4f}")
    print(f"{'rendering weight (previous best)':>34} {auc(WT, CONT):8.4f} "
          f"{within_auc(WT, CID, CONT):8.4f}")

    # ---- robust r0 vs the RMS spread we used before ----------------------------------------------
    q = 1.0 - a.eps
    r_rms = np.zeros(P); r_rob = np.zeros(P); Dmax = np.zeros(P)
    for i, c in enumerate(uc):
        s, e = start[i], start[i] + cnt[i]
        t = TH[s:e]
        r_rms[c] = np.sqrt((t ** 2).mean())
        r_rob[c] = np.quantile(t, q)
        Dmax[c] = t.max()
    have = np.zeros(P, bool); have[uc] = True
    print(f"\n=== r0 estimators over {int(have.sum()):,} cells")
    print(f"  RMS spread (all observations) : median {np.median(r_rms[have]):.4f}")
    print(f"  robust {q:.3f}-quantile        : median {np.median(r_rob[have]):.4f}   "
          f"({np.median(r_rob[have])/max(np.median(r_rms[have]),1e-9):.2f}x)")
    print(f"  D = max residual              : median {np.median(Dmax[have]):.4f}")

    T = torch.nn.functional.normalize(embed_class_names(names, "cuda").float(),
                                      dim=-1).cpu().numpy()
    m, lab = normalised_margin(Xh, T)
    lab = lab + 1
    correct = lab == cell_lab
    base = have & (cell_lab > 0) & vmask
    print(f"\n=== certificate coverage ({int(base.sum()):,} scoreable cells), eps={a.eps}")
    for nm, r0v, Dv in [("RMS r0, D=pi", r_rms, np.full(P, np.pi)),
                        ("RMS r0, D measured", r_rms, Dmax),
                        ("ROBUST r0, D=pi", r_rob, np.full(P, np.pi)),
                        ("ROBUST r0, D measured", r_rob, Dmax)]:
        cert = np.zeros(P, bool)
        db = 2 * r0v + (a.eps / (1 - a.eps)) * Dv
        cert[base] = 2 * np.sin(np.clip(db[base], 0, np.pi) / 2) < m[base]
        n = int(cert.sum())
        acc_c = correct[cert].mean() if n else float("nan")
        acc_u = correct[base & ~cert].mean() if (base & ~cert).any() else float("nan")
        print(f"  {nm:24s} certified {n:>7,} ({n/max(base.sum(),1):6.2%})   "
              f"acc cert {acc_c:.4f}   acc uncert {acc_u:.4f}")

    out = a.out.format(scene=a.scene)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, cell=CID.astype(np.int32), view=VID.astype(np.int32),
                        theta=TH.astype(np.float32), theta_loo=theta_loo.astype(np.float32),
                        weight=WT.astype(np.float32), cont=CONT,
                        r_rms=r_rms.astype(np.float32), r_rob=r_rob.astype(np.float32),
                        Dmax=Dmax.astype(np.float32), margin=m.astype(np.float32))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
