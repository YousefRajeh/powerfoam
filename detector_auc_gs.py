"""Contamination ground truth and our geometric detectors, for the 3DGS arm.

Mirror of detector_auc.py on Gaussians, so PLA's trust scores from the 3DGS sweep can be scored
against the same definition of "contaminated" used on foam:

    observation (gaussian j, view v) is CONTAMINATED iff the SAM mask it predominantly reads is
    dominated by GT points of a different class than gaussian j's own GT label.

Two things necessarily differ from the foam script, and both are forced by the representation:
  * ownership is NEAREST CENTRE -- a Gaussian mixture has no disjoint partition to ask "which
    primitive contains this point";
  * visibility uses gsplat's own expected depth (RGB+ED), not the foam renderer's, so each
    representation decides what it considers visible.
Cameras come from the shared bridge npz, so the rays are identical to the foam run.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0062_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--vis-tol", type=float, default=0.05)
    ap.add_argument("--min-gt-per-mask", type=int, default=20)
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--out", default="artifacts/detector_auc_gs.npz")
    a = ap.parse_args()

    from gsplat import rasterization

    from accumulate_hard_mask import load_masks
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from export_gsplat_operator import export_view_operator
    from oracle_labels import oracle_labels_nearest

    enable_determinism()
    ck = f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt"
    sp = torch.load(ck, map_location="cuda", weights_only=False)
    sp = sp["splats"] if "splats" in sp else sp
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]).reshape(-1)
    colors = sp["sh0"].reshape(len(means), 3)
    P = means.shape[0]

    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    K_ = len(names)
    cell_lab, _ = oracle_labels_nearest(means.detach().cpu().numpy().astype(np.float64),
                                        pts, gt_lab, K_ + 1)
    print(f"{P:,} gaussians, {int((cell_lab>0).sum()):,} with a GT label")

    cams = np.load(f"artifacts/participation/{a.scene}_cams_all.npz")
    Kk = torch.as_tensor(cams["K"], dtype=torch.float32, device="cuda")
    vms = torch.as_tensor(cams["viewmats"], dtype=torch.float32, device="cuda")
    view_ids = cams["view_ids"]
    W_, H_ = (int(x) for x in cams["wh"])
    Kn = np.asarray(cams["K"], np.float64)
    stems = sorted(p.stem for p in
                   (Path(a.features.format(scene=a.scene)).parent / "images").iterdir())
    feat_dir = a.features.format(scene=a.scene)

    P3 = np.asarray(pts, np.float64)
    S, W, NM, AR, CONT, CID, VID = [], [], [], [], [], [], []
    for k in range(vms.shape[0]):
        vi = int(view_ids[k])
        _, seg = load_masks(feat_dir, stems[vi], a.level, H_, W_)
        seg = seg.reshape(-1).numpy()
        M = int(seg.max()) + 1
        if M <= 0:
            continue
        with torch.no_grad():
            rc, ra, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                                      colors=colors, viewmats=vms[k][None], Ks=Kk[None],
                                      width=W_, height=H_, sh_degree=None,
                                      render_mode="RGB+ED", packed=False)
        z_img = rc[0, ..., -1].cpu().numpy()          # expected depth, camera-space z
        alp = ra[0, ..., 0].cpu().numpy()

        extr = np.asarray(cams["viewmats"][k], np.float64)
        pc = P3 @ extr[:3, :3].T + extr[:3, 3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = Kn[0, 0] * pc[:, 0] / z + Kn[0, 2]
            v = Kn[1, 1] * pc[:, 1] / z + Kn[1, 2]
        ui, vj = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        okp = (z > 1e-3) & (ui >= 0) & (ui < W_) & (vj >= 0) & (vj < H_) & (gt_lab > 0)
        idx = np.where(okp)[0]
        pix = vj[idx] * W_ + ui[idx]
        seen = (alp.reshape(-1)[pix] >= 0.5) & (np.abs(z_img.reshape(-1)[pix] - z[idx]) <= a.vis_tol)
        idx, pix = idx[seen], pix[seen]
        mid = seg[pix]
        good = mid >= 0
        hist = np.zeros((M, K_ + 1), np.int64)
        np.add.at(hist, (mid[good], gt_lab[idx][good]), 1)
        mask_lab = hist.argmax(1)
        mask_lab[hist.sum(1) < a.min_gt_per_mask] = -1

        r, c, val, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                               vms[k], Kk, W_, H_, max_hits_per_pixel=a.max_hits)
        rows = r.cpu().numpy(); cols = c.cpu().numpy(); vals = val.cpu().numpy().astype(np.float64)
        mm = seg[rows]
        keep = mm >= 0
        cols, vals, mm = cols[keep], vals[keep], mm[keep]
        if len(cols) == 0:
            continue
        Wv = np.bincount(cols, weights=vals, minlength=P)
        key = cols.astype(np.int64) * M + mm
        h = np.bincount(key, weights=vals, minlength=P * M).reshape(P, M)
        best_m = h.argmax(1); best_w = h.max(1)
        nmask = (h > 1e-9).sum(1)
        area = np.bincount(cols, minlength=P).astype(np.float64)
        pres = (Wv > 1e-9) & (cell_lab > 0) & (mask_lab[best_m] >= 0)
        if not pres.any():
            continue
        S.append(best_w[pres] / Wv[pres]); W.append(Wv[pres]); NM.append(nmask[pres])
        AR.append(area[pres]); CONT.append(mask_lab[best_m[pres]] != cell_lab[pres])
        CID.append(np.where(pres)[0]); VID.append(np.full(int(pres.sum()), vi))
        print(f"  view {vi}: {int(pres.sum()):,} obs, {CONT[-1].mean():.1%} contaminated",
              flush=True)

    S = np.concatenate(S); W = np.concatenate(W); NM = np.concatenate(NM)
    AR = np.concatenate(AR); CONT = np.concatenate(CONT)
    CID = np.concatenate(CID); VID = np.concatenate(VID)
    print(f"\n{len(S):,} judgeable (gaussian,view) observations; eps = {CONT.mean():.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, share=S.astype(np.float32), weight=W.astype(np.float32),
                        nmask=NM.astype(np.int32), area=AR.astype(np.float32), cont=CONT,
                        cell=CID.astype(np.int32), view=VID.astype(np.int32))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
