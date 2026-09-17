"""PLA's detector vs ours, at the two working min_cluster_size values, across all 10 ScanNet scenes.

SPEED. The sweep implementation re-exported the render operator once per min_cluster_size, which is
pure waste: the operator depends on geometry and cameras only, never on the clustering. It also built
a dense (P x M) weight table per view -- 51,610 x ~300 float64 = 124 MB allocated and arg-maxed per
view, to read out one number per cell. Both are fixed here:

  * ONE operator export per (scene, view), reused for the contamination ground truth, our geometric
    detectors, and every mcs value at once;
  * per-(cell, mask) aggregation via a COMPACT unique-key groupby over the ~2-3 M nonzeros instead
    of a P*M dense array, so cost scales with observations rather than with cells x masks.

Everything else is unchanged, so the numbers remain comparable to the single-scene sweep.

WHY THESE TWO VALUES. On 3DGS only mcs = 100 and 200 satisfied the condition our L1 bound requires
of any filter -- keep_dirty < keep_clean, hence eps actually falls. Their published default (500)
did not, on either representation. Running just those two across all scenes tests whether that
narrow window is a property of the method or an accident of one scene.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def group_dominant(cols, masks, vals, P):
    """Per cell: dominant mask, its weight share, total weight, and #masks touched.

    Compact groupby over the nonzeros -- no (P x M) allocation. Returns arrays of length P with
    zeros where the cell is absent from this view.
    """
    if len(cols) == 0:
        z = np.zeros(P)
        return z.astype(np.int64), z, z, z
    key = cols.astype(np.int64) * (int(masks.max()) + 2) + masks
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=vals)
    base = int(masks.max()) + 2
    ucell, umask = uk // base, uk % base
    order = np.lexsort((-w, ucell))                      # per cell, heaviest mask first
    uc_s, um_s, w_s = ucell[order], umask[order], w[order]
    first = np.ones(len(uc_s), bool)
    first[1:] = uc_s[1:] != uc_s[:-1]
    dom = np.zeros(P, np.int64)
    best = np.zeros(P)
    dom[uc_s[first]] = um_s[first]
    best[uc_s[first]] = w_s[first]
    tot = np.bincount(cols, weights=vals, minlength=P)
    nmask = np.bincount(ucell, minlength=P).astype(np.float64)
    return dom, best, tot, nmask


def top_cell_per_pixel(rows, cols, vals, npx):
    """Arg-max cell per pixel -- the arg-max of PLA's projected one-hot cluster image."""
    out = np.full(npx, -1, np.int64)
    if len(rows) == 0:
        return out
    order = np.lexsort((-vals, rows))
    r_s, c_s = rows[order], cols[order]
    first = np.ones(len(r_s), bool)
    first[1:] = r_s[1:] != r_s[:-1]
    out[r_s[first]] = c_s[first]
    return out


def auc(score, label):
    label = np.asarray(label, bool)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    if label.all() or (~label).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    a, b = int((~label).sum()), int(label.sum())
    return float((r[~label].sum() - a * (a + 1) / 2) / (a * b))


def within_auc(score, cell, cont):
    ok = ~np.isnan(score)
    score, cell, cont = score[ok], cell[ok], cont[ok]
    i = np.argsort(cell)
    score, cell, cont = score[i], cell[i], cont[i]
    b = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1], True])
    tot = conc = 0
    for k in range(len(b) - 1):
        lo, hi = b[k], b[k + 1]
        lab = cont[lo:hi]
        if lab.all() or (~lab).all():
            continue
        x, y = score[lo:hi][~lab], score[lo:hi][lab]
        tot += len(x) * len(y)
        conc += int((x[:, None] > y[None, :]).sum())
    return conc / max(tot, 1)


def cluster(feats, valid, mcs, n_comp=50):
    try:
        from cuml.cluster import HDBSCAN
        from cuml.decomposition import PCA
    except Exception:
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import PCA
    X = feats[valid].astype(np.float32)
    red = PCA(n_components=min(n_comp, X.shape[1])).fit_transform(X)
    lab = np.asarray(HDBSCAN(min_cluster_size=mcs).fit_predict(red)).astype(np.int32)
    out = np.full(len(feats), -1, np.int32)
    out[valid] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["foam", "gs"], required=True)
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--mcs", nargs="*", type=int, default=[100, 200])
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--vis-tol", type=float, default=0.05)
    ap.add_argument("--min-gt-per-mask", type=int, default=20)
    ap.add_argument("--out", default="artifacts/pla_multiscene_{arm}.json")
    a = ap.parse_args()

    from accumulate_hard_mask import load_masks
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    enable_determinism()
    rows_out = []

    for scene in a.scenes:
        feat_dir = f"data/scannet/{scene}_colmap/openclip_features_sam_blackboth"
        if not os.path.isdir(feat_dir):
            print(f"[miss] {scene} features"); continue
        pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
        K = len(names)
        P3 = np.asarray(pts, np.float64)

        # ---- representation-specific setup: model, features, ownership, per-view render ---------
        if a.arm == "foam":
            import configargparse
            import warp as wp
            from build_true_facet_graph import load_points_radii
            from configs import Params, add_group
            from data_loader import DataHandler
            from eval_surface_chamfer import cos_map
            from oracle_labels import oracle_labels
            from powerfoam.feature_operator import export_operator_for_views
            from powerfoam.rasterize import VisOptions
            from powerfoam.scene import PowerfoamScene

            ck = f"output/scannet_{scene}_truefrozen"
            fp = f"artifacts/scannet/{scene}/solved_geometric_median_truefrozen_ogl3.pt"
            if not (os.path.isdir(ck) and os.path.exists(fp)):
                print(f"[miss] {scene} foam"); continue
            wp.init()
            pr = configargparse.ArgParser(); add_group(pr, Params)
            pr.add_argument("-c", "--config", is_config_file=True)
            cargs = pr.parse_args(["-c", f"{ck}/config.yaml"])
            dh = DataHandler(cargs); dh.reload("all", downsample=cargs.downsample[-1])
            model = PowerfoamScene(cargs)
            model.initialize_from_dataset(dh, device="cuda")
            model.load_pt(f"{ck}/model.pt"); model.update_vis_cache()
            P = model.points.shape[0]
            cc, rr = load_points_radii(ck)
            cell_lab, _ = oracle_labels(np.asarray(cc, np.float64), np.asarray(rr, np.float64),
                                        pts, gt_lab, K + 1)
            vis = VisOptions(); vis.transmittance_threshold = 1e-3
            vis.max_intersections = 1024; vis.depth_quantile = 0.5
            vis.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)
            cams = list(dh.cameras)
            stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())

            def render(k):
                out = model.forward_visualization(cams[k], render_mode="rasterize",
                                                  vis_options=vis)
                d_ = out[1].detach().float().cpu().numpy()
                al = out[3].detach().float().cpu().numpy()
                d_ = d_[..., 0] if d_.ndim == 3 else d_
                al = al[..., 0] if al.ndim == 3 else al
                prm = cams[k].to_open3d()
                return (d_ * cos_map(cams[k]), al, np.asarray(prm.extrinsic, np.float64),
                        np.asarray(prm.intrinsic.intrinsic_matrix, np.float64))

            def operator(k):
                op = export_operator_for_views(model, [cams[k]], [k])
                return (op.row_indices.cpu().numpy(), op.col_indices.cpu().numpy(),
                        op.values.cpu().numpy().astype(np.float64))
            nviews = len(cams)
            HW = [(int(c.height), int(c.width)) for c in cams]
        else:
            from gsplat import rasterization
            from export_gsplat_operator import export_view_operator
            from oracle_labels import oracle_labels_nearest

            ckp = f"recon_remote/gs_froz/{scene}/ckpt.pt"
            fp = f"artifacts/scannet/{scene}/solved_geometric_median_gs_froz_ogl3.pt"
            camf = f"artifacts/participation/{scene}_cams_all.npz"
            if not (os.path.exists(ckp) and os.path.exists(fp) and os.path.exists(camf)):
                print(f"[miss] {scene} gs (need cams_all npz)"); continue
            sp = torch.load(ckp, map_location="cuda", weights_only=False)
            sp = sp["splats"] if "splats" in sp else sp
            means, quats = sp["means"], sp["quats"]
            scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]).reshape(-1)
            colors = sp["sh0"].reshape(len(means), 3)
            P = means.shape[0]
            cell_lab, _ = oracle_labels_nearest(means.detach().cpu().numpy().astype(np.float64),
                                                pts, gt_lab, K + 1)
            cz = np.load(camf)
            Kt = torch.as_tensor(cz["K"], dtype=torch.float32, device="cuda")
            vmt = torch.as_tensor(cz["viewmats"], dtype=torch.float32, device="cuda")
            Kn = np.asarray(cz["K"], np.float64)
            Wg, Hg = (int(x) for x in cz["wh"])
            stems = sorted(p.stem for p in
                           (Path(feat_dir).parent / "images").iterdir())
            nviews = vmt.shape[0]
            HW = [(Hg, Wg)] * nviews

            def render(k):
                with torch.no_grad():
                    rc, ra, _ = rasterization(means=means, quats=quats, scales=scales,
                                              opacities=opac, colors=colors,
                                              viewmats=vmt[k][None], Ks=Kt[None],
                                              width=Wg, height=Hg, sh_degree=None,
                                              render_mode="RGB+ED", packed=False)
                return (rc[0, ..., -1].cpu().numpy(), ra[0, ..., 0].cpu().numpy(),
                        np.asarray(cz["viewmats"][k], np.float64), Kn)

            def operator(k):
                r, c, v, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                                     vmt[k], Kt, Wg, Hg, max_hits_per_pixel=64)
                return (r.cpu().numpy(), c.cpu().numpy(), v.cpu().numpy().astype(np.float64))

        d = torch.load(fp, map_location="cpu", weights_only=True)
        F = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1).numpy()
        vmask = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(P, bool)
        clus = {m: cluster(F, vmask, m) for m in a.mcs}
        for m in a.mcs:
            print(f"[{scene}/{a.arm}] mcs={m}: {int(clus[m].max())+1} clusters, "
                  f"noise {(clus[m] < 0).mean():.1%}", flush=True)

        S, W, NM, CONT, CID = [], [], [], [], []
        TR = {m: [] for m in a.mcs}
        for k in range(nviews):
            H, Wd = HW[k]
            _, seg = load_masks(feat_dir, stems[k], a.level, H, Wd)
            seg = seg.reshape(-1).numpy()
            M = int(seg.max()) + 1
            if M <= 0:
                continue
            z_img, alp, extr, Kk = render(k)
            rows, cols, vals = operator(k)          # ONE export, reused by everything below

            pc = P3 @ extr[:3, :3].T + extr[:3, 3]
            z = pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = Kk[0, 0] * pc[:, 0] / z + Kk[0, 2]
                v = Kk[1, 1] * pc[:, 1] / z + Kk[1, 2]
            ui, vj = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            okp = (z > 1e-3) & (ui >= 0) & (ui < Wd) & (vj >= 0) & (vj < H) & (gt_lab > 0)
            idx = np.where(okp)[0]
            pix = vj[idx] * Wd + ui[idx]
            seen = (alp.reshape(-1)[pix] >= 0.5) & \
                   (np.abs(z_img.reshape(-1)[pix] - z[idx]) <= a.vis_tol)
            idx, pix = idx[seen], pix[seen]
            mid = seg[pix]
            g = mid >= 0
            hist = np.zeros((M, K + 1), np.int64)
            np.add.at(hist, (mid[g], gt_lab[idx][g]), 1)
            mask_lab = hist.argmax(1)
            mask_lab[hist.sum(1) < a.min_gt_per_mask] = -1

            mm = seg[rows]
            keep = mm >= 0
            r_, c_, v_, m_ = rows[keep], cols[keep], vals[keep], mm[keep]
            if len(c_) == 0:
                continue
            dom, best, tot, nmk = group_dominant(c_, m_, v_, P)
            pres = (tot > 1e-9) & (cell_lab > 0) & (mask_lab[dom] >= 0)
            if not pres.any():
                continue
            S.append(best[pres] / tot[pres]); W.append(tot[pres]); NM.append(nmk[pres])
            CONT.append(mask_lab[dom[pres]] != cell_lab[pres]); CID.append(np.where(pres)[0])

            topc = top_cell_per_pixel(r_, c_, v_, H * Wd)
            for mval in a.mcs:
                cl = clus[mval]
                nc = int(cl.max()) + 1
                pl = np.where(topc >= 0, cl[np.clip(topc, 0, P - 1)], -1)
                gg = (seg >= 0) & (pl >= 0)
                h = np.zeros((M, max(nc, 1)), np.int64)
                np.add.at(h, (seg[gg], pl[gg]), 1)
                dm = h.argmax(1); inter = h.max(1).astype(np.float64)
                ma = np.bincount(seg[seg >= 0], minlength=M).astype(np.float64)
                ca = np.bincount(pl[pl >= 0], minlength=max(nc, 1)).astype(np.float64)
                un = ma + ca[dm] - inter
                tr = np.where(un > 0, inter / np.maximum(un, 1e-9), 0.0)
                tr[h.sum(1) == 0] = 0.0
                TR[mval].append(tr[dom[pres]])

        if not S:
            print(f"[skip] {scene}: nothing judgeable"); continue
        S = np.concatenate(S); W = np.concatenate(W); NM = np.concatenate(NM)
        CONT = np.concatenate(CONT); CID = np.concatenate(CID)
        det = S * W / np.maximum(NM, 1)
        rec = {"scene": scene, "arm": a.arm, "n_obs": int(len(CONT)),
               "eps": float(CONT.mean()),
               "ours_pooled": auc(det, CONT), "ours_within": within_auc(det, CID, CONT)}
        for mval in a.mcs:
            T = np.concatenate(TR[mval])
            kp = T > 0.75
            kc = (~CONT & kp).sum() / max((~CONT).sum(), 1)
            kd = (CONT & kp).sum() / max(CONT.sum(), 1)
            rec[f"pla{mval}_pooled"] = auc(T, CONT)
            rec[f"pla{mval}_within"] = within_auc(T, CID, CONT)
            rec[f"pla{mval}_eps_after"] = float(CONT[kp].mean()) if kp.any() else float("nan")
            rec[f"pla{mval}_ok"] = bool(kd < kc)
            rec[f"pla{mval}_keep"] = float(kp.mean())
        rows_out.append(rec)
        print(f"  {scene}: eps {rec['eps']:.4f}  ours {rec['ours_pooled']:.4f}/"
              f"{rec['ours_within']:.4f}  " +
              "  ".join(f"pla{m} {rec[f'pla{m}_pooled']:.4f} eps->{rec[f'pla{m}_eps_after']:.4f} "
                        f"{'OK' if rec[f'pla{m}_ok'] else 'BAD'}" for m in a.mcs), flush=True)
        out = a.out.format(arm=a.arm)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rows_out, open(out, "w"), indent=1)

    if rows_out:
        print(f"\n=== {a.arm}: {len(rows_out)} scenes ===")
        f = lambda k: np.mean([r[k] for r in rows_out])
        print(f"  eps {f('eps'):.4f}   ours pooled {f('ours_pooled'):.4f}  "
              f"within {f('ours_within'):.4f}")
        for m in a.mcs:
            nok = sum(r[f'pla{m}_ok'] for r in rows_out)
            print(f"  PLA mcs={m}: pooled {f(f'pla{m}_pooled'):.4f}  within {f(f'pla{m}_within'):.4f}"
                  f"  eps->{f(f'pla{m}_eps_after'):.4f}  helps in {nok}/{len(rows_out)} scenes")


if __name__ == "__main__":
    main()
