"""Does the kappa reduction from ray-purity thresholding survive to mIoU?

WHAT THE BOUND SAYS AND WHAT IT DOES NOT. The exact identity makes the closed form's error a
co-visibility-weighted sum of disagreements, with coefficient kappa_j = 1 - (sum_i A_ij^2)/(sum_i
A_ij). Dropping rays that graze several primitives cuts kappa hard on foam -- median 0.2416 -> 0.0353
at p >= 0.9, and the provably-near-exact population (kappa < 0.05) grows 12.4% -> 55.7%. That is a
BIAS reduction. It is bought with VARIANCE: at that threshold 44% of rays survive and 6% of cells
lose support entirely. The bound predicts the direction of the first effect and says nothing about
the net, so the trade has to be measured.

Only DROPPING rays can move kappa. Scaling a row by p_i leaves its composition unchanged, and kappa
is a property of composition -- the apparent effect of soft weighting in the curve was an artefact of
breaking row-stochasticity, since kappa as written is not scale-invariant.

Cells left with no surviving ray get no feature and predict nothing (label 0), which is the honest
accounting: the threshold buys accuracy on the cells it keeps and pays for it in coverage.

Protocol as everywhere else: present classes, bare arg-max, opacity mask, exact power-cell ownership.
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
from run_loo_reweight_eval import seg_median, seg_sum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--features",
                    default=r"data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.0, 0.5, 0.7, 0.9])
    ap.add_argument("--out", default="artifacts/purity_miou.json")
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
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["scene"] for r in rows}
    taus = list(a.taus)

    for scene in a.scenes:
        if scene in done:
            print(f"[skip] {scene} already done"); continue
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
        centres = np.asarray(cc, np.float64); radii = np.asarray(rr, np.float64)
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
        owner = np.asarray(assign_points_to_power_cells(pts, centres, radii, valid=None, k=8))
        stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())

        acc = {t: {"cid": [], "emb": [], "wt": []} for t in taus}
        for k, cam in enumerate(dh.cameras):
            H, Wd = int(cam.height), int(cam.width)
            fmask, seg = load_masks(feat_dir, stems[k], a.level, H, Wd)
            fmask = fmask.numpy().astype(np.float32); seg = seg.reshape(-1).numpy()
            M = int(seg.max()) + 1
            if M <= 0:
                continue
            op = export_operator_for_views(model, [cam], [k])
            rws = op.row_indices.cpu().numpy(); cls = op.col_indices.cpu().numpy()
            vls = op.values.cpu().numpy().astype(np.float64)
            # per-ray purity, computed on the FULL row before any masking
            nr = int(rws.max()) + 1
            tot_r = np.bincount(rws, weights=vls, minlength=nr)
            sq_r = np.bincount(rws, weights=vls ** 2, minlength=nr)
            pur = np.zeros(nr)
            lv = tot_r > 1e-12
            pur[lv] = sq_r[lv] / (tot_r[lv] ** 2)
            mm = seg[rws]
            base_ok = mm >= 0
            for t in taus:
                sel = base_ok & (pur[rws] >= t)
                if not sel.any():
                    continue
                dom, best, tot, nmk = group_dominant(cls[sel], mm[sel], vls[sel], P)
                pres = tot > 1e-9
                ci = np.where(pres)[0]
                acc[t]["cid"].append(ci)
                acc[t]["emb"].append(fmask[dom[pres]])
                acc[t]["wt"].append(tot[pres])
            if k % 10 == 0:
                print(f"  view {k}: purity median {np.median(pur[lv]):.3f}", flush=True)

        def score(feat, uc):
            X = np.zeros((P, feat.shape[1]), np.float32)
            X[uc] = feat
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

        rec = {"scene": scene}
        for t in taus:
            if not acc[t]["cid"]:
                continue
            CID = np.concatenate(acc[t]["cid"]); EMB = np.concatenate(acc[t]["emb"])
            WT = np.concatenate(acc[t]["wt"])
            o = np.argsort(CID, kind="stable")
            CID, EMB, WT = CID[o], EMB[o], WT[o]
            uc, start = np.unique(CID, return_index=True)
            cnt = np.diff(np.r_[start, len(CID)])
            gi = np.repeat(np.arange(len(uc)), cnt)
            mfeat = seg_sum(EMB * WT[:, None].astype(np.float32), start) / \
                np.maximum(seg_sum(WT, start), 1e-12)[:, None]
            rec[f"p{t:g}_mean_mIoU"], _ = score(mfeat, uc)
            rec[f"p{t:g}_med_mIoU"], rec[f"p{t:g}_med_mAcc"] = score(
                seg_median(EMB, WT, start, gi, len(uc)), uc)
            rec[f"p{t:g}_cells"] = int(len(uc))
            rec[f"p{t:g}_obs"] = int(len(CID))
        rows.append(rec)
        base_cells = rec.get("p0_cells", 1)
        print(f"\n{scene}")
        print(f"{'threshold':>10} {'cells':>9} {'cell%':>7} {'obs':>10} "
              f"{'mean mIoU':>10} {'median mIoU':>12}")
        for t in taus:
            if f"p{t:g}_cells" not in rec:
                continue
            print(f"{f'p>={t:g}':>10} {rec[f'p{t:g}_cells']:>9,} "
                  f"{rec[f'p{t:g}_cells']/base_cells:>7.1%} {rec[f'p{t:g}_obs']:>10,} "
                  f"{rec[f'p{t:g}_mean_mIoU']:>10.2f} {rec[f'p{t:g}_med_mIoU']:>12.2f}")
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    if len(rows) > 1:
        print(f"\n=== {len(rows)} scenes ===")
        print(f"{'threshold':>10} {'mean mIoU':>10} {'median mIoU':>12} {'d vs p>=0':>10}")
        base = np.mean([r["p0_med_mIoU"] for r in rows if "p0_med_mIoU" in r])
        for t in taus:
            k = f"p{t:g}_med_mIoU"
            v = [r[k] for r in rows if k in r]
            if v:
                print(f"{f'p>={t:g}':>10} "
                      f"{np.mean([r[f'p{t:g}_mean_mIoU'] for r in rows if f'p{t:g}_mean_mIoU' in r]):>10.2f} "
                      f"{np.mean(v):>12.2f} {np.mean(v)-base:>+10.2f}")


if __name__ == "__main__":
    main()
