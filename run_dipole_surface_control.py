"""Density-matched control for the dipole-surface row.

THE CONFOUND. Replacing the predicted side with the extracted dipole surface improves the
completeness direction a lot (`mae_gt2pred`: 17.55 -> 15.67 cm frozen, 14.06 -> 9.80 cm unfrozen).
But `mae_gt2pred(c)` is "distance from each true class-c point to the NEAREST predicted class-c
point", and that shrinks monotonically as the predicted set gets denser, for free. The dipole
surface carries 1.7M-7.0M samples against 81k-373k GT points -- a 5-50x density advantage. So the
improvement as measured is not attributable to the surface being better placed.

THE CONTROL. Subsample the extracted surface, PER CLASS, to exactly the number of points that
class had under the point-based prediction, then recompute. Any gain that survives is a gain from
WHERE the surface is, not from how many samples it has. Subsampling is uniform without replacement
at a fixed seed; classes where the surface has fewer samples than the point prediction are left
whole and counted, since padding them would invent geometry.

Reports all three arms side by side: point-based, surface, surface-density-matched.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import TAU, GTSurfaceIndex
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from run_dipole_surface import ARMS, DENSITY_THRESH, extract_surface
from run_percell_masked import OPACITY_THRESH, SPLIT, primitive_alpha

SCENES = list(SPLIT)


def pair_metrics(gpts, ppts, tau=TAU):
    d_p2g, _ = cKDTree(gpts).query(ppts, k=1, workers=-1)
    d_g2p, _ = cKDTree(ppts).query(gpts, k=1, workers=-1)
    prec, rec = float((d_p2g <= tau).mean()), float((d_g2p <= tau).mean())
    return {"mae_pred2gt": float(d_p2g.mean()), "mae_gt2pred": float(d_g2p.mean()),
            "scd": float((d_p2g.mean() + d_g2p.mean()) / 2),
            "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
            "boundary_f1": float(2 * prec * rec / max(prec + rec, 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(ARMS))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--out", default="artifacts/scannet/dipole_surface_control.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    rows = []

    for recon in a.recons.split(","):
        mtmpl, ftmpl = ARMS[recon]
        for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
            mp, fp = mtmpl.format(s=scene), ftmpl.format(s=scene)
            apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            if not all(os.path.exists(p) for p in (mp, fp, apth)):
                continue
            m = torch.load(mp, map_location="cpu", weights_only=False)
            dens = F.softplus(m["density"].float().to(dev), beta=100)
            d = torch.load(fp, map_location=dev, weights_only=True)
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].to(dev)
            unit = torch.zeros_like(feats)
            unit[valid] = F.normalize(feats[valid], dim=-1)
            live = (dens > DENSITY_THRESH) & valid
            surf_pts, surf_own, _ = extract_surface(m, live, a.grid, dev)
            del m

            alpha = primitive_alpha(recon, scene, 2.0)
            assign = np.load(apth)
            owned = assign >= 0
            gt_pts, raw, names_all = load_scannet_pointcept_gt(
                rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
            pts = np.asarray(gt_pts, dtype=np.float64)
            n2i = {n: q for q, n in enumerate(names_all)}
            present = set(np.unique(raw).tolist())
            low = np.zeros(len(pts), dtype=bool)
            low[owned] = alpha[assign[owned]] < OPACITY_THRESH

            names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            gt[low] = 0
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1
            index = GTSurfaceIndex(pts, gt, nc)

            # point-based prediction, exactly as the table's other rows build it
            sc = owned.copy()
            sc[owned] = valid.cpu().numpy()[assign[owned]]
            pred_pt = np.zeros(len(gt), dtype=np.int64)
            pred_pt[sc] = cls[assign[sc]]
            surf_cls = cls[surf_own]

            acc = {k: {"mae_pred2gt": [], "mae_gt2pred": [], "scd": [], "hd95": [],
                       "boundary_f1": []} for k in ("point", "surface", "matched")}
            n_short = 0
            for c in index.classes():
                gpts = index.gt_pts[c]
                pm_pt, pm_sf = pred_pt == c, surf_cls == c
                if pm_pt.sum() == 0 or pm_sf.sum() == 0:
                    continue                       # missed by one side: not a paired comparison
                p_pt, p_sf = pts[pm_pt], surf_pts[pm_sf]
                n_target = len(p_pt)
                if len(p_sf) > n_target:
                    p_mt = p_sf[rng.choice(len(p_sf), n_target, replace=False)]
                else:
                    p_mt, n_short = p_sf, n_short + 1
                for tag, pp in (("point", p_pt), ("surface", p_sf), ("matched", p_mt)):
                    for k, v in pair_metrics(gpts, pp).items():
                        acc[tag][k].append(v)

            rec = {"recon": recon, "scene": scene, "class_set": a.class_set,
                   "n_classes_paired": len(acc["point"]["scd"]), "n_short": n_short}
            for tag in acc:
                for k, v in acc[tag].items():
                    rec[f"{tag}_{k}"] = float(np.mean(v)) if v else None
            rows.append(rec)
            print(f"[{recon}/{scene}] classes={rec['n_classes_paired']} "
                  f"gt2pred point={rec['point_mae_gt2pred']*100:.2f} "
                  f"surf={rec['surface_mae_gt2pred']*100:.2f} "
                  f"matched={rec['matched_mae_gt2pred']*100:.2f} cm", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
