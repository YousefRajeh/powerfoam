"""Surface metrics for the CROSS-RECONSTRUCTION diffusion arms (all representations).

WHY THIS BLOCK FIRST. `diffusion_cross_recon.json` is the only 10-scene source covering every
representation, so it is what the BEST sheet needs in order to put surface quality beside the
headline mIoU for PowerFoam, RadFoam and 3DGS on equal terms.

SCOPE, and what is deliberately excluded. Only arms with >= 10 scenes are computed, per the
standing rule that a sub-10-scene number is a pilot. That excludes rf_froz / rf_unfroz here:
diffusion_cross_recon only ever covered 4 scenes for those two (their solved features live on
the remote box), so their 48 rows stay blank rather than being filled with a 4-scene number
that would sit next to 10-scene ones in the same column.

ASSIGNMENT PROTOCOL. This uses the cached `geometric` assignment (nearest primitive regardless
of whether it carries a feature) for every arm, and tags the rows accordingly. The
`nearest_valid` variant is NOT computed here: it is defined by
assign_points_to_power_cells(valid=...), which is a power-diagram query and has no meaning for
Gaussians, whose assignment is exact Mahalanobis. Mixing the two would compare representations
under different protocols -- exactly the confound the `assignment` column exists to prevent.

The graph comes from the cached Delaunay CSR rather than model.pt, since Gaussian arms have no
adjacency stored in their checkpoint.
"""
import argparse
import os
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

DB = "artifacts/ablation.sqlite"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
SCENES = list(SPLIT)
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

# feature paths, verbatim from run_diffusion_cross_recon.py so the arms line up
FEATURES = {
    "pf_nonfroz": "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt",
    "pf_tfroz":   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt",
    "gs_froz":    "artifacts/scannet/{s}/solved_weighted_gs_froz_ogl3.pt",
    "gs_unfroz":  "artifacts/scannet/{s}/solved_weighted_gs_unfroz_ogl3.pt",
}
METHOD = {"base": "percell-argmax",
          "diffused": "diffusion(truefacet,s1000_a0.9)"}


def diffuse(p0, adjacent, offsets, n, alpha=0.9, iters=60):
    dev = p0.device
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    src = torch.repeat_interleave(torch.arange(n, device=dev), offsets[1:] - offsets[:-1])
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[adjacent])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(FEATURES))
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    con = sqlite3.connect(DB, timeout=120)
    written, skipped = 0, []

    for recon in a.recons.split(","):
        for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
            fp = FEATURES[recon].format(s=scene)
            apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            gp = f"artifacts/ablation_cache/{scene}_{recon}_delaunay.pt"
            if not all(os.path.exists(p) for p in (fp, apth, gp)):
                skipped.append(f"{recon}/{scene}: missing artifact")
                continue
            t0 = time.time()
            d = torch.load(fp, map_location=dev, weights_only=True)
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].cpu().numpy()
            vt = torch.from_numpy(valid).to(dev)
            unit = torch.zeros_like(feats)
            unit[vt] = F.normalize(feats[vt], dim=-1)
            n_prim = feats.shape[0]
            g = torch.load(gp, map_location="cpu", weights_only=True)
            adjacent = g["adjacent"].to(dev).long()
            offsets = g["offsets"].to(dev).long()
            if len(offsets) - 1 != n_prim:
                skipped.append(f"{recon}/{scene}: graph {len(offsets)-1} vs feats {n_prim}")
                continue
            assign = np.load(apth)
            owned = assign >= 0

            gt_pts, raw, names_all = load_scannet_pointcept_gt(
                rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
            pts = np.asarray(gt_pts, dtype=np.float64)
            n2i = {n: q for q, n in enumerate(names_all)}
            present = set(np.unique(raw).tolist())

            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                index = GTSurfaceIndex(pts, gt, nc)
                sim = unit @ text.T
                p0 = torch.softmax(1000.0 * sim, dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, adjacent, offsets, n_prim)

                for tag, cls, live in (("base", sim.argmax(-1).cpu().numpy() + 1, valid),
                                       ("diffused", pd.argmax(-1).cpu().numpy() + 1,
                                        (pd.sum(-1) > 0).cpu().numpy())):
                    sc = owned.copy()
                    sc[owned] = live[assign[owned]]
                    pred = np.zeros(len(gt), dtype=np.int64)
                    pred[sc] = cls[assign[sc]]
                    sm = semantic_surface_metrics(index, pred)
                    _, miou, _, macc = calculate_metrics(
                        torch.from_numpy(gt).long(), torch.from_numpy(pred).long(), nc)
                    con.execute(
                        "INSERT OR IGNORE INTO results_unified (scene,recon,features,solver,"
                        "method,family,class_set,n_classes,miou,macc,masked,scd,mae_pred2gt,"
                        "mae_gt2pred,hd95,boundary_f1,n_missed,assignment,source,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (scene, recon, "ogl3", "geometric_median", METHOD[tag],
                         "diffusion" if tag == "diffused" else "percell", cs, nc - 1,
                         float(miou) * 100, float(macc) * 100, 0,
                         sm.get("scd"), sm.get("mae_pred2gt"), sm.get("mae_gt2pred"),
                         sm.get("hd95"), sm.get("boundary_f1"), sm.get("n_missed"),
                         "geometric", "backfill_surface_cross_recon.py",
                         time.strftime("%Y-%m-%d %H:%M:%S")))
                    written += 1
            con.commit()
            print(f"  [{recon}/{scene}] {n_prim:,} prims {(time.time()-t0)/60:.1f} min",
                  flush=True)

    print(f"\nwrote {written} rows")
    if skipped:
        print(f"SKIPPED {len(skipped)} (logged, never silent):")
        for s in skipped[:12]:
            print("  ", s)
    print(f"\n{'recon':<13}{'method':<34}{'mIoU':>7}{'scd cm':>9}{'hd95 cm':>9}{'bF1':>7}{'n':>4}")
    for r in con.execute(
            "SELECT recon, method, ROUND(AVG(miou),2), ROUND(AVG(scd)*100,2), "
            "ROUND(AVG(hd95)*100,2), ROUND(AVG(boundary_f1),3), COUNT(DISTINCT scene) "
            "FROM results_unified WHERE source='backfill_surface_cross_recon.py' "
            "AND class_set='opengaussian19' GROUP BY recon, method "
            "ORDER BY recon, AVG(miou) DESC").fetchall():
        print(f"{r[0]:<13}{r[1]:<34}{r[2]:>7.2f}{r[3]:>9.2f}{r[4]:>9.2f}{r[5]:>7.3f}{r[6]:>4}")
    con.close()


if __name__ == "__main__":
    main()
