"""Compute semantic-surface metrics for the diffusion / mode-vote arms and write them to the DB.

WHY. `results_unified` carries scd / mae_pred2gt / mae_gt2pred / hd95 / boundary_f1 / n_missed
for the 5,130 rows that came from the original ablation table, but the JSON-sourced rows
(mode-voting, posterior diffusion, the stack) have those cells BLANK -- including for our best
method, modevote(truefacet)+diffusion(truefacet,s1000,a0.9). So the results sheet cannot show
surface quality next to the headline number for the very row that matters most.

WHAT THESE ADD OVER mIoU. mIoU counts points and cannot tell "the predicted region is slightly
the wrong size" from "the predicted region is on the far side of the room". The surface metrics
measure DISTANCE between predicted and true regions of the same cloud, so a low scd with a
mediocre mIoU means the errors are boundary slop rather than teleported regions. That is
exactly the question to ask of a SMOOTHING method: diffusion could plausibly raise mIoU while
smearing regions across boundaries, and scd is what would reveal it.

The metric code is imported from ablation_surface.py rather than reimplemented, so these
numbers are directly comparable with the ones already in the table (that module's definitions
are verified elementwise against eval_semantic_surface.py in test_ablation_surface.py).

METHOD NAMES written here match the backfill exactly, so the rows join rather than duplicate.
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
from point_cloud_query import assign_points_to_power_cells
from feature_foam_lifting.operator import AccumulatedFeatureStats
from run_normlift_refine_eval import mode_vote_refine

DB = "artifacts/ablation.sqlite"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
          "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

DIFF = "diffusion(truefacet,s1000_a0.9)"          # exact strings used by the backfill
METHODS = {"percell": "percell-argmax",
           "modevote": "modevote(truefacet)",
           "diffusion": DIFF,
           "modevote+diffusion": f"modevote(truefacet)+{DIFF}"}


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
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recon", default="pf_nonfroz")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    con = sqlite3.connect(DB)
    written = 0

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy"
        if not all(os.path.exists(p) for p in (mp, fp, stp, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        gt_pts_pre, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = P.shape[0]
        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        # TWO assignment protocols, scored side by side so the choice can be made later by
        # filtering rather than by rerunning:
        #   geometric      nearest cell regardless of feature (ablation_assign.py, valid=None),
        #                  then points whose owner is featureless are dropped -> ~88% coverage
        #   nearest_valid  nearest cell THAT HAS A FEATURE (what run_simplex_diffusion_eval.py
        #                  does, valid=valid_mask) -> every owned point classifiable, ~100%
        # Worth ~4 mIoU on scene0347; not a bug in either, but they are different tasks.
        nv_cache = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if os.path.exists(nv_cache):
            assign_nv = np.load(nv_cache)
        else:
            centers = m["points"].float().numpy().astype(np.float64)
            radii_np = F.softplus(m["radii"].float().squeeze(), beta=100).numpy().astype(np.float64)
            assign_nv = np.asarray(assign_points_to_power_cells(
                gt_pts_pre, centers, radii_np, valid=valid, k=64))
            np.save(nv_cache, assign_nv)
        ASSIGNMENTS = {"geometric": np.load(apth), "nearest_valid": assign_nv}
        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        refined = mode_vote_refine(unit, R, P, adjacent, offsets)
        pts = np.asarray(gt_pts_pre, dtype=np.float64)
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            index = GTSurfaceIndex(pts, gt, nc)          # built ONCE per (scene, class set)

            preds = {}
            for tag, u in (("percell", unit), ("modevote", refined)):
                sim = u @ text.T
                preds[tag] = (sim.argmax(-1).cpu().numpy() + 1, valid.copy())
                p0 = torch.softmax(1000.0 * sim, dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, adjacent, offsets, n_prim)
                key = "diffusion" if tag == "percell" else "modevote+diffusion"
                preds[key] = (pd.argmax(-1).cpu().numpy() + 1,
                              (pd.sum(-1) > 0).cpu().numpy())

            for (aname, assign), (tag, (cls, live)) in [
                    (av, pv) for av in ASSIGNMENTS.items() for pv in preds.items()]:
                owned = assign >= 0
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                sm = semantic_surface_metrics(index, pred)
                _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                                     torch.from_numpy(pred).long(), nc)
                method = METHODS[tag]
                # ALWAYS insert under THIS script's own source, never UPDATE rows owned by
                # another source. An earlier version updated on (scene, recon, method,
                # class_set) with no source predicate and stamped one set of surface metrics
                # onto 192 rows across 9 sources whose mIoU differs from what was measured
                # here -- this scorer does not reproduce simplex_*'s numbers exactly (e.g.
                # scene0347 diffusion 46.10 here vs 47.47 there), so the metrics belong only
                # to the predictions this script actually computed.
                if True:
                    con.execute(
                        "INSERT OR IGNORE INTO results_unified (scene,recon,features,solver,"
                        "method,family,class_set,n_classes,miou,macc,masked,scd,mae_pred2gt,"
                        "mae_gt2pred,hd95,boundary_f1,n_missed,assignment,source,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (scene, a.recon, "ogl3", "geometric_median", method,
                         "modevote+diffusion" if "+" in method else
                         ("diffusion" if "diffusion" in method else
                          ("modevote" if "modevote" in method else "percell")),
                         cs, nc - 1, float(miou) * 100, float(macc) * 100, 0,
                         sm.get("scd"), sm.get("mae_pred2gt"), sm.get("mae_gt2pred"),
                         sm.get("hd95"), sm.get("boundary_f1"), sm.get("n_missed"), aname,
                         "backfill_surface_metrics.py",
                         time.strftime("%Y-%m-%d %H:%M:%S")))
                written += 1
                if cs == "opengaussian19":
                    print(f"  {scene} {aname:<14}{tag:<20} mIoU={miou*100:6.2f} "
                          f"scd={sm.get('scd', float('nan'))*100:6.2f}cm "
                          f"hd95={sm.get('hd95', float('nan'))*100:6.2f}cm "
                          f"bF1={sm.get('boundary_f1', float('nan')):.3f} "
                          f"missed={sm.get('n_missed')}", flush=True)
        con.commit()
        print(f"[done] {scene} {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\nwrote/updated {written} rows")
    q = con.execute(
        "SELECT assignment || ' | ' || method, ROUND(AVG(miou),2), ROUND(AVG(scd)*100,2), "
        "ROUND(AVG(hd95)*100,2), "
        "ROUND(AVG(boundary_f1),3), COUNT(DISTINCT scene) FROM results_unified "
        "WHERE class_set='opengaussian19' AND scd IS NOT NULL AND method IN (?,?,?,?) "
        "AND source='backfill_surface_metrics.py' GROUP BY assignment, method "
        "ORDER BY assignment, AVG(miou) DESC", tuple(METHODS.values())).fetchall()
    print(f"\n{'method':<52}{'mIoU':>7}{'scd cm':>9}{'hd95 cm':>9}{'bF1':>7}{'n':>4}")
    for r in q:
        print(f"{r[0]:<70}{r[1]:>7.2f}{r[2]:>9.2f}{r[3]:>9.2f}{r[4]:>7.3f}{r[5]:>4}")
    con.close()


if __name__ == "__main__":
    main()
