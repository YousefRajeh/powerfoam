"""Surface metrics for the simplex-diffusion arms, gated on REPRODUCING their mIoU.

THE PROBLEM WITH JUST COMPUTING THEM. Surface metrics describe a specific set of per-point
predictions. The rows in `simplex_10scene` / `simplex_stack10` were produced by
run_simplex_diffusion_eval.py; if this script's reproduction of that pipeline differs at all,
its surface metrics would describe DIFFERENT predictions than the mIoU sitting in the same
row. An earlier pass in this project did exactly that and stamped one measurement across 192
rows from 9 sources.

THE GATE. Each arm is recomputed, and its mIoU is compared against the value already stored
in the row. Surface metrics are written ONLY where the two agree to within --tol (default
0.05 mIoU, i.e. essentially exact). Everything else is left blank and logged. That makes the
filled cells trustworthy by construction rather than by assumption.

WHY IT SHOULD MATCH. run_simplex_diffusion_eval.py assigns GT points with
assign_points_to_power_cells(valid=valid_mask) -- the `nearest_valid` protocol -- and then
predicts for every owned point. Reproducing that exactly already gave 36.53 for
percell-argmax at 19cls, matching the stored 36.53 to the digit.

GRAPHS: `truefacet` is the cached Delaunay CSR (the exact dual of the cell decomposition,
verified jaccard 1.0000 against radfoam's own CUDA Delaunay); `cech` is the cached AABB-overlap
graph, which is NOT a superset of the facet graph (51.4% edge agreement) and is kept only
because the published rows used it.
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
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from run_normlift_refine_eval import mode_vote_refine

DB = "artifacts/ablation.sqlite"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]

# method name -> (base features: 'unit' or 'modevote', graph or None, alpha)
ARMS = {
    "percell-argmax":                                       ("unit", None, None),
    "nonfrozen_10scene_avg_geometric_median:per_scene":     ("unit", None, None),
    "diffusion(truefacet,s1000_a0.9)":                      ("unit", "delaunay", 0.9),
    "diffusion(truefacet,s1000_a0.95)":                     ("unit", "delaunay", 0.95),
    "diffusion(cech,a0.9)":                                 ("unit", "cech", 0.9),
    "diffusion(cech,a0.95)":                                ("unit", "cech", 0.95),
    "modevote(truefacet)":                                  ("modevote", None, None),
    "modevote(truefacet)+diffusion(truefacet,s1000_a0.9)":  ("modevote", "delaunay", 0.9),
    "modevote(truefacet)+diffusion(cech,a0.9)":             ("modevote", "cech", 0.9),
}


def diffuse(p0, adjacent, offsets, n, alpha, iters=60):
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
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |reproduced - stored| mIoU to accept the reproduction")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    con = sqlite3.connect(DB, timeout=180)
    written, rejected, missing = 0, [], []

    todo = con.execute(
        "SELECT DISTINCT source, method FROM results_unified WHERE scd IS NULL "
        "AND recon='pf_nonfroz' AND source IN ('simplex_10scene','simplex_stack10',"
        "'nonfrozen_10scene_avg_geometric_median.json')").fetchall()
    print(f"{len(todo)} (source, method) combos to reproduce", flush=True)

    for scene in SPLIT:
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        if not all(os.path.exists(p) for p in (mp, fp, stp)):
            missing.append(f"{scene}: artifacts"); continue
        t0 = time.time()
        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        n_prim = P.shape[0]
        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        # the simplex protocol: nearest cell THAT HAS A FEATURE
        nv = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if os.path.exists(nv):
            assign = np.load(nv)
        else:
            centers = m["points"].float().numpy().astype(np.float64)
            radii = F.softplus(m["radii"].float().squeeze(), beta=100).numpy().astype(np.float64)
            assign = np.asarray(assign_points_to_power_cells(gt_pts, centers, radii,
                                                             valid=valid, k=64))
            np.save(nv, assign)
        owned = assign >= 0
        pts = np.asarray(gt_pts, dtype=np.float64)
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        graphs = {}
        for g in ("delaunay", "cech"):
            gp = f"artifacts/ablation_cache/{scene}_pf_nonfroz_{g}.pt"
            if os.path.exists(gp):
                gg = torch.load(gp, map_location="cpu", weights_only=True)
                graphs[g] = (gg["adjacent"].to(dev).long(), gg["offsets"].to(dev).long())

        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        adj_m = m["adjacency"].long().to(dev)
        off_m = m["adjacency_offsets"].long().to(dev)
        bases = {"unit": unit, "modevote": mode_vote_refine(unit, R, P, adj_m, off_m)}

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            index = GTSurfaceIndex(pts, gt, nc)

            for source, method in todo:
                if method not in ARMS:
                    missing.append(f"{method}: no recipe"); continue
                bkey, gkey, alpha = ARMS[method]
                if gkey and gkey not in graphs:
                    missing.append(f"{scene}/{method}: no {gkey} graph"); continue
                u = bases[bkey]
                sim = u @ text.T
                if gkey:
                    p0 = torch.softmax(1000.0 * sim, dim=-1)
                    p0[~vt] = 0.0
                    pr = diffuse(p0, *graphs[gkey], n_prim, alpha)
                    cls = pr.argmax(-1).cpu().numpy() + 1
                    live = (pr.sum(-1) > 0).cpu().numpy()
                else:
                    cls = sim.argmax(-1).cpu().numpy() + 1
                    live = valid
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                _, miou, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                  torch.from_numpy(pred).long(), nc)
                mine = float(miou) * 100

                row = con.execute(
                    "SELECT id, miou FROM results_unified WHERE source=? AND method=? "
                    "AND scene=? AND class_set=? AND recon='pf_nonfroz'",
                    (source, method, scene, cs)).fetchone()
                if row is None:
                    continue
                if abs(row[1] - mine) > a.tol:
                    rejected.append(f"{scene}/{cs[11:]}/{method[:40]}: "
                                    f"stored {row[1]:.2f} vs reproduced {mine:.2f}")
                    continue
                sm = semantic_surface_metrics(index, pred)
                con.execute(
                    "UPDATE results_unified SET scd=?,mae_pred2gt=?,mae_gt2pred=?,hd95=?,"
                    "boundary_f1=?,n_missed=?,assignment='nearest_valid' WHERE id=?",
                    (sm.get("scd"), sm.get("mae_pred2gt"), sm.get("mae_gt2pred"),
                     sm.get("hd95"), sm.get("boundary_f1"), sm.get("n_missed"), row[0]))
                written += 1
        con.commit()
        print(f"  [{scene}] {(time.time()-t0)/60:.1f} min  written so far {written}", flush=True)

    print(f"\nwrote surface metrics to {written} rows (mIoU reproduced within {a.tol})")
    if rejected:
        print(f"REJECTED {len(rejected)} -- reproduction did not match, left blank:")
        for r in rejected[:15]:
            print("  ", r)
    if missing:
        print(f"MISSING {len(missing)}:")
        for r in dict.fromkeys(missing[:10]):
            print("  ", r)
    con.close()


if __name__ == "__main__":
    main()
