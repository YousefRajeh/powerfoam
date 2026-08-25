"""Full ScanNet segmentation ablation, scene by scene, into SQLite.

ORDER OF WORK IS DELIBERATE: one scene is taken all the way through every reconstruction and
every method combination before the next scene starts. That way a partial run is a valid
ablation over a prefix of scenes rather than a partial column for all of them, and the
per-scene GT / assignment / adjacency stay resident instead of being reloaded per cell.

PER (scene, recon), COMPUTED ONCE AND CACHED IN THE DB:
  * the GT-point -> primitive assignment (power cell / nearest centre / exact Mahalanobis)
  * the adjacency graph for each complex (delaunay, alpha, and cech where powerfoam cached it)
Both are keyed by (scene, recon), so every method cell in that scene reads the SAME
correspondence -- a delta between two rows can never be an artefact of a re-derived mapping.

THEN, FOR EACH RECON THAT HAS LIFTED FEATURES: solver x grouping x class set.
Reconstructions without lifted features are recorded in `failures` with stage='no_features',
so a gap in `results` is never mistaken for "not run yet". Lifting is representation-specific
(powerfoam's accumulator, radfoam's own feature_operator, splat-distiller's distill) and is
NOT performed here.

Run:  D:\\conda\\envs\\powerfoam\\python.exe run_ablation.py [--scenes a,b] [--recons a,b]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import ablation_db as DB
from ablation_adjacency import build_for, load_cech_powerfoam
from ablation_assign import RECONS, compute_assignment, load_primitives
from determinism import enable_determinism
from diagnose_scannet_miou import spherical_kmeans
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from run_cluster_classify_eval import (SCENES, euclidean_kmeans, pool_classify_broadcast,
                                       two_level_position_aware)

CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
COMPLEXES = ["delaunay", "alpha", "cech"]
SOLVERS = ["geometric_median", "weighted", "ridge", "inverse_variance"]
CACHE = os.path.join("artifacts", "ablation_cache")

# Features currently lifted per recon. powerfoam-nonfrozen has the OpenGaussian-protocol L3
# artifacts; the other arms need their own representation's lifting first.
FEATURE_TAG = "ogl3"
FEATURE_PATH = "artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
HAS_FEATURES = {"pf_nonfroz"}


def cache_path(*parts):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, "_".join(parts))


def get_assignment(con, scene, recon, gt_points, log):
    row = con.execute("SELECT * FROM assignments WHERE scene=? AND recon=?",
                      (scene, recon)).fetchone()
    if row and os.path.exists(row["path"]):
        return np.load(row["path"]), row["method"]
    a, method, secs = compute_assignment(recon, scene, gt_points, progress=log)
    p = cache_path(scene, recon, "assign.npy")
    np.save(p, a)
    con.execute("INSERT OR REPLACE INTO assignments (scene,recon,method,n_points,n_owned,"
                "path,seconds,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (scene, recon, method, len(a), int((a >= 0).sum()), p, secs, time.time()))
    con.commit()
    log(f"    assign[{recon}] {method} owned={int((a>=0).sum()):,}/{len(a):,} {secs:.1f}s")
    return a, method


def get_adjacency(con, scene, recon, prim, complex_, log):
    row = con.execute("SELECT * FROM adjacency WHERE scene=? AND recon=? AND complex=?",
                      (scene, recon, complex_)).fetchone()
    if row and os.path.exists(row["path"]):
        d = torch.load(row["path"], map_location="cpu", weights_only=True)
        return d["adjacent"], d["offsets"]

    if complex_ == "cech":
        got = load_cech_powerfoam(recon, scene)
        if got is None:
            return None
        adjacent, offsets, st = got
    else:
        adjacent, offsets, st = build_for(prim["kind"], prim, complex_)

    p = cache_path(scene, recon, complex_ + ".pt")
    torch.save({"adjacent": adjacent, "offsets": offsets}, p)
    con.execute("INSERT OR REPLACE INTO adjacency (scene,recon,complex,n_edges,mean_degree,"
                "max_degree,path,seconds,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (scene, recon, complex_, st["n_edges"], st["mean_degree"], st["max_degree"],
                 p, st["seconds"], time.time()))
    con.commit()
    log(f"    adj[{recon}/{complex_}] E={st['n_edges']:,} deg={st['mean_degree']:.2f} "
        f"max={st['max_degree']} {st['seconds']:.1f}s")
    return adjacent, offsets


def solve_features(scene, recon, solver, log):
    """Per-primitive features under the requested solver.

    geometric_median reads the already-solved artifact. The other solvers need the per-view
    stats, which are multi-GB and deleted after each solve, so they are attempted only when
    the stats file happens to be present -- otherwise the cell is recorded as a gap rather
    than silently substituting a different solver's features.
    """
    if solver == "geometric_median":
        p = FEATURE_PATH.format(scene=scene)
        if not os.path.exists(p):
            return None, f"missing {p}"
        d = torch.load(p, map_location="cuda", weights_only=True)
        return d, None
    stats = f"artifacts/scannet/{scene}/stats_{FEATURE_TAG}.pt"
    if not os.path.exists(stats):
        return None, f"no stats for solver={solver} (deleted after solve)"
    return None, f"solver {solver} not wired"


def groupings_for(unit, positions, adjacency, log):
    """-> list of (name, labels, n_labels, complex_used)."""
    out = []
    flat, _ = spherical_kmeans(unit, 320, seed=0)
    out.append(("kmeans320", flat, 320, None))
    pos = two_level_position_aware(positions, unit, seed=0)
    out.append(("pos_aware_64x5", pos, 320, None))
    return out


def run_scene(con, run_id, scene, recons, log):
    split = SCENES[scene]
    gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present_ids = set(np.unique(raw_labels).tolist())
    log(f"\n===== {scene} ({split}) GT={len(gt_points):,} points =====")

    for recon in recons:
        try:
            prim = load_primitives(recon, scene, device="cuda")
        except (FileNotFoundError, KeyError) as e:
            log(f"  [skip] {recon}: {e}")
            DB.record_failure(con, run_id, scene, recon, "load", e)
            continue
        n_prim = prim["centers"].shape[0]
        con.execute("INSERT OR REPLACE INTO reconstructions (scene,recon,kind,n_primitives,"
                    "ckpt_path,created_at) VALUES (?,?,?,?,?,?)",
                    (scene, recon, prim["kind"], n_prim,
                     RECONS[recon][1].format(scene=scene), time.time()))
        con.commit()
        log(f"  {recon}: kind={prim['kind']} N={n_prim:,}")

        assigned, method = get_assignment(con, scene, recon, gt_points, log)

        for cx in COMPLEXES:
            if cx == "cech" and not recon.startswith("pf_"):
                continue          # cech is powerfoam's renderer graph only
            try:
                get_adjacency(con, scene, recon, prim, cx, log)
            except Exception as e:
                log(f"    [adj fail] {recon}/{cx}: {e}")
                DB.record_failure(con, run_id, scene, recon, f"adjacency:{cx}", e)

        if recon not in HAS_FEATURES:
            DB.record_failure(con, run_id, scene, recon, "no_features",
                              "lifting not yet run for this representation")
            log(f"    [no features] {recon}: assignment+adjacency cached; method sweep skipped")
            continue

        for solver in SOLVERS:
            solved, err = solve_features(scene, recon, solver, log)
            if solved is None:
                DB.record_failure(con, run_id, scene, recon, f"solve:{solver}", err)
                continue
            feats = solved["primitive_features"].cuda().float()
            valid = solved["valid_mask"].cpu().numpy()
            vidx = np.where(valid)[0]
            unit = F.normalize(feats[torch.from_numpy(vidx).cuda()], dim=-1)
            positions = prim["centers"][torch.from_numpy(vidx).cuda()].float()
            owned = assigned >= 0

            for gname, labels, nlab, cx in groupings_for(unit, positions, None, log):
                for cs in CLASS_SETS:
                    if DB.have_result(con, scene, recon, FEATURE_TAG, solver, gname, cs):
                        continue
                    t0 = time.time()
                    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                            if name_to_id[n] in present_ids]
                    tids = [i for i, _ in kept]
                    tnames = [n for _, n in kept]
                    gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
                    text = embed_class_names(tnames, "cuda")
                    cls_v = pool_classify_broadcast(labels, unit, nlab, text).cpu().numpy()
                    # A primitive with no lifted features cannot be classified. Leaving its
                    # points at prim_cls=0 would emit class 1 after the +1 shift, i.e. silently
                    # predict the FIRST class for every such point -- 7.07% of GT points on
                    # scene0062_00 pf_nonfroz, all becoming false positives for one class.
                    # They are marked 0 = "no prediction" instead, which the metric treats as
                    # unpredicted: it costs those points' true classes recall, without
                    # fabricating precision errors for an unrelated class.
                    #
                    # NOTE this differs from run_cluster_classify_eval.py, which reassigns such
                    # points to the nearest VALID primitive. That convention gives every point
                    # a prediction, but makes the correspondence depend on valid_mask, hence on
                    # the solver -- so it cannot be the single stored assignment shared across
                    # the ablation. The two differ by 0.65 mIoU on scene0062_00 (34.66 vs
                    # 35.31); the convention is recorded per row via the `grouping` provenance.
                    prim_valid = np.zeros(n_prim, dtype=bool)
                    prim_valid[vidx] = True
                    prim_cls = np.zeros(n_prim, dtype=np.int64)
                    prim_cls[vidx] = cls_v
                    pred = np.zeros(len(gt_points), dtype=np.int64)
                    scorable = owned.copy()
                    scorable[owned] = prim_valid[assigned[owned]]
                    pred[scorable] = prim_cls[assigned[scorable]] + 1
                    ious, miou, acc, macc = calculate_metrics(
                        gt_t, torch.from_numpy(pred).long(), len(tnames) + 1)
                    per_class = {tnames[c - 1]: float(ious[c]) for c in range(1, len(tnames) + 1)}
                    DB.put_result(con, run_id, scene, recon, FEATURE_TAG, solver, gname, cx,
                                  cs, len(tnames), miou, macc, acc, per_class,
                                  time.time() - t0)
                    log(f"    {recon}/{solver}/{gname}/{cs}: mIoU={miou*100:.2f} "
                        f"mAcc={macc*100:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--recons", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    enable_determinism()
    con = DB.connect()
    run_id = DB.start_run(con, a.note)
    scenes = [s for s in (a.scenes.split(",") if a.scenes else list(SCENES)) if s]
    recons = [r for r in (a.recons.split(",") if a.recons else list(RECONS)) if r]

    def log(m):
        print(m, flush=True)

    log(f"run_id={run_id}  scenes={len(scenes)}  recons={recons}")
    for scene in scenes:
        try:
            run_scene(con, run_id, scene, recons, log)
        except Exception as e:
            import traceback
            log(f"[SCENE FAIL] {scene}: {e}")
            DB.record_failure(con, run_id, scene, "-", "scene", traceback.format_exc())
    log("\n=== results so far (19cls) ===")
    for r in DB.summary(con, "opengaussian19"):
        log(f"  {r['recon']:<12}{r['solver']:<18}{r['grouping']:<16}"
            f"n={r['n']:<3} mIoU={r['miou']:.2f} mAcc={r['macc']:.2f}")


if __name__ == "__main__":
    main()
