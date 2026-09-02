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
from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from ablation_opacity import DEFAULT_THRESHOLD, mask_low_opacity, primitive_alpha
from ablation_assign import RECONS, compute_assignment, load_primitives
from determinism import enable_determinism
from diagnose_scannet_miou import spherical_kmeans
from run_region_grow_eval import batched_region_grow
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from run_cluster_classify_eval import (SCENES, euclidean_kmeans, pool_classify_broadcast,
                                       two_level_position_aware)

CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
# None = score every GT point; 0.1 = OpenGaussian/NormLift, which DELETE points whose
# primitive is below that opacity. Both are always run: the masked number is the one
# comparable to published baselines, the unmasked one is the honest representation
# comparison, and they differ by +1.08 (pf_nonfroz) to +11.09 (rf_unfroz) mIoU.
MASKS = [None, DEFAULT_THRESHOLD]
# OpenGaussian/LUDVIG zero any cluster with fewer than 2 members (eval_scannet.py:139).
MIN_LEAF_SUPPORT = 2
COMPLEXES = ["delaunay", "alpha", "cech"]
SOLVERS = ["geometric_median", "weighted", "ridge", "inverse_variance"]
# `weighted` is the solver splat-distiller uses for Gaussians; including it for the
# foams too is what turns solver from a confound into an axis.
CACHE = os.path.join("artifacts", "ablation_cache")

# Features currently lifted per recon. powerfoam-nonfrozen has the OpenGaussian-protocol L3
# artifacts; the other arms need their own representation's lifting first.
FEATURE_TAG = "ogl3"
# Per-recon solved-feature paths. Lifting is representation-specific: powerfoam uses its own
# accumulator, radfoam its feature_operator (run remotely, since the CUDA extension will not
# build here), gaussians would need splat-distiller's distill. An arm joins the sweep the
# moment its file appears -- no code change, so a partial lift yields a valid partial ablation
# rather than an all-or-nothing wait.
# Per (recon, solver). SOLVER IS A REAL AXIS, not a fixed choice: splat-distiller's Gaussian
# lifting divides by accumulated weight (a weighted mean) while the foam lifts use a geometric
# median, so a gaussian-vs-foam row read under one solver each would confound representation
# with solver. Every representation is therefore solved under every solver from the SAME
# stats, and both appear in the table.
FEATURE_PATHS = {
    "pf_nonfroz": "artifacts/scannet/{scene}/solved_{solver}_nonfrozen_ogl3.pt",
    "pf_tfroz":   "artifacts/scannet/{scene}/solved_{solver}_truefrozen_ogl3.pt",
    "rf_froz":    "artifacts/scannet/{scene}/solved_{solver}_rf_froz_ogl3.pt",
    "rf_unfroz":  "artifacts/scannet/{scene}/solved_{solver}_rf_unfroz_ogl3.pt",
    "gs_froz":    "artifacts/scannet/{scene}/solved_{solver}_gs_froz_ogl3.pt",
    "gs_unfroz":  "artifacts/scannet/{scene}/solved_{solver}_gs_unfroz_ogl3.pt",
    "rf20k_froz":   "artifacts/scannet/{scene}/solved_{solver}_rf20k_froz_ogl3.pt",
    "rf20k_unfroz": "artifacts/scannet/{scene}/solved_{solver}_rf20k_unfroz_ogl3.pt",
}


def feature_file(recon, scene, solver):
    """Path for one (recon, scene, solver), tolerating the pre-solver-axis powerfoam name."""
    tmpl = FEATURE_PATHS.get(recon)
    if tmpl is None:
        return None
    p = tmpl.format(scene=scene, solver=solver)
    if os.path.exists(p):
        return p
    # legacy: geometric-median powerfoam artifacts predate the {solver} slot
    if solver == "geometric_median":
        legacy = tmpl.format(scene=scene, solver="geometric_median")
        if os.path.exists(legacy):
            return legacy
    return None


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


def get_adjacency_cached(con, scene, recon, complex_):
    """Cached CSR only -- never rebuilds, so the grouping stage cannot pay hull cost."""
    row = con.execute("SELECT path FROM adjacency WHERE scene=? AND recon=? AND complex=?",
                      (scene, recon, complex_)).fetchone()
    if not row or not os.path.exists(row["path"]):
        return None
    d = torch.load(row["path"], map_location="cpu", weights_only=True)
    return d["adjacent"], d["offsets"]


def solve_features(scene, recon, solver, log):
    """Per-primitive features under the requested solver.

    geometric_median reads the already-solved artifact. The other solvers need the per-view
    stats, which are multi-GB and deleted after each solve, so they are attempted only when
    the stats file happens to be present -- otherwise the cell is recorded as a gap rather
    than silently substituting a different solver's features.
    """
    p = feature_file(recon, scene, solver)
    if p is None:
        return None, f"not solved yet: {recon}/{solver}"
    d = torch.load(p, map_location="cuda", weights_only=True)
    return d, None


# Cosine floors for the thresholded k-means arm. Plain k-means is a pure argmax with NO
# floor, so a primitive whose best centroid sits at cosine 0.1 is bound to that cluster as
# firmly as one at 0.99 and inherits its label. These arms leave such a primitive UNCLUSTERED
# (label -1 -> no prediction), the same principle already applied to cells with no features:
# a primitive with no good evidence should not be forced to a label.
#
# PRIOR POINTS THE OTHER WAY. The "softness law" in this project has five confirmations that
# making the estimator more decisive loses monotonically (squared weights -1.21, top1 -3.31,
# vMF 55->51%, consensus vote -1.46 to -2.04, tau-reweighting null). A hard floor is a
# decisiveness move. But every one of those was measured on the AGGREGATION side; this is a
# gate on cluster MEMBERSHIP, a different mechanism, so it is measured rather than assumed.
# VALUES SET FROM THE MEASURED DISTRIBUTION, not chosen a priori. The first attempt used
# 0.5-0.8 and was completely inert -- 100% of primitives kept at every level -- because the
# lifted CLIP features sit in a very narrow cone. Measured best-centroid cosine on
# scene0062_00 pf_nonfroz: p0 0.8088, p1 0.9612, p25 0.9929, median 0.9969, p90 0.9995.
# So a floor only discriminates above ~0.97. Retention at these values:
#     0.97 -> 98.0%   0.98 -> 94.6%   0.99 -> 83.0%   0.995 -> 64.9%
KMEANS_THRESHOLDS = [0.97, 0.98, 0.99, 0.995]


# Feature-similarity floors for facet-adjacency region growing. Growing is the structural
# alternative to k-means: a region is built by WALKING EDGES of the adjacency graph, so its
# regions are connected subgraphs by construction. That matters because k-means regions are
# measurably incoherent -- only 6.6% of the 320 are a single connected piece, the median is
# scattered over 15.5 fragments, and one pooled feature is broadcast across all of it.
# Verified in test_region_grow_invariants.py: connectivity holds exactly (every region spans
# 1 component), the gate is monotone in the threshold (4,503 -> 5,950 -> 9,083 regions at
# 0.80/0.90/0.95), and the labels are reproducible ONLY under enable_determinism() -- without
# it, atomics in the coherence reduction perturb the seed ordering and region counts drift
# about 0.5% between runs.
GROW_THRESHOLDS = [0.85, 0.90, 0.95]


def groupings_for(unit, positions, adjacency, log):
    """-> list of (name, labels, n_labels, complex_used).

    labels may contain -1, meaning "no cluster"; run_scene maps that to no prediction.
    """
    out = []
    flat, cent = spherical_kmeans(unit, 320, seed=0)
    out.append(("kmeans320", flat, 320, None))
    pos = two_level_position_aware(positions, unit, seed=0)
    out.append(("pos_aware_64x5", pos, 320, None))

    # Reuse the SAME clustering for every threshold -- only membership is masked, so the arms
    # cost one extra eval each rather than a re-solve, and differ from kmeans320 in exactly
    # one respect.
    best_sim = (unit @ cent.T).max(dim=1).values
    for tau in KMEANS_THRESHOLDS:
        masked = flat.clone()
        masked[best_sim < tau] = -1
        kept = int((masked >= 0).sum())
        out.append((f"kmeans320_thr{tau:g}", masked, 320, None))
        log(f"    grouping kmeans320_thr{tau:g}: {kept:,}/{len(masked):,} primitives kept "
            f"({100*kept/max(len(masked),1):.1f}%)")

    # Facet-adjacency growing, one arm per complex x threshold.
    for cx, (adjacent, offsets, vmask) in (adjacency or {}).items():
        for thr in GROW_THRESHOLDS:
            try:
                lab, nreg = batched_region_grow(adjacent, offsets, unit, vmask, thr)
            except Exception as e:
                log(f"    [grow fail] {cx} thr={thr}: {e}")
                continue
            out.append((f"grow_{cx}_thr{thr:g}", lab, int(nreg), cx))
            log(f"    grouping grow_{cx}_thr{thr:g}: {nreg:,} regions")
    return out


def run_scene(con, run_id, scene, recons, log):
    split = SCENES[scene]
    gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present_ids = set(np.unique(raw_labels).tolist())
    gt_pts64 = np.asarray(gt_points, dtype=np.float64)
    # One GT KD-tree set per class-set, reused by EVERY method cell in this scene -- the whole
    # point of the cached-GT surface implementation.
    surf_index = {}
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
        alpha = primitive_alpha(recon, scene)
        if alpha is None:
            log(f"    [no alpha] {recon}: opacity-masked rows will be skipped")

        for cx in COMPLEXES:
            if cx == "cech" and not recon.startswith("pf_"):
                continue          # cech is powerfoam's renderer graph only
            try:
                get_adjacency(con, scene, recon, prim, cx, log)
            except Exception as e:
                log(f"    [adj fail] {recon}/{cx}: {e}")
                DB.record_failure(con, run_id, scene, recon, f"adjacency:{cx}", e)

        have_any = any(feature_file(recon, scene, sv) for sv in SOLVERS)
        if not have_any:
            feat_file = FEATURE_PATHS.get(recon, "?").format(scene=scene, solver="<solver>")
            DB.record_failure(con, run_id, scene, recon, "no_features",
                              f"not lifted yet: {feat_file}")
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

            # Growing runs on the VALID sub-graph, in the same index space as `unit`:
            # remap CSR node ids to positions within vidx and drop edges touching an
            # unlifted cell, so a grown region can never include an unobservable primitive.
            adj_for_grow = {}
            remap = np.full(n_prim, -1, dtype=np.int64)
            remap[vidx] = np.arange(len(vidx))
            for cx_name in COMPLEXES:
                got = get_adjacency_cached(con, scene, recon, cx_name)
                if got is None:
                    continue
                a_np = got[0].cpu().numpy().astype(np.int64)
                o_np = got[1].cpu().numpy().astype(np.int64)
                if len(o_np) - 1 != n_prim:
                    continue
                src_np = np.repeat(np.arange(n_prim), np.diff(o_np))
                keep = (remap[src_np] >= 0) & (remap[a_np] >= 0)
                rs, rd = remap[src_np[keep]], remap[a_np[keep]]
                order = np.argsort(rs, kind="stable")
                rs, rd = rs[order], rd[order]
                cnt = np.bincount(rs, minlength=len(vidx))
                new_off = np.concatenate([[0], np.cumsum(cnt)])
                adj_for_grow[cx_name] = (
                    torch.from_numpy(rd).cuda().long(),
                    torch.from_numpy(new_off).cuda().long(),
                    torch.ones(len(vidx), dtype=torch.bool, device="cuda"))

            for gname, labels, nlab, cx in groupings_for(unit, positions, adj_for_grow, log):
                for cs in CLASS_SETS:
                  for mask in MASKS:
                    if mask is not None and alpha is None:
                        continue
                    if DB.have_result(con, scene, recon, FEATURE_TAG, solver, gname, cs, mask):
                        continue
                    t0 = time.time()
                    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                            if name_to_id[n] in present_ids]
                    tids = [i for i, _ in kept]
                    tnames = [n for _, n in kept]
                    gt_np = remap_gt_labels(raw_labels, tids)
                    kept_frac = 1.0
                    if mask is not None:
                        gt_np, _n_masked, kept_frac = mask_low_opacity(gt_np, assigned, alpha, mask)
                    gt_t = torch.from_numpy(gt_np).long()
                    text = embed_class_names(tnames, "cuda")
                    # A grouping may decline to cluster a primitive (label -1). Pool over the
                    # clustered ones only, and mark the rest unclassified so they reach the
                    # metric as "no prediction" rather than being folded into some cluster.
                    # OpenGaussian zeroes the language feature of any leaf occupied by fewer
                    # than MIN_LEAF_SUPPORT primitives (`eval_scannet.py:139`,
                    # `leaf_lang_feat[leaf_occu_count < 2] *= 0.0`), and LUDVIG inherits their
                    # eval verbatim. It is a minimum-support rule on the CLUSTER, not the
                    # primitive: a singleton leaf's pooled feature is one observation with no
                    # agreement behind it. We had no equivalent, and it matters most for the
                    # grow arms, which produce thousands of tiny regions.
                    in_cluster = (labels >= 0)
                    if MIN_LEAF_SUPPORT > 1:
                        counts = torch.bincount(labels[in_cluster], minlength=nlab)
                        weak = counts < MIN_LEAF_SUPPORT
                        if bool(weak.any()):
                            in_cluster = in_cluster & ~weak[labels.clamp_min(0)]
                    cls_v = np.full(len(vidx), -1, dtype=np.int64)
                    if bool(in_cluster.any()):
                        sub = pool_classify_broadcast(labels[in_cluster], unit[in_cluster],
                                                      nlab, text).cpu().numpy()
                        cls_v[in_cluster.cpu().numpy()] = sub
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
                    prim_valid[vidx] = cls_v >= 0     # unclustered primitives are not scorable
                    prim_cls = np.zeros(n_prim, dtype=np.int64)
                    prim_cls[vidx] = np.maximum(cls_v, 0)
                    pred = np.zeros(len(gt_points), dtype=np.int64)
                    scorable = owned.copy()
                    scorable[owned] = prim_valid[assigned[owned]]
                    pred[scorable] = prim_cls[assigned[scorable]] + 1
                    ious, miou, acc, macc = calculate_metrics(
                        gt_t, torch.from_numpy(pred).long(), len(tnames) + 1)
                    per_class = {tnames[c - 1]: float(ious[c]) for c in range(1, len(tnames) + 1)}
                    skey = (cs, mask)
                    if skey not in surf_index:
                        surf_index[skey] = GTSurfaceIndex(gt_pts64, gt_t.numpy(), len(tnames) + 1)
                    surf = semantic_surface_metrics(surf_index[skey], pred)
                    DB.put_result(con, run_id, scene, recon, FEATURE_TAG, solver, gname, cx,
                                  cs, len(tnames), miou, macc, acc, per_class,
                                  time.time() - t0, surface=surf,
                                  mask_opacity=mask, kept_fraction=kept_frac)
                    mtag = "raw " if mask is None else f"m{mask:g}"
                    log(f"    {recon}/{solver}/{gname}/{cs}[{mtag}]: mIoU={miou*100:.2f} "
                        f"mAcc={macc*100:.2f} scd={surf.get('scd', float('nan'))*100:.2f}cm "
                        f"bF1={surf.get('boundary_f1', float('nan')):.3f} "
                        f"missed={surf.get('n_missed', 0)}")


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
