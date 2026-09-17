"""Position-aware cluster-then-classify evaluation across all 10 ScanNet scenes.

Replicates the SHAPE of OpenGaussian's two-level codebook protocol (their eval never
classifies per-primitive): 64 root clusters on primitive POSITION, then up to 5 leaf
clusters on solved CLIP features within each root (64 x 5 = 320 leaves, their exact
codebook size), mean-pool the unit features per leaf, classify each pooled feature once
against the class-set text embeddings (plain cosine argmax, matching OpenGaussian
protocol), broadcast the leaf's class to every member primitive, then score points via
the usual power-cell assignment.

Also runs feature-only spherical k-means (k=320) as the controlled ablation -- isolating
"clustering at all" (already measured +3.7 mIoU on scene0000_00) from "position-aware
clustering" (this run's actual question).

Solved features / checkpoints: nonfrozen + geometric-median, same artifacts as
run_scannet_nonfrozen_eval.py, so numbers are directly comparable to the per-primitive
baseline averages (19cls 30.98/59.99, 15cls 30.98/60.70, 10cls 34.10/65.23).
"""
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import os
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS,
    calculate_metrics, remap_gt_labels, embed_class_names, classify_primitives,
    load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam, spherical_kmeans

SCENES = {
    "scene0000_00": "train", "scene0062_00": "train", "scene0070_00": "train",
    "scene0097_00": "train", "scene0140_00": "train", "scene0200_00": "train",
    "scene0347_00": "train", "scene0400_00": "train", "scene0590_00": "train",
    "scene0645_00": "val",
}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
NUM_ROOTS = 64
LEAVES_PER_ROOT = 5

# Canonical HARDEST-FIRST scene order for sweeps. Put the scenes that would FALSIFY an idea
# first, so a bad idea is killed in minutes instead of an hour, and state a kill criterion
# before launching.
#
# Derivation (2026-08-20): the first three are where coherence-gated geodesic growing
# collapsed outright -- scene0347_00 1.84, scene0070_00 0.42, scene0140_00 3.67 mIoU against a
# ~40 baseline. 0645/0590 carry the lowest baseline mIoU (28.40 / 35.54) and the largest cell
# counts (352k / 223k init points), so they stress memory and clustering together.
#
# The default (numeric) order is actively misleading: it starts with scene0000_00, the single
# scene where the foam-only method won MOST. A +1.75 mIoU pilot on it reversed to -12.3 over
# ten scenes. Easy-first ordering manufactures false progress.
HARD_FIRST = [
    "scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
    "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00",
]

K_FLAT = NUM_ROOTS * LEAVES_PER_ROOT  # 320, matching OpenGaussian's codebook size

# --- reported-configuration switches (all default to the previous behaviour) ---
RECON = os.environ.get("RECON", "nonfrozen").strip() or "nonfrozen"
"""Checkpoint arm: `output/scannet_<scene>_<RECON>`. ablation.sqlite's reported row is
`pf_tfroz` = `truefrozen`, NOT the `nonfrozen` this script defaulted to."""

GT_OPACITY_MASK = os.environ.get("GT_OPACITY_MASK", "0") == "1"
OPACITY_THRESHOLD = float(os.environ.get("OPACITY_THRESHOLD", "0.1"))
FOAM_ALPHA_LENGTH = os.environ.get("FOAM_ALPHA_LENGTH", "").strip()


def foam_alpha_from_density(density, radii):
    """alpha = 1 - exp(-density * L); L = 2*radius unless overridden.

    Same formula as evaluate_point_cloud_miou.py:229-231, taking density as an array because
    `load_foam(..., return_density=True)` already hands it over. The foam stores unbounded
    volumetric density, not a sigmoid opacity, so alpha only exists once a path length is fixed --
    and THAT choice is ours, not OpenGaussian's, so any number reported under it must say so.
    Using a different L here than the other script uses would make the two quietly incomparable,
    which is the failure mode this session kept hitting.
    """
    import numpy as _np
    d = _np.asarray(density, dtype=_np.float64).reshape(-1)
    L = float(FOAM_ALPHA_LENGTH) if FOAM_ALPHA_LENGTH else _np.asarray(radii).reshape(-1) * 2.0
    return 1.0 - _np.exp(-d * L)


def euclidean_kmeans(x, k, iters=25, seed=0):
    """Plain k-means on positions. x: (N, 3)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    centroids = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(x, centroids)
        labels = d.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).unsqueeze(1)
        keep = counts.squeeze(1) > 0
        new_centroids[keep] = new_centroids[keep] / counts[keep]
        new_centroids[~keep] = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:int((~keep).sum())]]
        centroids = new_centroids
    return torch.cdist(x, centroids).argmin(dim=1)


def two_level_position_aware(positions, unit_feats, seed=0, leaf_init="randperm"):
    """64 position roots x up-to-5 feature leaves -> global leaf label per primitive."""
    root = euclidean_kmeans(positions, NUM_ROOTS, seed=seed)
    leaf_global = torch.zeros_like(root)
    for r in range(NUM_ROOTS):
        idx = (root == r).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = min(LEAVES_PER_ROOT, idx.numel())
        if k == 1:
            leaf = torch.zeros(idx.numel(), dtype=torch.long, device=root.device)
        else:
            leaf, _ = spherical_kmeans(unit_feats[idx], k, seed=seed + r, init=leaf_init)
        leaf_global[idx] = r * LEAVES_PER_ROOT + leaf
    return leaf_global


# OpenGaussian zeroes any leaf occupied by fewer than 2 primitives (`eval_scannet.py:139`,
# `leaf_lang_feat[leaf_occu_count < 2] *= 0.0`); LUDVIG inherits their eval verbatim. It is a
# minimum-support rule on the CLUSTER: a singleton leaf's pooled feature is a single
# observation with no agreement behind it. Set MIN_LEAF_SUPPORT=1 to disable.
MIN_LEAF_SUPPORT = int(os.environ.get("MIN_LEAF_SUPPORT", "1"))


def pool_classify_broadcast(labels, unit_feats, num_labels, text_feats, weights=None):
    """weights: optional (N,) per-primitive pooling weight (e.g. #GT points owned) --
    aligns the pooled mean with what the point-level metric actually reads."""
    pooled = torch.zeros(num_labels, unit_feats.shape[1], device=unit_feats.device)
    pooled.index_add_(0, labels, unit_feats * weights[:, None] if weights is not None else unit_feats)
    if weights is not None:
        # A cluster all of whose members were gated out would pool to zero and receive NO
        # prediction, which is a protocol artefact rather than a result. Re-pool those clusters
        # unweighted so gating can only change WHICH members dominate, never silently delete a
        # cluster. Counted and reported by the caller.
        counts = torch.bincount(labels, minlength=num_labels)
        dead = (pooled.norm(dim=-1) < 1e-8) & (counts > 0)
        if dead.any():
            fallback = torch.zeros_like(pooled)
            fallback.index_add_(0, labels, unit_feats)
            pooled[dead] = fallback[dead]
            pool_classify_broadcast.last_fallback = int(dead.sum())
        else:
            pool_classify_broadcast.last_fallback = 0
    norms = pooled.norm(dim=-1, keepdim=True)
    nonempty = norms.squeeze(-1) > 1e-8
    pooled = pooled / norms.clamp_min(1e-8)
    cls = torch.full((num_labels,), -1, dtype=torch.long, device=unit_feats.device)
    cls[nonempty] = classify_primitives(pooled[nonempty], text_feats)
    if MIN_LEAF_SUPPORT > 1:
        counts = torch.bincount(labels, minlength=num_labels)
        cls[counts < MIN_LEAF_SUPPORT] = -1      # under-supported leaf -> no prediction
    return cls[labels]


def main():
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    device = "cuda"
    results = {m: {cs: {} for cs in CLASS_SETS} for m in ("feat_kmeans320", "pos_aware_64x5")}

    # ONLY_SCENES restricts the sweep to a comma-separated subset, for A/B-ing a feature
    # construction on one scene before paying for all ten. A single-scene delta is a PILOT,
    # never a conclusion -- eleven single-scene results in this project have reversed at
    # 10-scene scale, most recently hubness (+9.6 on one scene, -5.6 across ten).
    suffix = os.environ.get("FEAT_SUFFIX", "_l3")   # also read per-scene below; hoisted so the
                                                    # summary filename is defined even if the
                                                    # scene loop selects nothing.
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = {k: v for k, v in SCENES.items() if k in only} if only else SCENES

    for scene, split in scenes.items():
        ckpt_dir = f"output/scannet_{scene}_{RECON}"
        # SUFFIX selects which SAM-level construction to score. Default '' is the all-levels
        # normalized sum (splat-distiller/NormLift's construction); '_l3' is level-3 (large)
        # only, which is what OpenGaussian uses for LeRF. Their ScanNet script uses level 0.
        # DEFAULT IS _l3. Measured 10-scene, protocol-correct plain argmax, nonfrozen:
        #   all-levels  27.64/28.62/34.47 mIoU (45.87/45.91/52.40 mAcc)
        #   level 3     32.84/34.38/41.13 mIoU (56.46/58.21/63.34 mAcc)
        # i.e. +5.2/+5.8/+6.7 mIoU and ~+11 mAcc on every class set and both clusterings.
        # The all-levels normalized sum is splat-distiller's loader default, which NormLift
        # inherits -- but OpenGaussian uses a SINGLE level (0 for ScanNet, 3 for LeRF), never
        # the sum. Set FEAT_SUFFIX='' to score the all-levels construction as an ablation row.
        suffix = os.environ.get("FEAT_SUFFIX", "_l3")
        # FEAT_FILE overrides the whole stem, so a different SOLVER can be scored under an
        # otherwise identical protocol (same features, same views, same clustering, same class
        # sets). Used by run_solver_variants.py to compare Richardson-k and the surface-block
        # preconditioner against the closed form. Unset -> unchanged default.
        feat_file = os.environ.get("FEAT_FILE", "")
        features_path = (f"artifacts/scannet/{scene}/{feat_file}.pt" if feat_file else
                         f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen{suffix}.pt")
        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"

        print(f"\n===== {scene} =====", flush=True)
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        if GT_OPACITY_MASK:
            centers, radii, density = load_foam(ckpt_dir, device, return_density=True)
            prim_alpha = foam_alpha_from_density(density, radii)
        else:
            centers, radii = load_foam(ckpt_dir, device)
            prim_alpha = None
        solved = torch.load(features_path, map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
        owned = assigned >= 0

        valid_idx_np = np.where(valid_mask)[0]
        valid_idx = torch.from_numpy(valid_idx_np).to(device)
        unit = F.normalize(feats[valid_idx], dim=-1)
        positions = torch.from_numpy(centers[valid_idx_np]).to(device).float()

        # POOL_WEIGHT=norm weights the within-cluster pooling by ||x_j||, the pre-normalisation
        # lifted-feature norm. B rows are unit CLIP vectors, so x_j is a weighted AVERAGE of unit
        # vectors and ||x_j|| <= 1, with equality iff every contributing view agreed. It is
        # therefore a per-primitive multi-view CONSISTENCY score that F.normalize discards.
        # Measured on scene0347_00: mean 0.781, p25 0.701, 11% below 0.5, and Spearman only +0.058
        # against rho_j -- i.e. information that is both real and independent of the operator-side
        # reliability. Unset -> unchanged uniform pooling.
        # GRAPH_DIFFUSE=covis smooths the lifted features over G = A^T A, the co-visibility
        # graph, before clustering and pooling. Unlike Richardson (whose fixed point is x_hat and
        # which measured harmful), this is personalised PageRank: anchored to the input, fixed
        # point a SMOOTHING of it. G is the only neighbourhood here that is not chosen from
        # geometry -- it is the coupling the renderer defines, and Cor. 2 says the lifting error
        # lives on it. Edges are restricted to valid-valid so unobserved cells cannot pull
        # neighbours toward zero.
        if os.environ.get("GRAPH_DIFFUSE", "") == "covis":
            from covis_graph import covis_graph, ppr_diffuse
            P_all = centers.shape[0]
            gk = int(os.environ.get("GD_K", "30"))
            ga = float(os.environ.get("GD_ALPHA", "0.5"))
            gi = int(os.environ.get("GD_ITERS", "20"))
            gsrc, gdst, gw = covis_graph(scene, P_all, K=gk, device=device)
            vmask = torch.zeros(P_all, dtype=torch.bool, device=device)
            vmask[valid_idx] = True
            keep_e = vmask[gsrc] & vmask[gdst]
            gsrc, gdst, gw = gsrc[keep_e], gdst[keep_e], gw[keep_e]
            full = torch.zeros(P_all, unit.shape[1], device=device)
            full[valid_idx] = unit
            gsph = os.environ.get("GD_SPHERE", "") == "1"
            full = ppr_diffuse(full, gsrc, gdst, P_all, alpha=ga, iters=gi, weights=gw,
                               sphere=gsph)
            moved = float((F.normalize(full[valid_idx], dim=-1) * unit).sum(-1).mean())
            unit = F.normalize(full[valid_idx], dim=-1)
            print(f"  GRAPH_DIFFUSE=covis K={gk} alpha={ga} iters={gi} sphere={gsph}: "
                  f"{keep_e.sum():,} valid edges, mean cos to undiffused {moved:.4f}", flush=True)

        pool_w = None
        mode = os.environ.get("POOL_WEIGHT", "")
        if mode:
            # CONF_FILE decouples the CONFIDENCE SIGNAL from the FEATURES being classified.
            #
            # WHY. The geometric median (the reported solver) returns EXACTLY unit vectors
            # (sd 0.0000), so `||x_j||` carries zero information and every gate variant is a
            # bit-identical no-op on it. But `||x_j||` from a raw (un-normalised) lift is the best
            # per-primitive confidence predictor measured: AUC vs correctness 0.738 / 0.643 /
            # 0.697 on scene0062/0347/0070, BEATING the normalised lift's agreement signal
            # (0.723 / 0.640 / 0.675), and it survives stratification by agreement (AUC > 0.5 in
            # 13 of 15 quintile strata), so it is not merely re-expressing multi-view agreement.
            #
            # Pointing CONF_FILE at the raw lift lets pooling be weighted by that signal while
            # classification still uses the reported solver's features -- so using the gate no
            # longer requires switching solver, which is what made the trade net-negative.
            conf_file = os.environ.get("CONF_FILE", "").strip()
            if conf_file:
                cf = f"artifacts/scannet/{scene}/{conf_file}"
                cf = cf if cf.endswith(".pt") else cf + ".pt"
                cd_ = torch.load(cf, map_location=device, weights_only=True)
                cfeat = cd_["primitive_features"].to(device).float()
                assert cfeat.shape[0] == feats.shape[0], (cfeat.shape, feats.shape)
                nrm = cfeat[valid_idx].norm(dim=-1)
                # scale-free: percentile rank, so POOL_THRESH means the same thing whether the
                # confidence file is unit-norm (~0.8) or raw-magnitude (~13.7).
                nrm = nrm / nrm.median().clamp_min(1e-12)
                print(f"  CONF_FILE={conf_file}: confidence from a SEPARATE lift, "
                      f"median-normalised (raw median {float(cfeat[valid_idx].norm(dim=-1).median()):.4f})",
                      flush=True)
            else:
                nrm = feats[valid_idx].norm(dim=-1)
            if mode == "norm":
                pool_w = nrm
            elif mode == "gate":
                # hard include/exclude: cells whose views cancelled do not vote at all
                t = float(os.environ.get("POOL_THRESH", "0.5"))
                pool_w = (nrm >= t).float()
            elif mode == "randgate":
                # CONTROL for `gate`: keep a RANDOM subset of the same size. Gating on ||x||
                # confounds "keep confident cells" with "keep fewer cells"; this isolates the
                # first. Seeded per scene so the control is reproducible.
                keep_frac = float(os.environ.get("POOL_KEEP", "0.53"))
                gsel = torch.Generator(device="cpu").manual_seed(abs(hash(scene)) % (2**31))
                r = torch.rand(nrm.numel(), generator=gsel).to(nrm.device)
                pool_w = (r < keep_frac).float()
            elif mode == "normgate":
                t = float(os.environ.get("POOL_THRESH", "0.5"))
                pool_w = nrm * (nrm >= t).float()
            else:
                raise SystemExit(f"unknown POOL_WEIGHT={mode!r}")
            kept = float((pool_w > 0).float().mean())
            print(f"  POOL_WEIGHT={mode} thresh={os.environ.get('POOL_THRESH','-')}: "
                  f"||x|| mean {nrm.mean():.4f}, keeps {kept:.1%} of primitives", flush=True)

        # cluster ONCE per scene (class-set independent)
        flat_labels, _ = spherical_kmeans(unit, K_FLAT, seed=0)
        pos_labels = two_level_position_aware(positions, unit, seed=0)
        print(f"  clustered: flat k={K_FLAT}, pos-aware roots={NUM_ROOTS} x leaves<={LEAVES_PER_ROOT}", flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
            target_ids = [i for i, _ in kept]
            target_names = [n for _, n in kept]
            num_classes = len(target_ids)
            gt_np = remap_gt_labels(raw_labels, target_ids)
            if GT_OPACITY_MASK:
                # deletes points whose owning cell is near-transparent; label 0 is the ignore
                # label, so these leave the IoU union entirely rather than counting as errors
                from evaluate_point_cloud_miou import apply_gt_opacity_mask
                gt_np, _ = apply_gt_opacity_mask(
                    gt_np, assigned, prim_alpha, OPACITY_THRESHOLD, f"{scene}/{cs}")
            gt_t = torch.from_numpy(gt_np).long()
            text_feats = embed_class_names(target_names, device)

            for method, labels in (("feat_kmeans320", flat_labels), ("pos_aware_64x5", pos_labels)):
                prim_cls_valid = pool_classify_broadcast(labels, unit, K_FLAT, text_feats,
                                                         weights=pool_w).cpu().numpy()
                prim_class = np.zeros(centers.shape[0], dtype=np.int64)
                prim_class[valid_idx_np] = prim_cls_valid
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = prim_class[assigned[owned]] + 1
                _, miou, acc, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), num_classes + 1)
                results[method][cs][scene] = {"mIoU": miou, "mAcc": macc, "overall_acc": acc}
                print(f"  {cs} {method}: mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)

    n_scenes = len(scenes)
    tag = "10scene" if n_scenes == len(SCENES) else f"{n_scenes}scene_" + "_".join(scenes)
    print(f"\n\n=== {n_scenes}-scene average"
          f"{'s' if n_scenes > 1 else ' (PILOT -- not a conclusion)'}"
          f" ({RECON}, feats={os.path.basename(features_path)}, "
          f"opacity_mask={GT_OPACITY_MASK}@{OPACITY_THRESHOLD}) ===")
    print("baseline per-primitive: 19cls 30.98/59.99  15cls 30.98/60.70  10cls 34.10/65.23")
    summary = {}
    for method, per_cs in results.items():
        summary[method] = {}
        line = [method]
        for cs in CLASS_SETS:
            mious = [m["mIoU"] for m in per_cs[cs].values()]
            maccs = [m["mAcc"] for m in per_cs[cs].values()]
            summary[method][cs] = {
                "num_scenes": len(mious),
                "mean_mIoU": float(np.mean(mious)),
                "mean_mAcc": float(np.mean(maccs)),
                "per_scene": per_cs[cs],
            }
            line.append(f"{cs} {np.mean(mious)*100:.2f}/{np.mean(maccs)*100:.2f}")
        print("  ".join(line))

    # Filename carries the scene set AND the feature suffix. A single-scene ONLY_SCENES run
    # previously overwrote the real ten-scene artifact with a one-scene number still labelled
    # "10-scene averages", which is exactly how a pilot gets mistaken for a confirmed result.
    out = f"artifacts/scannet/cluster_classify_{tag}_avg{suffix}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
