"""Where do we actually stand on the BASE protocol, with every refinement stripped out?

The champion stack has accumulated five additions (mode-vote refinement, FPS seeding, partial
centering, reliability-weighted label voting, and the L3 feature fix). This measures the bare
OpenGaussian recipe -- L3 SAM+CLIP features -> group the primitives -> pool -> raw class-name
argmax -- across every grouping strategy we have, on identical features, so the groupings can
be compared without any downstream help.

THE MEASUREMENT THAT MATTERS: for each grouping we report BOTH

  oracle  : label every group with the class that maximises accuracy for that group (i.e.
            the majority GT label among the points it owns). This is the CEILING the grouping
            allows -- the best any classifier could ever do given these groups.
  actual  : what raw-name CLIP cosine argmax actually achieves.

The gap between them is the classification loss; the distance from oracle to 100 is the
grouping loss. Chasing better features when the oracle is already low is wasted effort, and
chasing better grouping when actual is already near oracle is equally wasted. Every idea
tried in this campaign was aimed at one of these two without knowing which was binding.

Also reported per grouping: number of non-empty groups, median connected components per
group on the real facet graph (spatial coherence), and the mean group purity (what fraction
of a group's points share its majority label) -- purity is the per-group version of the
oracle and shows whether the ceiling is limited by a few bad groups or by all of them.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam, spherical_kmeans
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES
from diagnose_region_locality import connected_components


def group_purity_and_oracle(groups_per_point, gt, n_groups, n_classes):
    """Oracle mIoU for a grouping, plus mean group purity.

    Oracle = assign each group its majority GT label, then score with the SAME metric used
    everywhere else, so the ceiling is directly comparable to the reported numbers.
    """
    valid = groups_per_point >= 0
    g = groups_per_point[valid]
    y = gt[valid]
    # counts[group, class]
    counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(counts, (g, y), 1)
    best = counts.argmax(1)
    tot = counts.sum(1)
    purity = np.divide(counts.max(1), np.maximum(tot, 1), dtype=np.float64)
    pred = np.zeros_like(gt)
    pred[valid] = best[g]
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                         torch.from_numpy(pred).long(), n_classes)
    live = tot > 0
    return float(miou), float(macc), float(purity[live].mean()), int(live.sum())


def main():
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    scene, variant = a.scene, a.variant
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit_full = torch.zeros_like(feats)
    unit_full[vi] = F.normalize(feats[vi], dim=-1)
    unit = unit_full[vi]                      # NO refinement -- bare solved features
    positions = torch.from_numpy(centers).to(device).float()
    pos_v = positions[vi]

    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{variant}.pt",
                     map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    cs = a.class_set
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
    tids, tnames = [i for i, _ in kept], [n for _, n in kept]
    gt_t = remap_gt_labels(raw_labels, tids)
    n_classes = len(tids) + 1
    text = embed_class_names(tnames, device)

    # cell -> group, for each strategy. NO refinement, NO centering, NO voting anywhere.
    groupings = {}
    groupings["per-primitive (no grouping)"] = (torch.arange(unit.shape[0], device=device),
                                                unit.shape[0])
    lbl, _ = spherical_kmeans(unit, K_FLAT, seed=0)
    groupings["flat spherical k-means 320"] = (lbl, K_FLAT)
    groupings["two-level 64x5 (OpenGaussian)"] = (
        two_level_position_aware(pos_v, unit, seed=0, leaf_init="randperm"), K_FLAT)
    groupings["two-level 64x5, FPS leaves"] = (
        two_level_position_aware(pos_v, unit, seed=0, leaf_init="fps"), K_FLAT)
    # pure spatial control: does feature information help the grouping at all?
    from diagnose_scannet_miou import fps_features  # noqa: F401  (kept for parity)
    from run_cluster_classify_eval import euclidean_kmeans
    groupings["pure spatial k-means 320"] = (euclidean_kmeans(pos_v, K_FLAT, seed=0), K_FLAT)

    rows = {}
    hdr = (f"{'grouping':<32}{'groups':>8}{'actual':>9}{'oracle':>9}{'gap':>8}"
           f"{'purity':>8}{'comps':>7}")
    print(f"\n=== {scene} / {cs} / {variant} -- BASE protocol, no refinements ===")
    print(hdr); print("-" * len(hdr))
    for name, (lab, n_groups) in groupings.items():
        # actual: pooled feature -> raw-name argmax
        pooled = torch.zeros(n_groups, unit.shape[1], device=device)
        pooled.index_add_(0, lab, unit)
        vcls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1)
        pc = np.zeros(centers.shape[0], dtype=np.int64)
        pc[vi.cpu().numpy()] = vcls[lab].cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = pc[assigned[owned]] + 1
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_t).long(),
                                             torch.from_numpy(pred).long(), n_classes)

        # oracle: same groups, best possible label per group
        gpp = np.full(len(gt_t), -1, dtype=np.int64)
        lab_full = np.full(centers.shape[0], -1, dtype=np.int64)
        lab_full[vi.cpu().numpy()] = lab.cpu().numpy()
        gpp[owned] = lab_full[assigned[owned]]
        o_miou, o_macc, purity, live = group_purity_and_oracle(gpp, gt_t, n_groups, n_classes)

        # spatial coherence of the grouping on the real facet graph
        if n_groups < unit.shape[0]:
            lf = torch.full((centers.shape[0],), -1, dtype=torch.long, device=device)
            lf[vi] = lab
            comp = connected_components(lf, adjacent, offsets, n_groups, device)
            ncomp = []
            for r in torch.unique(lab):
                m = lf == r
                ncomp.append(int(torch.unique(comp[m]).numel()))
            med_comp = float(np.median(ncomp))
        else:
            med_comp = 1.0
        rows[name] = {"groups": live, "actual_mIoU": float(miou), "actual_mAcc": float(macc),
                      "oracle_mIoU": o_miou, "oracle_mAcc": o_macc,
                      "purity": purity, "median_components": med_comp}
        print(f"{name:<32}{live:>8}{miou*100:>9.2f}{o_miou*100:>9.2f}"
              f"{(o_miou-miou)*100:>8.2f}{purity:>8.3f}{med_comp:>7.1f}")

    print("\nreading this table:")
    print("  oracle  = ceiling this grouping allows (best possible label per group)")
    print("  gap     = what the raw-name CLIP classifier loses BELOW that ceiling")
    print("  a low oracle means the GROUPING is binding; a large gap means the "
          "FEATURES/TEXT side is binding.")
    if a.output:
        json.dump({"scene": scene, "class_set": cs, "variant": variant, "rows": rows},
                  open(a.output, "w"), indent=2)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
