"""Does OpenGaussian's low-opacity GT masking change our 10-scene ScanNet numbers?

WHAT THE RULE IS. `OpenGaussian/scripts/eval_scannet.py:127-129`:

    ignored_pts = sigmoid(vertex_data["opacity"]) < 0.1
    updated_gt_labels[ignored_pts] = 0

Label 0 is their ignore label (`gt != 0` gates both the IoU union and the present-class list), so
this DELETES those points from the metric. NormLift copies it verbatim. It is therefore part of
the protocol we are benchmarking against, not an optional extra.

WHY IT NEEDS TRANSLATING. Their rule indexes GT points with a per-Gaussian mask, which is only
meaningful because `--frozen_init_pts` puts exactly one Gaussian per GT vertex in matching order.
We densify (3x), and the foam never had that correspondence at all. The faithful generalisation is
to mask the point whose ASSIGNED primitive is transparent -- under a frozen checkpoint that is
literally their rule.

AND THE FOAM HAS NO sigmoid(opacity). A Gaussian's opacity is already an alpha in (0,1) and "0.1"
means one-tenth opaque. The foam stores an unbounded volumetric density, so alpha only exists once
a ray path length is fixed: alpha = 1 - exp(-density * L). We take L = 2*radius, "how opaque is
this cell to a ray crossing it". That choice is OURS, not the protocol's, so this script sweeps L
rather than hard-coding one value -- if the verdict is stable across L it does not hinge on the
choice, and if it is not, the number is not reportable.

WHY THIS COMPARISON IS CLEAN. The mask changes only the GT vector; predictions, clustering and
assignment are byte-identical between arms. So the delta is PAIRED and exact -- none of the
~1.5 mIoU seed-noise band that has reversed ten single-scene results in this project applies to it.
Whatever this measures, it measures for real. The one thing to watch is that a mask which deletes
points cannot be judged by mIoU alone: deleting hard points RAISES mIoU without any method getting
better, so this reports the dropped fraction alongside every score.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_scannet_miou import load_foam, spherical_kmeans
from run_cluster_classify_eval import pool_classify_broadcast
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from determinism import enable_determinism

SCENES = {
    "scene0000_00": "train", "scene0062_00": "train", "scene0070_00": "train",
    "scene0097_00": "train", "scene0140_00": "train", "scene0200_00": "train",
    "scene0347_00": "train", "scene0400_00": "train", "scene0590_00": "train",
    "scene0645_00": "val",   # the one scene of the ten that is NOT in Pointcept's train split
}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
K_FLAT = 320
GT_ROOT = os.environ.get("SCANNET_GT_ROOT", r"D:\Downloads\scannet_pointcept")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--lengths", default="radius2,0.01,0.05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="artifacts/scannet/gt_opacity_mask_10scene.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip scenes already present in --output instead of recomputing them.")
    p.add_argument("--variant", choices=["nonfrozen", "frozen"], default="nonfrozen",
                   help="frozen = position-frozen, no densification, so cell i IS GT point i and "
                        "OpenGaussian's rule applies literally by index.")
    p.add_argument("--mask-mode", choices=["assigned", "index"], default="assigned",
                   help="assigned: mask the point whose power cell is transparent (the only "
                        "option off frozen). index: mask point i by cell i's alpha -- this is "
                        "OpenGaussian's own indexing, valid ONLY under the 1:1 frozen identity.")
    a = p.parse_args()
    enable_determinism()
    device = "cuda"
    lengths = a.lengths.split(",")

    # Resume: the JSON is rewritten after every scene, so a crash late in the list (e.g. the
    # scene0645_00 split typo) costs only the scenes that had not finished.
    out = json.load(open(a.output)) if os.path.exists(a.output) and a.resume else {}
    for scene, split in SCENES.items():
        if scene in out:
            print(f"\n===== {scene} (cached) =====", flush=True)
            continue
        print(f"\n===== {scene} =====", flush=True)
        gt_dir = os.path.join(GT_ROOT, split, scene)
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        # scene0000_00's corrected position-frozen run lives under `truefrozen`; the other nine
        # were retrained in place, so their plain `_frozen` dirs ARE the corrected ones (the
        # position-drifting originals were preserved as `_frozen_STALE_posdrift`).
        if a.variant == "frozen":
            tag = "truefrozen" if scene == "scene0000_00" else "frozen"
        else:
            tag = "nonfrozen"
        centers, radii, density = load_foam(
            f"output/scannet_{scene}_{tag}", device, return_density=True)

        solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{tag}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
        owned = assigned >= 0

        valid_idx_np = np.where(valid_mask)[0]
        unit = F.normalize(feats[torch.from_numpy(valid_idx_np).to(device)], dim=-1)
        flat_labels, _ = spherical_kmeans(unit, K_FLAT, seed=a.seed)

        rad = radii.reshape(-1)
        if a.mask_mode == "index" and centers.shape[0] != gt_points.shape[0]:
            raise SystemExit(
                f"{scene}: --mask-mode index needs the 1:1 frozen identity, but there are "
                f"{centers.shape[0]} primitives and {gt_points.shape[0]} GT points. "
                f"OpenGaussian's own indexing is only meaningful under --frozen_init_pts.")

        def low_mask(alpha):
            """Which GT points does this alpha field delete?

            index: OpenGaussian's literal rule -- point i is masked by cell i's own alpha, with no
              assignment step at all. Only defined under the frozen 1:1 identity, guarded above.
            assigned: the generalisation that survives densification -- mask the point whose
              containing power cell is transparent.
            """
            if a.mask_mode == "index":
                return alpha < a.threshold
            m = np.zeros(gt_points.shape[0], dtype=bool)
            m[owned] = alpha[assigned[owned]] < a.threshold
            return m
        rec = {"n_points": int(gt_points.shape[0]),
               "density_pct": {q: float(np.percentile(density, q)) for q in (1, 50, 99)},
               "arms": {}}

        for Lname in lengths:
            L = rad * 2.0 if Lname == "radius2" else float(Lname)
            alpha = 1.0 - np.exp(-density * L)
            rec["arms"][Lname] = {
                "alpha_pct": {q: float(np.percentile(alpha, q)) for q in (1, 50, 99)},
                "low_prim_pct": float((alpha < a.threshold).mean() * 100),
                "scores": {},
            }
            print(f"  L={Lname}: alpha median {np.median(alpha):.4f}, "
                  f"{(alpha < a.threshold).mean()*100:.2f}% of primitives below {a.threshold}",
                  flush=True)

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            target_ids, target_names = [i for i, _ in kept], [n for _, n in kept]
            gt_np = remap_gt_labels(raw_labels, target_ids)
            text_feats = embed_class_names(target_names, device)

            prim_cls_valid = pool_classify_broadcast(
                flat_labels, unit, K_FLAT, text_feats).cpu().numpy()
            prim_class = np.zeros(centers.shape[0], dtype=np.int64)
            prim_class[valid_idx_np] = prim_cls_valid
            pred = np.zeros(gt_points.shape[0], dtype=np.int64)
            pred[owned] = prim_class[assigned[owned]] + 1
            pred_t = torch.from_numpy(pred).long()

            # unmasked reference
            _, miou0, acc0, macc0 = calculate_metrics(
                torch.from_numpy(gt_np).long(), pred_t, len(target_ids) + 1)
            rec.setdefault("unmasked", {})[cs] = {"mIoU": miou0, "mAcc": macc0}

            for Lname in lengths:
                L = rad * 2.0 if Lname == "radius2" else float(Lname)
                alpha = 1.0 - np.exp(-density * L)
                low = low_mask(alpha)
                gtm = gt_np.copy()
                before = int((gtm != 0).sum())
                gtm[low] = 0
                dropped = before - int((gtm != 0).sum())
                _, miou1, acc1, macc1 = calculate_metrics(
                    torch.from_numpy(gtm).long(), pred_t, len(target_ids) + 1)
                rec["arms"][Lname]["scores"][cs] = {
                    "mIoU": miou1, "mAcc": macc1, "d_mIoU": miou1 - miou0,
                    "dropped": dropped, "scored_before": before,
                    "dropped_pct": dropped / max(before, 1) * 100,
                }
                print(f"  {cs} L={Lname}: {miou0*100:.2f} -> {miou1*100:.2f} "
                      f"({(miou1-miou0)*100:+.2f}) dropping {dropped/max(before,1)*100:.2f}% "
                      f"of scored points", flush=True)
        out[scene] = rec
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
        json.dump(out, open(a.output, "w"), indent=2)

    print("\n\n=== 10-scene means ===")
    for Lname in lengths:
        for cs in CLASS_SETS:
            b = [out[s]["unmasked"][cs]["mIoU"] for s in out]
            m = [out[s]["arms"][Lname]["scores"][cs]["mIoU"] for s in out]
            d = [out[s]["arms"][Lname]["scores"][cs]["dropped_pct"] for s in out]
            print(f"L={Lname:8s} {cs}: unmasked {np.mean(b)*100:.2f} -> masked "
                  f"{np.mean(m)*100:.2f} ({(np.mean(m)-np.mean(b))*100:+.2f}), "
                  f"mean {np.mean(d):.2f}% of points deleted")
    json.dump(out, open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
