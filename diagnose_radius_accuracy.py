"""How does the POWER RADIUS relate to segmentation accuracy?

The power radius is foam's own free parameter: in the Laguerre tessellation a cell with
a larger radius wins the argmin(||x-c||^2 - r^2) contest over a larger volume, so it both
(a) integrates its lifted feature over more space and (b) owns more GT points, which
amplifies any single mistake. Nothing analogous exists for Gaussians, so if radius
predicts correctness it is a Feature-Foam-only lever.

Measured here (per scene, and pooled across scenes):
  1. Per-CELL accuracy by radius decile (does a big bubble classify worse?)
  2. Per-POINT accuracy by owner-radius decile (what the metric actually sees, since big
     cells carry many points)
  3. ERROR MASS by radius decile (where the wrong points actually live)
  4. Spearman correlations: radius vs correctness, radius vs #owned points, radius vs
     reliability, radius vs label purity (does a big cell straddle class boundaries?)
  5. Per-class radius profile for the dominant classes (wall/floor vs small objects)

Predictions are the validated raw-only stack minus refinement (pooled 64x5 FPS clustering
+ raw class names + partial centering), which needs only the solved features.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def run_scene(scene, variant, gt_root, lam, cs_name, acc):
    device = "cuda"
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit = F.normalize(feats[vi], dim=-1)
    positions = torch.from_numpy(centers).to(device).float()
    leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    P = centers.shape[0]
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs_name] if n2i[n] in present]
    tids = [i for i, _ in kept]
    tnames = [n for _, n in kept]
    K = len(tids)
    gt = remap_gt_labels(raw_labels, tids)

    text = embed_class_names(tnames, device)
    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
    pooled.index_add_(0, leaf, unit)
    pooled = F.normalize(pooled, dim=-1)
    sim = pooled @ text.T
    cell_cls = (sim - lam * sim.mean(0, keepdim=True)).argmax(-1)
    pc = np.zeros(P, dtype=np.int64)
    pc[vi.cpu().numpy()] = cell_cls[leaf].cpu().numpy() + 1  # 1..K

    # per-cell majority GT label, owned counts, purity
    sel = gt[owned] > 0
    cop = assigned[owned][sel]
    lop = gt[owned][sel]
    vote = np.zeros((P, K + 1), dtype=np.int64)
    np.add.at(vote, (cop, lop), 1)
    npts = vote[:, 1:].sum(1)
    maj = vote.argmax(1)
    owner = maj > 0
    purity = np.where(npts > 0, vote.max(1) / np.maximum(npts, 1), 0.0)
    correct = (pc == maj) & owner

    acc["radius"].append(radii[owner])
    acc["correct"].append(correct[owner].astype(np.float64))
    acc["npts"].append(npts[owner].astype(np.float64))
    acc["purity"].append(purity[owner])
    acc["majlabel"].append(maj[owner])
    acc["names"] = tnames
    # per-point view: repeat each owner cell by how many points it owns
    acc["pt_radius"].append(np.repeat(radii[owner], npts[owner]))
    acc["pt_correct"].append(np.repeat(correct[owner].astype(np.float64), npts[owner]))
    print(f"  [{scene}] owner cells={owner.sum()} pts={npts[owner].sum()} "
          f"cell-acc={correct[owner].mean():.3f} "
          f"point-acc={np.repeat(correct[owner], npts[owner]).mean():.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--classes", default="opengaussian19")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    acc = {k: [] for k in ["radius", "correct", "npts", "purity", "majlabel",
                           "pt_radius", "pt_correct"]}
    for scene in args.scenes.split(","):
        run_scene(scene, args.variant, args.gt_root, args.lam, args.classes, acc)

    r = np.concatenate(acc["radius"]); c = np.concatenate(acc["correct"])
    n = np.concatenate(acc["npts"]); pu = np.concatenate(acc["purity"])
    ml = np.concatenate(acc["majlabel"])
    pr = np.concatenate(acc["pt_radius"]); pcx = np.concatenate(acc["pt_correct"])

    print(f"\n=== radius vs accuracy ({args.classes}, {len(r)} owner cells, {len(pr)} points) ===")
    qs = np.quantile(r, np.linspace(0, 1, 11))
    print(f"radius deciles (m): " + " ".join(f"{q:.3f}" for q in qs))
    print("\n1) PER-CELL accuracy by radius decile (low radius -> high):")
    print("   " + " ".join(f"{c[(r >= qs[i]) & (r <= qs[i+1])].mean():.3f}" for i in range(10)))
    print("2) mean #points owned by radius decile:")
    print("   " + " ".join(f"{n[(r >= qs[i]) & (r <= qs[i+1])].mean():.1f}" for i in range(10)))
    pqs = np.quantile(pr, np.linspace(0, 1, 11))
    print("3) PER-POINT accuracy by owner-radius decile (what the metric sees):")
    print("   " + " ".join(f"{pcx[(pr >= pqs[i]) & (pr <= pqs[i+1])].mean():.3f}" for i in range(10)))
    err = 1.0 - pcx
    tot_err = err.sum()
    print("4) share of ALL point errors by owner-radius decile:")
    print("   " + " ".join(f"{err[(pr >= pqs[i]) & (pr <= pqs[i+1])].sum()/max(tot_err,1)*100:.1f}%" for i in range(10)))
    print("5) label purity of owner cells by radius decile (do big cells straddle classes?):")
    print("   " + " ".join(f"{pu[(r >= qs[i]) & (r <= qs[i+1])].mean():.3f}" for i in range(10)))

    print("\nSpearman correlations (owner cells):")
    print(f"   radius vs correct      : {spearman(r, c):+.3f}")
    print(f"   radius vs #points      : {spearman(r, n):+.3f}")
    print(f"   radius vs purity       : {spearman(r, pu):+.3f}")
    print(f"   #points vs correct     : {spearman(n, c):+.3f}")
    print(f"   purity vs correct      : {spearman(pu, c):+.3f}")

    names = acc["names"]
    print("\n6) per-class median radius and accuracy (majority-label classes):")
    for k in range(1, len(names) + 1):
        m = ml == k
        if m.sum() < 50:
            continue
        print(f"   {names[k-1]:>16}: cells={int(m.sum()):>6} med_radius={np.median(r[m]):.4f} "
              f"cell_acc={c[m].mean():.3f} med_pts={np.median(n[m]):.0f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"radius_deciles": qs.tolist(),
                       "cell_acc_by_decile": [float(c[(r >= qs[i]) & (r <= qs[i+1])].mean()) for i in range(10)],
                       "point_acc_by_decile": [float(pcx[(pr >= pqs[i]) & (pr <= pqs[i+1])].mean()) for i in range(10)],
                       "err_share_by_decile": [float(err[(pr >= pqs[i]) & (pr <= pqs[i+1])].sum()/max(tot_err,1)) for i in range(10)],
                       "spearman_radius_correct": spearman(r, c),
                       "spearman_radius_npts": spearman(r, n),
                       "spearman_radius_purity": spearman(r, pu)}, f, indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
