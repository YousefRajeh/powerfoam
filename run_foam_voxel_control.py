"""CONTROL: score the foam through the Gaussians' extraction path, to separate representation
quality from extractor sharpness.

The exact-polygon foam scores BF1 0.375 on scene0062_00 against 0.024-0.151 for every Gaussian
method. Two explanations fit that gap and they have opposite consequences for the paper:

  (a) the foam's geometry really is sharper, or
  (b) exact planar polygons beat a 2 cm marching-cubes grid, and the gap is an artifact of the
      EXTRACTOR rather than a property of the representation -- in which case the comparison is
      unfair and the number must be restated.

This run distinguishes them. The foam is voxelised on the same grid at the same 2 cm resolution
(`foam_volume`, exact power-cell ownership per voxel) and marched with the same `isosurface_samples`
the Gaussian side uses. Everything else -- classification with rejection, per-class scoring against
the class-restricted GT mesh -- is unchanged. If BF1 stays near 0.375 the sharpness is the
representation; if it collapses toward 0.08 it was the extractor.

Known and deliberate: marching cubes over a PIECEWISE-CONSTANT field over-estimates area by ~+10% at
every resolution tested (see foam_exact_surface's docstring), so this control is biased AGAINST the
foam. That is the right direction for a control whose purpose is to avoid flattering our own method.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--patch-radius", type=float, default=1.0,
                    help="compact support in multiples of local spacing; 0 disables")
    ap.add_argument("--out", default="artifacts/cf_foam_voxel_control.json")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from run_camera_free_surface import local_spacing
    from surface_extract import foam_volume, grid_from_bbox, isosurface_samples

    enable_determinism()
    rows = []
    for scene in a.scenes:
        ck = f"output/scannet_{scene}_{a.arm}"
        sp = f"artifacts/scannet/{scene}/{a.solved}"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / {a.arm} (voxelised control) ===", flush=True)
        c, r = load_points_radii(ck)
        c = np.asarray(c); r = np.asarray(r)
        sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
        alpha = 1.0 - np.exp(-sigma * 2.0 * r)

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        _, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        if len(V) != len(raw):
            print(f"  [MISALIGNED] {scene}", flush=True)
            continue
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
        text = embed_class_names(names, "cuda")
        neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
        idx = MeshSurfaceIndex(scene, gt_lab, len(names) + 1)

        d = torch.load(sp, map_location="cpu", weights_only=True)
        pcls = classify_with_rejection(d["primitive_features"].float().cuda(), text,
                                       neg, a.margin).cpu().numpy()
        pcls[~d["valid_mask"].numpy()] = 0

        lo, dims = grid_from_bbox(V.min(0), V.max(0), a.voxel)
        print(f"  grid {dims} = {np.prod(dims)/1e6:.1f}M voxels, {len(c):,} cells", flush=True)
        # voxelise the per-cell OPACITY, not sigma, so the level `al` means the same thing it does
        # on the Gaussian side: how opaque a voxel-sized step must be to count as occupied.
        occ = np.where(pcls > 0, alpha, 0.0)
        md = None
        if a.patch_radius > 0:
            adj = sd["adjacency"].numpy().astype(np.int64)
            offs = sd["adjacency_offsets"].numpy().astype(np.int64)
            md = local_spacing(c, adj, offs) * a.patch_radius
        dens, cvol = foam_volume(c, r, occ, pcls, lo, dims, a.voxel, max_dist=md)
        print(f"  occupancy max {dens.max():.3f} mean {dens.mean():.4f}", flush=True)

        for al in a.alphas:
            pts, cls, areas = isosurface_samples(dens, cvol, lo, a.voxel, al)
            if len(pts) == 0:
                print(f"  alpha {al:<5g} EMPTY", flush=True)
                continue
            m = semantic_surface_metrics_mesh(idx, pts, cls)
            rec = {"scene": scene, "arm": a.arm, "alpha": al, "extractor": "voxel+mc",
                   "voxel": a.voxel, "area_m2": float(sum(areas.values())),
                   "gt_area_m2": float(mesh.get_surface_area())}
            rec.update({k: float(m[k]) for k in
                        ("scd", "hd95", "boundary_f1", "n_missed")
                        if isinstance(m.get(k), (int, float))})
            rows.append(rec)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"  alpha {al:<5g} area {rec['area_m2']:7.1f} (GT {rec['gt_area_m2']:.1f})  "
                  f"SCD {100*rec['scd']:6.2f}cm  HD95 {100*rec['hd95']:7.2f}cm  "
                  f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed', 0):.0f}", flush=True)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
