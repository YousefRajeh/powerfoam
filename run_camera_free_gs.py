"""Camera-free per-class surface metric for a GAUSSIAN arm, matching run_camera_free_surface.py.

WHY VOXELS HERE AND NOT FOR THE FOAM. A foam primitive carries its own surface (the dipole plane
clipped to its power cell), so the foam side is analytic. A Gaussian has no surface -- only a
density -- so the only camera-free way to ask "what geometry does this method assert" is to
evaluate the density field over the whole scene box and march its isosurface. That deliberately
includes Gaussians no camera ever sees, which is the point: a floater behind a wall is geometry the
method claims and a depth-based metric structurally cannot charge it for.

Marching cubes is appropriate on this side and not on the foam side, and that asymmetry is
measured, not assumed: on an analytic sphere a SMOOTH field (what a Gaussian mixture gives)
recovers the area to -0.1% at h = 2 cm, while a PIECEWISE-CONSTANT field (what a foam density is)
overestimates by ~+10% at every resolution tested.

The class of a voxel is that of its largest single contributor there, resolved order-independently
by a scatter-amax followed by an equality match. Per class c, the class-c isosurface is marched,
sampled uniformly by area, and scored against the GT class-c mesh -- the same reference and the same
per-class averaging as the foam side.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")
RECON = {"gs_froz": "recon_remote/gs_froz/{s}/ckpt.pt",
         "gs_unfroz": "recon_remote/gs_unfroz/{s}/ckpt.pt"}
FEATS = {"gs_froz": "artifacts/scannet/{s}/solved_geometric_median_gs_froz_ogl3.pt",
         "gs_unfroz": "artifacts/scannet/{s}/solved_geometric_median_gs_unfroz_ogl3.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arm", default="gs_froz", choices=list(RECON))
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.3, 0.5, 0.9])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", default="artifacts/camera_free_gs.json")
    a = ap.parse_args()

    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import gaussian_volume, grid_from_bbox, isosurface_samples

    enable_determinism()
    rows = []
    for scene in a.scenes:
        ck = RECON[a.arm].format(s=scene)
        fp = FEATS[a.arm].format(s=scene)
        if not (os.path.exists(ck) and os.path.exists(fp)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / {a.arm} ===", flush=True)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sp = sd["splats"] if "splats" in sd else sd
        means = sp["means"].float().numpy()
        scales = torch.exp(sp["scales"].float()).numpy()
        quats = sp["quats"].float().numpy()
        opac = torch.sigmoid(sp["opacities"].float()).numpy().reshape(-1)

        d = torch.load(fp, map_location="cpu", weights_only=True)
        feats = d["primitive_features"].float()
        vm = d["valid_mask"].numpy()

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        if len(V) != len(raw):
            print(f"  [MISALIGNED] mesh {len(V):,} vs labels {len(raw):,}", flush=True)
            continue
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
        nc = len(names) + 1
        text = embed_class_names(names, "cuda")
        neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
        pcls = classify_with_rejection(feats.cuda(), text, neg, a.margin).cpu().numpy()
        pcls[~vm] = 0
        print(f"  [reject] keeps {(pcls > 0).mean():.1%} of gaussians", flush=True)
        idx = MeshSurfaceIndex(scene, gt_lab, nc)

        lo, dims = grid_from_bbox(V.min(0), V.max(0), a.voxel)
        print(f"  grid {dims} = {np.prod(dims)/1e6:.1f}M voxels, {len(means):,} gaussians",
              flush=True)
        dens, cvol = gaussian_volume(means, scales, quats, opac, pcls, lo, dims, a.voxel)
        print(f"  density max {dens.max():.3f} mean {dens.mean():.4f}", flush=True)

        for al in a.alphas:
            # THRESHOLD THE ACCUMULATED OPACITY DIRECTLY. `gaussian_volume` returns
            # sum_i alpha_i * exp(-0.5 * mahalanobis^2), which is already an opacity-like sum, not
            # a volumetric density in 1/m. Converting it with alpha = 1 - exp(-dens * h) -- correct
            # for the foam, whose sigma IS a density -- is dimensionally wrong here and made every
            # level above 0.3 empty (field max 20.06 against a required sigma of 34.7 for alpha
            # 0.5). Both sides now mean the same thing: "opacity of a voxel-sized step".
            lvl = al
            pts, cls, areas = isosurface_samples(dens, cvol, lo, a.voxel, lvl)
            if len(pts) == 0:
                print(f"  alpha {al:<5g} EMPTY at sigma level {lvl:.1f}", flush=True)
                continue
            m = semantic_surface_metrics_mesh(idx, pts, cls)
            rec = {"scene": scene, "arm": a.arm, "alpha": al, "sigma_level": float(lvl),
                   "class_set": a.class_set, "voxel": a.voxel,
                   "area_m2": float(sum(areas.values())),
                   "gt_area_m2": float(mesh.get_surface_area()), "n_pred": int(len(pts))}
            rec.update({k: float(m[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(m.get(k), (int, float))})
            rows.append(rec)
            print(f"  alpha {al:<5g} area {rec['area_m2']:8.1f} (GT {rec['gt_area_m2']:.1f})  "
                  f"SCD {100*rec['scd']:7.2f}cm  HD95 {100*rec['hd95']:7.2f}cm  "
                  f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed',0):.0f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
