"""Camera-free per-class surface metric: score the geometry a method HAS, not the geometry a
camera happens to show.

WHY NOT BACK-PROJECTED DEPTH. The previous version back-projected each renderer's depth and scored
those points. That is camera-specific by construction: a floater hidden behind a surface, or thin
enough that the depth render walks past it, is never scored even though it is exactly the geometric
error that makes a representation worse. For 3DGS that is the dominant failure mode, so the metric
was structurally blind to the thing it was meant to catch. Gating by `front_prim_idx` made it worse
-- the "excess" surface that gating removed IS the signal.

THE TWO SIDES, both camera-independent and both per class: the predicted surface labelled c is
compared against the GT mesh restricted to class c, and the result averaged over classes present.

  foam       Each primitive already carries its own surface -- the displaced dipole plane the
             renderer clips rays to -- so its patch is that plane clipped to its power cell. Exact
             planar polygons, no grid, no isovalue (foam_exact_surface, verified to 2.96e-16 on
             closed-form cases). Every primitive is eligible, whether or not a camera sees it.
  gaussians  No per-primitive surface exists, so the density field is voxelised over the scene box
             -- ALL Gaussians, including ones no camera sees -- and the class-c isosurface is
             marched from it. Marching cubes is accurate here because a Gaussian mixture is smooth
             (measured -0.1% area error on an analytic sphere); the +10% step-function bias that
             ruled it out for the foam does not apply.

TWO MEMBERSHIP RULES for the foam, run side by side because they answer different questions:

  --rule opaque     every primitive with alpha >= alpha_min contributes its patch. Simplest and
                    most punishing: an opaque cell sitting inside a wall is counted as surface that
                    should not be there, which is the same charge a 3DGS floater incurs.
  --rule boundary   only primitives on the boundary of their class contribute, i.e. those with at
                    least one power-diagram neighbour of a different class (or a transparent one).
                    Closer to "the boundary of the matter": a cell fully enclosed by same-class
                    neighbours is an internal partition of a space-filling tessellation, not a
                    surface. Strictly a subset of `opaque`.
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


def local_spacing(centers, adjacency, offsets):
    """Mean distance from each cell to its power-diagram neighbours -- the local sampling scale."""
    deg = np.diff(offsets)
    out = np.zeros(len(centers))
    for i in range(len(centers)):
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        if len(nb):
            out[i] = float(np.linalg.norm(centers[nb] - centers[i], axis=1).mean())
    return out


def boundary_mask(prim_class, alpha, adjacency, offsets, alpha_min):
    """True for primitives on the boundary of their class: some neighbour differs in class or is
    transparent. A cell fully surrounded by same-class matter is interior, not surface."""
    deg = np.diff(offsets)
    live = alpha >= alpha_min
    out = np.zeros(len(prim_class), dtype=bool)
    for i in np.nonzero(live & (prim_class > 0))[0]:
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        if len(nb) == 0:
            out[i] = True
            continue
        differing = (prim_class[nb] != prim_class[i]) | (~live[nb])
        out[i] = bool(differing.any())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--rules", nargs="*", default=["opaque", "boundary", "interface"])
    # Swept, not fixed, so the foam is on the SAME knob as the Gaussian side: there `alpha` is the
    # accumulated opacity of a voxel-sized step, here it is the per-cell opacity 1-exp(-sigma*2r).
    # Both answer "how confident must matter be before we call it surface", and comparing the foam
    # at one alpha against a Gaussian sweep was the reason the first comparison was lopsided.
    ap.add_argument("--alpha-min", nargs="*", type=float, default=[0.1])
    ap.add_argument("--margin", type=float, default=0.0)
    # Compact support, in multiples of the local point spacing. 0 disables it, which
    # lets an air-facing cell own its plane out to the scene box (measured: one cell
    # claiming 8.9 m2 against an 18 m2 GT).
    ap.add_argument("--patch-radius", type=float, default=1.0)
    ap.add_argument("--out", default="artifacts/camera_free_surface.json")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from foam_exact_surface import foam_dipole_surface, foam_isosurface
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection

    enable_determinism()
    rows = []
    for scene in a.scenes:
        ck = f"output/scannet_{scene}_{a.arm}"
        sp = f"artifacts/scannet/{scene}/{a.solved}"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / {a.arm} ===", flush=True)
        c, r = load_points_radii(ck)
        sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
        adj = sd["adjacency"].numpy().astype(np.int64)
        off = sd["adjacency_offsets"].numpy().astype(np.int64)
        q = sd["quaternions"].float()
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nrm = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z),
                           2 * (x * z - w * y)], dim=-1).numpy()
        hgt = (sd["texel_height"].float().mean(dim=1) * torch.as_tensor(r).float()).numpy()
        alpha = 1.0 - np.exp(-sigma * 2.0 * r)
        spacing = local_spacing(np.asarray(c), adj, off)

        d = torch.load(sp, map_location="cpu", weights_only=True)
        feats = d["primitive_features"].float()
        vm = d["valid_mask"].numpy()

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        blo, bhi = V.min(0), V.max(0)
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
        print(f"  [reject] keeps {(pcls > 0).mean():.1%} of cells", flush=True)
        idx = MeshSurfaceIndex(scene, gt_lab, nc)

        for rule, amin in ((ru, am) for ru in a.rules for am in a.alpha_min):
            if rule == "interface":
                # THE ACTUAL ANALOGUE OF THE GAUSSIAN ISOSURFACE. A marched isosurface at level
                # `al` is the boundary of the occupied set, d{x : occ(x) >= al}. Its foam equivalent
                # is not a dipole patch but the union of power-diagram FACETS separating a live cell
                # from a dead one -- same set, computed exactly instead of marched. `opaque` and
                # `boundary` both emit a patch per CELL, so they charge interior partitions of a
                # space-filling tessellation as surface, which is why they report ~43x the GT area.
                occ = np.where(pcls > 0, alpha, -1.0)
                pts, cls, areas = foam_isosurface(c, r, occ, pcls, adj, off, amin, blo, bhi)
                keep = alpha >= amin
            elif rule == "opaque":
                keep = (alpha >= amin) & (pcls > 0)
            elif rule == "boundary":
                keep = boundary_mask(pcls, alpha, adj, off, amin)
            else:
                raise SystemExit(f"unknown rule {rule}")
            if rule != "interface":
                gate = np.where(keep, 1.0, 0.0)
                mr = (spacing * a.patch_radius) if a.patch_radius > 0 else None
                pts, cls, areas = foam_dipole_surface(c, r, nrm, hgt, gate, pcls, adj, off,
                                                      blo, bhi, alpha_min=0.5, max_radius=mr)
            m = semantic_surface_metrics_mesh(idx, pts, cls)
            rec = {"scene": scene, "arm": a.arm, "rule": rule, "alpha": amin,
                   "class_set": a.class_set,
                   "n_prims": int(len(c)), "n_contrib": int(keep.sum()),
                   "area_m2": float(sum(areas.values())),
                   "gt_area_m2": float(mesh.get_surface_area()), "n_pred": int(len(pts))}
            rec.update({k: float(m[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(m.get(k), (int, float))})
            rows.append(rec)
            print(f"  {rule:<9} contrib {rec['n_contrib']:>8,}/{rec['n_prims']:,}  "
                  f"area {rec['area_m2']:8.1f} (GT {rec['gt_area_m2']:.1f})  "
                  f"SCD {100*rec['scd']:7.2f}cm  HD95 {100*rec['hd95']:7.2f}cm  "
                  f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed',0):.0f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
