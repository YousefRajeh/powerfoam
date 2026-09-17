"""Per-class surface metrics from the foam's VISIBLE dipole surface, against the GT class meshes.

TWO CORRECTIONS OVER run_foam_surface_compare.py, both of which changed what was being measured.

1. THE METRIC IS PER CLASS. The surface predicted as class A is compared against the GT mesh
   restricted to class A, and the result is averaged over the classes present -- exactly
   `mesh_surface.semantic_surface_metrics_mesh`. The earlier comparison passed a constant class and
   so measured one class-agnostic blob; useful for choosing an extractor, but not the metric.

2. ONLY VISIBLE PRIMITIVES CONTRIBUTE. A foam tiles ALL of space, so emitting a patch per opaque
   primitive emits the entire interior: measured 1085-3519 m^2 of surface against a 41.6 m^2 fused
   surface and an 18 m^2 GT mesh, with accuracy stuck at ~32 cm across a 9x sweep of the opacity
   threshold and across two different surface definitions. That insensitivity is the tell -- the
   thing being emitted is invisible geometry, and no threshold on this axis removes it.

   The renderer already answers the question. `rasterize.py:890-893` tracks, per ray, the primitive
   with the largest `alpha * trans` and reports it as `front_prim_idx`. Taking the union of those
   indices over all views gives exactly the set of primitives that are front-most for some camera,
   i.e. the visible envelope. Only those contribute a dipole patch.

WHAT A PATCH IS. `foam_exact_surface.dipole_polygon` clips a primitive's displaced dipole plane
(`n . x = n . p + h`, the same plane `ray_plane_intersect` uses) by its own power cell. Exact
planar polygons, no grid, no isovalue -- verified to 2.96e-16 on closed-form cases.
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
POINTCEPT = r"D:\Downloads\scannet_pointcept"


def _ray_maps(cam):
    """(H, W, 6) origins+directions, resolved exactly as camera.to_warp() does for the rasterizer."""
    rm = cam.ray_maps
    if rm is None and getattr(cam, "cam_ray_dirs", None) is not None:
        rm = cam._build_ray_maps_from_basis()
    if rm is None:
        rm = cam._build_pinhole_ray_maps()
    return rm


def rendered_surface(model, data, max_views=None, stride=1):
    """Back-project `front_t_surf` to world points, labelled by `front_prim_idx`.

    This IS the surface the renderer produces: for each ray, `front_prim_idx` is the primitive
    carrying the largest `alpha * trans` (rasterize.py:890-893) and `front_t_surf` is where that
    ray meets its displaced dipole plane. The returned points are therefore the rendered surface
    samples, one per pixel that hit anything, with no reconstruction step in between.

    `stride` subsamples pixels; the surface is massively oversampled at full resolution and the
    metric only needs enough points to represent each class region.
    """
    cams = data.cameras if max_views is None else data.cameras[:max_views]
    model.update_vis_cache()
    pts, prim = [], []
    for i, cam in enumerate(cams):
        with torch.no_grad():
            # forward_visualization, NOT forward: the front-surface outputs exist only on the
            # visualization path (rasterize.py:2385-2505); the training forward returns None there.
            out = model.forward_visualization(cam)
        front_idx, front_t = out[7], out[8]
        if front_idx is None or front_t is None:
            raise RuntimeError("front_prim_idx / front_t_surf not returned by the rasterizer")
        rm = _ray_maps(cam).to(front_t.device)
        o, d = rm[..., :3], rm[..., 3:]
        hit = (front_idx >= 0) & torch.isfinite(front_t) & (front_t > 0)
        if stride > 1:
            m = torch.zeros_like(hit)
            m[::stride, ::stride] = True
            hit &= m
        if not bool(hit.any()):
            continue
        p = o[hit] + front_t[hit].unsqueeze(-1) * d[hit]
        pts.append(p.detach().cpu().numpy())
        prim.append(front_idx[hit].detach().cpu().numpy().astype(np.int64))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(cams)} views, {sum(len(x) for x in pts):,} surface points",
                  flush=True)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(pts), np.concatenate(prim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-sets", default="opengaussian19")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--stride", type=int, default=4,
                    help="pixel stride when back-projecting; the surface is heavily oversampled")
    ap.add_argument("--out", default="artifacts/foam_visible_surface.json")
    a = ap.parse_args()

    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from foam_exact_surface import foam_dipole_surface
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh

    enable_determinism()
    wp.init()
    rows = []

    for scene in a.scenes:
        ck = f"output/scannet_{scene}_{a.arm}"
        sp = f"artifacts/scannet/{scene}/{a.solved}"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / {a.arm} ===", flush=True)

        p = configargparse.ArgParser()
        add_group(p, Params)
        p.add_argument("-c", "--config", is_config_file=True)
        args = p.parse_args(["-c", f"{ck}/config.yaml"])
        dh = DataHandler(args)
        dh.reload("all", downsample=args.downsample[-1])
        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device="cuda")
        model.load_pt(f"{ck}/model.pt")

        surf_pts, surf_prim = rendered_surface(model, dh, a.max_views, a.stride)
        vis = np.unique(surf_prim)
        c, r = load_points_radii(ck)
        sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        adj = sd["adjacency"].numpy().astype(np.int64)
        off = sd["adjacency_offsets"].numpy().astype(np.int64)
        q = sd["quaternions"].float()
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nrm = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z),
                           2 * (x * z - w * y)], dim=-1).numpy()
        hgt = (sd["texel_height"].float().mean(dim=1) * torch.as_tensor(r).float()).numpy()
        print(f"  {len(surf_pts):,} rendered surface points from {len(vis):,} / {len(c):,} "
              f"primitives ({100*len(vis)/len(c):.1f}%)", flush=True)

        d = torch.load(sp, map_location="cpu", weights_only=True)
        feats = d["primitive_features"].float()
        vm = d["valid_mask"].numpy()

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        blo, bhi = V.min(0), V.max(0)
        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        if len(V) != len(raw):
            print(f"  [MISALIGNED] mesh {len(V):,} vertices vs {len(raw):,} GT labels", flush=True)
            continue
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())

        for cs in a.class_sets.split(","):
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, "cuda")
            # classify_primitives returns -1 for a primitive with no feature, so those become 0
            # (no prediction) and contribute no surface -- the same rule the point metric uses.
            pcls = classify_primitives(feats.cuda(), text).cpu().numpy() + 1
            pcls[~vm] = 0

            # label each rendered surface point by its front primitive's class; points whose
            # primitive has no feature (class 0) are dropped rather than defaulting to a class
            cls = pcls[surf_prim]
            keep_pt = cls > 0
            pts, cls = surf_pts[keep_pt], cls[keep_pt]
            areas = {}
            print(f"  {cs}: {len(pts):,} surface points carry a class "
                  f"({100*keep_pt.mean():.1f}% of rendered), {len(names)} classes present",
                  flush=True)
            idx = MeshSurfaceIndex(scene, gt_lab, nc)
            m = semantic_surface_metrics_mesh(idx, pts, cls)
            rec = {"scene": scene, "arm": a.arm, "class_set": cs,
                   "visible_frac": float(len(vis) / len(c)),
                   "gt_area_m2": float(mesh.get_surface_area()),
                   "n_pred": int(len(pts)), "stride": a.stride}
            rec.update({k: float(m[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(m.get(k), (int, float))})
            if a.per_class:
                pc_rows = []
                for cid, mm in sorted(m.get("per_class", {}).items()):
                    nm = names[cid - 1] if 0 < cid <= len(names) else str(cid)
                    pc_rows.append((nm, mm))
                    print(f"      {nm:<16} n_pred {mm.get('n_pred',0):>9,}  "
                          f"area {mm.get('area_m2',0):6.2f}  "
                          + ("MISSED" if mm.get("missed") else
                             f"acc {100*mm['mae_pred2gt']:6.1f}cm  comp {100*mm['mae_gt2pred']:6.1f}cm"
                             f"  bF1 {mm['boundary_f1']:.3f}"), flush=True)
                rec["per_class"] = {k: {kk: (float(vv) if isinstance(vv,(int,float)) else vv)
                                        for kk, vv in v.items()} for k, v in pc_rows}
            rows.append(rec)
            print(f"  -> SCD {100*rec['scd']:.2f}cm  HD95 {100*rec['hd95']:.2f}cm  "
                  f"BF1 {rec['boundary_f1']:.3f}  acc {100*rec['mae_pred2gt']:.2f}cm  "
                  f"comp {100*rec['mae_gt2pred']:.2f}cm  missed {rec.get('n_missed', 0):.0f}",
                  flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
