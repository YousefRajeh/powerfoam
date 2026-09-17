"""The foam's surface as the paper defines it: displaced dipole faces, weighted by opacity.

This replaces two approximations in run_camera_free_surface.py, both of which were mine and neither
of which is in the model:

  1. FLAT PATCHES. The old code took `texel_height.mean(dim=1)`, i.e. a single plane per primitive.
     The paper (arXiv:2604.24994, Eq. 3) puts k learnable DETAIL SITES on each dipole face carrying
     a displacement field combined by a soft Voronoi,
         d(x) = sum_i exp(-tau ||x - s_i||^2 / r^2) d_i / sum_i exp(...)
     with tau = 10 (rasterize.py:64), sites stored in units of radius in the (tangent, bitangent)
     frame and heights in units of radius (scene.py:397-410). Averaging the heights is the tau -> 0
     limit of that softmax, so the old extraction deleted exactly the high-frequency geometry the
     detail sites exist to carry. `foam_exact_surface.soft_voronoi_height` now reproduces the
     renderer's own loop to 3e-18.

  2. A HARD OPACITY GATE PLUS AN AD-HOC SIZE CAP. The old code kept cells with alpha >= 0.1 and then
     clipped each patch to a disc of k x the local spacing. The cap worked but has no counterpart in
     the model, and it was papering over the real issue: a cell that spans the empty room is not a
     defect. The paper fixes the OUTSIDE half-space of every dipole to zero density, so such a cell
     is a correct description of "wall on one side, void on the other" -- and it is nearly
     transparent, which is why it never appears in a render (measured: the largest 5% of patches
     have mean alpha 0.055 and only 1.2% are opaque). Counting it at full weight was the error.
     Here each patch instead contributes in proportion to its own opacity, which is what decides
     whether the renderer shows it. No cap, no threshold, no free parameter.

Effective area is therefore sum_i alpha_i * A_i, and points are drawn per patch in proportion to
alpha_i * A_i, so a near-transparent metre-wide patch contributes a metre-wide patch's worth of
almost nothing rather than all of it.
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


def frames(quats):
    """Normal, tangent and bitangent exactly as scene.get_normals / get_tangents build them."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    t = torch.stack([2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)], -1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = torch.cross(n, t, dim=-1)
    return n.numpy(), t.numpy(), b.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true",
                    help="label cells from GT instead of CLIP: the metric ceiling")
    ap.add_argument("--subdiv", type=int, default=3)
    # Locality, in multiples of the local point spacing. MEASURED TO BE NECESSARY: alpha-weighting
    # alone leaves 1361.9 m2 of claimed surface against an 18 m2 GT, because the large patches that
    # survive are mostly OPAQUE, not transparent. Opacity and locality constrain different things.
    # BOUND: how each cell is limited. "sphere" is the model's own Ball(p_i, r_i) -- the
    # defining feature of a BOUNDED power diagram and a learned parameter, so it costs nothing
    # to disclose. "spacing" is the ad-hoc k x local-spacing disc used before I read the paper;
    # it was an approximation of the sphere (median r/spacing = 0.563, best k empirically 0.5).
    ap.add_argument("--bound", default="sphere", choices=["sphere", "spacing", "none"])
    ap.add_argument("--patch-radius", type=float, default=1.0,
                    help="only used when --bound spacing")
    ap.add_argument("--samples-per-m2", type=float, default=2500.0)
    ap.add_argument("--min-alpha", type=float, default=1e-3,
                    help="numerical cutoff only: below this a patch contributes <0.1% of a sample")
    # THE POWER CELL MUST BE BUILT FROM THE TRUE FACET GRAPH. model.pt's `adjacency` is the
    # renderer's traversal structure (degree 7.48 on scene0062/truefrozen); the regular
    # triangulation gives degree 11.39, and 56.1% of true facets are ABSENT from the stored list.
    # Clipping a cell with only half its half-spaces leaves it under-constrained, so the patch is
    # far larger than the real cell -- which is where the "giant air-facing patches" came from.
    ap.add_argument("--adjacency", default="artifacts/scannet/{scene}/adjacency_true_facet_frozen.pt",
                    help="true facet CSR; pass 'model' to use model.pt's traversal adjacency")
    ap.add_argument("--out", default="artifacts/cf_dipole_alpha.json")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from foam_exact_surface import dipole_polygon, displaced_patch, sample_triangles
    from run_camera_free_surface import local_spacing
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection

    enable_determinism()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []   # resume
    done = {(r["scene"], r["arm"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} (scene, arm) rows already done", flush=True)
    for scene in a.scenes:
        if (scene, a.arm) in done:
            continue
        ck = f"output/scannet_{scene}_{a.arm}"
        sp = f"artifacts/scannet/{scene}/{a.solved}"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / {a.arm} (displaced dipole, alpha-weighted) ===", flush=True)
        c, r = load_points_radii(ck)
        c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
        sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
        if a.adjacency == "model":
            adj = sd["adjacency"].numpy().astype(np.int64)
            off = sd["adjacency_offsets"].numpy().astype(np.int64)
        else:
            tf = torch.load(a.adjacency.format(scene=scene), map_location="cpu",
                            weights_only=False)
            if int(tf["num_primitives"]) != len(c):
                print(f"  [SKIP] facet graph is for {int(tf['num_primitives']):,} primitives, "
                      f"this arm has {len(c):,}", flush=True)
                continue
            adj = np.asarray(tf["adjacent"]).astype(np.int64)
            off = np.asarray(tf["offsets"]).astype(np.int64)
        deg = np.diff(off)
        print(f"  adjacency: {a.adjacency if a.adjacency=='model' else 'true facets'}, "
              f"mean degree {len(adj)/len(c):.2f}", flush=True)
        nrm, tan, bit = frames(sd["quaternions"].float())
        # sites are (N, k, 2) in units of radius in the (tangent, bitangent) frame; heights are in
        # units of radius -- scene.py:397-410 converts both exactly this way.
        s2 = sd["texel_sites"].float().numpy()
        sites_w = (c[:, None, :] + r[:, None, None] *
                   (s2[..., 0:1] * tan[:, None, :] + s2[..., 1:2] * bit[:, None, :]))
        heights_w = sd["texel_height"].float().numpy() * r[:, None]
        alpha = 1.0 - np.exp(-sigma * 2.0 * r)

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        blo, bhi = V.min(0), V.max(0)
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
        if a.oracle:
            # ORACLE: replace the lifted semantics with ground truth, so any remaining
            # error is the extractor's, not the features'. This is the ceiling of the
            # metric on this representation.
            from oracle_labels import oracle_labels
            pcls, ostat = oracle_labels(c, r, V, gt_lab, len(names) + 1)
            print(f"  [oracle] {ostat['n_cells_with_gt']:,}/{ostat['n_cells']:,} cells "
                  f"({ostat['frac_cells_with_gt']:.1%}) own GT; "
                  f"{ostat['mean_gt_per_labelled_cell']:.1f} GT pts/cell; "
                  f"vote purity {ostat['vote_purity']:.3f}", flush=True)

        zero_h = np.zeros(len(c))
        if a.bound == 'sphere':
            mr = r.copy()                    # base face n Ball(p_i, r_i) is a disc of radius r_i
        elif a.bound == 'spacing':
            mr = local_spacing(c, adj, off) * a.patch_radius if a.patch_radius > 0 else None
        else:
            mr = None
        live = np.nonzero((pcls > 0) & (alpha >= a.min_alpha))[0]
        print(f"  {len(live):,}/{len(c):,} cells contribute", flush=True)
        tris, wts, cls_of = [], [], []
        raw_area = 0.0
        for i in live:
            # The BASE face sits at h = 0: `plane_intersection_fwd_local` calls
            # ray_plane_intersect without a height for the first intersection, and only the
            # SECOND call carries the soft-Voronoi displacement. So the polygon is the undisplaced
            # face clipped to the cell, and `displaced_patch` applies d(x) on top of it.
            poly = dipole_polygon(int(i), c, r, nrm, zero_h,
                                  adj[off[i]:off[i] + deg[i]], blo, bhi,
                                  None if mr is None else mr[i])
            if len(poly) < 3:
                continue
            tri, ar = displaced_patch(poly, c[i], nrm[i], sites_w[i], heights_w[i], r[i],
                                      subdiv=a.subdiv)
            if ar <= 0:
                continue
            tris.append(tri); wts.append(alpha[i] * ar); cls_of.append(int(pcls[i]))
            raw_area += ar
        wts = np.array(wts)
        eff = float(wts.sum())
        print(f"  raw patch area {raw_area:8.1f} m2  ->  alpha-weighted {eff:7.1f} m2  "
              f"(GT {mesh.get_surface_area():.1f})", flush=True)

        rng = np.random.default_rng(0)
        pts, cls = [], []
        n_tot = max(int(eff * a.samples_per_m2), 20000)
        share = wts / max(wts.sum(), 1e-12)
        for t, sh, k in zip(tris, share, cls_of):
            n = int(round(n_tot * sh))
            if n <= 0:
                continue
            p = sample_triangles(t, n, rng)
            if len(p):
                pts.append(p); cls.append(np.full(len(p), k, dtype=np.int64))
        if not pts:
            print("  EMPTY", flush=True)
            continue
        pts = np.concatenate(pts); cls = np.concatenate(cls)
        m = semantic_surface_metrics_mesh(idx, pts, cls)
        rec = {"scene": scene, "arm": a.arm, "extractor": "displaced_dipole_alpha_weighted", "patch_radius": a.patch_radius, "bound": a.bound,
               "n_contrib": int(len(tris)), "raw_area_m2": raw_area, "area_m2": eff,
               "gt_area_m2": float(mesh.get_surface_area()), "n_pred": int(len(pts))}
        rec.update({k: float(m[k]) for k in ("scd", "hd95", "boundary_f1", "n_missed")
                    if isinstance(m.get(k), (int, float))})
        rows.append(rec)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"  SCD {100*rec['scd']:6.2f}cm  HD95 {100*rec['hd95']:7.2f}cm  "
              f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed', 0):.0f}", flush=True)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
