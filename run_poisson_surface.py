"""Reconstruct a watertight surface from PowerFoam's dipole faces as ORIENTED POINTS.

THE IDEA. PowerFoam hands you oriented surface elements for free: every cell carries a face centre
p_i, a normal n_i, a soft-Voronoi displacement field, and a density. The dipole face is a genuine
oriented surfel -- the interface between the cell's dense half-space and its zero-density half. That
is exactly the input screened Poisson surface reconstruction consumes, so a watertight mesh comes
out WITHOUT TSDF fusion, without depth rendering, and without any camera. No Gaussian method can do
that: a Gaussian has no interface, only a density, so its mesh has to come from marching a
voxelised field or from fusing rendered depth.

WHY NOT THE UNION-OF-SOLIDS BOUNDARY (foam_union_mesh). That extraction is exact but produces a
surface that does not look like a scanned mesh, for reasons that are measured rather than guessed:

  * 85% of its faces are SPHERE CAPS (1,011,706 of 1,191,717 on scene0062_00). The bounding sphere
    is a rasterization tractability device -- `L_connect` actively shrinks sphere overlap -- not a
    claim about where matter ends, so treating caps as surface is a category error.
  * it thresholds occupancy at alpha >= 0.1, but the renderer never makes a binary decision; what a
    camera sees is where accumulated transmittance falls, which depends on chord length and sigma
    jointly.
  * ~29k independent polytopes at 2.5 cm spacing give a union that is faceted at cell scale.

Here only the dipole faces contribute, they carry the soft-Voronoi displacement, and Poisson fits a
single smooth indicator function through all of them at once -- which is what removes the cell-scale
faceting instead of hiding it.

Normal orientation: the dense half-space is {n_i.(x - p_i) <= h}, so +n_i points from matter into
void, i.e. outward. The paper's `L_normal` trains normals to face the camera, so they are globally
consistent already and no orientation propagation is needed.
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
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--subdiv", type=int, default=2)
    ap.add_argument("--samples-per-m2", type=float, default=20000.0)
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    ap.add_argument("--density-quantile", type=float, default=0.05,
                    help="trim Poisson vertices below this support quantile")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="artifacts/poisson_surface.json")
    ap.add_argument("--outdir", default="artifacts/surface_viz/poisson")
    a = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from foam_exact_surface import dipole_polygon, displaced_patch, sample_triangles
    from mesh_surface import (MeshSurfaceIndex, face_labels, load_mesh,
                              semantic_surface_metrics_mesh, _sample_mesh_uniform)
    from run_dipole_surface import frames
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    scene = a.scene

    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    adj = sd["adjacency"].numpy().astype(np.int64)
    off = sd["adjacency_offsets"].numpy().astype(np.int64)
    deg = np.diff(off)
    nrm, tan, bit = frames(sd["quaternions"].float())
    nrm = nrm.astype(np.float64)
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
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    nc = len(names) + 1
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
    idx = MeshSurfaceIndex(scene, gt_lab, nc)

    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    pcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    pcls[~d["valid_mask"].numpy()] = 0
    if a.oracle:
        from oracle_labels import oracle_labels
        pcls, ostat = oracle_labels(c, r, V, gt_lab, nc)
        print(f"[oracle] {ostat['n_cells_with_gt']:,} cells own GT, "
              f"purity {ostat['vote_purity']:.3f}", flush=True)

    live = np.nonzero((alpha >= a.alpha) & (pcls > 0))[0]
    print(f"contributing cells: {len(live):,}/{len(c):,}", flush=True)

    zero_h = np.zeros(len(c))
    P, N, L = [], [], []
    rng = np.random.default_rng(0)
    for count, i in enumerate(live):
        if count % 5000 == 0:
            print(f"    {count:,}/{len(live):,}", flush=True)
        nb = adj[off[i]:off[i] + deg[i]]
        poly = dipole_polygon(int(i), c, r, nrm, zero_h, nb, blo, bhi, r[i])
        if len(poly) < 3:
            continue
        tri, ar = displaced_patch(poly, c[i], nrm[i], sites_w[i], heights_w[i], r[i],
                                  subdiv=a.subdiv, bound_radius=r[i])
        if ar <= 0:
            continue
        k = max(int(round(ar * a.samples_per_m2)), 1)
        pts = sample_triangles(tri, k, rng)
        if not len(pts):
            continue
        P.append(pts)
        N.append(np.repeat(nrm[i][None, :], len(pts), axis=0))
        L.append(np.full(len(pts), int(pcls[i]), dtype=np.int64))
    P = np.concatenate(P); N = np.concatenate(N); L = np.concatenate(L)
    print(f"oriented samples: {len(P):,}", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    pmesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=a.depth, linear_fit=False)
    dens = np.asarray(dens)
    if a.density_quantile > 0:
        # Poisson extrapolates a closed surface into regions with no support; trimming by the
        # support density removes those invented sheets. This is the standard post-step, not a
        # tuning knob for the score -- without it the mesh balloons far outside the room.
        keep = dens >= np.quantile(dens, a.density_quantile)
        pmesh.remove_vertices_by_mask(~keep)
    pmesh.remove_degenerate_triangles()
    pmesh.remove_unreferenced_vertices()
    MV = np.asarray(pmesh.vertices); MF = np.asarray(pmesh.triangles)
    print(f"poisson mesh: {len(MV):,} verts, {len(MF):,} tris, "
          f"area {pmesh.get_surface_area():.1f} m2 (GT {mesh.get_surface_area():.1f})", flush=True)
    if len(MF) == 0:
        raise SystemExit("empty Poisson mesh")

    # carry the class labels onto the reconstructed mesh: nearest oriented sample per vertex
    vcls = L[cKDTree(P).query(MV, k=1)[1]]
    fl = face_labels(MF, vcls)

    pts_all, cls_all = [], []
    for cid in np.unique(fl):
        if cid <= 0:
            continue
        Fc = MF[fl == cid]
        tri = MV[Fc]
        area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                   tri[:, 2] - tri[:, 0]), axis=1).sum())
        if area <= 0:
            continue
        s = _sample_mesh_uniform(MV, Fc, max(500, int(area * 2500)), 0)
        pts_all.append(s); cls_all.append(np.full(len(s), cid, dtype=np.int64))
    pts_all = np.concatenate(pts_all); cls_all = np.concatenate(cls_all)

    m = semantic_surface_metrics_mesh(idx, pts_all, cls_all)
    tag = "oracle" if a.oracle else "predicted"
    rec = {"scene": scene, "arm": a.arm, "extractor": "poisson_from_dipole_surfels",
           "labels": tag, "depth": a.depth, "density_quantile": a.density_quantile,
           "n_samples": int(len(P)), "n_tris": int(len(MF)),
           "area_m2": float(pmesh.get_surface_area()),
           "gt_area_m2": float(mesh.get_surface_area())}
    rec.update({k: float(m[k]) for k in ("scd", "hd95", "boundary_f1", "n_missed")
                if isinstance(m.get(k), (int, float))})
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    rows.append(rec)
    json.dump(rows, open(a.out, "w"), indent=1)
    o3d.io.write_triangle_mesh(f"{a.outdir}/{scene}_poisson_{tag}.ply", pmesh)
    print(f"\n  SCD {100*rec['scd']:6.2f}cm  HD95 {100*rec['hd95']:7.2f}cm  "
          f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed', 0):.0f}", flush=True)
    print(f"wrote {a.outdir}/{scene}_poisson_{tag}.ply")


if __name__ == "__main__":
    main()
