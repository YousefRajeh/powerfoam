"""Shaded MESH renders of each extracted class -- real triangles, not scatter plots.

The scatter panels answered "is this fog or a surface" but they cannot show whether a surface is
flat, oriented correctly, or riddled with holes, because every point is drawn the same regardless of
the geometry it came from. Here each class is turned into an actual triangle mesh and rendered with
flat shading, so orientation and continuity are visible:

  GT     the ScanNet mesh restricted to triangles whose vertices all carry class c
  foam   the exact dipole polygons of class-c cells, fan-triangulated (compact-support capped)
  3DGS   marching cubes on the class-c density field, the same extraction the metric scores

Each class also gets a .ply mesh so it can be opened directly. Shading is a fixed headlight dotted
with the facet normal -- no external renderer, so this works headless.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from export_surface_viz import PALETTE, POINTCEPT, SPLIT  # noqa: E402


def write_mesh_ply(path, verts, faces, rgb):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(verts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {len(faces)}\n"
                 "property list uchar int vertex_indices\nend_header\n").encode())
        v = np.empty(len(verts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        v["r"], v["g"], v["b"] = rgb[0], rgb[1], rgb[2]
        f.write(v.tobytes())
        fa = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        fa["n"] = 3
        fa["a"], fa["b"], fa["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
        f.write(fa.tobytes())


def polygons_to_mesh(polys):
    """Fan-triangulate a list of convex polygons into one (verts, faces) pair."""
    verts, faces, off = [], [], 0
    for p in polys:
        if len(p) < 3:
            continue
        verts.append(p)
        for k in range(1, len(p) - 1):
            faces.append((off, off + k, off + k + 1))
        off += len(p)
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts), np.array(faces, dtype=np.int64)


def render(ax, verts, faces, base_rgb, lo, hi, max_faces=90000, seed=0):
    """Flat-shaded render of a triangle soup, viewed down -y (the panels' front view)."""
    from matplotlib.collections import PolyCollection
    if len(faces) == 0:
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        return
    if len(faces) > max_faces:
        faces = faces[np.random.default_rng(seed).choice(len(faces), max_faces, replace=False)]
    tri = verts[faces]                                            # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(ln, 1e-12, None)
    light = np.array([0.35, -0.85, 0.4]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)                       # abs: patches are two-sided
    depth = tri[:, :, 1].mean(axis=1)
    order = np.argsort(-depth)                                    # painter's algorithm
    tri, shade = tri[order], shade[order]
    cols = np.clip((base_rgb / 255.0)[None, :] * shade[:, None], 0, 1)
    ax.add_collection(PolyCollection(tri[:, :, [0, 2]], facecolors=cols,
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gs-alpha", type=float, default=0.9)
    ap.add_argument("--bound", default="sphere", choices=["sphere", "spacing", "none"])
    ap.add_argument("--patch-radius", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true",
                    help="label cells from GT instead of CLIP: the metric ceiling")
    ap.add_argument("--subdiv", type=int, default=3)
    # Build cells from the TRUE facet graph. model.pt's `adjacency` is the renderer traversal
    # structure and omits 56.1% of real power-diagram facets, so cells clipped with it are
    # under-constrained and their patches spill past the neighbours they should stop at.
    ap.add_argument("--adjacency",
                    default="artifacts/scannet/{scene}/adjacency_true_facet_frozen.pt",
                    help="pass 'model' for the old traversal adjacency")
    ap.add_argument("--classes", nargs="*", default=None, help="default: every present class")
    ap.add_argument("--outdir", default="artifacts/surface_viz/meshes")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from foam_exact_surface import dipole_polygon, displaced_patch
    from mesh_surface import load_mesh
    from run_camera_free_surface import boundary_mask, local_spacing
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import gaussian_volume, grid_from_bbox

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    scene = a.scene

    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c); r = np.asarray(r)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    if a.adjacency == "model":
        adj = sd["adjacency"].numpy().astype(np.int64)
        off = sd["adjacency_offsets"].numpy().astype(np.int64)
    else:
        _tf = torch.load(a.adjacency.format(scene=scene), map_location="cpu", weights_only=False)
        assert int(_tf["num_primitives"]) == len(c), (int(_tf["num_primitives"]), len(c))
        adj = np.asarray(_tf["adjacent"]).astype(np.int64)
        off = np.asarray(_tf["offsets"]).astype(np.int64)
    deg = np.diff(off)
    print(f"adjacency: {'model traversal' if a.adjacency=='model' else 'TRUE facets'}, "
          f"mean degree {len(adj)/len(c):.2f}", flush=True)
    q = sd["quaternions"].float(); q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nrm = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z),
                       2 * (x * z - w * y)], dim=-1).numpy()
    from run_dipole_surface import frames
    nrm, tan, bit = frames(sd["quaternions"].float())
    s2 = sd["texel_sites"].float().numpy()
    sites_w = (c[:, None, :] + r[:, None, None] *
               (s2[..., 0:1] * tan[:, None, :] + s2[..., 1:2] * bit[:, None, :]))
    heights_w = sd["texel_height"].float().numpy() * r[:, None]
    alpha = 1.0 - np.exp(-sigma * 2.0 * r)

    mesh = load_mesh(scene)
    V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
    blo, bhi = V.min(0), V.max(0)
    _, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")

    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    pcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
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
    # Match the SCORER's membership (run_dipole_surface): every classified cell above the alpha
    # floor. The exporter used to hardcode the `boundary` rule at alpha 0.9, so its meshes showed
    # a different subset of cells than the numbers next to them were computed from.
    keep = (pcls > 0) & (alpha >= a.alpha)
    # patch_radius <= 0 means NO cap. Multiplying by zero would clip every patch to a
    # zero-radius disc and silently produce an empty mesh.
    if a.bound == 'sphere':
        mr = r.copy()
    elif a.bound == 'spacing':
        mr = local_spacing(c, adj, off) * a.patch_radius if a.patch_radius > 0 else None
    else:
        mr = None

    # foam patches, grouped by class. The base face is at h = 0 and every vertex is then pushed
    # along the normal by the soft-Voronoi displacement (Eq. 3), exactly as run_dipole_surface.py
    # scores it -- an earlier version of this exporter fan-triangulated the FLAT mean-height plane,
    # so the meshes it wrote were the tau -> 0 collapse of the real surface and carried none of the
    # detail sites' geometry.
    zero_h = np.zeros(len(c))
    foam_tris, foam_alpha = {}, {}
    for i in np.nonzero(keep)[0]:
        poly = dipole_polygon(int(i), c, r, nrm, zero_h, adj[off[i]:off[i] + deg[i]],
                              blo, bhi, None if mr is None else mr[i])
        if len(poly) < 3:
            continue
        t, ar = displaced_patch(poly, c[i], nrm[i], sites_w[i], heights_w[i], r[i],
                                subdiv=a.subdiv,
                                bound_radius=r[i] if a.bound == 'sphere' else None)
        if ar > 0:
            foam_tris.setdefault(int(pcls[i]), []).append(t)
            foam_alpha.setdefault(int(pcls[i]), []).append(np.full(len(t), alpha[i]))

    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    spg = sdg["splats"] if "splats" in sdg else sdg
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    lo, dims = grid_from_bbox(blo, bhi, a.voxel)
    dens, cvol = gaussian_volume(spg["means"].detach().float().numpy(),
                                 torch.exp(spg["scales"].detach().float()).numpy(),
                                 spg["quats"].detach().float().numpy(),
                                 torch.sigmoid(spg["opacities"].detach().float()).numpy().reshape(-1),
                                 gcls, lo, dims, a.voxel)

    want = a.classes or names
    sel = [(k, nm) for k, nm in enumerate(names, start=1) if nm in want]
    fig, axes = plt.subplots(len(sel), 3, figsize=(7.6, 2.3 * len(sel)), dpi=170)
    axes = np.atleast_2d(axes)

    for row, (k, nm) in enumerate(sel):
        col = PALETTE[k % len(PALETTE)].astype(float)

        vmask = gt_lab == k
        tmask = vmask[T].all(axis=1)
        gv, gf = V, T[tmask]

        ts = foam_tris.get(k, [])
        if ts:
            allt = np.concatenate(ts, axis=0)
            fv = allt.reshape(-1, 3)
            ff = np.arange(len(fv), dtype=np.int64).reshape(-1, 3)
        else:
            fv, ff = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        dk = np.where(cvol == k, dens, 0.0)
        if float(dk.max()) >= a.gs_alpha:
            gvv, gff, _, _ = measure.marching_cubes(dk, level=a.gs_alpha)
            gvv = lo + (gvv + 0.5) * a.voxel
        else:
            gvv, gff = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        for ax, (vv, ffc, ttl) in zip(axes[row], [(gv, gf, "GT"), (fv, ff, "foam"),
                                                  (gvv, gff, "3DGS")]):
            render(ax, vv, ffc, col, blo, bhi)
            if row == 0:
                ax.set_title(ttl, fontsize=8)
        axes[row, 0].set_ylabel(f"{nm}", fontsize=8)
        axes[row, 1].set_xlabel(f"{len(ff):,} tris", fontsize=6)
        axes[row, 2].set_xlabel(f"{len(gff):,} tris", fontsize=6)
        axes[row, 0].set_xlabel(f"{len(gf):,} tris", fontsize=6)

        for nmm, vv, ffc in (("gt", gv, gf), ("foam", fv, ff), ("gs", gvv, gff)):
            if len(ffc):
                write_mesh_ply(f"{a.outdir}/{scene}_{nm.replace(' ', '_')}_{nmm}.ply",
                               vv, ffc, PALETTE[k % len(PALETTE)])
        print(f"  {nm:<16} GT {len(gf):>7,} tris | foam {len(ff):>8,} | 3DGS {len(gff):>8,}",
              flush=True)

    fig.suptitle(f"{scene} - per-class extracted MESH (foam boundary a={a.alpha}, cap "
                 f"{a.patch_radius}x spacing; 3DGS marching cubes a={a.gs_alpha})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = f"{a.outdir}/{scene}_class_meshes.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"\nwrote {png} and per-class PLY meshes -> {a.outdir}")


if __name__ == "__main__":
    main()
