"""Render the boundary of the union of PowerFoam's rendered solids, per class.

This replaces the per-cell dipole-facet extraction used by export_class_meshes.py. The renderer
integrates a chord through the convex solid R_i = Ball n radical half-spaces n dipole half-space
(verified in tests/test_foam_solid.py against both kernels), so the visible geometry is the boundary
of the union of those solids, and faces shared between two occupied cells are interior and dropped.
That is what makes the result a connected mesh rather than a pile of disjoint patches.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from export_class_meshes import render, write_mesh_ply  # noqa: E402
from export_surface_viz import PALETTE, POINTCEPT, SPLIT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--sphere-dirs", type=int, default=48)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--no-sphere-caps", action="store_true",
                    help="drop sphere-cap faces, keeping only radical and dipole faces")
    ap.add_argument("--outdir", default="artifacts/surface_viz/meshes_union")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from foam_union_mesh import union_boundary
    from mesh_surface import load_mesh
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
    q = sd["quaternions"].float(); q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nrm = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z),
                       2 * (x * z - w * y)], dim=-1).numpy().astype(np.float64)
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
        from oracle_labels import oracle_labels
        pcls, ostat = oracle_labels(c, r, V, gt_lab, len(names) + 1)
        print(f"[oracle] {ostat['n_cells_with_gt']:,}/{ostat['n_cells']:,} cells own GT, "
              f"purity {ostat['vote_purity']:.3f}", flush=True)

    occupied = (alpha >= a.alpha) & (pcls > 0)
    print(f"occupied cells: {occupied.sum():,}/{len(c):,}", flush=True)
    MV, MF, MC, st = union_boundary(c, r, nrm, adj, off, occupied, pcls,
                                    n_sphere_dirs=a.sphere_dirs,
                                    keep_sphere_caps=not a.no_sphere_caps, progress=5000)
    print(f"faces {st['n_faces']:,}  (interior dropped {st['n_interior_dropped']:,}, "
          f"exposed-neighbour {st['n_exposed_neighbour']:,}, dipole {st['n_dipole']:,}, "
          f"sphere {st['n_sphere']:,}, skipped cells {st['n_skipped']:,})", flush=True)
    print(f"union area {st['area_m2']:.1f} m2  (GT {mesh.get_surface_area():.1f})", flush=True)

    sel = list(enumerate(names, start=1))
    fig, axes = plt.subplots(len(sel), 2, figsize=(5.4, 2.3 * len(sel)), dpi=170)
    axes = np.atleast_2d(axes)
    for row, (k, nm) in enumerate(sel):
        col = PALETTE[k % len(PALETTE)].astype(float)
        vmask = gt_lab == k
        gf = T[vmask[T].all(axis=1)]
        fk = MF[MC == k]
        render(axes[row, 0], V, gf, col, blo, bhi)
        render(axes[row, 1], MV, fk, col, blo, bhi)
        if row == 0:
            axes[row, 0].set_title("GT", fontsize=8)
            axes[row, 1].set_title("foam (union boundary)", fontsize=8)
        axes[row, 0].set_ylabel(nm, fontsize=8)
        axes[row, 0].set_xlabel(f"{len(gf):,} tris", fontsize=6)
        axes[row, 1].set_xlabel(f"{len(fk):,} tris", fontsize=6)
        if len(fk):
            write_mesh_ply(f"{a.outdir}/{scene}_{nm.replace(' ', '_')}_union.ply",
                           MV, fk, PALETTE[k % len(PALETTE)])
        print(f"  {nm:<16} GT {len(gf):>7,} | union {len(fk):>8,}", flush=True)

    tag = "oracle" if a.oracle else "predicted"
    fig.suptitle(f"{scene} - UNION BOUNDARY of the rendered solids ({tag}, alpha>={a.alpha})",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = f"{a.outdir}/{scene}_union_{tag}.png"
    fig.savefig(png, bbox_inches="tight")
    write_mesh_ply(f"{a.outdir}/{scene}_union_all_{tag}.ply", MV, MF, np.array([180, 180, 185]))
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
