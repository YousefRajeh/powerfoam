"""Per-CLASS semantic surface: what each method actually extracts when asked for one object.

The camera-free metric averages SCD/HD95/BF1 over the classes present in a scene, so a single number
hides which objects were extracted cleanly and which were smeared across the room. This renders one
ROW per class -- GT class-c surface, then each method's class-c surface -- and prints the per-class
metrics next to it, which is the form in which an open-vocabulary result is actually used: ask for
"chair", get the chair.

Cells are classified with rejection (semantic_reject), so a cell that claims no queried class is
dropped rather than forced into the nearest one.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

from export_surface_viz import PALETTE, SPLIT, POINTCEPT, write_ply  # noqa: E402


def tile(ax, pts, lo, hi, color, title, cap=40000, seed=0):
    if len(pts) > cap:
        k = np.random.default_rng(seed).choice(len(pts), cap, replace=False)
        pts = pts[k]
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 2], s=0.5, marker=".", linewidths=0, c=[color / 255.0])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--rule", default="boundary", choices=["opaque", "boundary", "interface"])
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--gs-alpha", type=float, default=0.9)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--patch-radius", type=float, default=1.0)
    ap.add_argument("--outdir", default="artifacts/surface_viz")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from foam_exact_surface import foam_dipole_surface, foam_isosurface
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from run_camera_free_surface import boundary_mask, local_spacing
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import gaussian_volume, grid_from_bbox, isosurface_samples

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    scene = a.scene

    ck = f"output/scannet_{scene}_{a.arm}"
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

    if a.rule == "interface":
        occ = np.where(pcls > 0, alpha, -1.0)
        fp, fc, _ = foam_isosurface(c, r, occ, pcls, adj, off, a.alpha, blo, bhi)
    else:
        keep = ((alpha >= a.alpha) & (pcls > 0) if a.rule == "opaque"
                else boundary_mask(pcls, alpha, adj, off, a.alpha))
        mr = (local_spacing(c, adj, off) * a.patch_radius) if a.patch_radius > 0 else None
        fp, fc, _ = foam_dipole_surface(c, r, nrm, hgt, np.where(keep, 1.0, 0.0), pcls, adj, off,
                                        blo, bhi, alpha_min=0.5, max_radius=mr)

    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    sp = sdg["splats"] if "splats" in sdg else sdg
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    lo, dims = grid_from_bbox(blo, bhi, a.voxel)
    dens, cvol = gaussian_volume(sp["means"].float().numpy(),
                                 torch.exp(sp["scales"].float()).numpy(),
                                 sp["quats"].float().numpy(),
                                 torch.sigmoid(sp["opacities"].float()).numpy().reshape(-1),
                                 gcls, lo, dims, a.voxel)
    gp, gc, _ = isosurface_samples(dens, cvol, lo, a.voxel, a.gs_alpha)

    mf = semantic_surface_metrics_mesh(idx, fp, fc)
    mg = semantic_surface_metrics_mesh(idx, gp, gc)
    pcf, pcg = mf.get("per_class", {}), mg.get("per_class", {})

    rows, recs = [], []
    for k, nm in enumerate(names, start=1):
        gtm = V[gt_lab == k]
        fm, gm = fp[fc == k], gp[gc == k]
        rows.append((nm, gtm, fm, gm, PALETTE[k % len(PALETTE)]))
        recs.append({"scene": scene, "class": nm,
                     "gt_pts": int(len(gtm)), "foam_pts": int(len(fm)), "gs_pts": int(len(gm)),
                     "foam_scd": pcf.get(k, {}).get("scd"), "gs_scd": pcg.get(k, {}).get("scd")})

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.1 * n), dpi=165)
    axes = np.atleast_2d(axes)
    for i, (nm, gtm, fm, gm, col) in enumerate(rows):
        f_s, g_s = recs[i]["foam_scd"], recs[i]["gs_scd"]
        tile(axes[i, 0], gtm, blo, bhi, col, "GT" if i == 0 else None)
        tile(axes[i, 1], fm, blo, bhi, col,
             f"foam {a.rule}" if i == 0 else None)
        tile(axes[i, 2], gm, blo, bhi, col, "3DGS" if i == 0 else None)
        axes[i, 0].set_ylabel(f"{nm}\n{len(gtm):,} gt", fontsize=7)
        axes[i, 1].set_xlabel(f"{len(fm):,} pts"
                              + (f"  SCD {100*f_s:.0f}cm" if f_s is not None else "  (missed)"),
                              fontsize=6)
        axes[i, 2].set_xlabel(f"{len(gm):,} pts"
                              + (f"  SCD {100*g_s:.0f}cm" if g_s is not None else "  (missed)"),
                              fontsize=6)
    fig.suptitle(f"{scene} - per-class extracted surface "
                 f"(foam {a.rule} a={a.alpha}, 3DGS a={a.gs_alpha}, negatives-rejected)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = f"{a.outdir}/{scene}_perclass.png"
    fig.savefig(png, bbox_inches="tight")
    json.dump(recs, open(f"{a.outdir}/{scene}_perclass.json", "w"), indent=1)
    for rec in recs:
        fs = f"{100*rec['foam_scd']:.1f}" if rec["foam_scd"] is not None else "  --"
        gs = f"{100*rec['gs_scd']:.1f}" if rec["gs_scd"] is not None else "  --"
        print(f"  {rec['class']:<16} gt {rec['gt_pts']:>7,}  foam {rec['foam_pts']:>9,} "
              f"SCD {fs:>6}cm   gs {rec['gs_pts']:>8,} SCD {gs:>6}cm", flush=True)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
