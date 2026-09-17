"""Dump the SEMANTIC surfaces the camera-free metric actually scores, so they can be looked at.

The metric reports one number per configuration; that number said the foam claims 1925 m2 of surface
inside a 17.8 m3 room, which is a statement about interior partitions of the tessellation that is
much easier to confirm by eye than by argument. This writes, for one scene:

  gt.ply                   GT mesh vertices coloured by their ScanNet class
  foam_<rule>_a<alpha>.ply the foam surface under each membership rule
  gs_a<alpha>.ply          the Gaussian isosurface at each level

Every point carries the colour of the class the METHOD assigned it, so a wall painted "chair" is
visible as a chair-coloured slab. Same code paths as run_camera_free_surface.py / run_camera_free_gs.py
-- this is a viewer for the scored geometry, not a second implementation of it.

A PNG contact sheet is rendered alongside for a quick look; the PLYs are the real artefact.
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

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")

# ScanNet's own 20-class palette, so the dumps match every figure the dataset ships.
PALETTE = np.array([
    [200, 200, 200], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
    [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213],
    [148, 103, 189], [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141],
    [255, 127, 14], [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194],
    [82, 84, 163]], dtype=np.uint8)


def write_ply(path, pts, cls):
    col = PALETTE[np.clip(cls, 0, len(PALETTE) - 1)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = col[:, 0], col[:, 1], col[:, 2]
        f.write(rec.tobytes())


def panel(ax, pts, cls, title, lo, hi, cap=60000, seed=0):
    if len(pts) > cap:                       # thin for rendering only; PLYs keep every point
        k = np.random.default_rng(seed).choice(len(pts), cap, replace=False)
        pts, cls = pts[k], cls[k]
    if len(pts):
        order = np.argsort(-pts[:, 1])       # painter's order, far to near
        pts, cls = pts[order], cls[order]
        ax.scatter(pts[:, 0], pts[:, 2], s=0.35, marker=".", linewidths=0,
                   c=PALETTE[np.clip(cls, 0, len(PALETTE) - 1)] / 255.0)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.9])
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--outdir", default="artifacts/surface_viz")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from foam_exact_surface import foam_dipole_surface, foam_isosurface
    from mesh_surface import load_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from run_camera_free_surface import boundary_mask
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
    text = embed_class_names(names, "cuda")

    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
    pcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    pcls[~d["valid_mask"].numpy()] = 0
    print(f"[reject] foam keeps {(pcls > 0).mean():.1%} of cells", flush=True)

    tiles = []
    write_ply(f"{a.outdir}/{scene}_gt.ply", V, gt_lab)
    tiles.append((V, gt_lab, f"GT mesh\n{mesh.get_surface_area():.1f} m2"))
    print(f"[gt] {len(V):,} verts, {mesh.get_surface_area():.1f} m2", flush=True)

    for rule in ("opaque", "boundary", "interface"):
        for al in a.alphas:
            if rule == "interface":
                # a rejected cell claims no class, so it is not occupied for the purpose
                # of a semantic surface -- mask it out of the occupancy field entirely
                occ = np.where(pcls > 0, alpha, -1.0)
                pts, cls, areas = foam_isosurface(c, r, occ, pcls, adj, off, al, blo, bhi)
            else:
                keep = ((alpha >= al) & (pcls > 0) if rule == "opaque"
                        else boundary_mask(pcls, alpha, adj, off, al))
                pts, cls, areas = foam_dipole_surface(
                    c, r, nrm, hgt, np.where(keep, 1.0, 0.0), pcls, adj, off,
                    blo, bhi, alpha_min=0.5)
            ar = float(sum(areas.values()))
            write_ply(f"{a.outdir}/{scene}_foam_{rule}_a{al}.ply", pts, cls)
            tiles.append((pts, cls, f"foam {rule} a={al}\n{ar:.0f} m2"))
            print(f"[foam/{rule}/a{al}] {len(pts):,} pts, {ar:.1f} m2", flush=True)

    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    sp = sdg["splats"] if "splats" in sdg else sdg
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    print(f"[reject] gs keeps {(gcls > 0).mean():.1%} of gaussians", flush=True)
    lo, dims = grid_from_bbox(blo, bhi, a.voxel)
    dens, cvol = gaussian_volume(sp["means"].float().numpy(), torch.exp(sp["scales"].float()).numpy(),
                                 sp["quats"].float().numpy(),
                                 torch.sigmoid(sp["opacities"].float()).numpy().reshape(-1),
                                 gcls, lo, dims, a.voxel)
    for al in a.alphas:
        pts, cls, areas = isosurface_samples(dens, cvol, lo, a.voxel, al)
        ar = float(sum(areas.values()))
        write_ply(f"{a.outdir}/{scene}_gs_a{al}.ply", pts, cls)
        tiles.append((pts, cls, f"3DGS a={al}\n{ar:.0f} m2"))
        print(f"[gs/a{al}] {len(pts):,} pts, {ar:.1f} m2", flush=True)

    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.6), dpi=190)
    for ax, (p, cl, t) in zip(np.atleast_1d(axes), tiles):
        panel(ax, p, cl, t, blo, bhi)
    fig.suptitle(f"{scene} - semantic surface scored by the camera-free metric "
                 f"(front view, ScanNet palette)", fontsize=9)
    fig.tight_layout()
    png = f"{a.outdir}/{scene}_panel.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"\nwrote {png} and {n} PLYs -> {a.outdir}")


if __name__ == "__main__":
    main()
