"""Alpha-composited SEMANTIC RENDERS: the segmentation as a camera would actually see it.

A per-class alpha mask answers "where is the mass"; this answers "what does the segmentation look
like", which is the thing worth putting in a paper. Each ray composites CLASS COLOUR weighted by the
opacity it accumulates:

    C(pixel) = sum_k  w_k * palette[class_k],    w_k = alpha_k * T_k,   T_k = prod_{l<k} (1 - alpha_l)

so a representation that commits to a surface returns saturated, flat class colour, while one whose
mass is spread through empty space returns a blend of everything the ray passed through -- the haze.
Compositing is what exposes it: an arg-max render would hide the problem by picking a winner per
pixel and throwing away the distribution that makes it fuzzy.

Both representations are voxelised on the same grid and marched by the same routine, so the only
asymmetry is the real one: the foam's density is piecewise constant on exact power-cell ownership,
a Gaussian mixture's is a sum of overlapping kernels.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--res", type=int, default=520)
    ap.add_argument("--step", type=float, default=0.006)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--outdir", default="artifacts/surface_viz/semantic")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from mesh_surface import load_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import foam_volume, gaussian_volume, grid_from_bbox

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    scene = a.scene
    mesh = load_mesh(scene)
    V = np.asarray(mesh.vertices)
    lo, dims = grid_from_bbox(V.min(0), V.max(0), a.voxel)
    _, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    nc = len(names) + 1
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")

    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sig_f = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    fcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    fcls[~d["valid_mask"].numpy()] = 0
    print("voxelising foam ...", flush=True)
    fdens, fcvol = foam_volume(c, r, sig_f, fcls, lo, dims, a.voxel, max_dist=r)

    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    sp = sdg["splats"] if "splats" in sdg else sdg
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    print("voxelising gaussians ...", flush=True)
    gdens, gcvol = gaussian_volume(sp["means"].detach().float().numpy(),
                                   torch.exp(sp["scales"].detach().float()).numpy(),
                                   sp["quats"].detach().float().numpy(),
                                   torch.sigmoid(sp["opacities"].detach().float()).numpy().reshape(-1),
                                   gcls, lo, dims, a.voxel)
    gdens = gdens / a.voxel     # accumulated opacity -> density per metre, so both march the same

    cen = 0.5 * (V.min(0) + V.max(0))
    ext = V.max(0) - V.min(0)
    cams = [(cen + np.array([0, 0, 0.25]), np.array([1.0, 0.0, 0.0])),
            (cen + np.array([0, 0, 0.25]), np.array([0.0, 1.0, 0.0])),
            (cen + np.array([0, 0, 0.45]), np.array([0.6, 0.6, -0.35]))][:a.views]
    W = a.res
    diag = float(np.linalg.norm(ext))
    ts = np.arange(0.0, diag, a.step)
    shape = np.array(dims, dtype=np.int64)
    pal = PALETTE.astype(np.float64) / 255.0

    def render(dens, cvol, eye, fwd):
        f = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        if abs(f @ up) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(f, up); right /= np.linalg.norm(right); up = np.cross(right, f)
        g = np.linspace(-0.72, 0.72, W)
        uu, vv = np.meshgrid(g, g, indexing="xy")
        dirs = (f + uu[..., None] * right + vv[..., None] * up)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs.reshape(-1, 3)
        o = np.broadcast_to(eye, dirs.shape)
        col = np.zeros((len(dirs), 3)); T = np.ones(len(dirs))
        # EARLY RAY TERMINATION. Without it every ray marches the full scene diagonal even after its
        # transmittance has collapsed, and most rays hit a surface within the first 10-20% of their
        # range -- so the naive loop does roughly 5-10x the necessary work. Rays are dropped once
        # they can no longer contribute more than `t_min` of colour, and the per-step arrays shrink
        # with them. This is the difference between minutes and tens of minutes for the same image.
        t_min = 0.01
        alive = np.arange(len(dirs))
        for t in ts:
            if len(alive) == 0:
                break
            P = o[alive] + t * dirs[alive]
            iv = np.floor((P - lo) / a.voxel).astype(np.int64)
            ok = ((iv >= 0) & (iv < shape[None, :])).all(axis=1)
            if ok.any():
                sub = alive[ok]
                ii = iv[ok]
                s = dens[ii[:, 0], ii[:, 1], ii[:, 2]]
                cl = cvol[ii[:, 0], ii[:, 1], ii[:, 2]]
                al = 1.0 - np.exp(-s * a.step)
                w = al * T[sub]
                col[sub] += w[:, None] * pal[np.clip(cl, 0, len(pal) - 1)]
                T[sub] *= (1.0 - al)
            # keep only rays that are still inside the box and still able to contribute
            alive = alive[ok][T[alive[ok]] > t_min] if ok.any() else alive[np.zeros(0, int)]
        col = col + T[:, None] * 1.0            # composite over white
        return np.clip(col, 0, 1).reshape(W, W, 3), (1 - T).reshape(W, W)

    rows = []
    for vi, (eye, fwd) in enumerate(cams):
        print(f"rendering view {vi} ...", flush=True)
        rows.append((render(fdens, fcvol, eye, fwd), render(gdens, gcvol, eye, fwd)))

    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(4.3 * n, 9.0), dpi=190)
    ax = np.atleast_2d(ax)
    if n == 1:
        ax = ax.reshape(2, 1)
    for vi, ((fc_, fa), (gc_, ga)) in enumerate(rows):
        ax[0, vi].imshow(fc_); ax[1, vi].imshow(gc_)
        for r_ in (0, 1):
            ax[r_, vi].set_xticks([]); ax[r_, vi].set_yticks([])
            for sp_ in ax[r_, vi].spines.values():
                sp_.set_linewidth(0.6); sp_.set_color("0.7")
    ax[0, 0].set_ylabel("PowerFoam (ours)", fontsize=15, labelpad=10)
    ax[1, 0].set_ylabel("3D Gaussian Splatting", fontsize=15, labelpad=10)
    handles = [Patch(facecolor=pal[k % len(pal)], edgecolor="0.5", label=nm)
               for k, nm in enumerate(names, start=1)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(names), 7),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Alpha-composited semantic render  -  {scene}", fontsize=16)
    fig.tight_layout(rect=[0, 0.045, 1, 0.98])
    p = f"{a.outdir}/{scene}_semantic_render.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
