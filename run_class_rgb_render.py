"""The RECONSTRUCTION itself, masked to one class at a time, alpha-composited.

WHAT THIS SHOWS THAT THE OTHER FIGURES DO NOT. A per-class alpha mask says where the opacity is; a
class-coloured render says which label won. Neither looks like the scene. Here each ray composites
the primitives' OWN colour, but only from primitives of the queried class:

    C_c(pixel) = sum_{k in class c} w_k * rgb_k,    w_k = alpha_k * T_k

so the image is the reconstruction as its viewer would draw it, with everything but one class
removed. Because the weights are alpha*T rather than an arg-max, a representation whose mass is
spread through empty space paints faint colour over large regions -- the haze -- while one that
commits to a surface paints a solid object and leaves the rest of the frame clean. That pairing is
the point: RGB for realism, alpha weighting to expose the blur that an arg-max render would hide.

COLOUR IS THE DC TERM, not full view-dependent SH: 3DGS `sh0` (rgb = 0.2821*sh0 + 0.5), PowerFoam's
Spherical-Voronoi lobes averaged to a base albedo, RadFoam `color_dc`. View-dependence changes
shading, not where the mass sits, so it does not affect what these figures are about -- but they are
albedo renders and should be labelled as such rather than passed off as the papers' viewers.

PERFORMANCE. The cost here is NOT the ray marching, which was the wrong diagnosis the first time: it
is voxelising ~20 M voxels twice, and the foam side runs an exact power-cell query per voxel. The
volumes are therefore cached to disk keyed by (scene, arm, voxel), so only the first run pays.
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

from export_surface_viz import POINTCEPT, SPLIT  # noqa: E402

SH_C0 = 0.28209479177387814


def cached(path, fn):
    if os.path.exists(path):
        z = np.load(path)
        return z["dens"], z["idx"]
    dens, idx = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, dens=dens, idx=idx)
    return dens, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--step", type=float, default=0.008)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--outdir", default="artifacts/surface_viz/class_rgb")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
    cache = f"artifacts/volcache/{scene}_v{int(a.voxel*1000)}mm"

    # ---- foam: density + OWNER INDEX volume (so colour and class both come from the owner) -----
    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sig_f = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    frgb = sd["texel_sv_rgb"].float().numpy().reshape(len(c), -1, 3).mean(axis=1)
    frgb = 1.0 / (1.0 + np.exp(-frgb))                       # lobes -> base albedo
    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    fcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    fcls[~d["valid_mask"].numpy()] = 0
    print("foam volume ...", flush=True)
    fdens, fidx = cached(f"{cache}_foam.npz",
                         lambda: foam_volume(c, r, sig_f, np.arange(len(c)), lo, dims,
                                             a.voxel, max_dist=r))

    # ---- gaussians ------------------------------------------------------------------------------
    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    sp = sdg["splats"] if "splats" in sdg else sdg
    gm = sp["means"].detach().float().numpy()
    grgb = np.clip(SH_C0 * sp["sh0"].detach().float().numpy().reshape(len(gm), 3) + 0.5, 0, 1)
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    print("gaussian volume ...", flush=True)
    gdens, gidx = cached(f"{cache}_gs.npz",
                         lambda: gaussian_volume(gm, torch.exp(sp["scales"].detach().float()).numpy(),
                                                 sp["quats"].detach().float().numpy(),
                                                 torch.sigmoid(sp["opacities"].detach().float()).numpy().reshape(-1),
                                                 np.arange(len(gm)), lo, dims, a.voxel))
    gdens = gdens / a.voxel

    cen = 0.5 * (V.min(0) + V.max(0))
    eye = cen + np.array([0.0, 0.0, 0.25])
    fwd = np.array([1.0, 0.0, 0.0])
    W = a.res
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    ts = np.arange(0.0, diag, a.step)
    shape = np.array(dims, dtype=np.int64)

    f_ = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f_, up); right /= np.linalg.norm(right); up = np.cross(right, f_)
    g = np.linspace(-0.72, 0.72, W)
    uu, vv = np.meshgrid(g, g, indexing="xy")
    dirs = (f_ + uu[..., None] * right + vv[..., None] * up)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs.reshape(-1, 3)
    o = np.broadcast_to(eye, dirs.shape)

    def render(dens, idxvol, rgb, cls, keep_class):
        """keep_class=None -> the full reconstruction; else only that class contributes colour."""
        col = np.zeros((len(dirs), 3)); T = np.ones(len(dirs))
        alive = np.arange(len(dirs))
        for t in ts:
            if len(alive) == 0:
                break
            P = o[alive] + t * dirs[alive]
            iv = np.floor((P - lo) / a.voxel).astype(np.int64)
            ok = ((iv >= 0) & (iv < shape[None, :])).all(axis=1)
            if ok.any():
                sub = alive[ok]; ii = iv[ok]
                s = dens[ii[:, 0], ii[:, 1], ii[:, 2]]
                pid = idxvol[ii[:, 0], ii[:, 1], ii[:, 2]]
                al = 1.0 - np.exp(-s * a.step)
                w = al * T[sub]
                valid = pid >= 0
                if keep_class is not None:
                    valid &= (cls[np.clip(pid, 0, len(cls) - 1)] == keep_class)
                if valid.any():
                    col[sub[valid]] += (w[valid][:, None]
                                        * rgb[np.clip(pid[valid], 0, len(rgb) - 1)])
                # transmittance always accumulates: occluders still block, even when masked out
                T[sub] *= (1.0 - al)
            alive = alive[ok][T[alive[ok]] > 0.01] if ok.any() else alive[np.zeros(0, int)]
        return np.clip(col + T[:, None], 0, 1).reshape(W, W, 3)

    want = a.classes or ["wall", "floor", "toilet", "sink"]
    sel = [(k, nm) for k, nm in enumerate(names, start=1) if nm in want]
    cols = 1 + len(sel)
    fig, ax = plt.subplots(2, cols, figsize=(3.7 * cols, 7.9), dpi=185)
    for row, (dens, idxv, rgb, cls, lab) in enumerate(
            [(fdens, fidx, frgb, fcls, "PowerFoam (ours)"),
             (gdens, gidx, grgb, gcls, "3D Gaussian Splatting")]):
        print(f"rendering {lab} ...", flush=True)
        ax[row, 0].imshow(render(dens, idxv, rgb, cls, None))
        ax[row, 0].set_title("full reconstruction" if row == 0 else "", fontsize=11)
        ax[row, 0].set_ylabel(lab, fontsize=13)
        for j, (k, nm) in enumerate(sel, start=1):
            ax[row, j].imshow(render(dens, idxv, rgb, cls, k))
            if row == 0:
                ax[row, j].set_title(f"only: {nm}", fontsize=11)
        for j in range(cols):
            ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
    fig.suptitle(f"Reconstruction masked to one class, alpha-composited  -  {scene}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{a.outdir}/{scene}_class_rgb.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
