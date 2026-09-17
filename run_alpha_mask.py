"""2-D per-class ALPHA MASKS, and how far each method's opaque mass sits from the true surface.

WHY THIS IS THE RIGHT PICTURE FOR A SURFACE METRIC. Chamfer-style numbers compare an extracted
surface against the GT mesh, so they inherit every arbitrary choice the extractor made -- and this
project has now built six extractors that disagree on the mesh while agreeing to ~0.5 cm on the
score. An alpha mask sidesteps extraction entirely: march a ray, accumulate `alpha * T` per class,
and look at WHERE the opaque mass actually is. That is a property of the representation, not of a
meshing decision.

Two images per class, both camera-space:

  alpha mask   A_c(pixel) = sum over the ray of alpha_k * T_k for cells/Gaussians of class c.
               A crisp representation gives a near-binary mask; a fuzzy one gives soft, spread-out
               support, which is exactly the over-segmentation that 3DGS is expected to show.

  spread map   S(pixel) = sum_k w_k |t_k - t_gt| / sum_k w_k,  w_k = alpha_k * T_k
               the contribution-weighted distance of the accumulated mass from the GT surface along
               the same ray. Small means the opacity sits ON the surface; large means it is smeared
               in depth even if the mask looks fine in 2-D.

FAIRNESS. Both representations are voxelised on the SAME grid and marched by the SAME routine, so
nothing here depends on the foam's exact extractor or on marching cubes. The only asymmetry is the
one the representations genuinely have: the foam's density is piecewise constant on power cells
(exact ownership), while a Gaussian mixture's is a sum of overlapping kernels.
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
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--res", type=int, default=280)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="artifacts/alpha_mask.json")
    ap.add_argument("--outdir", default="artifacts/surface_viz/alpha")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

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

    # ---- foam volume ---------------------------------------------------------------------------
    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sigma_f = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    fcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    fcls[~d["valid_mask"].numpy()] = 0
    if a.oracle:
        from oracle_labels import oracle_labels
        fcls, _ = oracle_labels(c, r, V, gt_lab, nc)
    print("voxelising foam ...", flush=True)
    fdens, fcvol = foam_volume(c, r, sigma_f, fcls, lo, dims, a.voxel, max_dist=r)

    # ---- gaussian volume -----------------------------------------------------------------------
    sdg = torch.load(f"recon_remote/{a.gs_arm}/{scene}/ckpt.pt", map_location="cpu",
                     weights_only=False)
    sp = sdg["splats"] if "splats" in sdg else sdg
    dg = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{a.gs_arm}_ogl3.pt",
                    map_location="cpu", weights_only=True)
    gcls = classify_with_rejection(dg["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    gcls[~dg["valid_mask"].numpy()] = 0
    if a.oracle:
        from oracle_labels import oracle_labels
        gm = sp["means"].detach().float().numpy().astype(np.float64)
        gr = np.full(len(gm), a.voxel)          # oracle for GS: nearest GT vertex label
        from scipy.spatial import cKDTree
        gcls = gt_lab[cKDTree(V).query(gm)[1]]
    print("voxelising gaussians ...", flush=True)
    gdens, gcvol = gaussian_volume(sp["means"].detach().float().numpy(),
                                   torch.exp(sp["scales"].detach().float()).numpy(),
                                   sp["quats"].detach().float().numpy(),
                                   torch.sigmoid(sp["opacities"].detach().float()).numpy().reshape(-1),
                                   gcls, lo, dims, a.voxel)
    # gaussian_volume returns an accumulated OPACITY-like sum; convert to a density per metre so
    # both fields mean the same thing when marched with the same step
    gdens = gdens / a.voxel

    # ---- camera --------------------------------------------------------------------------------
    cen = 0.5 * (V.min(0) + V.max(0))
    eye = cen + np.array([0.0, 0.0, 0.20])
    fwd = np.array([1.0, 0.0, 0.0]); fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    W = a.res
    g = np.linspace(-0.7, 0.7, W)
    uu, vv = np.meshgrid(g, g, indexing="xy")
    dirs = (fwd + uu[..., None] * right + vv[..., None] * up)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs.reshape(-1, 3)
    origins = np.broadcast_to(eye, dirs.shape)

    gs = o3d.t.geometry.RaycastingScene()
    gs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = o3d.core.Tensor(np.concatenate([origins, dirs], -1).astype(np.float32))
    t_gt = gs.cast_rays(rays)["t_hit"].numpy().astype(np.float64)

    # ---- march both fields with the SAME routine -----------------------------------------------
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    ts = np.arange(0.0, diag, a.step)
    shape = np.array(dims, dtype=np.int64)

    def march_field(dens, cvol):
        R = len(dirs)
        A = np.zeros((R, nc)); Trans = np.ones(R)
        wsum = np.zeros(R); wdev = np.zeros(R); wdepth = np.zeros(R)
        for t in ts:
            P = origins + t * dirs
            idxv = np.floor((P - lo) / a.voxel).astype(np.int64)
            ok = ((idxv >= 0) & (idxv < shape[None, :])).all(axis=1)
            if not ok.any():
                continue
            ii = idxv[ok]
            s = dens[ii[:, 0], ii[:, 1], ii[:, 2]]
            cl = cvol[ii[:, 0], ii[:, 1], ii[:, 2]]
            al = 1.0 - np.exp(-s * a.step)
            w = al * Trans[ok]
            np.add.at(A, (np.nonzero(ok)[0], cl), w)
            wsum[ok] += w
            wdepth[ok] += w * t
            fin = np.isfinite(t_gt[ok])
            dev = np.where(fin, np.abs(t - t_gt[ok]), 0.0)
            wdev[ok] += w * dev
            Trans[ok] *= (1.0 - al)
        return A, wsum, wdev, wdepth, Trans

    print("marching foam ...", flush=True)
    Af, wf, devf, depf, Tf = march_field(fdens, fcvol)
    print("marching gaussians ...", flush=True)
    Ag, wg, devg, depg, Tg = march_field(gdens, gcvol)

    def spread(w, dev):
        out = np.full_like(w, np.nan)
        m = w > 1e-6
        out[m] = dev[m] / w[m]
        return out.reshape(W, W)

    Sf, Sg = spread(wf, devf), spread(wg, devg)
    hit = np.isfinite(t_gt).reshape(W, W)
    rec = {"scene": scene, "labels": "oracle" if a.oracle else "predicted",
           "foam_spread_cm": float(100 * np.nanmean(Sf[hit])),
           "gs_spread_cm": float(100 * np.nanmean(Sg[hit])),
           "foam_opacity": float(np.mean(1 - Tf)), "gs_opacity": float(np.mean(1 - Tg))}
    print(f"\nweighted |depth - GT| :  foam {rec['foam_spread_cm']:6.2f} cm   "
          f"3DGS {rec['gs_spread_cm']:6.2f} cm")
    print(f"mean accumulated alpha:  foam {rec['foam_opacity']:.3f}        "
          f"3DGS {rec['gs_opacity']:.3f}")

    # ---- figures --------------------------------------------------------------------------------
    order = np.argsort(-Af[:, 1:].sum(0))[:4] + 1
    fig, ax = plt.subplots(3, len(order) + 1, figsize=(3.0 * (len(order) + 1), 8.4), dpi=165)
    for k, cid in enumerate(order):
        nm = names[cid - 1]
        ax[0, k].imshow(Af[:, cid].reshape(W, W), cmap="magma", vmin=0, vmax=1)
        ax[0, k].set_title(f"foam - {nm}", fontsize=8)
        ax[1, k].imshow(Ag[:, cid].reshape(W, W), cmap="magma", vmin=0, vmax=1)
        ax[1, k].set_title(f"3DGS - {nm}", fontsize=8)
        for r_ in (0, 1):
            ax[r_, k].set_xticks([]); ax[r_, k].set_yticks([])
    im0 = ax[2, 0].imshow(100 * Sf, cmap="turbo", vmin=0, vmax=30)
    ax[2, 0].set_title("foam |depth-GT| (cm)", fontsize=8)
    im1 = ax[2, 1].imshow(100 * Sg, cmap="turbo", vmin=0, vmax=30)
    ax[2, 1].set_title("3DGS |depth-GT| (cm)", fontsize=8)
    for k in range(2, len(order) + 1):
        ax[2, k].axis("off")
    ax[0, len(order)].imshow((1 - Tf).reshape(W, W), cmap="gray", vmin=0, vmax=1)
    ax[0, len(order)].set_title("foam total alpha", fontsize=8)
    ax[1, len(order)].imshow((1 - Tg).reshape(W, W), cmap="gray", vmin=0, vmax=1)
    ax[1, len(order)].set_title("3DGS total alpha", fontsize=8)
    for r_ in (0, 1):
        ax[r_, len(order)].set_xticks([]); ax[r_, len(order)].set_yticks([])
    for aa in (ax[2, 0], ax[2, 1]):
        aa.set_xticks([]); aa.set_yticks([])
    fig.colorbar(im1, ax=ax[2, 1], fraction=0.046)
    tag = "oracle" if a.oracle else "predicted"
    fig.suptitle(f"{scene} - per-class alpha masks and depth spread ({tag} labels)", fontsize=11)
    fig.tight_layout()
    p = f"{a.outdir}/{scene}_alpha_{tag}.png"
    fig.savefig(p, bbox_inches="tight")
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    rows.append(rec); json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
