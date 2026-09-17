"""Render the reconstruction with the REAL kernels, masked to one predicted class at a time.

The predictions are already per primitive, and both methods already have fast renderers, so a
class-masked view needs no voxel grid and no ray marching: silence every primitive that is not of
the queried class and call the renderer that the method ships with. That gives the actual
reconstruction -- real view-dependent appearance, real alpha compositing -- rather than the albedo
approximation a hand-rolled marcher produces, and it runs in the time a normal render takes.

Silencing is done on DENSITY (foam) and OPACITY (3DGS), not by deleting primitives, so the geometry,
adjacency and traversal order are untouched; only the contribution of off-class primitives goes to
zero. What survives is exactly that class's contribution to the image, alpha-composited.

That compositing is what makes the comparison informative. A representation that commits its mass to
a surface paints a solid object and leaves the rest of the frame clean. One whose mass is smeared
through empty space paints faint colour over large regions -- the haze -- because those primitives
still contribute alpha even where they are individually near-transparent. An arg-max render would
hide precisely that.

TWO PASSES, TWO ENVIRONMENTS. PowerFoam needs warp + fpsample (env `powerfoam`); gsplat is only
compiled in env `splat-distiller`. No single interpreter runs both, so `--side foam` and `--side gs`
each write an npz and `--side figure` composes them. The foam pass also writes the CAMERA it used
(viewmat/K, derived through camera_bridge from the very camera foam traversed), and the gs pass
consumes it, so the two rows are guaranteed to be the same viewpoint rather than two dataset
orderings that are assumed to agree.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")
SILENT = -50.0          # pre-softplus density that makes a cell effectively transparent
# Row order of the four-arm figure. Frozen arms first so the frozen/unfrozen pair reads down the
# column within each representation.
ARM_LABEL = {"pf_truefrozen": "PowerFoam (frozen)", "pf_nonfrozen": "PowerFoam (unfrozen)",
             "gs_froz": "3DGS (frozen)", "gs_unfroz": "3DGS (unfrozen)",
             "pf_lerf": "PowerFoam", "gs_lerf": "3D Gaussian Splatting"}


def lerf_gt(scene):
    """3D GT for a LERF-OVS scene: the 2D polygon labels back-projected by unproject_lerf_gt.py.

    LERF-OVS ships no 3D ground truth -- only polygon masks on a handful of annotated frames -- so
    the oracle needs those lifted into space first. That lifting uses ONE reconstruction's depth
    (PowerFoam's) for every arm, which keeps the labelled point set identical across arms the same
    way the shared camera keeps the viewpoint identical. It does mean the GT geometry is
    foam-derived; that is a real asymmetry and is the reason to read these panels as a comparison of
    where each arm puts its mass, not as an independent accuracy measurement.
    """
    import json
    z = np.load(f"artifacts/lerf_gt3d/{scene}_gt3d.npz", allow_pickle=True)
    id2name = json.loads(str(z["class_id_to_name"].item()))
    names = [id2name[str(i)] for i in range(1, len(id2name) + 1)]
    return z["points"].astype(np.float64), z["labels"].astype(np.int64), names


def lerf_frame_view(scene, frame):
    from featurefoam_lerf_bridge import load_manifest_index
    idx = load_manifest_index(scene)
    if frame not in idx:
        raise SystemExit(f"{frame} not in manifest; have {sorted(idx)[:4]} ...")
    return idx[frame]


def scene_gt(scene, class_set):
    """GT points and labels in 1..K, plus the class names, for the classes present in this scene."""
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in present]
    return pts, remap_gt_labels(raw, [n2i[n] for n in names]), names


def report(cls, names, tag):
    print(f"  [{tag}] " + "  ".join(f"{nm}:{int((cls == k).sum()):,}"
                                    for k, nm in enumerate(names, start=1))
          + f"  none:{int((cls == 0).sum()):,}", flush=True)


def labels_for(path, names, margin):
    from evaluate_point_cloud_miou import embed_class_names
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    d = torch.load(path, map_location="cpu", weights_only=True)
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
    cls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                  margin).cpu().numpy()
    cls[~d["valid_mask"].numpy()] = 0
    return cls


def oracle(a, centers, gt_pts, gt_lab, names, radii=None):
    """GT labels per primitive, under whichever ownership convention `--oracle-mode` selects.

    `owned` is the metric-ceiling oracle (a primitive owning no GT vertex stays unlabelled), and it
    uses each representation's own membership query: exact power-cell for foam, nearest centre for
    Gaussians. `nearest-gt` labels every primitive within `--oracle-dist` of the true surface, which
    is what makes arms with different primitive counts comparable -- see oracle_labels.py.
    """
    from oracle_labels import (oracle_labels, oracle_labels_by_nearest_gt, oracle_labels_nearest)
    if a.oracle_mode == "nearest-gt":
        cls, st = oracle_labels_by_nearest_gt(centers, gt_pts, gt_lab, a.oracle_dist)
        return cls, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}
    if radii is not None:
        cls, st = oracle_labels(centers, radii, gt_pts, gt_lab, len(names) + 1)
    else:
        cls, st = oracle_labels_nearest(centers, gt_pts, gt_lab, len(names) + 1)
    return cls, {"frac_cells_with_gt": round(st["frac_cells_with_gt"], 4),
                 "vote_purity": round(st["vote_purity"], 4)}


def selected(names, want):
    want = want or [n for n in ("wall", "floor", "toilet", "sink", "door") if n in names]
    return [(k, nm) for k, nm in enumerate(names, start=1) if nm in want]


def _load_foam(a):
    """Shared foam setup: returns (model, dh, VisOptions) with the options filled in explicitly."""
    import configargparse
    import warp as wp

    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.rasterize import VisOptions
    from powerfoam.scene import PowerfoamScene

    ckpt_dir = (f"output/lerf_ovs_{a.scene}" if a.dataset == "lerf"
                else f"output/scannet_{a.scene}_{a.variant}")
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ckpt_dir}/model.pt")
    model.update_vis_cache()

    # VisOptions is a warp struct: it ZERO-initializes, and Rasterizer.visualize()'s internal
    # default never fills these in. Left at zero, max_intersections=0 stops the traversal before it
    # accumulates anything and every pixel comes back black -- the same trap eval_surface_chamfer.py
    # documents for depth_quantile. Always construct it explicitly.
    vis = VisOptions()
    vis.transmittance_threshold = 1e-3
    vis.max_intersections = 1024
    vis.depth_quantile = 0.5
    vis.bkgd_color = wp.vec3f(1.0, 1.0, 1.0)     # white: masked-out regions read as empty, not black
    return model, dh, vis, ckpt_dir


def side_pick(a, gt_pts, gt_lab, names, sel):
    """Rank views by how much of each class is VISIBLE, not merely inside the frustum.

    A mask panel is only worth looking at if the class is actually on screen and unoccluded, so
    in-frustum counting is not enough -- in a bathroom the wall behind the camera projects into the
    image just as happily as the one in front of it. Each GT point is therefore depth-tested against
    the foam's OWN rendered depth: the reconstruction decides what it considers visible, rather than
    a hand-rolled occlusion test built on a mesh the renderer never saw.
    """
    from eval_surface_chamfer import cos_map

    model, dh, vis, _ = _load_foam(a)
    P = np.asarray(gt_pts, dtype=np.float64)
    lab = np.asarray(gt_lab)
    keys = [k for k, _ in sel]
    rows = []
    for vi, cam in enumerate(dh.cameras):
        with torch.no_grad():
            out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis)
        d = out[1].detach().float().cpu().numpy()
        al = out[3].detach().float().cpu().numpy()
        d = d[..., 0] if d.ndim == 3 else d
        al = al[..., 0] if al.ndim == 3 else al
        z_img = d * cos_map(cam)                       # ray distance -> planar z, camera's own map

        params = cam.to_open3d()
        extr = np.asarray(params.extrinsic, dtype=np.float64)
        K = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
        pc = P @ extr[:3, :3].T + extr[:3, 3]
        z = pc[:, 2]
        H, W = z_img.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (K[0, 0] * pc[:, 0] / z + K[0, 2])
            v = (K[1, 1] * pc[:, 1] / z + K[1, 2])
        ui, vi_ = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ok = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi_ >= 0) & (vi_ < H)
        vis_pt = np.zeros(len(P), bool)
        idx = np.where(ok)[0]
        zi = z_img[vi_[idx], ui[idx]]
        ai = al[vi_[idx], ui[idx]]
        vis_pt[idx] = (ai >= 0.5) & (np.abs(zi - z[idx]) <= a.vis_tol)
        cnt = {k: int((vis_pt & (lab == k)).sum()) for k in keys}
        rows.append((vi, cnt))

    print(f"\nvisible GT points per class (depth tol {a.vis_tol} m), {len(rows)} views", flush=True)
    hdr = "  ".join(f"{nm:>8}" for _, nm in sel)
    print(f"{'view':>5}  {hdr}   n_cls  total", flush=True)
    best = None
    for vi, cnt in rows:
        ncls = sum(1 for k in keys if cnt[k] >= a.vis_min)
        tot = sum(cnt.values())
        score = (ncls, tot)
        if best is None or score > best[0]:
            best = (score, vi)
        print(f"{vi:>5}  " + "  ".join(f"{cnt[k]:>8,}" for k in keys)
              + f"   {ncls:>5}  {tot:>6,}", flush=True)
    print(f"\nBEST view {best[1]}: {best[0][0]}/{len(keys)} classes over {a.vis_min} points, "
          f"{best[0][1]:,} visible GT points total", flush=True)


# ------------------------------------------------------------------------------------------------
def side_foam(a, gt_pts, gt_lab, names, sel):
    from camera_bridge import K_from_ray_dirs, viewmat_from_camera

    model, dh, vis, ckpt_dir = _load_foam(a)
    cam = dh.cameras[a.view % len(dh.cameras)]

    if a.oracle:
        from build_true_facet_graph import load_points_radii
        cc, rr = load_points_radii(ckpt_dir)
        fcls, st = oracle(a, np.asarray(cc, dtype=np.float64), gt_pts, gt_lab, names,
                          radii=np.asarray(rr, dtype=np.float64))
        print(f"oracle[{a.oracle_mode}]: {st}", flush=True)
    else:
        fcls = labels_for(f"artifacts/scannet/{a.scene}/{a.solved}", names, a.margin)
    print(f"{len(fcls):,} primitives, {int((fcls > 0).sum()):,} classified", flush=True)
    report(fcls, names, "foam")

    def render(keep=None):
        orig = model.density.data.clone()
        if keep is not None:
            m = torch.from_numpy(fcls == keep).to(orig.device)
            model.density.data[~m.reshape(orig.shape)] = SILENT
            model.update_vis_cache()
        with torch.no_grad():
            out = model.forward_visualization(cam, render_mode="rasterize", vis_options=vis)
        model.density.data.copy_(orig)
        model.update_vis_cache()
        img = out[0].detach().float().cpu().numpy()
        al = out[3].detach().float().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        print(f"    alpha mean {float(al.mean()):.3f}", flush=True)
        return np.clip(img[..., :3], 0, 1)

    panels = [render(None)] + [render(k) for k, _ in sel]
    K, info = K_from_ray_dirs(cam)
    vm = viewmat_from_camera(cam)
    print(f"K fit residual {info['max_resid_px']:.2e} px", flush=True)
    np.savez_compressed(f"{a.outdir}/{a.scene}_v{a.view}_pf_{a.variant}{a.tag}.npz",
                        panels=np.stack(panels).astype(np.float32),
                        K=np.asarray(K, dtype=np.float64),
                        viewmat=vm.cpu().numpy().astype(np.float64),
                        wh=np.array([int(cam.width), int(cam.height)]))


def side_gs(a, gt_pts, gt_lab, names, sel):
    from gsplat import rasterization

    cams = np.load(f"{a.outdir}/{a.scene}_v{a.view}_{a.cam_from}{a.tag}.npz")
    K = torch.as_tensor(cams["K"], dtype=torch.float32, device="cuda")[None]
    vm = torch.as_tensor(cams["viewmat"], dtype=torch.float32, device="cuda")
    vm = vm[None] if vm.ndim == 2 else vm
    W, H = (int(x) for x in cams["wh"])

    # same activation convention as eval_surface_chamfer_gaussian.load_splats: the .pt stores raw
    # parameters (log-scale, logit-opacity).
    gp = (f"artifacts/lerf_ovs/{a.scene}/3DGS/ckpts/ckpt_59999_rank0.pt" if a.dataset == "lerf"
          else f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt")
    ck = torch.load(gp, map_location="cuda", weights_only=False)["splats"]
    gm, gq = ck["means"], ck["quats"]
    gs, go = torch.exp(ck["scales"]), torch.sigmoid(ck["opacities"]).reshape(-1)
    gc = torch.cat([ck["sh0"], ck["shN"]], dim=1)
    if a.oracle:
        gcls, st = oracle(a, gm.detach().cpu().numpy().astype(np.float64),
                          gt_pts, gt_lab, names)
        print(f"oracle[{a.oracle_mode}]: {st}", flush=True)
    else:
        gcls = labels_for(f"artifacts/scannet/{a.scene}/"
                          f"solved_geometric_median_{a.gs_arm}_ogl3.pt", names, a.margin)
    print(f"{len(gcls):,} gaussians, {int((gcls > 0).sum()):,} classified", flush=True)
    report(gcls, names, "gs")

    def render(keep=None):
        """Off-class Gaussians get opacity 0 -- geometry and ordering untouched, contribution zero."""
        op = go.clone()
        if keep is not None:
            op[~torch.from_numpy(gcls == keep).cuda()] = 0.0
        with torch.no_grad():
            rc, ra, _ = rasterization(means=gm, quats=gq, scales=gs, opacities=op, colors=gc,
                                      viewmats=vm, Ks=K, width=W, height=H, sh_degree=3,
                                      render_mode="RGB", packed=False)
        al = ra[0, ..., 0:1]
        print(f"    alpha mean {float(al.mean()):.3f}", flush=True)
        # composite over white, matching the foam side's bkgd_color
        return (rc[0, ..., :3] + (1.0 - al)).clamp(0, 1).cpu().numpy()

    panels = [render(None)] + [render(k) for k, _ in sel]
    np.savez_compressed(f"{a.outdir}/{a.scene}_v{a.view}_{a.gs_arm}{a.tag}.npz",
                        panels=np.stack(panels).astype(np.float32))


def side_figure(a, gt_pts, gt_lab, names, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["full reconstruction"] + [f"only: {nm}" for _, nm in sel]
    rows = []
    for key in a.arms:
        p = f"{a.outdir}/{a.scene}_v{a.view}_{key}{a.tag}.npz"
        if os.path.exists(p):
            rows.append((np.load(p)["panels"], ARM_LABEL.get(key, key)))
        else:
            print(f"[miss] {p}", flush=True)
    n = len(titles)
    fig, ax = plt.subplots(len(rows), n, figsize=(3.6 * n, 3.7 * len(rows)), dpi=190)
    ax = np.atleast_2d(ax)
    for r, (panels, lab) in enumerate(rows):
        for j in range(n):
            ax[r, j].imshow(panels[j])
            if r == 0:
                ax[r, j].set_title(titles[j], fontsize=11)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=13)
    what = "GROUND-TRUTH class (oracle labels)" if a.oracle else "predicted class"
    fig.suptitle(f"Reconstruction masked to a {what} - {a.scene} (view {a.view})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{a.outdir}/{a.scene}_v{a.view}_class_masked{a.tag}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["pick", "foam", "gs", "figure"], required=True)
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--vis-tol", type=float, default=0.05,
                    help="pick: |rendered z - GT z| under this counts the GT point as visible")
    ap.add_argument("--vis-min", type=int, default=300,
                    help="pick: a class counts as 'on screen' above this many visible GT points")
    ap.add_argument("--arms", nargs="*", default=list(ARM_LABEL),
                    help="figure: which arm npzs to stack as rows")
    ap.add_argument("--cam-from", default="pf_truefrozen",
                    help="gs: which arm's npz supplies the shared camera")
    ap.add_argument("--dataset", choices=["scannet", "lerf"], default="scannet")
    ap.add_argument("--frame", default=None,
                    help="lerf: annotated frame name, e.g. frame_00041.jpg (sets --view)")
    ap.add_argument("--oracle-mode", choices=["nearest-gt", "owned"], default="nearest-gt")
    ap.add_argument("--oracle-dist", type=float, default=0.10,
                    help="nearest-gt: primitives farther than this from the GT surface stay unlabelled")
    ap.add_argument("--oracle", action="store_true",
                    help="replace the CLIP predictions with GT labels, to isolate geometry")
    ap.add_argument("--outdir", default="artifacts/surface_viz/class_masked")
    a = ap.parse_args()

    from determinism import enable_determinism
    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    a.tag = "_oracle" if a.oracle else ""
    if a.dataset == "lerf":
        gt_pts, gt_lab, names = lerf_gt(a.scene)
        if a.frame:
            a.view = lerf_frame_view(a.scene, a.frame)
        a.variant, a.gs_arm = "lerf", "gs_lerf"
    else:
        gt_pts, gt_lab, names = scene_gt(a.scene, a.class_set)
    sel = selected(names, a.classes)
    print(f"{a.side}{a.tag}: {a.scene} view {a.view}, classes {[nm for _, nm in sel]}", flush=True)
    {"pick": side_pick, "foam": side_foam, "gs": side_gs, "figure": side_figure}[a.side](
        a, gt_pts, gt_lab, names, sel)


if __name__ == "__main__":
    main()
