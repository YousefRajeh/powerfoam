"""PROJECTED oracle: perfect per-pixel labels from GT, then lift. The contamination is REAL.

WHY THE PREVIOUS ORACLE WAS THE WRONG QUESTION. `oracle_stream.py` built its observations as
B = A Z T -- it labelled every primitive from its nearest GT point and then RENDERED that through
the same operator being solved. The exact solution is then Z by construction, and a ray through a
mug in front of a table has a "correct" answer that is already the blend of mug and table. The
occlusion contamination that motivated the whole question was defined to be the target instead of
being an error.

THIS ORACLE, per the user's design. Assume CLIP is perfect on what it is shown and SAM's masks are
perfect: then the observation at a pixel is the embedding of the class of the surface ACTUALLY
VISIBLE there. A ray through a mug reports MUG. The table primitive behind it still collects that
mug evidence through its own A_ij, and that shows up as error rather than as truth. No solver can
be exactly correct here, which is the point -- the shortfall measures the lifting problem itself.

PER-PIXEL TRUTH, BY RAYCASTING THE GT MESH. `scenes10_points3d/<scene>/points3d.ply` is the
labelled cloud WITH connectivity -- its vertices coincide with the Pointcept GT points to 0.00e+00 m
-- so one ray per pixel against the triangles gives exact visibility, and the hit triangle's
nearest vertex by barycentric coordinate gives the class with no transfer step.

The alternative (splatting the points and z-buffering) was BUILT AND MEASURED against this, in
`test_perpixel_labels.py`, and rejected: it covers only 49.6% of pixels against the mesh's 88.7%,
its depth is wrong by 0.128 m (10x the 0.0131 m point spacing), and 90.08% of its pixels sit IN
FRONT of the true surface because a large disc at a near point's depth wins the z-buffer over
pixels whose true surface is farther back. Labels agreed 98.65% where both fired, so the labels
were not the problem -- the coverage loss was, and it is biased toward near surfaces, which is
precisely the bias this oracle exists to measure.

Pixels the mesh does not cover are UNLABELLED and contribute no evidence; their share is reported
as `coverage` rather than being filled in by guessing.

This depends only on the scene and the cameras, never on the reconstruction, so every arm is
solved against a byte-identical upstream. Using an arm's own rendered depth would have been denser
but would have made B arm-dependent, which is exactly the confound being avoided.

CLASS SPACE. Each pixel carries ONE class, so B = S T with S one-hot and A^T B = (A^T S) T. The
solve therefore runs in C (7-19) dimensions and never materialises a (rays x 512) tensor; the
readout <x, t_c> is (A^T S / D) @ (T T^T) exactly.

SCORING, per the user's design: only primitives that actually received evidence (D > 0) are
assigned a class at all; each is argmax'd and checked against the class of its nearest GT point.
Point-level mIoU over the fixed GT sample is reported alongside, because the per-primitive
denominator is not comparable across arms whose primitive counts differ by 12x (A32).
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from ablation_opacity import primitive_alpha
from diagnose_holes import SCENES, GT_ROOT

MESH_ROOT = r"D:\Downloads\scenes10_points3d"
LABEL2D_ROOT = r"D:\Downloads\scannet_2dlabels"
NYU20 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def official_lut(kept, n2i, max_raw=1500):
    """raw ScanNet category id -> our 1..C class index (0 = not in the evaluated set).

    ScanNet's released 2D masks carry RAW category ids (1, 14, 27, ... 1163, 1176), which the
    official `scannetv2-labels.combined.tsv` maps to nyu40. The 20-class benchmark is a fixed subset
    of nyu40, and our per-scene `kept` list is the subset of those actually present. Composing the
    three gives one lookup table, built once per scene.
    """
    import csv
    raw2n40 = {}
    with open(os.path.join(LABEL2D_ROOT, "scannetv2-labels.combined.tsv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="	"):
            try:
                raw2n40[int(row["id"])] = int(row["nyu40id"])
            except (ValueError, KeyError):
                continue
    n40_to_nyuidx = {v: i for i, v in enumerate(NYU20)}
    nyuidx_to_ours = {n2i[n]: k + 1 for k, n in enumerate(kept)}
    lut = np.zeros(max_raw, np.int64)
    for rawid, n40 in raw2n40.items():
        if 0 <= rawid < max_raw:
            lut[rawid] = nyuidx_to_ours.get(n40_to_nyuidx.get(n40, -1), 0)
    return lut


def official_label_image(scene, vi, H, W, lut, frame_stems, dev):
    """ScanNet's OWN released per-frame mask (`_2d-label-filt`), at native 968x1296.

    Measured against our GT-mesh raycast on scene0097: 99.29% agreement where both are labelled,
    coverage 88.8% vs 88.7%, boundary density 0.452% vs 0.471%. So this changes no number
    materially -- it is used because it is the dataset's own artefact, which is a stronger
    provenance claim than rendering our own, and it removes our camera/ray convention from the
    upstream entirely.
    """
    from PIL import Image
    # UNFILTERED `label/`, not `label-filt/`. Measured against our segment20-derived raycast over
    # all 38 views of scene0097: raw agrees 99.760%, filtered only 99.418%, and the gap is
    # concentrated in the rare classes that dominate class-averaged mIoU (counter 97.7% vs 94.9%,
    # sink 97.8% vs 95.6%). The filtering is the discrepancy; the raw render tracks the mesh
    # annotation that the 3D scoring target also comes from.
    fp = os.path.join(LABEL2D_ROOT, scene, "label", f"{frame_stems[vi]}.png")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    a = np.array(Image.open(fp)).astype(np.int64)
    if a.shape != (H, W):                     # only if the run is downsampled
        a = np.array(Image.fromarray(a.astype(np.int32)).resize((W, H), Image.NEAREST))
    return torch.from_numpy(lut[np.clip(a, 0, lut.shape[0] - 1)].reshape(-1)).to(dev)


def mesh_label_image(scene, vi, cam, c2w, H, W, rc, tri, vert_cls, dev, cache_dir,
                     mode="vertex", gt_tree=None, gt_cls=None):
    """Exact per-pixel class by raycasting the GT mesh. Cached: it does not depend on the arm.

    THREE LABEL MODES, because the mesh is coarse (median edge 2.05 cm; a triangle projects to ~48
    px, so a label boundary can only be resolved to ~7 px, nothing like a SAM mask):

      vertex : label = the NEAREST VERTEX of the hit triangle, by barycentric weight. Fast, but
               inside a triangle whose three vertices disagree the boundary follows the barycentric
               bisectors -- a geometric artifact unrelated to the object edge. 1.99% of hit pixels
               sit in such triangles.
      knn    : label = the nearest GT POINT to the 3-D hit position, over the whole cloud rather
               than the hit triangle's three corners. Removes the arbitrary bisector split; still
               limited by the ~2 cm point spacing.
      strict : `vertex`, but any pixel whose hit triangle has DISAGREEING vertex labels is marked
               UNLABELLED. Follows the rule already used for mesh misses and unlabelled vertices:
               where the ground truth does not determine an answer, contribute no evidence rather
               than a coin flip. The most conservative option, and the one that bounds the artifact.

    Every arm consumes the identical cached image, so whichever mode is used the boundary noise is
    common-mode and cannot bias a foam-vs-3DGS comparison.
    """
    fp = os.path.join(cache_dir, f"v{vi}_{H}x{W}_{mode}.npy")
    if os.path.exists(fp):
        return torch.from_numpy(np.load(fp).astype(np.int64)).to(dev)
    import open3d as o3d
    d_cam = cam.cam_ray_dirs.reshape(-1, 3).double().cpu().numpy()
    dirs = d_cam @ c2w[:3, :3].numpy().T
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    orig = np.broadcast_to(c2w[:3, 3].numpy(), dirs.shape)
    ans = rc.cast_rays(o3d.core.Tensor(np.ascontiguousarray(
        np.concatenate([orig, dirs], 1), dtype=np.float32)))
    t_hit = ans["t_hit"].numpy(); pid = ans["primitive_ids"].numpy(); uv = ans["primitive_uvs"].numpy()
    hit = np.isfinite(t_hit)
    out = np.zeros(H * W, np.uint8)
    if hit.any():
        if mode == "knn":
            X = orig[hit] + t_hit[hit][:, None] * dirs[hit]
            _, nn = gt_tree.query(X, k=1, workers=-1)
            out[hit] = gt_cls[nn].astype(np.uint8)
        else:
            vtx = tri[pid[hit]]                                  # (N,3) vertex ids
            b = np.stack([1.0 - uv[hit, 0] - uv[hit, 1], uv[hit, 0], uv[hit, 1]], 1)
            lab = vert_cls[vtx[np.arange(b.shape[0]), b.argmax(1)]]
            if mode == "strict":
                tl = vert_cls[vtx]
                lab = np.where(tl.min(1) == tl.max(1), lab, 0)   # ambiguous -> no evidence
            out[hit] = lab.astype(np.uint8)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(fp, out)
    return torch.from_numpy(out.astype(np.int64)).to(dev)


def one(scene, recon, n_views, class_set, cap, squeeze=None, label_mode="vertex",
        visible_only=True, dev="cuda", tfloor=1e-3, dump_stats=None):
    from camera_bridge import K_from_ray_dirs
    is_gs = recon.startswith("gs_")
    cfg_recon = "nonfrozen" if is_gs else recon
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{cfg_recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    if is_gs:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ckpt = torch.load(f"recon_remote/{recon}/{scene}/ckpt.pt", map_location=dev,
                          weights_only=False)
        sp = ckpt["splats"] if "splats" in ckpt else ckpt
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gsc = torch.exp(sp["scales"].to(dev))
        _ol = sp["opacities"].to(dev).reshape(-1)
        gop = torch.sigmoid(_ol * squeeze) if squeeze else torch.sigmoid(_ol)
        gcol = torch.zeros((gm.shape[0], 1), device=dev)
        centers = gm.detach().cpu().numpy()
        model = None
    else:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
        centers = model.points.detach().cpu().numpy()
        radii = model.get_radii().detach().cpu().numpy()
    P = centers.shape[0]

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    lab_pts, lab_cls = gt_pts[gt_lab > 0], gt_lab[gt_lab > 0]

    from scipy.spatial import cKDTree
    tree = cKDTree(lab_pts)
    # each primitive's nearest labelled GT point -- the SCORING target, per the user's design
    _, nn = tree.query(centers, k=1, workers=-1)
    emit = torch.from_numpy(lab_cls[nn].astype(np.int64)).to(dev)

    T = embed_class_names(kept, dev)
    TT = T @ T.T

    # ---- GT mesh: exact per-pixel visibility, identical for every arm ----
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, scene, "points3d.ply"))
    Vm = np.asarray(mesh.vertices); tri = np.asarray(mesh.triangles)
    if Vm.shape[0] != gt_pts.shape[0] or float(np.abs(Vm - gt_pts).max()) > 1e-6:
        # the label lookup below indexes vertices BY POSITION in the GT array, so a mismatch here
        # would silently mislabel every pixel rather than fail
        raise RuntimeError(f"mesh/GT mismatch for {scene}: V {Vm.shape[0]} vs GT {gt_pts.shape[0]}")
    vert_cls = gt_lab.astype(np.int64)
    gt_tree_all = cKDTree(gt_pts)          # ALL points: an unlabelled nearest
    gt_cls_all = gt_lab.astype(np.int64)   # neighbour correctly yields class 0
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cache_dir = os.path.join("artifacts", "scannet", scene, "gtlabels")
    # ScanNet's own released masks; `frame_stems[vi]` because the loader orders cameras by the same
    # lexicographic sort of the image filenames (0.jpg, 100.jpg, 120.jpg, ...), so index vi and
    # frame stem correspond. Verified by the 99.29% agreement against our raycast.
    off_lut = frame_stems = None
    if label_mode in ("official", "sam_gt", "sam_clip"):
        off_lut = official_lut(kept, n2i)
        frame_stems = [os.path.splitext(f)[0]
                       for f in sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))]

    ncam = len(dh.cameras)
    sel = (list(range(ncam)) if n_views <= 0 else
           np.linspace(0, ncam - 1, min(n_views, ncam)).astype(int).tolist())

    # --- plot statistics (only when --dump-stats). Counts, NOT mass: the paper's purity plot is
    # defined as (rays of the top-contributing class) / (total rays) per primitive, which is a
    # different quantity from the mass-weighted ev_top_share and is kept separately.
    AtN = torch.zeros(P, C, device=dev) if dump_stats else None
    rays_per_prim = torch.zeros(P, dtype=torch.int64, device=dev) if dump_stats else None
    PPR_MAX = int(cap) + 2
    hist_ppr = torch.zeros(PPR_MAX, dtype=torch.int64, device=dev) if dump_stats else None

    AtS = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
    AtS2 = torch.zeros(P, C, device=dev); D2 = torch.zeros(P, device=dev)
    gz = torch.zeros(P, C, device=dev); gw = torch.zeros(P, device=dev)
    cov_s = agree_n = agree_d = 0.0
    nv = 0
    R_tot = nnz = 0

    def sph_norm(Y):
        return ((Y @ TT) * Y).sum(-1).clamp_min(0).sqrt()

    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        K, _ = K_from_ray_dirs(cam)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        vm = torch.linalg.inv(c2w).float().to(dev)

        if label_mode in ("sam_gt", "sam_clip"):
            # RUNGS OF THE UPSTREAM LADDER. Each replaces one real stage while leaving the rest
            # oracular, so the mIoU drop from `official` is that stage's cost IN THE METRIC WE
            # REPORT -- not in pixel accuracy, which is a different denominator.
            #   sam_gt   real SAM regions, each emitting its own majority GT class
            #            -> costs exactly what SAM's coverage + boundaries cost
            #   sam_clip the same regions, each emitting argmax of its own CLIP feature
            #            -> adds exactly what CLIP's naming costs
            from upstream_modes import sam_label_image
            _gt = official_label_image(scene, vi, H, W, off_lut, frame_stems, dev)
            cls_img, _ = sam_label_image(scene, frame_stems[vi], H, W,
                                         _gt.cpu().numpy(), C, dev, label_mode, TT=T)
            del _gt
        elif label_mode == "official":
            cls_img = official_label_image(scene, vi, H, W, off_lut, frame_stems, dev)
        else:
            cls_img = mesh_label_image(scene, vi, cam, c2w.float(), H, W, rc, tri, vert_cls,
                                       dev, cache_dir, label_mode, gt_tree_all, gt_cls_all)
        cov_s += float((cls_img > 0).float().mean()); nv += 1
        if not bool((cls_img > 0).any()):
            continue

        if is_gs:
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gsc, gop, gcol, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=tfloor)
        else:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096,
                                           transmittance_threshold=tfloor)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        r_ = ri.to(torch.int64).to(dev); c_ = ci.to(torch.int64).to(dev); v_ = vv.float().to(dev)
        del ri, ci, vv
        if dump_stats:
            cnt_row = torch.bincount(r_, minlength=H * W)
            nzr = cnt_row[cnt_row > 0].clamp(max=PPR_MAX - 1)
            hist_ppr += torch.bincount(nzr, minlength=PPR_MAX)
            rays_per_prim += torch.bincount(c_, minlength=P)
            del cnt_row, nzr

        keep = cls_img[r_] > 0                      # unlabelled pixels contribute no evidence
        r_, c_, v_ = r_[keep], c_[keep], v_[keep]
        if r_.numel() == 0:
            continue
        R_tot += H * W; nnz += v_.numel()
        k_ = cls_img[r_] - 1

        if dump_stats:
            AtN.index_put_((c_, k_), torch.ones_like(v_), accumulate=True)
        AtS.index_put_((c_, k_), v_, accumulate=True)
        D.index_add_(0, c_, v_)
        v2 = v_ * v_
        AtS2.index_put_((c_, k_), v2, accumulate=True)
        D2.index_add_(0, c_, v2)

        # DIAGNOSTIC: how much ray mass lands on a primitive whose own nearest-GT class already
        # matches the visible class. 1.0 would mean no contamination is possible at all.
        agree_n += float(v_[(emit[c_] - 1) == k_].sum()); agree_d += float(v_.sum())

        yv = torch.zeros(P, C, device=dev); yv.index_put_((c_, k_), v_, accumulate=True)
        wv = torch.zeros(P, device=dev).index_add_(0, c_, v_)
        seen = (wv > 0) & (sph_norm(yv) > 1e-20)
        fv = torch.zeros_like(yv)
        fv[seen] = yv[seen] / sph_norm(yv[seen]).unsqueeze(-1)
        init = seen & (gw <= 0); upd = seen & (gw > 0)
        gz[init] = fv[init]; gw[init] = wv[init]
        if upd.any():
            zp, wp_, wn = gz[upd], gw[upd], wv[upd]
            eta = (wn / (wp_ + wn)).clamp_max(1.0)
            cos = ((fv[upd] @ TT) * zp).sum(-1, keepdim=True)
            zn = zp + eta[:, None] * (fv[upd] - cos * zp)
            gz[upd] = zn / sph_norm(zn).clamp_min(1e-30).unsqueeze(-1); gw[upd] = wp_ + wn
        del cls_img, r_, c_, v_, v2, k_, yv, wv, fv
        torch.cuda.empty_cache()

    live = D > 0
    Wp = torch.zeros(P, C, device=dev); Wp[live] = AtS[live] / D[live].unsqueeze(-1)
    Wt = torch.zeros(P, C, device=dev); Wt[live] = AtS2[live] / D2[live].clamp_min(1e-30).unsqueeze(-1)

    # ---- TWO point->primitive assignment rules, scored side by side ----
    # CENTRE: Euclidean nearest centre. Applies to both representations unchanged, and is what
    #         `evaluate_point_cloud_miou.evaluate_gaussians` uses for splats.
    # SHAPE : the representation's own notion of ownership -- power-cell containment for foam
    #         (what `evaluate_powerfoam` uses), exact Mahalanobis argmin for Gaussians. Giving only
    #         the splat arm a shape-aware rule was the asymmetry flagged in A32.4; running both
    #         rules on both arms removes it instead of trading one bias for its mirror image.
    # VISIBLE-ONLY SCORING SET. A labelled GT point that no ray ever saw (out of frustum, or
    # occluded in every view) deposited evidence on nothing, so scoring it charges the lift for the
    # capture's coverage rather than for the lift. The mask comes from the GT mesh + cameras alone,
    # so the SAME points are removed for every arm -- a fair restriction, not a favourable one.
    # Measured: 9.15% of labelled points are never seen (1.75% out of frustum, 7.40% occluded).
    vis_fp = os.path.join("artifacts", "scannet", scene, "gt_visible.npy")
    vis_mask = None
    if visible_only and os.path.exists(vis_fp):
        vfull = np.load(vis_fp)
        vis_mask = vfull[gt_lab > 0]

    live_np = live.cpu().numpy(); live_idx = np.nonzero(live_np)[0]
    pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)
    assigns = {}
    shape_owner_dead = float("nan")
    # OpenGaussian's low-opacity GT masking (`eval_scannet.py:127-129`): a point whose owning
    # primitive is transparent is DELETED from the metric rather than scored. Reported as a
    # separate column, never as the default -- it deletes 12.1% of labelled points for foam but
    # 24.9-25.0% for the splat arms, so it exempts twice as much of the hard geometry from one
    # side of the comparison as from the other.
    try:
        alpha_np = primitive_alpha(recon, scene)
        alpha_np = None if alpha_np is None else np.asarray(alpha_np).reshape(-1)
    except Exception as e:
        alpha_np = None
        print(f"    [alpha unavailable: {type(e).__name__}: {e}]", flush=True)
    if live_idx.size:
        _, loc = cKDTree(centers[live_idx]).query(lab_pts, k=1, workers=-1)
        assigns["centre"] = torch.from_numpy(live_idx[loc].astype(np.int64)).to(dev)
        # The shape-aware assignment is ALREADY CACHED by the ablation pipeline, keyed by
        # (scene, arm), and verified here to be exactly power-cell containment for foam
        # (100.00% match) and exact Mahalanobis argmin for splats (100.00%). Loading it both
        # avoids a costly recomputation and guarantees the oracle scores against the SAME
        # correspondence the paper's tables use.
        #
        # The cache is over ALL primitives, while only `live` ones get a feature. A point whose
        # owner never received evidence therefore has NO prediction and is counted wrong -- which
        # is what `evaluate_powerfoam` does too (`pred_labels[owned]`, rest stay 0). That penalty
        # is real and is reported as `shape_owner_dead`.
        CACHE_ARM = {"truefrozen": "pf_tfroz", "nonfrozen": "pf_nonfroz",
                     "gs_froz": "gs_froz", "gs_unfroz": "gs_unfroz"}
        try:
            cp = os.path.join("artifacts", "ablation_cache",
                              f"{scene}_{CACHE_ARM[recon]}_assign.npy")
            if os.path.exists(cp):
                full = np.load(cp)
                if full.shape[0] != gt_pts.shape[0]:
                    raise RuntimeError(f"cache len {full.shape[0]} != GT {gt_pts.shape[0]}")
                sh = full[gt_lab > 0]
            elif is_gs:
                from ablation_maha import prepare, assign_exact
                mu, sc_, Rm, _, _ = prepare(sp["means"], sp["scales"], sp["quats"])
                idx, _ = assign_exact(torch.from_numpy(np.ascontiguousarray(lab_pts)).float().to(dev),
                                      mu.to(dev), sc_.to(dev), Rm.to(dev), device=dev)
                sh = idx.cpu().numpy()
            else:
                sh = assign_points_to_power_cells(lab_pts, centers, radii, valid=None, k=64)
            dead = ~live_np[np.clip(sh, 0, P - 1)]
            shape_owner_dead = float(dead.mean())
            sh = np.where(sh >= 0, sh, live_idx[loc])
            assigns["shape"] = torch.from_numpy(sh.astype(np.int64)).to(dev)
        except Exception as e:
            shape_owner_dead = float("nan")
            print(f"    [shape-assign unavailable: {type(e).__name__}: {e}]", flush=True)

    # Restrict the scoring set to points that were actually VISIBLE in some view (see above). Done
    # once, after every assignment is built, so pt_gt and all assignment vectors stay index-aligned.
    if vis_mask is not None:
        vm = torch.from_numpy(vis_mask).to(dev)
        pt_gt = pt_gt[vm]
        for _k in list(assigns):
            assigns[_k] = assigns[_k][vm]

    # Filled by the purity block below; score() runs after it, so the closure sees it.
    # Splits the scored points by how AMBIGUOUS their owner's evidence was, which is the
    # forced-vs-contested decomposition: a primitive receiving one class has no decision to make.
    BUCKETS = {}

    def score(Wmat):
        pred = torch.zeros(P, dtype=torch.long, device=dev)
        pred[live] = (Wmat[live] @ TT).argmax(1) + 1
        acc = float((pred[live] == emit[live]).float().mean())
        _, miou, _, macc = calculate_metrics(emit[live].cpu(), pred[live].cpu(), C + 1)
        out = {"acc": acc, "miou": float(miou), "macc": float(macc)}
        for nm, av in assigns.items():
            pp = pred[av]
            _, pmi, _, _ = calculate_metrics(pt_gt.cpu(), pp.cpu(), C + 1)
            out[f"pt_miou_{nm}"] = float(pmi)
            out[f"pt_acc_{nm}"] = float((pp == pt_gt).float().mean())
            if nm == "centre" and BUCKETS:
                ok = (pp == pt_gt).float()
                for bname, bm in BUCKETS.items():
                    n = float(bm.sum())
                    out[f"b_{bname}_n"] = int(n)          # raw count, so scenes can be pooled
                    out[f"b_{bname}_frac"] = n / max(float(bm.numel()), 1.0)
                    out[f"b_{bname}_acc"] = float(ok[bm].mean()) if n else float("nan")
                    # mIoU restricted to this bucket's points. NOT decomposable like accuracy --
                    # it averages over the classes PRESENT IN THE BUCKET, so the denominator
                    # differs per bucket and the buckets do not recombine into the overall mIoU.
                    # Report it as "mIoU if only these points were scored", never as a share.
                    if n:
                        _g, _p = pt_gt[bm], pp[bm]
                        _, _bm_mi, _, _ = calculate_metrics(_g.cpu(), _p.cpu(), C + 1)
                        out[f"b_{bname}_miou"] = float(_bm_mi)
                        out[f"b_{bname}_ncls"] = int((torch.unique(_g) != 0).sum())
                    else:
                        out[f"b_{bname}_miou"] = float("nan")
                        out[f"b_{bname}_ncls"] = 0
            if alpha_np is not None:
                low = torch.from_numpy(
                    alpha_np[np.clip(av.cpu().numpy(), 0, alpha_np.shape[0] - 1)] < 0.1).to(dev)
                gtm = pt_gt.clone(); gtm[low] = 0
                _, mmi, _, _ = calculate_metrics(gtm.cpu(), pp.cpu(), C + 1)
                out[f"pt_miou_{nm}_ogmask"] = float(mmi)
                out[f"pt_dropped_{nm}"] = float((low & (pt_gt > 0)).float().sum()
                                                / max(float((pt_gt > 0).sum()), 1.0))
        return out

    # ---- PURITY: how many DISTINCT labels does each primitive receive? ----
    # A primitive can emit exactly one class. If the GT points it owns carry more than one label it
    # is IMPURE and no solver can be right about all of them -- that is a geometry limit, not a
    # solve limit. Reported as the label-count distribution (the direct question), plus the
    # point-weighted majority share and the mIoU a perfect solver would reach if every primitive
    # took its own majority class. Computed on the SAME scoring set as the metric.
    row_extra = {}
    purity = ceil_acc = ceil_miou = float("nan")
    pure_frac = mean_labels = float("nan")
    ev_pure_frac = ev_mean = ev_top = float("nan")
    lab_hist = {}
    if "centre" in assigns and assigns["centre"].numel():
        own = assigns["centre"]
        cnt = torch.zeros(P, C + 1, device=dev)
        cnt.index_put_((own, pt_gt), torch.ones(pt_gt.numel(), device=dev), accumulate=True)
        occ = cnt[:, 1:]                                   # class 0 is the ignore slot
        nlab = (occ > 0).sum(1)                            # distinct labels per primitive
        has = nlab > 0
        pure_frac = float((nlab[has] == 1).float().mean())
        mean_labels = float(nlab[has].float().mean())
        for k in (1, 2, 3, 4):
            lab_hist[k] = float((nlab[has] == k).float().mean()) if k < 4 else                 float((nlab[has] >= 4).float().mean())
        purity = float(occ.max(1).values.sum() / occ.sum().clamp_min(1))

        # EVIDENCE purity: how many DISTINCT class-features the rays actually deposited on each
        # live primitive (the nonzero entries of AtS_j). This is the other half of the question --
        # a primitive can own GT points of a single class (pure geometry) yet still receive several
        # different features (contaminated evidence), or straddle a boundary and own several
        # classes (impure geometry) no matter how clean the evidence is. The first is a solve
        # problem, the second is a geometry problem, and only both together explain the shortfall.
        ev = (AtS[live] > 0).sum(1).float()
        ev_pure_frac = float((ev == 1).float().mean())
        ev_mean = float(ev.mean())
        # mass-weighted: what share of a primitive's evidence belongs to its own dominant class
        ev_top = float((AtS[live].max(1).values / AtS[live].sum(1).clamp_min(1e-30)).mean())
        del ev
        # Per-primitive evidence ambiguity, broadcast to the points each primitive owns.
        # ev_cnt: how many distinct classes deposited ANY mass (count view).
        # ev_top: what share of the mass belongs to the dominant class (mass view) -- a primitive
        # with two classes at 0.999/0.001 is forced in practice but impure by count, so both.
        ev_cnt_p = torch.zeros(P, device=dev)
        ev_cnt_p[live] = (AtS[live] > 0).sum(1).float()
        ev_top_p = torch.zeros(P, device=dev)
        ev_top_p[live] = AtS[live].max(1).values / AtS[live].sum(1).clamp_min(1e-30)
        ec, et = ev_cnt_p[own], ev_top_p[own]
        BUCKETS["ev1"] = ec == 1
        BUCKETS["ev2"] = ec == 2
        BUCKETS["ev3p"] = ec >= 3
        # EXACT per-k decomposition (k = 1..C, which is complete since a primitive cannot receive
        # more classes than exist). The 1/2/>=3 grouping above is only a presentation choice; these
        # let the same measurement be redrawn as a histogram, or re-binned, without re-running.
        for _k in range(1, C + 1):
            BUCKETS[f"evk{_k}"] = ec == _k
        # WHY a single-evidence point can still be wrong. Exactly two causes, and they are
        # different problems:
        #   imp  the owner straddles a label boundary (owns >1 GT label). One class cannot serve
        #        all its points -- a CAPACITY limit of the partition. No solver can fix it.
        #   occ  the owner is GT-pure, so its one evidence class is simply the WRONG class: every
        #        ray that reached it came from an occluder. A VISIBILITY limit, also unfixable
        #        by reweighting, since there is no competing class to promote.
        # Neither is a solve failure, which is why all three solvers score these identically.
        own_nlab = nlab[own].float()
        BUCKETS["ev1imp"] = (ec == 1) & (own_nlab > 1)
        BUCKETS["ev1pur"] = (ec == 1) & (own_nlab == 1)
        BUCKETS["ev2imp"] = (ec >= 2) & (own_nlab > 1)
        BUCKETS["ev2pur"] = (ec >= 2) & (own_nlab == 1)
        BUCKETS["m999"] = et >= 0.999
        BUCKETS["m99"] = (et >= 0.99) & (et < 0.999)
        BUCKETS["m90"] = (et >= 0.90) & (et < 0.99)
        BUCKETS["m70"] = (et >= 0.70) & (et < 0.90)
        BUCKETS["m50"] = (et >= 0.50) & (et < 0.70)
        BUCKETS["mlo"] = et < 0.50
        # Is a FORCED primitive even right? Its single class need not be its own majority GT class
        # (all its rays may have come from an occluder). Without this the 'easy' set is not a bound.
        e1 = (ev_cnt_p == 1) & live
        row_extra["ev1_prim_frac"] = float(e1.float().sum() / max(float(live.sum()), 1.0))
        if int(e1.sum()):
            maj_all = occ.argmax(1) + 1
            hasgt = (occ.sum(1) > 0) & e1
            row_extra["ev1_prim_agrees_gt"] = float(
                ((AtS[hasgt].argmax(1) + 1) == maj_all[hasgt]).float().mean()) if int(hasgt.sum()) else float("nan")
        del ev_cnt_p, ev_top_p, ec, et, own_nlab
        maj = occ.argmax(1) + 1
        ceil_pred = maj[own]
        ceil_acc = float((ceil_pred == pt_gt).float().mean())
        _, _cm, _, _ = calculate_metrics(pt_gt.cpu(), ceil_pred.cpu(), C + 1)
        ceil_miou = float(_cm)
        del cnt, occ, nlab, maj, ceil_pred

    S7, S24, SGM = score(Wp), score(Wt), score(gz)
    row = dict(scene=scene, recon=recon, views=len(sel), P=int(P), live=int(live.sum()),
               rays=int(R_tot), nnz=int(nnz), C=C, n_gt_pts=int(lab_pts.shape[0]),
               n_scored=int(pt_gt.numel()),
               purity=purity, ceiling_acc=ceil_acc, ceiling_miou=ceil_miou,
               pure_frac=pure_frac, mean_labels=mean_labels,
               lab1=lab_hist.get(1, float("nan")), lab2=lab_hist.get(2, float("nan")),
               lab3=lab_hist.get(3, float("nan")), lab4plus=lab_hist.get(4, float("nan")),
               ev_pure_frac=ev_pure_frac, ev_mean_classes=ev_mean, ev_top_share=ev_top,
               coverage=cov_s / max(nv, 1), front_agree=agree_n / max(agree_d, 1e-30),
               assign_rules=sorted(assigns.keys()), shape_owner_dead=shape_owner_dead,
               label_mode=label_mode, tfloor=tfloor,
               squeeze=(squeeze or 0.0))
    if dump_stats:
        # Per-primitive arrays for the paper's plots. Accuracy is attributed to the primitive that
        # OWNS each scored GT point (centre rule, same as the metric), so a purity-vs-accuracy curve
        # is weighted the way the reported mIoU is.
        os.makedirs(dump_stats, exist_ok=True)
        # foam carries an exact facet adjacency; 3DGS does not, so dump empties there and the
        # offline script falls back to kNN on centres for the comparison arm.
        try:
            _adj_flat = model.adjacency.detach().cpu().numpy().astype(np.int32)
            _adj_off = model.adjacency_offsets.detach().cpu().numpy().astype(np.int64)
        except Exception:
            _adj_flat = np.zeros(0, np.int32); _adj_off = np.zeros(0, np.int64)
        _tot = AtN.sum(1)
        _pur = AtN.max(1).values / _tot.clamp_min(1)          # count-based purity (the paper's)
        _mass = AtS.sum(1)
        _purm = AtS.max(1).values / _mass.clamp_min(1e-30)    # mass-weighted, for comparison
        # Do the two definitions even agree on WHICH class is top? They can disagree: a primitive
        # reached by 60 rays of class A and 40 of class B is 0.60-pure for A by count, but if each
        # B ray deposits 10x the weight it is 0.87-pure for B by mass -- opposite answers. The
        # solver reads MASS (W = AtS/D), so where they disagree only the mass version is causally
        # tied to the prediction, and a count-purity x-axis cannot explain that primitive.
        _topc = AtN.argmax(1); _topm = AtS.argmax(1)
        _agree_top = (_topc == _topm)
        _pred = torch.zeros(P, dtype=torch.long, device=dev)
        _pred[live] = (Wp[live] @ TT).argmax(1) + 1
        _npts = torch.zeros(P, device=dev)
        _ncor = torch.zeros(P, device=dev)
        if "centre" in assigns and assigns["centre"].numel():
            _own = assigns["centre"]
            _ok = (_pred[_own] == pt_gt).float()
            _npts.index_add_(0, _own, torch.ones_like(_ok))
            _ncor.index_add_(0, _own, _ok)
        np.savez_compressed(
            os.path.join(dump_stats, f"{recon}_{scene}.npz"),
            hist_prims_per_ray=hist_ppr.cpu().numpy(),
            rays_per_prim=rays_per_prim.cpu().numpy().astype(np.int32),
            purity_count=_pur.cpu().numpy().astype(np.float32),
            purity_mass=_purm.cpu().numpy().astype(np.float32),
            top_class_agree=_agree_top.cpu().numpy(),
            top_class_count=_topc.cpu().numpy().astype(np.int16),
            top_class_mass=_topm.cpu().numpy().astype(np.int16),
            n_evidence_rays=_tot.cpu().numpy().astype(np.int64),
            n_points=_npts.cpu().numpy().astype(np.int32),
            n_correct=_ncor.cpu().numpy().astype(np.int32),
            live=live.cpu().numpy(),
            # --- everything a facet-propagation readout needs, so the method can be iterated
            # OFFLINE without re-running the oracle. AtS is the per-primitive class evidence
            # (P x C); `own`/`pt_gt` are the scored points' owner and true class, so mIoU can be
            # recomputed for any relabelling; `adj`/`adj_off` is foam's power-diagram facet graph
            # (the Cech complex already in the checkpoint) -- 3DGS has no such graph.
            AtS=AtS.cpu().numpy().astype(np.float32),
            own=(assigns["centre"].cpu().numpy().astype(np.int32)
                 if "centre" in assigns else np.zeros(0, np.int32)),
            pt_gt=pt_gt.cpu().numpy().astype(np.int16),
            adj=_adj_flat, adj_off=_adj_off,
            meta=np.array([P, int(live.sum()), C, len(sel)], dtype=np.int64))
        del _tot, _pur, _mass, _purm, _pred, _npts, _ncor, _topc, _topm, _agree_top

    row.update(row_extra)
    for tag, res in (("closed", S7), ("tikhonov", S24), ("geomed", SGM)):
        row[f"acc_{tag}"] = res["acc"]; row[f"miou_{tag}"] = res["miou"]
        row[f"macc_{tag}"] = res["macc"]
        for nm in assigns:
            row[f"pt_miou_{nm}_{tag}"] = res[f"pt_miou_{nm}"]
            row[f"pt_acc_{nm}_{tag}"] = res[f"pt_acc_{nm}"]
            if f"pt_miou_{nm}_ogmask" in res:
                row[f"pt_miou_{nm}_ogmask_{tag}"] = res[f"pt_miou_{nm}_ogmask"]
                row[f"pt_dropped_{nm}"] = res[f"pt_dropped_{nm}"]
        # forced-vs-contested buckets (only known keys are copied above, so pass these through)
        for _k, _v in res.items():
            if _k.startswith("b_"):
                row[f"{_k}_{tag}"] = _v
    if "centre" in assigns and "shape" in assigns:
        a, b = assigns["centre"], assigns["shape"]
        row["assign_same_frac"] = float((a == b).float().mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--views", type=int, default=-1)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--squeeze", type=float, default=None)
    ap.add_argument("--all-points", action="store_true",
                    help="score every labelled GT point, including ones no view ever saw")
    ap.add_argument("--label-mode", default="vertex",
                    # sam_gt / sam_clip are the upstream ladder rungs; see the dispatch above
                    choices=["vertex", "knn", "strict", "official", "sam_gt", "sam_clip"])
    ap.add_argument("--tfloor", type=float, default=1e-3,
                    help="stop marching a ray once transmittance falls below this. The default "
                         "1e-3 lets a ray deposit on everything until 99.9%% of its light is gone, "
                         "so primitives BEHIND the visible surface receive that surface's class. "
                         "Raising it truncates the ray nearer the first surface.")
    ap.add_argument("--dump-stats", default=None,
                    help="directory to write per-primitive arrays for the paper plots")
    ap.add_argument("--out", default="artifacts/scannet/oracle_projected.json")
    a = ap.parse_args()
    from determinism import enable_determinism
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    rows = []
    if os.path.exists(a.out):
        try:
            rows = json.load(open(a.out))
        except Exception:
            rows = []
    done = {(r["recon"], r["scene"]) for r in rows}
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            if (rec, sc) in done:
                print(f"[{rec}/{sc}] cached", flush=True); continue
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.squeeze, a.label_mode,
                        not a.all_points, tfloor=a.tfloor, dump_stats=a.dump_stats)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] v{r['views']} P {r['P']:,} live {r['live']:,} "
                  f"labelled-px {r['coverage']:.1%} front_agree {r['front_agree']:.3f} || "
                  f"PER-PRIMITIVE Eq6 {r['miou_closed']*100:6.2f} Eq18 {r['miou_tikhonov']*100:6.2f} "
                  f"GeoMed {r['miou_geomed']*100:6.2f} || PER-POINT(centre) Eq6 {r.get('pt_miou_centre_closed', float('nan'))*100:6.2f} "
                  f"Eq18 {r.get('pt_miou_centre_tikhonov', float('nan'))*100:6.2f} "
                  f"[shape {r.get('pt_miou_shape_tikhonov', float('nan'))*100:6.2f}, "
                  f"same-assign {r.get('assign_same_frac', float('nan')):.1%}]", flush=True)
            torch.cuda.empty_cache()
    if rows:
        print("")
        print("NEW (projected) oracle.")
        print("  PER-PRIMITIVE = mIoU over SEGMENTABLE primitives; truth = that primitive's "
              "nearest GT point.")
        print("  PER-POINT     = mIoU over the fixed GT point set, under two ownership rules:")
        print("      centre  Euclidean nearest centre                 (both arms)")
        print("      shape   power-cell containment (foam) / exact Mahalanobis argmin (3DGS)")
        hdr = f"{'arm':<12}{'segmentable':>12}{'same':>7}|"
        print("")
        print(hdr + f"{'PER-PRIMITIVE':^28}|{'PER-POINT centre':^28}|{'PER-POINT shape':^28}")
        print(" " * len(hdr) + f"{'Eq6':>9}{'Eq18':>9}{'GeoMed':>10}" * 3)
        for rec in a.recons.split(","):
            rs = [r for r in rows if r["recon"] == rec]
            if not rs:
                continue
            def f(k):
                v = [r[k] for r in rs if k in r and r[k] == r[k]]
                return float(np.mean(v)) if v else float("nan")
            cells = ""
            for pre in ("miou", "pt_miou_centre", "pt_miou_shape"):
                for sol in ("closed", "tikhonov", "geomed"):
                    cells += f"{f(pre + '_' + sol) * 100:>9.2f}" if sol != "geomed" else                              f"{f(pre + '_' + sol) * 100:>10.2f}"
            print(f"{rec:<12}{f('live'):>12,.0f}{f('assign_same_frac'):>7.1%}|" + cells
                  + f"  n={len(rs)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
