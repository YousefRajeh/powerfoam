"""Joint (deconvolved) solve on the PROJECTED oracle -- can it recover the contaminated 64%?

WHY THIS IS WORTH RUNNING NOW, AND WAS NOT BEFORE. Under the OLD oracle (`B = A Z T`) an exact
solution existed by construction, so `G^-1 A^T B` recovered it trivially and the win said nothing.
Under the PROJECTED oracle each pixel carries the class of the surface ACTUALLY VISIBLE there, so
`A Y = S` has no exact solution and this is a genuine least-squares problem.

`diagnose_oracle_gap.py` measured the opportunity: of the primitives whose weighted-plurality class
is wrong, ~64% still carry NONZERO mass for their true class (out-voted, not absent) while ~36%
carry none at all (occluded wherever visible -- a visibility limit no solver can beat). So a joint
solve has roughly 6 of the ~9 mIoU of contamination loss available to it, and 3 is a hard floor.

THE SOLVE. Eq. 6 is the per-primitive average `Y = D^-1 A^T S`: it sees only j's own rays and cannot
know that the "wall" mass on them was already explained by the wall in front. The joint objective

    min_Y || A Y - S ||^2        (normal equations  G Y = A^T S,  G = A^T A)

subtracts exactly that. G is P x P and dense-ish, so it is never formed: `G Y = A^T (A Y)` is applied
by two streaming passes over the cached operator. The iteration is preconditioned Richardson,

    Y <- Y - eta_j * ( A^T (A Y) - A^T S ),      eta_j = 1 / sum_k G_jk   (Gershgorin)

whose step is guaranteed non-expansive row by row without any tuning, starting FROM the Eq. 6
solution so it can only improve on it, and guarded by tracking the true objective and keeping the
best iterate -- semi-convergence is real here (A27: the exact least-squares optimum is WORSE than
the closed form on real features, because near-null directions blow up).

COST. Unlike Eq. 6 this needs A on every iteration, so the operator is cached in memory --
restricted to labelled pixels, which is ~84% of nnz. Use small scenes or few views for the big ones.
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
from diagnose_holes import SCENES, GT_ROOT
from oracle_projected import mesh_label_image, MESH_ROOT


def one(scene, recon, n_views, class_set, cap, label_mode, iters, dev="cuda"):
    from scipy.spatial import cKDTree
    import open3d as o3d
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    wp.init()

    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
    centers = model.points.detach().cpu().numpy()
    P = centers.shape[0]

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    lab_pts, lab_cls = gt_pts[gt_lab > 0], gt_lab[gt_lab > 0]
    tree = cKDTree(lab_pts)
    emit = torch.from_numpy(lab_cls[tree.query(centers, k=1, workers=-1)[1]].astype(np.int64)).to(dev)
    T = embed_class_names(kept, dev); TT = T @ T.T

    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, scene, "points3d.ply"))
    tri = np.asarray(mesh.triangles); vert_cls = gt_lab.astype(np.int64)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cache_dir = os.path.join("artifacts", "scannet", scene, "gtlabels")
    gt_tree_all = cKDTree(gt_pts); gt_cls_all = gt_lab.astype(np.int64)

    ncam = len(dh.cameras)
    sel = list(range(ncam)) if n_views <= 0 else \
        np.linspace(0, ncam - 1, min(n_views, ncam)).astype(int).tolist()

    # ---- cache the operator, labelled pixels only, with COMPACT global ray ids ----
    rows, cols, vals, kls = [], [], [], []
    roff = 0
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        cls_img = mesh_label_image(scene, vi, cam, c2w.float(), H, W, rc, tri, vert_cls,
                                   dev, cache_dir, label_mode, gt_tree_all, gt_cls_all)
        if not bool((cls_img > 0).any()):
            continue
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        r_ = op.row_indices.to(torch.int64).to(dev)
        c_ = op.col_indices.to(torch.int64).to(dev)
        v_ = op.values.float().to(dev)
        del op
        keep = cls_img[r_] > 0
        r_, c_, v_ = r_[keep], c_[keep], v_[keep]
        if r_.numel() == 0:
            del cls_img; continue
        # compact this view's surviving rays to 0..n-1 so the (R, C) residual stays small
        uniq, inv = torch.unique(r_, return_inverse=True)
        rows.append(inv + roff); cols.append(c_); vals.append(v_)
        kls.append(cls_img[uniq] - 1)
        roff += uniq.numel()
        del cls_img, r_, c_, v_, uniq, inv
        torch.cuda.empty_cache()

    row = torch.cat(rows); col = torch.cat(cols); val = torch.cat(vals)
    kcl = torch.cat(kls)                                   # (R,) class of each kept ray
    del rows, cols, vals, kls
    R, nnz = int(roff), int(val.numel())

    AtS = torch.zeros(P, C, device=dev).index_put_((col, kcl[row]), val, accumulate=True)
    D = torch.zeros(P, device=dev).index_add_(0, col, val)
    live = D > 0
    Y0 = torch.zeros(P, C, device=dev); Y0[live] = AtS[live] / D[live].unsqueeze(-1)

    # Gershgorin row step: eta_j = 1 / sum_k G_jk = 1 / (A^T (A 1))_j
    rowsum = torch.zeros(R, device=dev).index_add_(0, row, val)
    Gs = torch.zeros(P, device=dev).index_add_(0, col, val * rowsum[row])
    eta = torch.zeros(P, device=dev); eta[live] = 1.0 / Gs[live].clamp_min(1e-20)

    # ROW-CHUNKED gradient. Materialising the full (R, C) residual OOMs on the large scenes
    # (R reaches 269M rays -> 7.5 GB at C=7, on top of a multi-GB operator). Sorting the nnz by row
    # makes each row-block's entries contiguous, so the residual only ever exists for one block.
    order = torch.argsort(row)
    row, col, val = row[order], col[order], val[order]
    del order
    bounds = torch.searchsorted(row, torch.arange(0, R + 1, device=dev))

    def grad_and_obj(Y, block=1 << 22):
        g = torch.zeros(P, C, device=dev)
        obj = 0.0
        for a0 in range(0, R, block):
            a1 = min(a0 + block, R)
            s0, s1 = int(bounds[a0]), int(bounds[a1])
            if s1 <= s0:
                continue
            rl = row[s0:s1] - a0                       # local row ids within the block
            cl, vl = col[s0:s1], val[s0:s1]
            ay = torch.zeros(a1 - a0, C, device=dev)
            ay.index_add_(0, rl, vl.unsqueeze(-1) * Y[cl])
            ay[torch.arange(a1 - a0, device=dev), kcl[a0:a1]] -= 1.0
            obj += float(ay.pow(2).sum())
            g.index_add_(0, cl, vl.unsqueeze(-1) * ay[rl])
            del ay, rl, cl, vl
        return g, obj

    def objective(Y):
        return grad_and_obj(Y)[1]

    Y = Y0.clone()
    j0 = objective(Y0); jb, Yb, bit = j0, Y0.clone(), 0
    for it in range(iters):
        grad, _ = grad_and_obj(Y)
        Y = Y - eta.unsqueeze(-1) * grad
        Y[~live] = 0.0
        j = objective(Y)
        if j < jb:
            jb, Yb, bit = j, Y.clone(), it + 1
        del grad

    live_np = live.cpu().numpy(); li = np.nonzero(live_np)[0]
    _, loc = cKDTree(centers[li]).query(lab_pts, k=1, workers=-1)
    owner = torch.from_numpy(li[loc].astype(np.int64)).to(dev)
    pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)

    def score(Ym):
        pr = torch.zeros(P, dtype=torch.long, device=dev)
        pr[live] = (Ym[live] @ TT).argmax(1) + 1
        _, mi, _, _ = calculate_metrics(emit[live].cpu(), pr[live].cpu(), C + 1)
        pp = pr[owner]
        _, pmi, _, _ = calculate_metrics(pt_gt.cpu(), pp.cpu(), C + 1)
        # how many primitives still disagree with their own class
        cont = float((pr[live] != emit[live]).float().mean())
        return float(mi), float(pmi), cont

    mi0, pmi0, c0 = score(Y0)
    mi1, pmi1, c1 = score(Yb)
    return dict(scene=scene, recon=recon, P=int(P), live=int(live.sum()), R=R, nnz=nnz, C=C,
                obj_init=j0, obj_best=jb, obj_drop=float(1.0 - jb / max(j0, 1e-30)), best_iter=bit,
                miou_eq7=mi0, pt_miou_eq7=pmi0, cont_eq7=c0,
                miou_deconv=mi1, pt_miou_deconv=pmi1, cont_deconv=c1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00,scene0062_00,scene0200_00")
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=-1)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--label-mode", default="vertex")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="artifacts/scannet/oracle_deconv.json")
    a = ap.parse_args()
    rows = []
    print(f"{'scene':<14}{'arm':<11}{'nnz(M)':>8}{'obj drop':>10}{'best@':>7}|"
          f"{'ppEq6':>8}{'ppDeconv':>10}{'d':>7}|{'ptEq6':>8}{'ptDeconv':>10}{'d':>7}|"
          f"{'contam Eq6':>12}{'deconv':>9}")
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.label_mode, a.iters)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"{sc:<14}{rec:<11}{r['nnz']/1e6:>8.0f}{r['obj_drop']:>10.2%}{r['best_iter']:>7}|"
                  f"{r['miou_eq7']*100:>8.2f}{r['miou_deconv']*100:>10.2f}"
                  f"{(r['miou_deconv']-r['miou_eq7'])*100:>+7.2f}|"
                  f"{r['pt_miou_eq7']*100:>8.2f}{r['pt_miou_deconv']*100:>10.2f}"
                  f"{(r['pt_miou_deconv']-r['pt_miou_eq7'])*100:>+7.2f}|"
                  f"{r['cont_eq7']:>12.2%}{r['cont_deconv']:>9.2%}", flush=True)
            torch.cuda.empty_cache()
    if rows:
        f = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\nMEAN over {len(rows)}: perPrim {f('miou_eq7')*100:.2f} -> "
              f"{f('miou_deconv')*100:.2f} ({(f('miou_deconv')-f('miou_eq7'))*100:+.2f})  |  "
              f"perPoint {f('pt_miou_eq7')*100:.2f} -> {f('pt_miou_deconv')*100:.2f} "
              f"({(f('pt_miou_deconv')-f('pt_miou_eq7'))*100:+.2f})  |  "
              f"contaminated {f('cont_eq7'):.2%} -> {f('cont_deconv'):.2%}")


if __name__ == "__main__":
    main()
