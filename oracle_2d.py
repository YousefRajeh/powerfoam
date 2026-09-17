"""2D scoring of the oracle: render the lifted labels back to the image plane, on HELD-OUT views.

WHY (user's proposal, and it fixes a real confound). Scoring in 3D assigns each GT point to a
primitive and compares labels there, which

  * needs an assignment rule (nearest live centre / power cell / Mahalanobis) and inherits its
    ambiguity -- the 0.81% OWNERSHIP bucket of A36 is entirely that,
  * and PUNISHES DEPTH ERROR. `nonfrozen`'s rendered surface sits 17 cm from the GT mesh, so its
    3D score mixes lifting error with geometry error and is not a fair read of its lift.

Rendering the prediction back to the image plane removes both. A depth error ALONG the ray does not
change which pixel shows which object -- only lateral displacement does -- so 2D scoring is
insensitive to the 17 cm offset while still penalising a floater that actually occludes something.
No assignment rule appears anywhere. And it is the metric the 2D open-vocabulary segmentation
literature uses, so the numbers are comparable to it.

THE CIRCULARITY TRAP, AND THE SPLIT THAT AVOIDS IT. Rendering predictions into the SAME views that
supplied the evidence measures reconstruction fit -- exactly the quantity `||A X - B||^2` that the
deconvolution minimises. Under that scoring a residual-minimising solver would win by construction
and the comparison would be worthless. So the views are SPLIT: the lift accumulates on train views
only, and the 2D metric is computed on test views the solve has never seen.

    train views  ->  A_tr, B_tr  ->  X = D^-1 A_tr^T B_tr        (the lift, unchanged)
    test views   ->  A_te        ->  per-pixel argmax of (A_te X) vs that view's GT label image

The rendered class map is the compositing of the per-primitive class distributions -- exactly how
the renderer would composite colour -- so nothing about the forward model is special-cased.
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


def one(scene, recon, class_set, cap, label_mode, stride, kappa_q, dev="cuda"):
    from scipy.spatial import cKDTree
    from camera_bridge import K_from_ray_dirs
    import open3d as o3d
    import warp as wp
    wp.init()
    is_gs = recon.startswith("gs_")
    cfg = "nonfrozen" if is_gs else recon
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{cfg}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    if is_gs:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{recon}/{scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gsc = torch.exp(sp["scales"].to(dev))
        gop = torch.sigmoid(sp["opacities"].to(dev).reshape(-1) * 1.2)
        gcol = torch.zeros((gm.shape[0], 1), device=dev)
        P = gm.shape[0]; model = None; adj = aoff = None
    else:
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
        P = model.points.shape[0]
        adj = model.adjacency.detach().long().to(dev)
        aoff = model.adjacency_offsets.detach().long().to(dev)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    T = embed_class_names(kept, dev); TT = T @ T.T

    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, scene, "points3d.ply"))
    tri = np.asarray(mesh.triangles); vert_cls = gt_lab.astype(np.int64)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cache_dir = os.path.join("artifacts", "scannet", scene, "gtlabels")
    gt_tree_all = cKDTree(gt_pts); gt_cls_all = gt_lab.astype(np.int64)

    ncam = len(dh.cameras)
    test = list(range(0, ncam, stride))                 # held out from the lift
    train = [v for v in range(ncam) if v not in set(test)]

    def operator(vi):
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        if is_gs:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gsc, gop, gcol, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=1e-3)
        else:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        return (ri.to(torch.int64).to(dev), ci.to(torch.int64).to(dev), vv.float().to(dev), H, W)

    def labels(vi, H, W):
        cam = dh.cameras[vi]
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        return mesh_label_image(scene, vi, cam, c2w.float(), H, W, rc, tri, vert_cls,
                                dev, cache_dir, label_mode, gt_tree_all, gt_cls_all)

    # ---- lift on TRAIN views only ----
    AtS = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
    for vi in train:
        r_, c_, v_, H, W = operator(vi)
        cl = labels(vi, H, W)
        keep = cl[r_] > 0
        r_, c_, v_ = r_[keep], c_[keep], v_[keep]
        if r_.numel():
            AtS.index_put_((c_, cl[r_] - 1), v_, accumulate=True)
            D.index_add_(0, c_, v_)
        del r_, c_, v_, cl, keep
        torch.cuda.empty_cache()
    live = D > 0
    X = torch.zeros(P, C, device=dev); X[live] = AtS[live] / D[live].unsqueeze(-1)

    # optional: the A37 shrinkage readout, foam only (needs the facet graph)
    Xs = None
    if kappa_q > 0 and adj is not None:
        src = torch.repeat_interleave(torch.arange(P, device=dev), aoff.diff())
        nS = torch.zeros(P, C, device=dev).index_add_(0, src, AtS[adj])
        nD = torch.zeros(P, device=dev).index_add_(0, src, D[adj])
        m = torch.zeros(P, C, device=dev); ok = nD > 0
        m[ok] = nS[ok] / nD[ok].unsqueeze(-1)
        lone = live & ~ok
        if bool(lone.any()):
            m[lone] = AtS[lone] / D[lone].clamp_min(1e-30).unsqueeze(-1)
        kap = float(torch.quantile(D[live], kappa_q))
        Xs = torch.zeros(P, C, device=dev)
        Xs[live] = (AtS[live] + kap * m[live]) / (D[live] + kap).unsqueeze(-1)

    # ---- render the predictions into HELD-OUT views and score in 2D ----
    def render_and_score(Xm):
        gts, prs = [], []
        for vi in test:
            r_, c_, v_, H, W = operator(vi)
            cl = labels(vi, H, W)
            acc = torch.zeros(H * W, C, device=dev)
            acc.index_add_(0, r_, v_.unsqueeze(-1) * Xm[c_])       # composite the class field
            sim = acc @ TT
            pr = torch.where(acc.sum(1) > 0, sim.argmax(1) + 1,
                             torch.zeros_like(cl))                  # 0 where nothing was rendered
            m_ = cl > 0                                             # score only labelled GT pixels
            gts.append(cl[m_].cpu()); prs.append(pr[m_].cpu())
            del r_, c_, v_, cl, acc, sim, pr
            torch.cuda.empty_cache()
        g = torch.cat(gts); q = torch.cat(prs)
        _, mi, acc_, _ = calculate_metrics(g, q, C + 1)
        cover = float((q > 0).float().mean())
        return float(mi), float(acc_), cover, int(g.numel())

    mi, ac, cov, npx = render_and_score(X)
    out = dict(scene=scene, recon=recon, P=int(P), live=int(live.sum()), C=C,
               n_train=len(train), n_test=len(test), px=npx,
               miou2d=mi, acc2d=ac, rendered_cover=cov)
    if Xs is not None:
        mi2, ac2, cov2, _ = render_and_score(Xs)
        out.update(miou2d_shrink=mi2, acc2d_shrink=ac2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--label-mode", default="vertex")
    ap.add_argument("--stride", type=int, default=5, help="every Nth view is HELD OUT")
    ap.add_argument("--kappa-q", type=float, default=0.25, help="0 disables the shrinkage arm")
    ap.add_argument("--out", default="artifacts/scannet/oracle_2d.json")
    a = ap.parse_args()
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
                continue
            try:
                r = one(sc, rec, a.class_set, a.cap, a.label_mode, a.stride, a.kappa_q)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            extra = (f"  shrink {r['miou2d_shrink']*100:6.2f}" if "miou2d_shrink" in r else "")
            print(f"[{rec}/{sc}] train {r['n_train']:>3} test {r['n_test']:>3}  "
                  f"2D mIoU {r['miou2d']*100:6.2f}  acc {r['acc2d']*100:6.2f}  "
                  f"rendered {r['rendered_cover']:5.1%}{extra}", flush=True)
            torch.cuda.empty_cache()
    if rows:
        print("")
        print(f"{'arm':<12}{'n':>4}{'2D mIoU':>10}{'2D acc':>9}{'rendered':>10}{'+shrink':>10}")
        for rec in a.recons.split(","):
            s = [r for r in rows if r["recon"] == rec]
            if not s:
                continue
            f = lambda k: float(np.mean([r[k] for r in s if k in r])) if any(k in r for r in s) else float("nan")
            print(f"{rec:<12}{len(s):>4}{f('miou2d')*100:>10.2f}{f('acc2d')*100:>9.2f}"
                  f"{f('rendered_cover'):>10.1%}{f('miou2d_shrink')*100:>10.2f}")


if __name__ == "__main__":
    main()
