"""Where do the ~11 points of per-point oracle shortfall actually go?

The projected oracle gives truefrozen ~88.6 per-point mIoU with a PERFECT upstream. This attributes
the missing mass to disjoint causes, per GT point, so the paper can say which of them is the
bottleneck instead of asserting "lifting error".

For each labelled GT point, with `j` = its owning primitive (nearest LIVE centre):

  DEAD        j never received evidence -> no prediction at all.
  OWNERSHIP   j is live, but j's OWN class (nearest-GT-point label, i.e. what j should be) already
              differs from this point's class. The point is owned by a primitive that legitimately
              belongs to something else; no solver could fix this, it is an assignment limit.
  EVIDENCE    j's own class is right, but the solved class distribution argmax (in CLASS space, no
              text embeddings) is wrong -> genuine contamination: rays deposited another class's
              mass on j.
  READOUT     the class-space argmax is RIGHT but argmax(W @ T T^T) flips it. CLIP text embeddings
              are mutually similar (<t_a,t_b> is far from 0), so a correct-but-mixed distribution
              can still decode to the wrong class. This is a property of the text head, NOT of the
              lift, and it would persist with a perfect solver.
  CORRECT     everything agreed.

The split matters because the four have different fixes: DEAD -> more views/coverage, OWNERSHIP ->
finer primitives, EVIDENCE -> a better solver or less overlap, READOUT -> a better text head.
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


def one(scene, recon, n_views, class_set, cap, label_mode, dev="cuda"):
    from camera_bridge import K_from_ray_dirs
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

    AtS = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
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
            continue
        AtS.index_put_((c_, cls_img[r_] - 1), v_, accumulate=True)
        D.index_add_(0, c_, v_)
        del cls_img, r_, c_, v_
        torch.cuda.empty_cache()

    live = D > 0
    Wp = torch.zeros(P, C, device=dev)
    Wp[live] = AtS[live] / D[live].unsqueeze(-1)

    # every labelled GT point -> its nearest LIVE primitive (the per-point metric's rule)
    live_np = live.cpu().numpy(); li = np.nonzero(live_np)[0]
    _, loc = cKDTree(centers[li]).query(lab_pts, k=1, workers=-1)
    owner = torch.from_numpy(li[loc].astype(np.int64)).to(dev)
    pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)

    pred_tt = torch.zeros(P, dtype=torch.long, device=dev)
    pred_raw = torch.zeros(P, dtype=torch.long, device=dev)
    pred_tt[live] = (Wp[live] @ TT).argmax(1) + 1
    pred_raw[live] = Wp[live].argmax(1) + 1              # class space, NO text head

    o_pred_tt = pred_tt[owner]; o_pred_raw = pred_raw[owner]; o_emit = emit[owner]
    correct = o_pred_tt == pt_gt
    dead = ~torch.from_numpy(live_np).to(dev)[owner]      # owner not live (can't happen: NN over live)
    ownership = (~correct) & (o_emit != pt_gt)
    evidence = (~correct) & (o_emit == pt_gt) & (o_pred_raw != pt_gt)
    readout = (~correct) & (o_emit == pt_gt) & (o_pred_raw == pt_gt)
    n = pt_gt.numel()

    # IS THE CONTAMINATION RECOVERABLE? For a primitive whose plurality is wrong, look at how much
    # of its own evidence mass belongs to its TRUE class:
    #   W_j[emit_j] == 0  -> the class never appeared in j's rays at all. No solver can invent it;
    #                        this is a VISIBILITY limit (j is occluded wherever it is seen).
    #   W_j[emit_j]  > 0  -> the signal is present but out-voted. A JOINT solve (G^-1 A^T B), which
    #                        subtracts what co-visible neighbours already explain, can in principle
    #                        recover it. This is the part a better solver could win.
    own_mass = torch.zeros(P, device=dev)
    own_mass[live] = Wp[live].gather(1, (emit[live] - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    contaminated = live & (pred_raw != emit) & (pred_raw > 0)
    cm = own_mass[contaminated]
    n_cont = int(contaminated.sum())
    frac_zero = float((cm <= 1e-12).float().mean()) if n_cont else float("nan")
    frac_runner = float((cm > 1e-12).float().mean()) if n_cont else float("nan")
    med_mass = float(cm.median()) if n_cont else float("nan")

    _, miou_tt, _, _ = calculate_metrics(pt_gt.cpu(), o_pred_tt.cpu(), C + 1)
    _, miou_raw, _, _ = calculate_metrics(pt_gt.cpu(), o_pred_raw.cpu(), C + 1)
    # ceiling: every primitive labelled by its OWN nearest-GT class (a perfect solver)
    _, miou_ceil, _, _ = calculate_metrics(pt_gt.cpu(), o_emit.cpu(), C + 1)

    return dict(scene=scene, recon=recon, P=int(P), live=int(live.sum()), n_pts=int(n), C=C,
                acc=float(correct.float().mean()),
                frac_ownership=float(ownership.float().mean()),
                frac_evidence=float(evidence.float().mean()),
                frac_readout=float(readout.float().mean()),
                miou_tt=float(miou_tt), miou_raw=float(miou_raw), miou_ceiling=float(miou_ceil),
                n_contaminated=n_cont, cont_frac_of_live=n_cont / max(int(live.sum()), 1),
                cont_zero_mass=frac_zero, cont_has_mass=frac_runner, cont_med_own_mass=med_mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00")
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=-1)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--label-mode", default="vertex")
    ap.add_argument("--out", default="artifacts/scannet/oracle_gap.json")
    a = ap.parse_args()
    rows = []
    print(f"{'scene':<14}{'arm':<11}{'acc':>7}{'ownership':>11}{'evidence':>10}{'readout':>9}|"
          f"{'mIoU':>7}{'ceiling':>9}|{'contam prims':>13}{'no signal':>11}{'has signal':>12}")
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.label_mode)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"{sc:<14}{rec:<11}{r['acc']:>7.2%}{r['frac_ownership']:>11.2%}"
                  f"{r['frac_evidence']:>10.2%}{r['frac_readout']:>9.2%}|"
                  f"{r['miou_tt']*100:>7.2f}{r['miou_ceiling']*100:>9.2f}|"
                  f"{r['cont_frac_of_live']:>13.2%}{r['cont_zero_mass']:>11.1%}{r['cont_has_mass']:>12.1%}",
                  flush=True)
            torch.cuda.empty_cache()
    if rows:
        f = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\nMEAN over {len(rows)}: acc {f('acc'):.2%} | ownership {f('frac_ownership'):.2%} "
              f"evidence {f('frac_evidence'):.2%} readout {f('frac_readout'):.2%} | "
              f"mIoU {f('miou_tt')*100:.2f} no-text {f('miou_raw')*100:.2f} "
              f"ceiling {f('miou_ceiling')*100:.2f}")


if __name__ == "__main__":
    main()
