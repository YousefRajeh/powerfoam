"""Two targeted fixes for A36's mechanism, tested against Eq. 6 on the projected oracle.

A36 measured WHY the joint least-squares solve loses: `||A Y - S||^2` weights each primitive by its
VISIBILITY, an occluded primitive contributes almost nothing to the residual, and so it can be moved
a long way -- including to a wrong argmax -- for a fractionally better fit elsewhere. But the
contaminated primitives ARE the low-visibility ones, so the objective spends its error budget in
inverse proportion to where the errors are.

Two fixes follow, attacking different halves of that sentence.

FIX 1 -- SIMPLEX-PROJECTED JOINT SOLVE (the solver half).
  Note first that the OBVIOUS reading of "weight primitives by their share of errors" is a no-op:
  rescaling the normal equations row-wise, `D^-1 G Y = D^-1 A^T S`, is Jacobi preconditioning. It
  changes the iteration, not the fixed point, so the optimum -- and the failure -- are unchanged.
  What is actually missing is a CONSTRAINT. `Y_j` is a class distribution: the truth is one-hot and
  Eq. 6's solution is a weighted average of one-hots, so both lie in the probability simplex. The
  unconstrained least-squares solution leaves it, and leaving it is precisely the "wild values"
  mode. So iterate projected gradient with a Euclidean projection onto the simplex after each step.
  This cannot produce a value that is not a distribution, which bounds how far an under-determined
  primitive can drift.

FIX 2 -- ABSTAINING READOUT (the readout half).
  A primitive with near-zero `D_j` has committed to an argmax over almost no evidence. Rather than
  trusting it, a GT point defers to the nearest primitive whose `D_j` exceeds a quantile threshold.
  Abstention is only sensible if the deferred-to neighbour is better, so the threshold is swept and
  tau=0 (defer to nobody) is reported as the baseline -- if no tau beats it, the idea is dead.

Both are measured against the SAME Eq. 6 baseline computed in this script, so no cross-run
comparison is involved.
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

TAUS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]


def project_simplex(Y):
    """Euclidean projection of each row onto the probability simplex (Duchi et al. 2008)."""
    C = Y.shape[1]
    u, _ = torch.sort(Y, dim=1, descending=True)
    css = u.cumsum(1) - 1.0
    ind = torch.arange(1, C + 1, device=Y.device, dtype=Y.dtype).unsqueeze(0)
    cond = u - css / ind > 0
    rho = cond.float().cumsum(1).argmax(1)                       # last True index
    theta = css.gather(1, rho.unsqueeze(1)) / (rho + 1).unsqueeze(1).to(Y.dtype)
    return (Y - theta).clamp_min(0.0)


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
    centers = model.points.detach().cpu().numpy(); P = centers.shape[0]

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

    # PER-VIEW CHUNKS. Concatenating into one array transiently DOUBLES ~9 GB on the large scenes,
    # then a global argsort and an (R+1) int64 `bounds` array add ~2 GB more -- that is what OOMed
    # scene0140/scene0000. Rays are already contiguous WITHIN a view, so keeping the per-view slices
    # removes the concatenation, the sort and the bounds entirely. Indices are int32 (rays < 2^31,
    # P << 2^31) and cast to int64 only inside the block that uses them.
    chunks = []      # (row_local int32, col int32, val f32, kcl int32) per view
    R = 0
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        cls_img = mesh_label_image(scene, vi, cam, c2w.float(), H, W, rc, tri, vert_cls,
                                   dev, cache_dir, label_mode, gt_tree_all, gt_cls_all)
        if not bool((cls_img > 0).any()):
            del cls_img; continue
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
        uniq, inv = torch.unique(r_, return_inverse=True)
        chunks.append((inv.to(torch.int32), c_.to(torch.int32), v_,
                       (cls_img[uniq] - 1).to(torch.int32)))
        R += int(uniq.numel())
        del cls_img, r_, c_, v_, uniq, inv, keep
        torch.cuda.empty_cache()

    AtS = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
    Gs = torch.zeros(P, device=dev)
    for rl, cl, vl, kl in chunks:
        rl_, cl_ = rl.long(), cl.long()
        AtS.index_put_((cl_, kl.long()[rl_]), vl, accumulate=True)
        D.index_add_(0, cl_, vl)
        rs = torch.zeros(int(rl_.max()) + 1, device=dev).index_add_(0, rl_, vl)
        Gs.index_add_(0, cl_, vl * rs[rl_])
        del rl_, cl_, rs
    live = D > 0
    Y0 = torch.zeros(P, C, device=dev); Y0[live] = AtS[live] / D[live].unsqueeze(-1)
    eta = torch.zeros(P, device=dev); eta[live] = 1.0 / Gs[live].clamp_min(1e-20)

    def grad_and_obj(Y):
        g = torch.zeros(P, C, device=dev); obj = 0.0
        for rl, cl, vl, kl in chunks:
            rl_, cl_ = rl.long(), cl.long()
            n = int(rl_.max()) + 1
            ay = torch.zeros(n, C, device=dev)
            ay.index_add_(0, rl_, vl.unsqueeze(-1) * Y[cl_])
            ay[torch.arange(n, device=dev), kl.long()] -= 1.0
            obj += float(ay.pow(2).sum())
            g.index_add_(0, cl_, vl.unsqueeze(-1) * ay[rl_])
            del ay, rl_, cl_
        return g, obj

    def solve(project):
        Y = Y0.clone()
        jb, Yb, bit = grad_and_obj(Y0)[1], Y0.clone(), 0
        for it in range(iters):
            g, _ = grad_and_obj(Y)
            Y = Y - eta.unsqueeze(-1) * g
            if project:
                Y[live] = project_simplex(Y[live])
            Y[~live] = 0.0
            j = grad_and_obj(Y)[1]
            if j < jb:
                jb, Yb, bit = j, Y.clone(), it + 1
            del g
        return Yb, bit

    Yd, bit_d = solve(project=False)     # unconstrained joint solve (the A36 arm)
    Yb, bit = solve(project=True)        # FIX 1: simplex-projected

    live_np = live.cpu().numpy(); li = np.nonzero(live_np)[0]
    pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)
    Dl = D[live]

    def preds(Ym):
        pr = torch.zeros(P, dtype=torch.long, device=dev)
        pr[live] = (Ym[live] @ TT).argmax(1) + 1
        return pr

    def pt_miou(pr, idx_pool):
        """idx_pool: global primitive ids a GT point may be assigned to."""
        _, loc = cKDTree(centers[idx_pool]).query(lab_pts, k=1, workers=-1)
        pp = pr[torch.from_numpy(idx_pool[loc].astype(np.int64)).to(dev)]
        _, mi, _, _ = calculate_metrics(pt_gt.cpu(), pp.cpu(), C + 1)
        return float(mi)

    fields = {"eq7": preds(Y0), "deconv": preds(Yd), "simplex": preds(Yb)}
    out = dict(scene=scene, recon=recon, P=int(P), live=int(live.sum()), R=R, C=C,
               best_iter=bit, best_iter_deconv=bit_d)
    # every field x every abstention threshold -> separate AND in conjunction
    for tau in TAUS:
        thr = torch.quantile(Dl, tau) if tau > 0 else torch.tensor(-1.0, device=dev)
        keepm = (D > thr) & live
        pool = np.nonzero(keepm.cpu().numpy())[0]
        out[f"kept_{tau:g}"] = float(keepm.sum()) / max(float(live.sum()), 1.0)
        for nm, pr in fields.items():
            out[f"pt_{nm}_tau{tau:g}"] = pt_miou(pr, pool) if pool.size else float("nan")
    for nm in fields:
        out[f"pt_{nm}"] = out[f"pt_{nm}_tau0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00,scene0062_00,scene0200_00")
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=-1)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--label-mode", default="vertex")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="artifacts/scannet/oracle_fixes.json")
    a = ap.parse_args()
    rows = []
    FIELDS = ("eq7", "deconv", "simplex")
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.label_mode, a.iters)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] " + "  ".join(
                f"{nm}={r[f'pt_{nm}']*100:.2f}" for nm in FIELDS)
                + f"   best tau: " + max(
                    ((t, max(r[f'pt_{nm}_tau{t:g}'] for nm in FIELDS)) for t in TAUS),
                    key=lambda z: z[1])[0].__repr__(), flush=True)
            torch.cuda.empty_cache()
    if not rows:
        return
    f = lambda k: float(np.mean([r[k] for r in rows]))
    base = f("pt_eq7")
    print("")
    print(f"=== {len(rows)} scenes, per-point mIoU (Eq6 baseline {base*100:.2f}) ===")
    print(f"{'tau':>6}{'kept':>8}" + "".join(f"{nm:>12}" for nm in FIELDS))
    for t in TAUS:
        line = f"{t:>6g}{f(f'kept_{t:g}'):>8.1%}"
        for nm in FIELDS:
            v = f(f"pt_{nm}_tau{t:g}")
            line += f"{v*100:>8.2f}{(v-base)*100:>+5.2f}"
        print(line)
    print("")
    print("  columns: eq7 = closed form | deconv = unconstrained joint solve"
          " | simplex = joint solve projected onto the probability simplex")
    print("  rows:    tau = fraction of least-observed primitives a GT point may NOT be assigned to")


if __name__ == "__main__":
    main()
