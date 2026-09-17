"""Given a PERFECT upstream, does the bound predict segmentation? Verified exactly, not correlated.

A24 showed the bound does not predict mIoU across scenes. That is not a defect of the theory --
it bounds ||A X - B||^2, and on real data mIoU is dominated by upstream error the lift provably
does not cause (A20: 0.000% of rendered pixels lack support, region-level CLIP is 49.2%
accurate). The claim the theory actually makes is conditional: GIVEN that the per-view evidence
is correct, the lift's loss is controlled by the overlap mass.

That conditional is checkable exactly, by removing the upstream error instead of correlating
around it. Replace B with an ORACLE: for ray i, the text embedding of the TRUE class of the
surface that ray sees,

    b_i = t_{ c(front(i)) } ,   c = per-primitive ground-truth label

so the 2D model is, by construction, perfect. Then every label error in the lifted field is
attributable to the lift alone. Two predictions follow directly from the theorem:

  * where rays are disjoint, G = D, so X' = Xhat exactly and the oracle labels must be recovered
    essentially perfectly;
  * where rays overlap, each primitive receives a weighted blend of the class embeddings of the
    OTHER surfaces its rays saw first, and accuracy must fall with the overlap mass.

The mechanism is measured directly as well: `own_frac`, the share of a primitive's incoming ray
weight that comes from rays it is itself the front of. That is the per-primitive version of
"this ray is mine", and it is what o_i aggregates.

THE FRONT PRIMITIVE COMES FROM THE RASTERISER, NOT FROM THE OPERATOR'S ORDERING. A first version
took it to be the first nonzero of each operator row. That matches the rasteriser's own
`front_prim_idx` 86-90% of the time on the truefrozen arm but only **26-36%** on the nonfrozen
arm, where many primitives are hit per ray and traversal order is not depth order. Building the
oracle on it would have meant labelling two thirds of nonfrozen rays with the wrong surface --
and the arm comparison is the entire point of the experiment. `front_prim_idx` is exact and is
already produced by the same rasterisation, so it is used directly and the agreement with the
operator ordering is reported as a diagnostic only.

WHY THIS IS CHEAP. The oracle target is a CLASS one-hot, B = S T with S in {0,1}^(R x C), so
A^T B = (A^T S) T and the entire computation lives in C ~ 7-19 dimensions rather than the 512 of
a CLIP embedding -- a ~70x saving, exactly rather than approximately. Scoring reduces to
    argmax_c  <x_j, t_c>  =  argmax_c  [ (A^T S / D) (T T^T) ]_{jc}
and the row normalisation of x_j drops out of the argmax because it is a positive per-primitive
scalar -- the same per-primitive inertness that makes confidence weighting a no-op.
A first version of this script built B at full 512 dimensions and materialised A p at
(rays, channels): ~10 GB per matvec, several live at once inside CG. That is the exact failure
compute_beta.py warns about in its A^T A comment.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT

ARMS = {"truefrozen": "truefrozen", "nonfrozen": "nonfrozen"}


def cg(matvec, B, diag, iters=150, tol=1e-8):
    X = torch.zeros_like(B)
    R = B - matvec(X)
    Z = R / diag
    P = Z.clone()
    rz = (R * Z).sum()
    b0 = B.norm().clamp_min(1e-30)
    for _ in range(iters):
        AP = matvec(P)
        a = rz / (P * AP).sum().clamp_min(1e-30)
        X += a * P
        R -= a * AP
        Z = R / diag
        rz2 = (R * Z).sum()
        P = Z + (rz2 / rz.clamp_min(1e-30)) * P
        rz = rz2
        if R.norm() / b0 < tol:
            break
    return X, float(R.norm() / b0)


def one(scene, recon, n_views, class_set, cap, dev="cuda"):
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    wp.init()

    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ck}/model.pt")

    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    P = centers.shape[0]

    # ---- per-primitive ground truth ----
    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=None, k=64)
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = votes.argmax(1).astype(np.int64)
    prim_gt[votes.max(1) == 0] = 0
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)
    T = embed_class_names(kept, dev)                       # (C, F), unit norm

    # ---- operator A over a fixed, evenly spaced view subset ----
    model.update_vis_cache()
    c_ = model._vis_cache
    blank = torch.zeros(P, model.args.num_texel_sites, 3, device=dev)

    sel = np.linspace(0, len(dh.cameras) - 1, n_views).astype(int).tolist()
    rows, cols, vals, fronts, offs = [], [], [], [], 0
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        rows.append(op.row_indices.to(torch.int64) + offs)
        cols.append(op.col_indices.to(torch.int64))
        vals.append(op.values.float())
        del op
        # the visible surface of each pixel, straight from the rasteriser
        with torch.no_grad():
            o_ = model.rasterizer.visualize(cam, c_["points"], c_["radii"], c_["density"],
                                            c_["normals"], c_["texel_sites"], blank,
                                            c_["texel_height"], c_["adjacency"],
                                            c_["adjacency_offsets"])
        fp = o_[7].reshape(-1).long()
        al = o_[3].reshape(-1)
        fronts.append(torch.where((al > 0.05) & (fp >= 0), fp, torch.full_like(fp, -1)))
        offs += H * W
    row = torch.cat(rows).to(dev); col = torch.cat(cols).to(dev); val = torch.cat(vals).to(dev)
    del rows, cols, vals
    R = offs
    nnz = val.numel()
    order = torch.argsort(row, stable=True)
    row, col, val = row[order], col[order], val[order]
    starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
    rowsum = torch.zeros(R, device=dev).index_add_(0, row, val)
    livery = rowsum > 0

    # the visible surface per ray, exact
    front_r = torch.cat(fronts).to(dev)
    del fronts
    has_front = front_r >= 0
    front = front_r.clamp_min(0)

    # DIAGNOSTIC ONLY: how often the operator's first nonzero would have agreed
    first = starts[:-1].clamp(max=nnz - 1)
    guess = col[first]
    m = has_front & livery
    agree = float((guess[m] == front[m]).float().mean()) if bool(m.any()) else float("nan")

    # ---- ORACLE B: the true class of the surface each ray sees, as a class one-hot ----
    ray_cls = torch.where(has_front, prim_gt_t[front], torch.zeros_like(front))
    usable = livery & has_front & (ray_cls > 0)
    keep = usable[row]
    AtS = torch.zeros(P, C, device=dev)                     # A^T S, the C-dimensional target
    AtS.index_put_((col[keep], ray_cls[row[keep]] - 1), val[keep], accumulate=True)
    D = torch.zeros(P, device=dev).index_add_(0, col[keep], val[keep])
    TT = (T @ T.T)                                          # (C, C), turns class mass into scores

    # ---- own_frac: share of incoming weight from rays this primitive is the front of ----
    own = torch.zeros(P, device=dev)
    mine = keep & has_front[row] & (front[row] == col)
    own.index_add_(0, col[mine], val[mine])
    own_frac = own / D.clamp_min(1e-30)

    # ---- overlap mass on the SAME operator, split by whether it crosses a class boundary ----
    # o_i = (sum_j A_ij)^2 - sum_j A_ij^2 counts ALL shared ray weight. But blending two
    # primitives of the SAME class cannot move an argmax, so only the class-crossing part can
    # cost segmentation. With m_ic = sum_{j: c_j = c} A_ij,
    #     o_i^within = sum_c ( m_ic^2 - sum_{j: c_j=c} A_ij^2 ),   o_i^cross = s_i^2 - sum_c m_ic^2
    # and o_i = o_i^within + o_i^cross exactly. This is the refinement the segmentation question
    # forces: the operator-only o_i is class-blind, and class-blindness is why it fails to
    # predict mIoU (A24).
    rs = torch.zeros(R, device=dev).index_add_(0, row, val)
    rq = torch.zeros(R, device=dev).index_add_(0, row, val * val)
    o_ray = (rs * rs - rq).clamp_min(0)
    sum_o = float(o_ray.sum())
    mean_o = sum_o / float(rs.sum().clamp_min(1e-30))

    # The split must be taken on the SAME subset on both sides. Primitives with no GT label
    # carry ray weight but belong to no class, so if they are dropped from M while s_i still
    # sums over everything, their mass is silently charged to "cross" -- which produced shares
    # above 100%, an impossibility since cross is a PART of the total. Both terms are therefore
    # restricted to labelled primitives, and o_lab is reported beside o so the subset is visible.
    pc = prim_gt_t[col]                                   # class of each contributing primitive
    lab = pc > 0
    rs_lab = torch.zeros(R, device=dev).index_add_(0, row[lab], val[lab])
    rq_lab = torch.zeros(R, device=dev).index_add_(0, row[lab], val[lab] * val[lab])
    M = torch.zeros(R, C, device=dev)
    M.index_put_((row[lab], pc[lab] - 1), val[lab], accumulate=True)
    o_lab = (rs_lab * rs_lab - rq_lab).clamp_min(0)
    o_cross = (rs_lab * rs_lab - (M * M).sum(1)).clamp_min(0)
    denom = float(rs.sum().clamp_min(1e-30))
    sum_o_lab = float(o_lab.sum())
    sum_o_cross = float(o_cross.sum())
    assert sum_o_cross <= sum_o_lab * 1.001, (sum_o_cross, sum_o_lab)
    mean_o_lab = sum_o_lab / denom
    mean_o_cross = sum_o_cross / denom
    cross_share = sum_o_cross / max(sum_o_lab, 1e-30)

    # ---- X' and Xhat on the live subset ----
    live = D > 0
    idx = torch.nonzero(live, as_tuple=True)[0]
    remap = torch.full((P,), -1, dtype=torch.long, device=dev)
    remap[idx] = torch.arange(idx.numel(), device=dev)
    kk = keep & live[col]
    rr, cc, vv2 = row[kk], remap[col[kk]], val[kk]
    n = idx.numel()

    def Gmv(Y):
        # (rays, C) intermediate, harmless at C ~ 7-19 where it would be fatal at 512
        y = torch.zeros(R, Y.shape[1], device=dev)
        y.index_add_(0, rr, vv2.unsqueeze(-1) * Y[cc])
        z = torch.zeros(n, Y.shape[1], device=dev)
        z.index_add_(0, cc, vv2.unsqueeze(-1) * y[rr])
        return z

    diag = torch.zeros(n, device=dev).index_add_(0, cc, vv2 * vv2)
    Yp = AtS[idx] / D[idx].unsqueeze(-1)                    # closed form, in class space
    Yh, res = cg(Gmv, AtS[idx], diag.clamp_min(1e-30).unsqueeze(-1))

    def score(Y):
        pred = torch.zeros(P, dtype=torch.long, device=dev)
        pred[idx] = (Y @ TT).argmax(1) + 1
        m2 = (prim_gt_t > 0) & live
        acc = float((pred[m2] == prim_gt_t[m2]).float().mean())
        _, miou, _, macc = calculate_metrics(prim_gt_t[m2].cpu(), pred[m2].cpu(), C + 1)
        return acc, float(miou), float(macc)

    a_p, mi_p, ma_p = score(Yp)
    a_h, mi_h, ma_h = score(Yh)
    scored = int(((prim_gt_t > 0) & live).sum())
    return dict(scene=scene, recon=recon, views=n_views, P=int(P), rays=int(R), nnz=int(nnz),
                front_agree=agree, mean_o=mean_o, sum_o=sum_o,
                mean_o_cross=mean_o_cross, sum_o_cross=sum_o_cross, cross_share=cross_share,
                mean_o_lab=mean_o_lab, sum_o_lab=sum_o_lab,
                own_frac_mean=float(own_frac[live].mean()),
                own_frac_p50=float(own_frac[live].median()),
                acc_closed=a_p, miou_closed=mi_p, macc_closed=ma_p,
                acc_exact=a_h, miou_exact=mi_h, macc_exact=ma_h,
                cg_residual=res, scored=scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen,nonfrozen")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--out", default="artifacts/scannet/oracle_upstream.json")
    a = ap.parse_args()
    rows = []
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] mean_o {r['mean_o']:.4f}  o_lab {r['mean_o_lab']:.4f}  "
                  f"o_cross {r['mean_o_cross']:.4f} ({r['cross_share']:.1%} of labelled)  "
                  f"own_frac {r['own_frac_mean']:.3f}  "
                  f"|| ORACLE closed mIoU {r['miou_closed']*100:6.2f} acc {r['acc_closed']*100:6.2f}"
                  f"  exact mIoU {r['miou_exact']*100:6.2f}  (front-match {r['front_agree']:.3f}, "
                  f"cg {r['cg_residual']:.1e})", flush=True)
    if rows:
        print(f"\n=== oracle upstream, {a.views} views, mean over scenes ===")
        print(f"{'arm':<12}{'mean o':>9}{'o_cross':>10}{'cross%':>8}{'own_frac':>10}"
              f"{'closed mIoU':>13}{'closed acc':>12}{'exact mIoU':>12}{'n':>4}")
        for rec in a.recons.split(","):
            rs = [r for r in rows if r["recon"] == rec]
            if not rs:
                continue
            f = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"{rec:<12}{f('mean_o'):>9.4f}{f('mean_o_cross'):>10.4f}"
                  f"{f('cross_share'):>8.1%}{f('own_frac_mean'):>10.3f}"
                  f"{f('miou_closed')*100:>13.2f}{f('acc_closed')*100:>12.2f}"
                  f"{f('miou_exact')*100:>12.2f}{len(rs):>4}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
