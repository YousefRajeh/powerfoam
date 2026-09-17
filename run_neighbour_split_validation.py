"""Does the feature gradient point at the TRUE semantic boundary? Validated on facet neighbours.

WHY NOT GT POINTS INSIDE THE CELL. The obvious test -- project a cell's own ScanNet points along the
gradient and see whether it separates their classes -- is impossible on the frozen arm. There is one
primitive per GT vertex by construction, so a cell owns ~1.6 points and almost never contains two
labels; the filters left 2 scoreable cells out of 16,206 candidates (run_split_validation.py).

THE FIX. Validate against the cell's FACET NEIGHBOURS instead. Each cell has ~16 of them, each
carrying its own GT label, so there are enough samples at any primitive density. If the cell sits on
a semantic boundary, its neighbours carry >= 2 labels, and the question becomes:

    does the image-space feature gradient separate the differently-labelled neighbours?

Neighbour centres are projected into the same view and scored by
    s = (u - u_bar) cos(theta) + (v - v_bar) sin(theta)
with theta the principal direction of feature change. AUC of s against the two dominant neighbour
labels, folded to >= 0.5 because the gradient's sign is arbitrary.

WHAT IT DISCRIMINATES. The R^2 signal cannot currently tell genuine object straddling from a SAM
mask seam crossing the footprint -- both produce a clean feature-vs-position gradient. Only GT can.
    AUC ~ null  -> the gradient tracks something that is not the object boundary (mask edge,
                   shading, texture). Splitting a cell along it would be arbitrary.
    AUC >> null -> the gradient IS the boundary; a new site can be placed on it directly.

NULL. The identical AUC along a random image direction, same neighbours, same folding. Folding
inflates any direction above 0.5, so the null is the only meaningful reference point -- an absolute
AUC of 0.7 means nothing until compared against it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401

import configargparse
import numpy as np
import torch

OUT = "artifacts/scannet/neighbour_split"
R2_MIN = 0.5
MIN_NB = 6           # neighbours with a GT label
MIN_PER_CLASS = 2    # per each of the two dominant neighbour labels


def load_view_features(feat_dir, stem, H, W):
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[-1] if s.shape[0] <= 4 else s[..., -1]
    t = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(f)).float(), dim=-1)
    seg = torch.from_numpy(np.ascontiguousarray(s)).long()
    if seg.shape != (H, W):
        seg = torch.nn.functional.interpolate(
            seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), t


def auc(scores, pos):
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = torch.argsort(torch.argsort(scores)).float() + 1.0
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    ap.add_argument("--r2-min", type=float, default=R2_MIN,
                    help="0.0 accepts every cell so MAGNITUDE can be judged as the gate")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    from camera_bridge import K_from_ray_dirs
    from configs import Params, add_group
    from data_loader import DataHandler
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    from point_cloud_query import assign_points_to_power_cells
    from evaluate_point_cloud_miou import (embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels, OPENGAUSSIAN_CLASS_SETS,
                                           SCANNET20_CLASS_NAMES)

    cfg = f"output/scannet_{a.scene}_truefrozen/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
    stems = sorted(os.path.splitext(f)[0][:-2] for f in os.listdir(feat_dir) if f.endswith("_f.npy"))

    wp.init()
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"output/scannet_{a.scene}_truefrozen/model.pt")
    P = int(model.points.shape[0])
    centres = model.points.detach().float().to(dev)
    radii = model.get_radii().detach().float().to(dev).reshape(-1)

    names = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    T = torch.nn.functional.normalize(embed_class_names(names, dev).float(), dim=-1)
    D = T.shape[1]

    # ---- adjacency: the real shared-facet dual, same routine the renderer uses ----------------
    model.aabb_tree.update(model.points.detach(), model.get_radii().detach())
    adjacent, offsets = model.aabb_tree.build_cech_complex()
    adjacent = adjacent.long().to(dev)
    offsets = offsets.long().to(dev)
    deg = (offsets[1:] - offsets[:-1])
    print(f"  [adj] {adjacent.numel():,} edges, {float(deg.float().mean()):.2f} avg neighbours",
          flush=True)

    # ---- per-cell GT label -------------------------------------------------------------------
    for sub in ("train", "val", "test"):
        gt_dir = os.path.join(a.gt_root, sub, a.scene)
        if os.path.isdir(gt_dir):
            break
    gt_pts, gt_raw, _ = load_scannet_pointcept_gt(gt_dir)
    target_ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]
    gl_np = remap_gt_labels(gt_raw, target_ids) - 1
    owner_np = assign_points_to_power_cells(gt_pts.astype(np.float64),
                                            centres.detach().cpu().numpy(),
                                            radii.detach().cpu().numpy())
    keep = (owner_np >= 0) & (gl_np >= 0)
    own = torch.from_numpy(owner_np[keep]).long().to(dev)
    gl = torch.from_numpy(gl_np[keep]).long().to(dev)
    C_ = T.shape[0]
    votes = torch.zeros((P, C_), device=dev)
    votes.index_put_((own, gl), torch.ones(own.numel(), device=dev), accumulate=True)
    cell_gt = votes.argmax(1)
    has_gt = votes.sum(1) > 0
    print(f"  [gt] cells with a GT label {int(has_gt.sum()):,}/{P:,}", flush=True)

    aucs, nulls, oracles, mags, anis = [], [], [], [], []
    ex_grad, ex_const, r2s = [], [], []
    ANGLES = [i * 3.14159265 / 36 for i in range(36)]   # oracle sweep, 5-degree steps
    n_strong, n_scored = 0, 0
    rng = torch.Generator(device=dev).manual_seed(0)

    for vi in sel:
        cam = dh.cameras[vi]
        H, W_ = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                       max_intersections=4096)
        ri = op.row_indices.to(torch.int64)
        ci = op.col_indices.to(torch.int64)
        vv = op.values.float()
        del op
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W_)
        seg, tab = seg.to(dev), tab.to(dev)
        sid = seg[ri].clamp(0, tab.shape[0] - 1)
        py = (ri // W_).float()
        px = (ri % W_).float()
        Wc = torch.zeros(P, device=dev).index_add_(0, ci, vv)
        Wn = Wc.clamp_min(1e-8)
        mx = (torch.zeros(P, device=dev).index_add_(0, ci, vv * px)) / Wn
        my = (torch.zeros(P, device=dev).index_add_(0, ci, vv * py)) / Wn
        dx, dy = px - mx[ci], py - my[ci]
        Sxx = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dx)
        Syy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dy * dy)
        Sxy = torch.zeros(P, device=dev).index_add_(0, ci, vv * dx * dy)
        Fb = (torch.zeros((P, D), device=dev).index_add_(0, ci, vv[:, None] * tab[sid])) / Wn[:, None]
        Gx = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dx)[:, None] * tab[sid])
        Gy = torch.zeros((P, D), device=dev).index_add_(0, ci, (vv * dy)[:, None] * tab[sid])
        det = (Sxx * Syy - Sxy * Sxy)
        bx = (Syy[:, None] * Gx - Sxy[:, None] * Gy) / det.clamp_min(1e-12)[:, None]
        by = (Sxx[:, None] * Gy - Sxy[:, None] * Gx) / det.clamp_min(1e-12)[:, None]
        expl = (bx * Gx).sum(1) + (by * Gy).sum(1)
        total = Wc * (1.0 - (Fb * Fb).sum(1)).clamp_min(0)
        r2 = torch.where(total > 1e-9, (expl / total.clamp_min(1e-12)).clamp(0, 1),
                         torch.zeros_like(total))
        strong = (Wc > 1e-6) & (det > 1e-6) & (r2 > a.r2_min) & (deg >= MIN_NB)
        if not bool(strong.any()):
            del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by, Fb
            continue
        aa = (bx * bx).sum(1)
        bb2 = (bx * by).sum(1)
        cc = (by * by).sum(1)
        th = 0.5 * torch.atan2(2 * bb2, (aa - cc))
        gxv, gyv = torch.cos(th), torch.sin(th)
        # CONFIDENCE IN THE DIRECTION, from the same 2x2 Gram G = [[aa,bb],[bb,cc]] of [bx; by].
        #   MAGNITUDE  sqrt(lam1) * rms_radius = how much the feature changes ACROSS the footprint.
        #     R^2 alone cannot say this: a cell with almost no feature variance can explain a large
        #     FRACTION of it with position and still be pure noise.
        #   ANISOTROPY (lam1 - lam2) / (lam1 + lam2) = how ONE-DIMENSIONAL the variation is.
        #     A real boundary varies along a single direction, so lam1 >> lam2 and theta is well
        #     determined. Isotropic variation (lam1 ~ lam2) leaves theta arbitrary no matter how
        #     large R^2 or the magnitude are -- which is the failure mode a coin-flip result looks
        #     like from the outside.
        tr_ = aa + cc
        dt_ = (aa * cc - bb2 * bb2).clamp_min(0)
        disc = (tr_ * tr_ - 4 * dt_).clamp_min(0).sqrt()
        lam1 = 0.5 * (tr_ + disc)
        lam2 = 0.5 * (tr_ - disc)
        rms_r = ((Sxx + Syy) / Wn).clamp_min(1e-12).sqrt()
        gmag = lam1.clamp_min(0).sqrt() * rms_r
        aniso = ((lam1 - lam2) / (lam1 + lam2).clamp_min(1e-20))

        K, _ = K_from_ray_dirs(cam)
        K = K.to(dev).float()
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :4] = dh.c2ws[vi].double()
        w2c = torch.linalg.inv(c2w).float().to(dev)
        Xc = (w2c[:3, :3] @ centres.T + w2c[:3, 3:4]).T          # ALL cell centres, once
        z = Xc[:, 2]
        front = z > 1e-4
        uu = torch.where(front, K[0, 0] * Xc[:, 0] / z.clamp_min(1e-6) + K[0, 2],
                         torch.full_like(z, float("nan")))
        vv2 = torch.where(front, K[1, 1] * Xc[:, 1] / z.clamp_min(1e-6) + K[1, 2],
                          torch.full_like(z, float("nan")))
        ang = float(torch.rand(1, generator=rng, device=dev) * 3.14159265)

        cand = torch.nonzero(strong, as_tuple=True)[0]
        n_strong += int(cand.numel())
        for j in cand.tolist():
            s0, s1 = int(offsets[j]), int(offsets[j + 1])
            nb = adjacent[s0:s1]
            nb = nb[has_gt[nb] & front[nb]]
            if nb.numel() < MIN_NB:
                continue
            lj = cell_gt[nb]
            uq, cn = torch.unique(lj, return_counts=True)
            if uq.numel() < 2:
                continue
            top = torch.argsort(cn, descending=True)[:2]
            if int(cn[top[0]]) < MIN_PER_CLASS or int(cn[top[1]]) < MIN_PER_CLASS:
                continue
            k2 = (lj == uq[top[0]]) | (lj == uq[top[1]])
            nb2 = nb[k2]
            pos = lj[k2] == uq[top[0]]
            du = uu[nb2] - mx[j]
            dv = vv2[nb2] - my[j]
            A1 = auc(du * gxv[j] + dv * gyv[j], pos)
            A0 = auc(du * float(np.cos(ang)) + dv * float(np.sin(ang)), pos)
            if A1 is None or A0 is None:
                continue
            # ORACLE: the best AUC any direction achieves on these same neighbours. A single random
            # draw is a weak reference here -- neighbour labels are already spatially clustered, so
            # most directions separate them somewhat and the null sits near 0.84. What matters is
            # how close the gradient gets to the best available direction: gradient ~ oracle means
            # it is finding the boundary, not merely beating chance.
            best = 0.0
            for t_ in ANGLES:
                Ax = auc(du * float(np.cos(t_)) + dv * float(np.sin(t_)), pos)
                if Ax is not None:
                    best = max(best, max(Ax, 1 - Ax))
            aucs.append(max(A1, 1 - A1))
            nulls.append(max(A0, 1 - A0))
            oracles.append(best)
            mags.append(float(gmag[j]))
            anis.append(float(aniso[j]))
            r2s.append(float(r2[j]))
            n_scored += 1

            # ---- USE THE WHOLE GRADIENT, not just its direction --------------------------------
            # AUC is rank-based, so scaling the score by ||g|| cannot change it -- the magnitude is
            # invisible to the test above. The gradient's magnitude only means something inside the
            # model it came from: the regression predicts the FEATURE at any position,
            #     f_hat(u,v) = f_bar + bx (u - u_bar) + by (v - v_bar),
            # so extrapolate to each neighbour's pixel and classify the extrapolated feature. That
            # uses direction, magnitude and the full 512-d structure at once, and it is exactly the
            # quantity a split would need: the feature to hand to each side of the new boundary.
            # BASELINE is the same prediction with the gradient switched off (the constant model
            # f_hat = f_bar), i.e. what the cell already predicts today. Beating it is the claim.
            fh = (Fb[j][None, :]
                  + bx[j][None, :] * du[:, None]
                  + by[j][None, :] * dv[:, None])
            pred_g = (torch.nn.functional.normalize(fh, dim=1) @ T.T).argmax(1)
            pred_c = int((torch.nn.functional.normalize(Fb[j][None, :], dim=1) @ T.T).argmax(1))
            truth = cell_gt[nb2]
            ex_grad.append(float((pred_g == truth).float().mean()))
            ex_const.append(float((truth == pred_c).float().mean()))
        del ri, ci, vv, seg, tab, sid, Gx, Gy, bx, by, Fb

    A = torch.tensor(aucs)
    N = torch.tensor(nulls)
    O = torch.tensor(oracles)
    Mg = torch.tensor(mags)
    An = torch.tensor(anis)
    EG = torch.tensor(ex_grad)
    EC = torch.tensor(ex_const)
    R2v = torch.tensor(r2s)

    def buckets(key, name):
        """AUC quality as a function of a confidence signal -- the point of the whole exercise: if
        reliability rises with the signal, it is a usable gate for placing sites automatically."""
        if key.numel() < 40:
            return []
        qs = torch.quantile(key.double(), torch.tensor([.25, .5, .75], dtype=torch.float64)).float()
        out = []
        edges = [(-1e30, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e30)]
        for lo, hi in edges:
            m = (key > lo) & (key <= hi)
            if int(m.sum()) < 5:
                continue
            out.append({"signal": name, "lo": float(lo) if lo > -1e29 else None,
                        "n": int(m.sum()), "auc": float(A[m].mean()),
                        "null": float(N[m].mean()), "oracle": float(O[m].mean()),
                        "beats_random": float((A[m] > N[m]).float().mean()),
                        "ties_oracle": float((A[m] >= O[m] - 1e-6).float().mean())})
        return out
    def gate_compare(frac):
        """Which signal picks the better candidates at a FIXED budget? Take the top `frac` of
        (cell,view) pairs by magnitude and by R^2 separately and compare the quality of what each
        selects. This is the question directly: given that we can only act on some cells, which
        ones should we even look at?"""
        k = max(5, int(frac * A.numel()))
        out = {}
        for nm, key in (("magnitude", Mg), ("r2", R2v)):
            idx = torch.argsort(key, descending=True)[:k]
            out[nm] = {"n": k, "auc": float(A[idx].mean()),
                       "ties_oracle": float((A[idx] >= O[idx] - 1e-6).float().mean()),
                       "beats_random": float((A[idx] > N[idx]).float().mean()),
                       "extrap_delta": float((EG[idx] - EC[idx]).mean())}
        return out
    gates = {f"top{int(f*100)}pct": gate_compare(f) for f in (0.05, 0.10, 0.25, 0.50)}
    bmag = buckets(Mg, "gradient magnitude")
    bani = buckets(An, "anisotropy")
    res = {"scene": a.scene, "views": a.views, "r2_min": R2_MIN,
           "avg_degree": float(deg.float().mean()),
           "n_strong_cellviews": n_strong, "n_scored": n_scored,
           "auc_mean": float(A.mean()) if A.numel() else None,
           "auc_median": float(A.median()) if A.numel() else None,
           "null_mean": float(N.mean()) if N.numel() else None,
           "null_median": float(N.median()) if N.numel() else None,
           "frac_auc_gt_0p8": float((A > 0.8).float().mean()) if A.numel() else None,
           "frac_null_gt_0p8": float((N > 0.8).float().mean()) if N.numel() else None,
           "delta_mean": float(A.mean() - N.mean()) if A.numel() else None,
           "oracle_mean": float(O.mean()) if O.numel() else None,
           "grad_over_oracle": float((A / O.clamp_min(1e-9)).mean()) if O.numel() else None,
           "frac_grad_beats_random": float((A > N).float().mean()) if A.numel() else None,
           "frac_grad_ties_oracle": float((A >= O - 1e-6).float().mean()) if O.numel() else None,
           "buckets_magnitude": bmag, "buckets_anisotropy": bani, "gate_compare": gates,
           "extrap_acc_gradient": float(EG.mean()) if EG.numel() else None,
           "extrap_acc_constant": float(EC.mean()) if EC.numel() else None,
           "extrap_delta": float((EG - EC).mean()) if EG.numel() else None,
           "extrap_frac_gradient_better": float((EG > EC).float().mean()) if EG.numel() else None,
           "extrap_frac_gradient_worse": float((EG < EC).float().mean()) if EG.numel() else None}
    json.dump(res, open(f"{OUT}/{a.scene}.json", "w"), indent=1)
    print(f"[{a.scene}] strong (cell,view) {n_strong:,}   scored {n_scored:,}", flush=True)
    if A.numel():
        print(f"  AUC gradient  mean {res['auc_mean']:.4f}  median {res['auc_median']:.4f}  "
              f">0.8 {res['frac_auc_gt_0p8']*100:.1f}%", flush=True)
        print(f"  AUC RANDOM    mean {res['null_mean']:.4f}  median {res['null_median']:.4f}  "
              f">0.8 {res['frac_null_gt_0p8']*100:.1f}%", flush=True)
        print(f"  AUC ORACLE    mean {res['oracle_mean']:.4f}   (best of 36 directions)", flush=True)
        print(f"  DELTA grad-random {res['delta_mean']:+.4f}   grad/oracle "
              f"{res['grad_over_oracle']:.4f}", flush=True)
        print(f"  paired: gradient beats random on {res['frac_grad_beats_random']*100:.1f}%   "
              f"ties the oracle on {res['frac_grad_ties_oracle']*100:.1f}%", flush=True)
        print(f"  EXTRAPOLATION (gradient used in full, predicting neighbour labels):", flush=True)
        print(f"     with gradient {res['extrap_acc_gradient']*100:.2f}%   constant model "
              f"{res['extrap_acc_constant']*100:.2f}%   delta "
              f"{res['extrap_delta']*100:+.2f} pp", flush=True)
        print(f"     gradient better on {res['extrap_frac_gradient_better']*100:.1f}% of cells, "
              f"worse on {res['extrap_frac_gradient_worse']*100:.1f}%", flush=True)
        print("  GATE COMPARISON -- top-k by each signal, same candidates pool:", flush=True)
        for nm, g in gates.items():
            m_, r_ = g["magnitude"], g["r2"]
            print(f"    {nm:9s} n={m_['n']:4d} | MAG auc {m_['auc']:.4f} ties {m_['ties_oracle']*100:5.1f}%"
                  f" extrap {m_['extrap_delta']*100:+5.2f}pp | R2 auc {r_['auc']:.4f} "
                  f"ties {r_['ties_oracle']*100:5.1f}% extrap {r_['extrap_delta']*100:+5.2f}pp",
                  flush=True)
        for tag, bs in (("MAG", bmag), ("ANISO", bani)):
            for i, b in enumerate(bs):
                print(f"    [{tag} Q{i+1}] n={b['n']:4d}  auc {b['auc']:.4f}  null {b['null']:.4f}"
                      f"  beats {b['beats_random']*100:5.1f}%  ties-oracle "
                      f"{b['ties_oracle']*100:5.1f}%", flush=True)
    else:
        print("  nothing scoreable", flush=True)


if __name__ == "__main__":
    main()
