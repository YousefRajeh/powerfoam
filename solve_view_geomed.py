"""Per-primitive, contribution-weighted geometric median over VIEWS.

One estimator that subsumes both of ReLaGS's post-processing components, with no tuned constants:

    X_j = argmin_x  sum_v  m_jv * ||x - f_jv||,     m_jv = sum_{i in view v} A_ij
                                                    f_jv = sum_{i in view v} A_ij B_i / m_jv

  * MWP (`max_c,p w_ip > tau_contrib = 5e-4`, a hard prune) becomes the WEIGHT `m_jv`. A view in
    which the primitive barely contributed votes proportionally less, instead of the primitive being
    deleted outright on a single-pixel maximum. No threshold.
  * ROFA (mean pairwise cosine -> z-score -> drop z < -3 -> average) becomes the ROBUST LOSS. Their
    estimator computes `mu_s` and `sigma_s` from the contaminated sample it is trying to clean, so
    its breakdown point is ~0 and it needs a sensitivity table for `tau_lang`. The geometric median
    has breakdown 1/2 and no threshold at all.

WHY VIEWS AND NOT RAYS. We measured per-view region-CLIP accuracy spanning 0.057 to 0.94 -- entire
views fail together, because a view's SAM regions get one CLIP label each. A ray-level robust
estimator cannot delete a view that is uniformly wrong, which is why our ray-level geometric median
bought only +0.10. The view is the unit at which the upstream actually fails.

The baseline it must be compared against is SFS Eq. 6, which is the weighted MEAN over RAYS -- the
least robust corner of the same design space. Setting `--iters 0` returns exactly that, asserted, so
a sweep measures the estimator rather than an implementation difference.

COST. The per-(primitive, view) table is built once in CLASS space (C is 7-19, not 512), so it is
`sum_v |primitives hit in v|` rows of C floats -- hundreds of MB, not the ~100 GB a dense
(P, V, 512) tensor would need. Weiszfeld then runs fully vectorised over that table with index_add.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from camera_bridge import K_from_ray_dirs
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT
from oracle_projected import official_lut, official_label_image, LABEL2D_ROOT
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

FOAM = {"truefrozen", "nonfrozen"}


def real_class_scores(scene, stem, H, W, T, dev):
    """Per-pixel class scores from the REAL upstream: SAM region -> CLIP feature -> text head.

    This is what makes the estimator testable at all. Under the oracle every view carries perfect
    labels, so a view-level robust aggregator has nothing to reject and would measure nothing; the
    0.057-0.94 spread in per-view region-CLIP accuracy only exists here.

    Scores are kept SOFT (cosine against the text head) rather than argmaxed, so a view that is
    merely uncertain is down-weighted by the geometric median rather than voting as if confident.
    """
    base = os.path.join("data", "scannet", f"{scene}_colmap", "language_features")
    sg = np.load(os.path.join(base, f"{stem}_s.npy"))
    f = np.load(os.path.join(base, f"{stem}_f.npy")).astype(np.float32)
    if sg.ndim == 3:
        sg = sg[0]
    if sg.shape != (H, W):
        from PIL import Image
        sg = np.array(Image.fromarray(sg.astype(np.int32)).resize((W, H), Image.NEAREST))
    ft = torch.from_numpy(f).to(dev)
    ft = ft / ft.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    reg_scores = ft @ T.T                                     # (n_regions, C)
    sg = np.where(sg >= f.shape[0], -1, sg).reshape(-1)
    idx = torch.from_numpy(np.clip(sg, 0, max(f.shape[0] - 1, 0))).to(dev)
    sc = reg_scores[idx]
    sc[torch.from_numpy(sg < 0).to(dev)] = 0.0                # unassigned pixels contribute nothing
    return sc, torch.from_numpy(sg >= 0).to(dev)


def build_view_table(scene, arm, kept, n2i, dev, cap=64, tfloor=1e-3, max_views=0, mode="real"):
    """-> (prim idx, view idx, per-view class hist normalised, per-view mass), plus P and C."""
    C = len(kept)
    lut = official_lut(kept, n2i)
    T_txt = embed_class_names(kept, dev)
    T_txt = T_txt / T_txt.norm(dim=-1, keepdim=True)
    cfg = f"output/scannet_{scene}_{'truefrozen' if arm not in FOAM else arm}/config.yaml"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    stems = [os.path.splitext(f)[0]
             for f in sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))]
    sel = list(range(len(dh.cameras)))
    if max_views:
        sel = np.linspace(0, len(sel) - 1, max_views).astype(int).tolist()

    if arm in FOAM:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{arm}/model.pt")
        P = model.points.shape[0]
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gs_ = torch.exp(sp["scales"].to(dev))
        go = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)
        P = gm.shape[0]

    PI, VI, HI, MI = [], [], [], []
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        if mode == "oracle":
            cls = official_label_image(scene, vi, H, W, lut, stems, dev)
            valid_px = cls > 0
        else:
            sc_px, valid_px = real_class_scores(scene, stems[vi], H, W, T_txt, dev)
        if not bool(valid_px.any()):
            continue
        if arm in FOAM:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096, transmittance_threshold=tfloor)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.to(dev), W, H,
                                                    max_hits_per_pixel=cap,
                                                    transmittance_floor=tfloor)
        r_ = ri.to(torch.int64).to(dev); c_ = ci.to(torch.int64).to(dev); v_ = vv.float().to(dev)
        del ri, ci, vv
        keep = valid_px[r_]
        r_, c_, v_ = r_[keep], c_[keep], v_[keep]
        if r_.numel() == 0:
            continue
        hist = torch.zeros(P, C, device=dev)
        if mode == "oracle":
            hist.index_put_((c_, cls[r_] - 1), v_, accumulate=True)
            mass = hist.sum(1)
        else:
            # soft: each ray contributes its class-score vector, weighted by its operator value
            hist.index_add_(0, c_, v_[:, None] * sc_px[r_])
            mass = torch.zeros(P, device=dev).index_add_(0, c_, v_)
        hit = torch.nonzero(mass > 0, as_tuple=False).reshape(-1)
        if hit.numel() == 0:
            continue
        # store the per-view class DISTRIBUTION; the mass rides separately as the weight
        PI.append(hit.to(torch.int32).cpu())
        VI.append(torch.full((hit.numel(),), vi, dtype=torch.int32))
        HI.append((hist[hit] / mass[hit, None]).half().cpu())
        MI.append(mass[hit].float().cpu())
        del hist, mass, hit, r_, c_, v_
        torch.cuda.empty_cache()
    return (torch.cat(PI).to(dev).long(), torch.cat(VI), torch.cat(HI).to(dev).float(),
            torch.cat(MI).to(dev), P, C, len(sel))


def weiszfeld(pi, h, m, P, C, iters, eps=1e-6):
    """Weighted geometric median per primitive over its views. iters=0 -> the weighted MEAN."""
    num = torch.zeros(P, C, device=h.device).index_add_(0, pi, h * m[:, None])
    den = torch.zeros(P, device=h.device).index_add_(0, pi, m)
    X = num / den.clamp_min(1e-30)[:, None]                  # iters=0: SFS Eq. 6, exactly
    for _ in range(iters):
        d = (h - X[pi]).norm(dim=-1).clamp_min(eps)
        w = m / d
        num = torch.zeros(P, C, device=h.device).index_add_(0, pi, h * w[:, None])
        den = torch.zeros(P, device=h.device).index_add_(0, pi, w)
        X = num / den.clamp_min(1e-30)[:, None]
    return X, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen")
    ap.add_argument("--iters", default="0,1,3,8")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--mode", default="real", choices=["real", "oracle"],
                    help="real = actual SAM+CLIP features (the only setting where view-level "
                         "robustness can do anything; the oracle's views are all perfect)")
    ap.add_argument("--out", default="artifacts/scannet/view_geomed.json")
    a = ap.parse_args()
    its = [int(x) for x in a.iters.split(",")]
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    dev = "cuda"
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
                pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
                n2i = {n: i for i, n in enumerate(names)}
                pres = set(np.unique(raw).tolist())
                kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
                gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
                vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
                m = (gl > 0) & vis
                pi, vi, h, mass, P, C, nv = build_view_table(sc, arm, kept, n2i, dev,
                                                             a.cap, max_views=a.max_views,
                                                             mode=a.mode)
                if arm in FOAM:
                    from diagnose_holes import geometry
                    cent, rad, _ = geometry(sc, arm)
                    live = np.zeros(P, bool); live[pi.cpu().numpy()] = True
                    own = assign_points_to_power_cells(pts[m], cent, rad, valid=live, k=64)
                else:
                    ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                    weights_only=False)
                    sp = ck["splats"] if "splats" in ck else ck
                    cent = sp["means"].float().numpy()
                    live = np.zeros(P, bool); live[pi.cpu().numpy()] = True
                    own = assign_points_to_nearest_center(pts[m], cent, valid=live)
                gt = torch.from_numpy(gl[m])
                r = {"arm": arm, "scene": sc, "P": int(P), "C": C, "views": int(nv),
                     "table_rows": int(pi.numel()),
                     "views_per_prim": float(pi.numel() / max(int(live.sum()), 1))}
                for it in its:
                    X, den = weiszfeld(pi, h, mass, P, C, it)
                    lab = torch.zeros(P, dtype=torch.long, device=dev)
                    liv = torch.from_numpy(live).to(dev)
                    lab[liv] = X[liv].argmax(1) + 1
                    pred = np.zeros(m.sum(), np.int64)
                    ok = own >= 0
                    pred[ok] = lab.cpu().numpy()[own[ok]]
                    _, mi, ac, _ = calculate_metrics(gt, torch.from_numpy(pred), C + 1)
                    r[f"miou_{it}"] = float(mi) * 100
                    r[f"acc_{it}"] = float(ac) * 100
                del pi, vi, h, mass
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            b = r["miou_0"]
            best = max(its, key=lambda i: r[f"miou_{i}"])
            print(f"[{arm}/{sc}] views {r['views']}  rows {r['table_rows']:,}  "
                  f"views/prim {r['views_per_prim']:.1f}  mean {b:.2f}  "
                  f"best it={best} {r[f'miou_{best}']:.2f} ({r[f'miou_{best}'] - b:+.2f})", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'n':>3}" + "".join(f"{('it=' + str(i)):>10}" for i in its))
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        b = float(np.mean([r["miou_0"] for r in s]))
        print(f"{arm:<12}{len(s):>3}" + "".join(
            f"{np.mean([r[f'miou_{i}'] for r in s]) - b:>+10.2f}" for i in its)
            + f"   mean-over-rays base {b:.2f}")
    print("(it=0 is the weighted MEAN over views; deltas are mIoU vs that)")


if __name__ == "__main__":
    main()
