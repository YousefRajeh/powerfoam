"""Aggregate in SCORE space instead of FEATURE space.

The pipeline collapses each primitive to one 512-d vector and only then projects it onto a text
direction:  s_j(c) = < aggregate_v f_j^v , t_c >.  The order can be swapped -- project each view
first and aggregate the scalars:  s_j(c) = aggregate_v < f_j^v , t_c >.

Whether that changes anything is decided by the theory, not by taste. The lifting objective is
LINEAR in the features and touches the data only through A^T B, so for a linear aggregator
(the plain visibility-weighted mean) projection commutes with the solve and the two orders are
IDENTICAL -- `score_mean` below is the control that must reproduce the weighted-mean lift. The
orders can only differ where the aggregator is nonlinear, and then score space is strictly more
expressive: a primitive whose views genuinely disagree is bimodal, and no single vector can
represent two CLIP directions at once. Averaging two class directions lands between them, where
the nearest text embedding is frequently a third, unrelated class. Projecting first keeps the
two modes apart and lets a nonlinear aggregator pick one.

This matters here because the vote-table diagnostic says the true class is rank 1 for 60.5% of
primitives and within rank 2 for 75.7% -- the evidence holds the answer, the collapse to a mean
loses it.

SINGLE-QUERY SAFE. Every arm computes class c from that primitive's own per-view features and
t_c alone. No class competes with another, no scene-level statistic is used, so running one
query costs one accumulation pass over that one direction and behaves exactly as it does here.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import configargparse, warp as wp

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics,
                                       apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT


def one_scene(scene, recon, class_set, feat_dir, alpha_eps, opacity_threshold,
              gt_opacity_mask, taus, dev="cuda"):
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device=dev)
    m.load_pt(f"{ck}/model.pt"); m.update_vis_cache()

    sol = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{recon}_ogl3.pt",
                     map_location="cpu", weights_only=True)
    X = sol["primitive_features"].to(dev).float()
    valid = sol["valid_mask"].numpy()
    centers = m.points.detach().cpu().numpy()
    radii = m.get_radii().detach().cpu().numpy()
    density = m.get_density().detach().float().cpu().numpy().reshape(-1)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    text = embed_class_names(kept, dev)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    if gt_opacity_mask:
        alpha_p = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
        gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, alpha_p, opacity_threshold, scene)

    P = centers.shape[0]
    acc_w = torch.zeros(P, C, device=dev)            # sum_v w * sim   (linear control)
    acc_wt = torch.zeros(P, device=dev)              # sum_v w
    acc_max = torch.full((P, C), -2.0, device=dev)   # max_v sim
    lse = {t: torch.zeros(P, C, device=dev) for t in taus}   # sum_v w * exp(sim/t)

    names = sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))
    fdir = f"data/scannet/{scene}_colmap/{feat_dir}"
    c = m._vis_cache
    pts, rad = c["points"], c["radii"]
    rgb = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)
    nviews = 0
    for vi, nm in enumerate(names):
        stem = os.path.splitext(nm)[0]
        fp, sp = f"{fdir}/{stem}_f.npy", f"{fdir}/{stem}_s.npy"
        if not (os.path.exists(fp) and os.path.exists(sp)):
            continue
        nviews += 1
        with torch.no_grad():
            out = m.rasterizer.visualize(dh.cameras[vi], pts, rad, c["density"], c["normals"],
                                         c["texel_sites"], rgb, c["texel_height"],
                                         c["adjacency"], c["adjacency_offsets"])
        alpha, fpi = out[3], out[7].long()
        H, W = alpha.shape[-2], alpha.shape[-1]
        alpha, fpi = alpha.reshape(H, W), fpi.reshape(H, W)
        seg = torch.from_numpy(np.load(sp).astype(np.int64))
        seg = seg[0] if seg.ndim == 3 else seg
        seg = F.interpolate(seg[None, None].float(), size=(H, W),
                            mode="nearest")[0, 0].to(dev).long()
        fv = F.normalize(torch.from_numpy(np.load(fp)).to(dev).float(), dim=-1)
        simm = fv @ text.T
        keep = (alpha >= alpha_eps) & (fpi >= 0) & (seg >= 0)
        if not bool(keep.any()):
            continue
        j, s, w = fpi[keep], seg[keep], alpha[keep]
        sv = simm[s]
        acc_w.index_add_(0, j, sv * w.unsqueeze(-1))
        acc_wt.index_add_(0, j, w)
        acc_max.index_reduce_(0, j, sv, "amax", include_self=True)
        for t in taus:
            lse[t].index_add_(0, j, torch.exp(sv / t) * w.unsqueeze(-1))

    have = (acc_wt > 0).cpu().numpy()
    own = assigned >= 0

    def score(pred_cls):
        pc = pred_cls.copy(); pc[~have] = 0
        pl = np.zeros(len(gt_pts), np.int64)
        pl[own] = pc[assigned[own]]
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                             torch.from_numpy(pl).long(), C + 1)
        return float(miou), float(macc)

    res = {}
    # baseline: the method as reported -- scored on ALL primitives, not just front-visible ones
    pl = np.zeros(len(gt_pts), np.int64)
    plift = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    pl[own] = plift[assigned[own]]
    _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                     torch.from_numpy(pl).long(), C + 1)
    res["lifted_all"] = (float(mi), float(ma))
    res["lifted_front"] = score(plift)                  # same arm, restricted to voted primitives
    res["score_mean"] = score(acc_w.argmax(1).cpu().numpy() + 1)
    res["score_max"] = score(acc_max.argmax(1).cpu().numpy() + 1)
    for t in taus:
        res[f"score_lse{t}"] = score(lse[t].argmax(1).cpu().numpy() + 1)
    res["_nviews"] = nviews
    res["_front_frac"] = float(have.mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-sets", default="opengaussian19")
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--taus", default="0.01,0.03,0.1")
    ap.add_argument("--out", default="artifacts/scannet/score_aggregation.json")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    wp.init()
    allres = {}
    for cs in a.class_sets.split(","):
        rows = {}
        for sc in a.scenes.split(","):
            try:
                r = one_scene(sc, a.recon, cs, a.feat_dir, a.alpha_eps, a.opacity_threshold,
                              not a.no_gt_opacity_mask, taus)
            except Exception as e:
                print(f"[{cs}/{sc}] SKIP {type(e).__name__}: {e}"); continue
            rows[sc] = r
            print(f"[{cs}/{sc}] front {r['_front_frac']:.1%}  || " + "  ".join(
                f"{k} {v[0]*100:.2f}" for k, v in r.items() if not k.startswith("_")))
        allres[cs] = rows
        if rows:
            keys = [k for k in next(iter(rows.values())) if not k.startswith("_")]
            base = np.mean([rows[s]["lifted_front"][0] for s in rows]) * 100
            print(f"\n--- {cs}: mean over {len(rows)} scenes "
                  f"(delta vs lifted_front, the matched baseline) ---")
            for k in keys:
                mi = np.mean([rows[s][k][0] for s in rows]) * 100
                ma = np.mean([rows[s][k][1] for s in rows]) * 100
                wins = sum(rows[s][k][0] > rows[s]["lifted_front"][0] for s in rows)
                print(f"  {k:<16} mIoU {mi:6.2f} ({mi-base:+5.2f})  mAcc {ma:6.2f}  "
                      f"wins {wins}/{len(rows)}")
    json.dump(allres, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
