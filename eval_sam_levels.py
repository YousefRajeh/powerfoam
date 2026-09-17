"""The upstream defect, located: the lift consumes ONE SAM granularity, and it is the coarsest.

The region diagnostic found that SAM regions are clean (92.6% pixel-weighted purity, 77.4% of
them at least 90% pure) but that CLIP assigns the right class to a region only 49.2% of the
time. A region is given ONE feature and every primitive under it inherits it, so a region CLIP
gets wrong is wrong in every view that contains it. That is a bias, not noise, and it is why
every aggregation-side intervention has failed: averaging more views of the same mislabelled
region only makes the wrong answer more confident. It is also exactly what a hole in a mask
looks like -- a patch of cabinet whose region CLIP calls "door" is called door from every angle.

The reported configuration reads `openclip_features_sam_l3`: SAM level 3 alone, 14 regions for
a 968x1296 image. The multi-level exports on disk carry 142 regions over 4 nested levels, with
35 at the finest. This measures, per level and for the levels combined:

  region acc   does CLIP label the region with the region's own majority GT class
  mIoU         the lifted per-primitive prediction, scored in the reported point protocol

Scoring is done in SCORE space (sum over pixels of alpha * cosine). That is not a shortcut: the
lifting objective is linear in the features, so for a linear aggregator projection commutes with
the solve, and `score_mean` was measured to reproduce the feature-space lift to within +0.10
mIoU. It lets every level and every combination be read off one accumulation pass.

Combining levels is the cheapest possible decorrelation of the region bias: a primitive covered
by a wrong coarse region is usually also covered by a finer region that is right, and the two
errors are not the same error.
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
              gt_opacity_mask, dev="cuda"):
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
    votes = np.zeros((P, C + 1), np.int32)
    okm = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[okm], gt_lab[okm]), 1)
    prim_gt = votes.argmax(1); prim_gt[votes.max(1) == 0] = 0
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)

    probe = np.load(f"data/scannet/{scene}_colmap/{feat_dir}/"
                    f"{os.path.splitext(sorted(os.listdir(f'data/scannet/{scene}_colmap/images'))[0])[0]}_s.npy")
    nlev = probe.shape[0] if probe.ndim == 3 else 1

    acc = {L: torch.zeros(P, C, device=dev) for L in range(nlev)}
    wsum = {L: torch.zeros(P, device=dev) for L in range(nlev)}
    reg_hit = {L: [0.0, 0.0] for L in range(nlev)}

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
        segs = np.load(sp).astype(np.int64)
        if segs.ndim == 2:
            segs = segs[None]
        fv = F.normalize(torch.from_numpy(np.load(fp)).to(dev).float(), dim=-1)
        simm = fv @ text.T
        reg_cls = simm.argmax(1) + 1
        N = fv.shape[0]
        base = (alpha >= alpha_eps) & (fpi >= 0)
        for L in range(segs.shape[0]):
            seg = F.interpolate(torch.from_numpy(segs[L])[None, None].float(), size=(H, W),
                                mode="nearest")[0, 0].to(dev).long()
            keep = base & (seg >= 0)
            if not bool(keep.any()):
                continue
            j, s, w = fpi[keep], seg[keep], alpha[keep]
            acc[L].index_add_(0, j, simm[s] * w.unsqueeze(-1))
            wsum[L].index_add_(0, j, w)
            g = prim_gt_t[j]
            hm = g > 0
            if bool(hm.any()):
                hist = torch.bincount(s[hm] * (C + 1) + g[hm],
                                      minlength=N * (C + 1)).reshape(N, C + 1).float()
                tot = hist.sum(1)
                live = tot > 0
                reg_hit[L][0] += float(((reg_cls[live] == hist.argmax(1)[live]).float()
                                        * tot[live]).sum())
                reg_hit[L][1] += float(tot[live].sum())

    own = assigned >= 0

    def score(A, W):
        pc = (A.argmax(1).cpu().numpy() + 1)
        pc[(W <= 0).cpu().numpy()] = 0
        pl = np.zeros(len(gt_pts), np.int64)
        pl[own] = pc[assigned[own]]
        _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pl).long(), C + 1)
        return float(mi), float(ma)

    res = {}
    pl = np.zeros(len(gt_pts), np.int64)
    plift = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    pl[own] = plift[assigned[own]]
    _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                     torch.from_numpy(pl).long(), C + 1)
    res["lifted_l3_reported"] = (float(mi), float(ma))
    for L in range(nlev):
        res[f"L{L}"] = score(acc[L], wsum[L])
    # normalise each level to a mean cosine before summing, so a level that simply covers more
    # pixels does not outvote the others purely on mass
    tot = sum(acc[L] / wsum[L].clamp_min(1e-9).unsqueeze(-1) for L in range(nlev))
    res["all_levels"] = score(tot, sum(wsum[L] for L in range(nlev)))
    if nlev > 1:
        fine = sum(acc[L] / wsum[L].clamp_min(1e-9).unsqueeze(-1) for L in range(nlev - 1))
        res["levels_0_2"] = score(fine, sum(wsum[L] for L in range(nlev - 1)))
    res["_region_acc"] = {f"L{L}": (reg_hit[L][0] / reg_hit[L][1]) if reg_hit[L][1] else float("nan")
                          for L in range(nlev)}
    res["_nviews"] = nviews
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-sets", default="opengaussian19")
    ap.add_argument("--feat-dir", default="openclip_features_sam_whitepad")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/sam_levels.json")
    a = ap.parse_args()
    wp.init()
    allres = {}
    for cs in a.class_sets.split(","):
        rows = {}
        for sc in a.scenes.split(","):
            try:
                r = one_scene(sc, a.recon, cs, a.feat_dir, a.alpha_eps, a.opacity_threshold,
                              not a.no_gt_opacity_mask)
            except Exception as e:
                print(f"[{cs}/{sc}] SKIP {type(e).__name__}: {e}")
                continue
            rows[sc] = r
            print(f"[{cs}/{sc}] region CLIP acc " +
                  " ".join(f"{k} {v:.3f}" for k, v in r["_region_acc"].items()))
            print(f"[{cs}/{sc}] mIoU " + "  ".join(
                f"{k} {v[0]*100:.2f}" for k, v in r.items() if not k.startswith("_")))
        allres[cs] = rows
        if rows:
            keys = [k for k in next(iter(rows.values())) if not k.startswith("_")]
            base = np.mean([rows[s]["lifted_l3_reported"][0] for s in rows]) * 100
            print(f"\n--- {cs}: mean over {len(rows)} scenes (delta vs reported l3 lift) ---")
            for k in keys:
                mi = np.mean([rows[s][k][0] for s in rows]) * 100
                ma = np.mean([rows[s][k][1] for s in rows]) * 100
                wins = sum(rows[s][k][0] > rows[s]["lifted_l3_reported"][0] for s in rows)
                print(f"  {k:<20} mIoU {mi:6.2f} ({mi-base:+5.2f})  mAcc {ma:6.2f}  "
                      f"wins {wins}/{len(rows)}")
            ra = {L: np.mean([rows[s]["_region_acc"][L] for s in rows])
                  for L in next(iter(rows.values()))["_region_acc"]}
            print("  region CLIP acc: " + "  ".join(f"{k} {v:.3f}" for k, v in ra.items()))
    json.dump(allres, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
