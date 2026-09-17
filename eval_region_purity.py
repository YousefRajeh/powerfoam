"""Is the upstream defect VARIANCE or a systematic BIAS -- and if bias, where does it enter?

Every intervention that recombines the per-view evidence has failed: feature-space graph
smoothing, score-space max/LSE, hard majority voting, confidence gating, Potts smoothing. That
is the signature of a systematic bias rather than noise: averaging more observations of a
consistent error only makes the wrong answer more confident.

The natural place for such a bias to enter is the SAM region. A region is assigned ONE CLIP
feature, and every primitive under it inherits that feature. If a region straddles two objects,
every primitive beneath it is pulled toward a blend of two class directions -- in exactly the
same way in every view that contains the region, which is precisely a bias that no aggregator
can remove.

So this measures region MIXEDNESS directly: for each SAM region in each view, the distribution
of GT classes over the pixels it covers (GT read through the primitive each pixel sees, the same
target the 3D metric uses). Then accuracy is tabulated against the purity of the regions a
primitive inherited from. If accuracy collapses for primitives under mixed regions, the defect
is localised, measurable and upstream -- and the fix is a region proposal the lift can trust,
not a better lift.
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
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT


def one_scene(scene, recon, class_set, feat_dir, alpha_eps, dev="cuda"):
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

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    text = embed_class_names(kept, dev)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)

    P = centers.shape[0]
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = votes.argmax(1); prim_gt[votes.max(1) == 0] = 0
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)
    pred3d_t = torch.from_numpy((F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1).to(dev)

    # per-primitive accumulators: pixel-weighted region purity, and region GT-vs-CLIP agreement
    pur_sum = torch.zeros(P, device=dev)
    pur_w = torch.zeros(P, device=dev)
    reg_area = torch.zeros(P, device=dev)

    names = sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))
    fdir = f"data/scannet/{scene}_colmap/{feat_dir}"
    c = m._vis_cache
    pts, rad = c["points"], c["radii"]
    rgb = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)
    reg_stats = []                     # (purity, npix, region_clip_correct_vs_majority_gt)
    for vi, nm in enumerate(names):
        stem = os.path.splitext(nm)[0]
        fp, sp = f"{fdir}/{stem}_f.npy", f"{fdir}/{stem}_s.npy"
        if not (os.path.exists(fp) and os.path.exists(sp)):
            continue
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
        reg_cls = ((fv @ text.T).argmax(1) + 1)                       # (N,) CLIP label per region
        N = fv.shape[0]

        keep = (alpha >= alpha_eps) & (fpi >= 0) & (seg >= 0)
        if not bool(keep.any()):
            continue
        j, s = fpi[keep], seg[keep]
        g = prim_gt_t[j]
        lab = keep & False
        # region x class histogram over pixels that carry a GT label
        hm = g > 0
        if not bool(hm.any()):
            continue
        flat = s[hm] * (C + 1) + g[hm]
        hist = torch.bincount(flat, minlength=N * (C + 1)).reshape(N, C + 1).float()
        tot = hist.sum(1)
        purity = torch.where(tot > 0, hist.max(1).values / tot.clamp_min(1), torch.zeros_like(tot))
        maj = hist.argmax(1)
        live = tot > 0
        reg_stats.append(torch.stack([purity[live], tot[live],
                                      (reg_cls[live] == maj[live]).float()], 1).cpu().numpy())
        # push region purity down to the primitives that inherited it
        w = alpha[keep][hm]
        pur_sum.index_add_(0, j[hm], purity[s[hm]] * w)
        pur_w.index_add_(0, j[hm], w)
        reg_area.index_add_(0, j[hm], tot[s[hm]] * w)

    prim_pur = (pur_sum / pur_w.clamp_min(1e-9)).cpu().numpy()
    prim_area = (reg_area / pur_w.clamp_min(1e-9)).cpu().numpy()
    seen = (pur_w > 0).cpu().numpy()

    pred = pred3d_t.cpu().numpy()
    sel = seen & (prim_gt > 0)
    corr = (pred[sel] == prim_gt[sel])
    pur, area = prim_pur[sel], prim_area[sel]

    def by_quintile(v):
        q = np.quantile(v, np.linspace(0, 1, 6))
        out = []
        for b in range(5):
            mk = (v >= q[b]) & (v <= q[b + 1] if b == 4 else v < q[b + 1])
            out.append((float(q[b]), float(q[b + 1]), int(mk.sum()),
                        float(corr[mk].mean()) if mk.sum() else float("nan")))
        return out

    rs = np.concatenate(reg_stats) if reg_stats else np.zeros((0, 3))
    return dict(scene=scene, n_prims=int(sel.sum()),
                acc=float(corr.mean()),
                purity_q=by_quintile(pur), area_q=by_quintile(area),
                region_purity_mean=float(np.average(rs[:, 0], weights=rs[:, 1])) if len(rs) else float("nan"),
                region_clip_acc=float(np.average(rs[:, 2], weights=rs[:, 1])) if len(rs) else float("nan"),
                frac_regions_pure90=float(np.average(rs[:, 0] >= 0.9, weights=rs[:, 1])) if len(rs) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/scannet/region_purity.json")
    a = ap.parse_args()
    wp.init()
    rows = []
    for sc in a.scenes.split(","):
        try:
            r = one_scene(sc, a.recon, a.class_set, a.feat_dir, a.alpha_eps)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}"); continue
        rows.append(r)
        print(f"[{sc}] acc {r['acc']:.4f} | SAM regions: pixel-weighted purity "
              f"{r['region_purity_mean']:.3f}, >=90% pure {r['frac_regions_pure90']:.1%}, "
              f"CLIP label == region majority GT {r['region_clip_acc']:.3f}")
        print("   acc by inherited region purity:  " +
              "  ".join(f"Q{i+1}[{q[0]:.2f}-{q[1]:.2f}] {q[3]:.3f}"
                        for i, q in enumerate(r["purity_q"])))
    if rows:
        print(f"\n=== mean over {len(rows)} scenes ===")
        print(f"  acc {np.mean([r['acc'] for r in rows]):.4f}   "
              f"region purity {np.mean([r['region_purity_mean'] for r in rows]):.3f}   "
              f">=90% pure {np.mean([r['frac_regions_pure90'] for r in rows]):.1%}   "
              f"region CLIP acc {np.mean([r['region_clip_acc'] for r in rows]):.3f}")
        for lbl, key in [("purity", "purity_q"), ("region area", "area_q")]:
            arr = np.array([[q[3] for q in r[key]] for r in rows], float)
            print(f"  acc by {lbl:<12} " +
                  "  ".join(f"Q{i+1} {np.nanmean(arr[:, i]):.3f}" for i in range(5)))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
