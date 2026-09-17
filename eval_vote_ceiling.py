"""How much headroom is left in AGGREGATION, as opposed to in the upstream features?

The 2D check (eval_2d_ceiling.py) showed a single view's SAM-region CLIP features are much
WORSE than the lifted per-primitive features -- so the lift is already adding a great deal by
combining views, and "just use the 2D features" is not an option. That leaves the sharper
question: given these per-view features, is the current aggregation near the best one?

This walks every view once, maps each pixel to the primitive it sees (front_prim_idx) and to
its SAM region, and accumulates a per-primitive per-class VOTE table. From one table we read
three per-primitive predictions, all scored in the reported point protocol:

  lifted    argmax of the solved feature                  -- the method as it stands
  vote      argmax of the vote table                      -- hard per-view labelling, then a
                                                             weighted majority. This is the
                                                             FlashSplat-style aggregation, done
                                                             on the foam.
  oracle    correct iff the primitive's GT class received ANY vote at all

`oracle` is the ceiling of every scheme that decides a primitive's class from the per-view
evidence it actually received. If `lifted` is near `oracle`, aggregation is finished and the
upstream features are the wall. If the gap is wide, aggregation still has room and it is worth
attacking -- and `vote` says whether a different aggregation captures any of it.
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
              gt_opacity_mask, soft, dev="cuda"):
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
    votes = torch.zeros(P, C, device=dev)
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
        simm = fv @ text.T                                         # (N, C)
        keep = (alpha >= alpha_eps) & (fpi >= 0) & (seg >= 0)
        if not bool(keep.any()):
            continue
        j = fpi[keep]
        s = seg[keep]
        w = alpha[keep].unsqueeze(-1)
        if soft:
            contrib = simm[s] * w                                  # weighted cosine, soft
        else:
            contrib = F.one_hot(simm[s].argmax(-1), C).float() * w  # hard per-pixel label
        votes.index_add_(0, j, contrib)

    have = votes.sum(1) > 0
    pred_vote = votes.argmax(1).cpu().numpy() + 1
    pred_vote[~have.cpu().numpy()] = 0
    pred_lift = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    votes_np = votes.cpu().numpy()

    own = assigned >= 0
    res = {}
    for nmarm, pc in [("lifted", pred_lift), ("vote", pred_vote)]:
        pl = np.zeros(len(gt_pts), np.int64)
        pl[own] = pc[assigned[own]]
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                             torch.from_numpy(pl).long(), C + 1)
        res[nmarm] = (float(miou), float(macc))

    # oracle: give each point its GT class iff that class got any vote on its primitive
    pl = np.zeros(len(gt_pts), np.int64)
    g = gt_lab[own]
    j = assigned[own]
    hit = (g > 0) & (votes_np[j, np.clip(g - 1, 0, C - 1)] > 0)
    pl[own] = np.where(hit, g, pred_vote[j])
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                         torch.from_numpy(pl).long(), C + 1)
    res["oracle"] = (float(miou), float(macc))

    # How loose is that ceiling? Rank of the TRUE class in each primitive vote table. "any vote"
    # is a weak bar when a primitive is seen by many views, so the rank says whether a better
    # aggregator is plausible (true class sitting at rank 2-3) or whether the ceiling is an
    # artefact of counting a single stray pixel (true class buried at rank 8).
    pv = np.zeros(P, np.int64)
    np.add.at(pv, assigned[own][gt_lab[own] > 0], 1)
    pgt = np.zeros(P, np.int64)
    vb = np.zeros((P, C + 1), np.int64)
    gg, jj = gt_lab[own], assigned[own]
    ms = gg > 0
    np.add.at(vb, (jj[ms], gg[ms]), 1)
    pgt = vb.argmax(1); pgt[vb.max(1) == 0] = 0
    sel = (pgt > 0) & have.cpu().numpy()
    order = np.argsort(-votes_np[sel], axis=1)
    rank = (order == (pgt[sel] - 1)[:, None]).argmax(1) + 1
    got = votes_np[sel][np.arange(sel.sum()), pgt[sel] - 1] > 0
    rank = np.where(got, rank, C + 1)
    res["_rank"] = {f"top{k}": float((rank <= k).mean()) for k in (1, 2, 3, 5)}
    res["_rank"]["never"] = float((rank > C).mean())
    res["_rank"]["median"] = float(np.median(rank))
    res["_nviews"] = nviews
    res["_covered"] = float(have.float().mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--soft", action="store_true",
                    help="accumulate weighted cosine instead of a hard per-pixel label")
    ap.add_argument("--out", default="artifacts/scannet/vote_ceiling.json")
    a = ap.parse_args()
    wp.init()
    rows = {}
    for sc in a.scenes.split(","):
        try:
            r = one_scene(sc, a.recon, a.class_set, a.feat_dir, a.alpha_eps,
                          a.opacity_threshold, not a.no_gt_opacity_mask, a.soft)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}"); continue
        rows[sc] = r
        rk = r["_rank"]
        print(f"[{sc}] true-class rank in vote table: " +
              "  ".join(f"{k} {v:.3f}" for k, v in rk.items()))
        print(f"[{sc}] views {r['_nviews']:>3}  voted-on {r['_covered']:.1%}  || " +
              "  ".join(f"{k} {v[0]*100:6.2f}/{v[1]*100:6.2f}"
                        for k, v in r.items() if not k.startswith('_')))
    if rows:
        print(f"\n=== mean over {len(rows)} scenes (mIoU / mAcc) ===")
        for k in ["lifted", "vote", "oracle"]:
            mi = np.mean([rows[s][k][0] for s in rows]) * 100
            ma = np.mean([rows[s][k][1] for s in rows]) * 100
            print(f"  {k:<8} {mi:6.2f} / {ma:6.2f}")
        print("  true-class rank: " + "  ".join(
            f"{k} {np.mean([rows[s]['_rank'][k] for s in rows]):.3f}"
            for k in ["top1", "top2", "top3", "top5", "never", "median"]))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
