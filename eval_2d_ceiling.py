"""Is the bottleneck the LIFT or the UPSTREAM per-view features?

`diagnose_holes.py` attributes ~45% of scored points to "upstream", but that bucket is a
RESIDUAL -- whatever support and view-agreement failed to explain. This measures it directly.

The comparison is made pixel by pixel, on identical pixels, against an identical target:

  target   the GT label of the primitive the pixel sees (majority of the GT points inside that
           cell). This is exactly the label the 3D metric scores against, so neither arm is
           given an easier goal.
  3D arm   the lifted per-primitive feature of that same primitive, argmaxed against the text
           embeddings -- i.e. what the method actually predicts.
  2D arm   the SAM-region CLIP feature of that pixel in this very view, argmaxed against the
           same text embeddings -- i.e. the best any lifting scheme could hope to inherit,
           since it never has to commit a primitive to a single feature and never has to
           reconcile views.

If the 2D arm is far above the 3D arm, the lift is destroying information and is worth
attacking. If the two are close, the per-view features are the ceiling and no amount of solver,
aggregator or smoother can move the number -- which would say the pipeline is missing a better
upstream, not a better lift.
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


def iou_from_counts(inter, union, seen):
    v = seen > 0
    return float(np.mean(inter[v] / np.maximum(union[v], 1))) if v.any() else float("nan")


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
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])          # 0 = ignore, else 1..C
    text = embed_class_names(kept, dev)

    # per-primitive GT by majority vote of the GT points it owns
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    P = centers.shape[0]
    votes = np.zeros((P, C + 1), np.int32)
    ok = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[ok], gt_lab[ok]), 1)
    prim_gt = votes.argmax(1)
    prim_gt[votes.max(1) == 0] = 0

    pred3d = (F.normalize(X, dim=-1) @ text.T).argmax(1).cpu().numpy() + 1
    prim_gt_t = torch.from_numpy(prim_gt).to(dev)
    pred3d_t = torch.from_numpy(pred3d).to(dev)

    names = sorted(os.listdir(os.path.join(f"data/scannet/{scene}_colmap", "images")))
    stems = [os.path.splitext(n)[0] for n in names]
    fdir = f"data/scannet/{scene}_colmap/{feat_dir}"

    c = m._vis_cache
    pts, rad = c["points"], c["radii"]
    rgb = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)

    acc = {k: [0, 0] for k in ["3d", "2d", "2d_covered"]}
    cnt = {k: (np.zeros(C + 1), np.zeros(C + 1), np.zeros(C + 1)) for k in ["3d", "2d"]}
    nomask = [0, 0]
    for vi, stem in enumerate(stems):
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
        seg = F.interpolate(seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].to(dev).long()
        fv = F.normalize(torch.from_numpy(np.load(fp)).to(dev).float(), dim=-1)
        cls2d_per_mask = (fv @ text.T).argmax(1) + 1                 # (N,)

        j = fpi.clamp_min(0)
        g = prim_gt_t[j]
        keep = (alpha >= alpha_eps) & (fpi >= 0) & (g > 0)
        if not bool(keep.any()):
            continue
        g = g[keep]
        p3 = pred3d_t[j][keep]
        sm = seg[keep]
        covered = sm >= 0
        p2 = torch.where(covered, cls2d_per_mask[sm.clamp_min(0)], torch.zeros_like(sm))

        acc["3d"][0] += int((p3 == g).sum()); acc["3d"][1] += int(g.numel())
        acc["2d"][0] += int((p2 == g).sum()); acc["2d"][1] += int(g.numel())
        acc["2d_covered"][0] += int(((p2 == g) & covered).sum())
        acc["2d_covered"][1] += int(covered.sum())
        nomask[0] += int((~covered).sum()); nomask[1] += int(covered.numel())

        gn = g.cpu().numpy()
        for k, pr in [("3d", p3.cpu().numpy()), ("2d", p2.cpu().numpy())]:
            i_, u_, s_ = cnt[k]
            np.add.at(i_, gn[pr == gn], 1)
            np.add.at(s_, gn, 1)
            np.add.at(u_, gn, 1)
            np.add.at(u_, pr[pr > 0], 1)
    for k in cnt:
        i_, u_, s_ = cnt[k]
        u_ -= i_
    return dict(
        scene=scene, C=C,
        acc3d=acc["3d"][0] / max(acc["3d"][1], 1),
        acc2d=acc["2d"][0] / max(acc["2d"][1], 1),
        acc2d_covered=acc["2d_covered"][0] / max(acc["2d_covered"][1], 1),
        nomask=nomask[0] / max(nomask[1], 1),
        miou3d=iou_from_counts(*cnt["3d"]), miou2d=iou_from_counts(*cnt["2d"]),
        px=acc["3d"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/scannet/ceiling_2d.json")
    a = ap.parse_args()
    wp.init()
    rows = []
    for sc in a.scenes.split(","):
        try:
            r = one_scene(sc, a.recon, a.class_set, a.feat_dir, a.alpha_eps)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}"); continue
        rows.append(r)
        print(f"[{sc}] px {r['px']:>10,}  acc 3D {r['acc3d']:.4f}  2D {r['acc2d']:.4f}  "
              f"2D|covered {r['acc2d_covered']:.4f}  (no SAM mask {r['nomask']:.1%})  ||  "
              f"mIoU 3D {r['miou3d']*100:5.2f}  2D {r['miou2d']*100:5.2f}")
    if rows:
        print(f"\n=== mean over {len(rows)} scenes ===")
        for k in ["acc3d", "acc2d", "acc2d_covered", "miou3d", "miou2d", "nomask"]:
            print(f"  {k:<14} {np.mean([r[k] for r in rows]):.4f}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
