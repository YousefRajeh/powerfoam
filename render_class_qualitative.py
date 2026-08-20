"""Qualitative per-class figures: where each method succeeds and where it fails, in 3D.

The quantitative tables say foam beats 3DGS on mIoU and that the errors are metres deep
rather than centimetres. This renders what that actually looks like, per class, so the claim
can be seen instead of only read.

For each (method, class) it colours the SAME ScanNet GT point cloud four ways:
  TP  (green)  predicted this class, and correct
  FP  (red)    predicted this class, but the GT is something else  -> drives mae_pred2gt
  FN  (blue)   GT is this class, but we predicted something else   -> drives mae_gt2pred
  rest (grey)  context, drawn faint so the room stays readable

Red pixels far from any green are exactly the failure mode the semantic-Chamfer metric
measures: a class label painted metres from where that class lives. Reading the red/green
separation in these figures is the visual counterpart of the p->g column.

Classes are chosen per method by per-class IoU (best N and worst N among classes PRESENT in
the scene), so each method is shown at its own best and worst rather than on a fixed list --
which is the honest way to answer "what does it get right and what does it get wrong".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       classify_primitives, load_gaussian_means_opacities)
from point_cloud_query import assign_points_to_nearest_center
from run_cluster_classify_eval import SCENES, two_level_position_aware, K_FLAT
from eval_semantic_surface import predict_labels
import torch.nn.functional as F


def per_class_iou(gt, pred, n_classes):
    out = {}
    for c in range(1, n_classes):
        g, p = gt == c, pred == c
        if not g.any():
            continue
        inter = float((g & p).sum())
        union = float((g | p).sum())
        out[c] = inter / max(union, 1.0)
    return out


def gaussian_predictions(scene, gt_root, ckpt, features, device, class_sets,
                         protocol="opengaussian", opacity_threshold=0.1):
    """3DGS side under the OpenGaussian protocol, mirroring eval_semantic_surface_gaussian."""
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{gt_root}/{split}/{scene}", "segment20")
    means, opacities = load_gaussian_means_opacities(ckpt, device)
    valid = opacities >= opacity_threshold
    feats = torch.load(features, map_location=device, weights_only=False).float()
    assigned = assign_points_to_nearest_center(gt_points, means, valid=valid)
    owned = assigned >= 0
    vi = np.where(valid)[0]
    unit = F.normalize(feats[torch.from_numpy(vi).to(device)], dim=-1)
    pos = torch.from_numpy(np.asarray(means)[vi]).to(device).float()
    leaf = two_level_position_aware(pos, unit, seed=0, leaf_init="randperm")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    out = {}
    for cs in class_sets:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = remap_gt_labels(raw_labels, tids)
        text = embed_class_names(tnames, device)
        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, leaf, unit)
        vcls = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1)
        pcls = np.zeros(np.asarray(means).shape[0], dtype=np.int64)
        pcls[vi] = vcls[leaf].cpu().numpy()
        pred = np.zeros(gt_points.shape[0], dtype=np.int64)
        pred[owned] = pcls[assigned[owned]] + 1
        out[cs] = (gt_points, gt_t, pred, len(tids) + 1, tnames)
    return out


def render_panel(ax, pts, gt, pred, c, name, iou, max_pts=90000, seed=0):
    """One class on one method. Subsampled for legibility, but TP/FP/FN are subsampled
    INDEPENDENTLY of the grey context so small classes stay visible in a large room."""
    rng = np.random.default_rng(seed)
    g, p = gt == c, pred == c
    tp, fp, fn = g & p, (~g) & p, g & (~p)
    rest = ~(tp | fp | fn)
    idx_rest = np.where(rest)[0]
    if idx_rest.size > max_pts:
        idx_rest = rng.choice(idx_rest, max_pts, replace=False)
    ax.scatter(pts[idx_rest, 0], pts[idx_rest, 1], pts[idx_rest, 2],
               s=0.30, c="#d9d9d9", alpha=0.16, linewidths=0, rasterized=True)
    for mask, col, lab in ((fn, "#1f77ff", "FN"), (fp, "#e8112d", "FP"), (tp, "#12a150", "TP")):
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        if idx.size > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   s=0.7, c=col, alpha=0.75, linewidths=0, rasterized=True, label=lab)
    ax.set_title(f"{name}\nIoU {iou*100:.1f}  |  TP {int(tp.sum())}  FP {int(fp.sum())}  FN {int(fn.sum())}",
                 fontsize=8)
    ax.set_axis_off()
    ax.view_init(elev=62, azim=-72)
    try:
        ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--n-best", type=int, default=2)
    p.add_argument("--n-worst", type=int, default=2)
    p.add_argument("--gauss-ckpt", default=r"D:\Downloads\gaussians_scannet\scene0000_00\converted.ply")
    p.add_argument("--gauss-features", default=r"D:\Downloads\gaussians_scannet\scene0000_00\converted_features.pt")
    p.add_argument("--outdir", default="artifacts/scannet/qualitative")
    args = p.parse_args()

    device = "cuda"
    cs = args.class_set
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    methods = {}
    for label, variant, protocol in (
            ("foam nonfrozen / champion", "nonfrozen", "champion"),
            ("foam frozen / champion", "frozen", "champion"),
            ("foam nonfrozen / OpenGaussian protocol", "nonfrozen", "opengaussian")):
        try:
            preds = predict_labels(args.scene, variant, args.gt_root, device, [cs],
                                   uniform_R=(protocol == "opengaussian"), protocol=protocol)
            preds.pop("_uniform_reliability", None)
            methods[label] = preds[cs]
            print(f"[ok] {label}", flush=True)
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}", flush=True)

    try:
        methods["3DGS / OpenGaussian protocol (SFS lifting)"] = gaussian_predictions(
            args.scene, args.gt_root, args.gauss_ckpt, args.gauss_features, device, [cs])[cs]
        print("[ok] 3DGS", flush=True)
    except Exception as e:
        print(f"[skip] 3DGS: {type(e).__name__}: {e}", flush=True)

    summary = {}
    for label, (pts, gt, pred, ncls, tnames) in methods.items():
        ious = per_class_iou(gt, pred, ncls)
        ranked = sorted(ious.items(), key=lambda kv: kv[1], reverse=True)
        chosen = ranked[:args.n_best] + ranked[-args.n_worst:]
        summary[label] = {tnames[c - 1]: v for c, v in ranked}
        n = len(chosen)
        fig = plt.figure(figsize=(4.1 * n, 4.6))
        for j, (c, iou) in enumerate(chosen):
            ax = fig.add_subplot(1, n, j + 1, projection="3d")
            tag = "BEST" if j < args.n_best else "WORST"
            render_panel(ax, pts, gt, pred, c, f"[{tag}] {tnames[c-1]}", iou)
            if j == 0:
                ax.legend(loc="upper left", fontsize=7, markerscale=8, framealpha=0.85)
        fig.suptitle(f"{label}  --  {args.scene}, {cs}\n"
                     f"green = correct, red = predicted here but wrong, blue = missed",
                     fontsize=10)
        fig.tight_layout()
        safe = label.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        out = f"{args.outdir}/{args.scene}_{cs}_{safe}.png"
        fig.savefig(out, dpi=165, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    with open(f"{args.outdir}/{args.scene}_{cs}_per_class_iou.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote per-class IoU json")


if __name__ == "__main__":
    main()
