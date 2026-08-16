"""Experiment D: real ground-truth segmentation evaluation. Compares a
rendered discrete cluster map (from feature_foam_lifting.segment) against
Replica's real per-frame semantic ground truth on the held-out test views --
the actual mIoU this whole research question has been building toward,
replacing the self-referential clustering-cohesion proxy used everywhere
else in this project so far.

Since our clusters are unsupervised (no class names), each predicted cluster
is matched to whichever ground-truth class maximizes IoU with it, using
Hungarian assignment on the IoU matrix restricted to clusters/classes that
actually co-occur -- the standard protocol for evaluating unsupervised
segmentation against a labeled taxonomy it was never trained to predict.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment


def build_split(num_images):
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def load_gt_labels(data_dir, test_idx):
    labels = []
    for i in test_idx.tolist():
        arr = np.array(Image.open(Path(data_dir) / "semantic_class" / f"semantic_class_{i}.png"))
        labels.append(torch.from_numpy(arr.astype(np.int64)))
    return torch.stack(labels)  # (n_views, H, W)


def hungarian_miou(pred_labels, gt_labels, num_clusters):
    """pred_labels: (N,) flat predicted cluster ids, -1 for unassigned.
    gt_labels: (N,) flat ground-truth class ids. Returns (mean_iou,
    per_match list of (cluster, gt_class, iou), matched_fraction_of_pixels).
    """
    valid = (pred_labels >= 0) & (gt_labels >= 0)
    pred_v, gt_v = pred_labels[valid], gt_labels[valid]

    gt_classes = torch.unique(gt_v)
    n_gt = gt_classes.numel()
    class_to_col = {int(c): j for j, c in enumerate(gt_classes.tolist())}

    intersection = torch.zeros(num_clusters, n_gt)
    pred_area = torch.zeros(num_clusters)
    gt_area = torch.zeros(n_gt)
    for c in range(num_clusters):
        m = pred_v == c
        pred_area[c] = m.sum()
    for j, c in enumerate(gt_classes.tolist()):
        gt_area[j] = (gt_v == c).sum()

    # Build the intersection matrix by a single pass (bincount on a combined
    # index) rather than an O(num_clusters * n_gt) loop over full masks.
    combined = pred_v.clamp_min(0) * (int(gt_v.max()) + 1) + gt_v
    uniq, counts = torch.unique(combined, return_counts=True)
    for u, cnt in zip(uniq.tolist(), counts.tolist()):
        c = u // (int(gt_v.max()) + 1)
        g = u % (int(gt_v.max()) + 1)
        if 0 <= c < num_clusters and g in class_to_col:
            intersection[c, class_to_col[g]] = cnt

    union = pred_area[:, None] + gt_area[None, :] - intersection
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union))

    cost = -iou.numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if pred_area[r] > 0 or gt_area[c] > 0:
            matches.append({
                "cluster": int(r), "gt_class": int(gt_classes[c]),
                "iou": float(iou[r, c]), "pred_pixels": int(pred_area[r]), "gt_pixels": int(gt_area[c]),
            })
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0
    matched_pixel_fraction = float(valid.float().mean())
    return mean_iou, matches, matched_pixel_fraction


def superpixel_pool_labels(rendered_labels, row_view_ids, data_dir, test_idx, height, width,
                            n_segments=200, compactness=10.0):
    """Tier 1 idea #4: majority-vote pool the rendered per-pixel cluster labels within each
    frame's own SLIC superpixel regions (computed on the GT RGB image -- GT-independent,
    just the photo, so this can't leak ground-truth labels into the prediction). A classic
    segmentation post-processing step (SLIC superpixels + majority-vote pooling predates
    CLIP-3D work entirely) that suppresses single-pixel label noise without touching the 3D
    representation or solver at all. Unassigned (-1) pixels are left as -1 and excluded from
    each superpixel's vote.

    Room_0's real SAM masks aren't available for the held-out TEST views (extraction only
    covers the 787 train views, by construction of the train/test split), so SLIC (computed
    directly on the already-available GT RGB, no extra extraction needed) is the version of
    this idea that's actually testable here without a new extraction pass -- see
    ResearchVault/Ideas/general-mIoU-ideas.md Idea 8 for the SAM-mask variant this is a
    zero-extra-cost stand-in for.
    """
    from skimage.segmentation import slic

    pooled = rendered_labels.clone()
    num_pixels_per_view = height * width
    for view_idx, i in enumerate(test_idx.tolist()):
        rgb_path = Path(data_dir) / "rgb" / f"rgb_{i}.png"
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        segments = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0)
        segments = torch.from_numpy(segments.astype(np.int64)).reshape(-1)

        start = view_idx * num_pixels_per_view
        end = start + num_pixels_per_view
        view_labels = pooled[start:end]

        valid = view_labels >= 0
        if not valid.any():
            continue
        num_segments_actual = int(segments.max()) + 1
        for seg_id in range(num_segments_actual):
            in_segment = (segments == seg_id) & valid
            if not in_segment.any():
                continue
            votes = view_labels[in_segment]
            majority = torch.bincount(votes).argmax()
            view_labels[in_segment] = majority
        pooled[start:end] = view_labels
    return pooled


def main():
    p = argparse.ArgumentParser(description="Evaluate rendered cluster segmentation against real Replica ground truth")
    p.add_argument("--segmentation", required=True, help="Output of feature_foam_lifting.segment_cli (with rendered_labels)")
    p.add_argument("--data-dir", default=r"D:\Downloads\powerfoam\data\replica\room_0")
    p.add_argument("--output", required=True)
    p.add_argument("--superpixel-pool", action="store_true",
                    help="Tier 1 idea #4: majority-vote pool predicted labels within SLIC "
                    "superpixels (computed on GT RGB) before scoring -- see superpixel_pool_labels().")
    p.add_argument("--slic-n-segments", type=int, default=200)
    p.add_argument("--slic-compactness", type=float, default=10.0)
    args = p.parse_args()

    seg = torch.load(args.segmentation, map_location="cpu", weights_only=True)
    rendered_labels = seg["rendered_labels"]
    row_view_ids = seg["row_view_ids"]
    num_clusters = seg["report"]["num_clusters"]

    _, test_idx = build_split(900)
    n_views = int(row_view_ids.max().item()) + 1
    assert n_views == len(test_idx), f"{n_views} rendered views vs {len(test_idx)} expected test views"

    gt = load_gt_labels(args.data_dir, test_idx)  # (n_views, H, W)
    height, width = gt.shape[1], gt.shape[2]
    gt_flat = gt.reshape(-1)
    assert gt_flat.shape[0] == rendered_labels.shape[0], (
        f"gt has {gt_flat.shape[0]} pixels, rendered_labels has {rendered_labels.shape[0]} -- "
        f"resolution or row-order mismatch"
    )

    if args.superpixel_pool:
        rendered_labels = superpixel_pool_labels(
            rendered_labels, row_view_ids, args.data_dir, test_idx, height, width,
            n_segments=args.slic_n_segments, compactness=args.slic_compactness,
        )

    mean_iou, matches, matched_fraction = hungarian_miou(rendered_labels, gt_flat, num_clusters)
    matches.sort(key=lambda m: -m["iou"])

    report = {
        "segmentation_file": args.segmentation,
        "superpixel_pool": args.superpixel_pool,
        "num_clusters": num_clusters,
        "num_test_views": len(test_idx),
        "resolution": [height, width],
        "matched_pixel_fraction": matched_fraction,
        "mean_iou": mean_iou,
        "matches": matches,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, indent=2))
    print("\nTop matches (cluster -> gt_class, iou):")
    for m in matches[:num_clusters]:
        print(f"  cluster {m['cluster']:2d} -> class {m['gt_class']:3d}  IoU={m['iou']:.4f}  "
              f"pred_px={m['pred_pixels']}  gt_px={m['gt_pixels']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
