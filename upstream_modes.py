"""Decompose the upstream error U into named, separately-measurable causes.

The oracle replaces the whole image->feature pipeline with a perfect per-pixel class, and scores
~50 mIoU above the real pipeline. That 50 points is `U`, and quoting it as one number says only
"the upstream is bad". These modes take the real upstream apart one stage at a time, so the parts
can be attributed and a bound can carry a measured upstream term instead of a fitted constant.

The real chain is:   pixel -> SAM region -> CLIP feature of that region -> class

so substituting one stage at a time gives a ladder, each rung adding exactly one real component:

  official   per-pixel GT class                          the oracle: no upstream at all
  sam_gt     region -> MAJORITY GT CLASS of that region  + SAM's region geometry
  sam_clip   region -> argmax of that region's CLIP feat + CLIP's labelling of the region
  (real)     the full 512-d feature, not its argmax      + feature softness

Differences between consecutive rungs are the per-stage costs. Two properties matter:

  * every rung is a per-pixel CLASS image, so the class-space reduction (B = S T) still applies and
    all four are measured by the identical scorer -- the rungs differ only in what feeds `cls_img`.
  * region id -1 means SAM produced NO region for that pixel. Those pixels deposit no evidence at
    all, which is a distinct failure from labelling one wrongly, so they are counted separately
    rather than folded into the region-geometry term.

A region's "majority GT class" ignores unlabelled pixels: a region straddling labelled and
unlabelled ground is charged only for the labelled part it gets wrong.
"""
from __future__ import annotations
import os

import numpy as np
import torch

FEAT_DIRNAME = "language_features"   # the level-3 variant: s is (1, H, W), f is (n_regions, 512)


def _load_sam(scene, stem, H, W, feat_dir=FEAT_DIRNAME, level=0):
    """-> (region_id map flattened to (H*W,), region features (n_regions, 512))."""
    base = os.path.join("data", "scannet", f"{scene}_colmap", feat_dir)
    s = np.load(os.path.join(base, f"{stem}_s.npy"))
    f = np.load(os.path.join(base, f"{stem}_f.npy")).astype(np.float32)
    if s.ndim == 3:
        s = s[level]
    if s.shape != (H, W):
        from PIL import Image
        s = np.array(Image.fromarray(s.astype(np.int32)).resize((W, H), Image.NEAREST))
    return s.reshape(-1).astype(np.int64), f


def sam_region_stats(s_flat, gt_flat, n_reg, C):
    """Per-region histogram of GT classes (labelled pixels only). -> (n_reg, C) float64."""
    m = (s_flat >= 0) & (gt_flat > 0)
    if not m.any():
        return np.zeros((n_reg, C), np.float64)
    idx = s_flat[m] * C + (gt_flat[m] - 1)
    return np.bincount(idx, minlength=n_reg * C).reshape(n_reg, C).astype(np.float64)


def sam_label_image(scene, stem, H, W, gt_flat, C, dev, mode, TT=None,
                    feat_dir=FEAT_DIRNAME, level=0):
    """Per-pixel class image with ONE upstream stage substituted in.

    mode 'sam_gt'   : each SAM region emits the majority GT class of its own labelled pixels.
                      Everything CLIP does is replaced by an oracle, so the only loss relative to
                      the per-pixel oracle is SAM's region geometry (plus unassigned pixels).
    mode 'sam_clip' : each SAM region emits argmax over the text head of its own CLIP feature.
                      Adds exactly one thing to 'sam_gt': whether CLIP names the region correctly.

    Returns (cls_img_flat LongTensor on dev, stats dict).
    """
    s_flat, f = _load_sam(scene, stem, H, W, feat_dir, level)
    n_reg = int(f.shape[0])
    # ids beyond the feature table cannot be resolved; treat as unassigned rather than crash
    s_flat = np.where(s_flat >= n_reg, -1, s_flat)
    unassigned = s_flat < 0

    occ = sam_region_stats(s_flat, gt_flat, n_reg, C)
    reg_has = occ.sum(1) > 0
    reg_gt = np.where(reg_has, occ.argmax(1) + 1, 0)          # majority GT class per region

    if mode == "sam_gt":
        reg_cls = reg_gt
    elif mode == "sam_clip":
        ft = torch.from_numpy(f).to(dev)
        ft = ft / ft.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        reg_cls = (ft @ TT.T).argmax(1).cpu().numpy() + 1     # TT rows already normalised
        reg_cls = np.where(reg_has | True, reg_cls, 0)        # CLIP labels every region it produced
    else:
        raise ValueError(mode)

    cls = np.where(unassigned, 0, reg_cls[np.clip(s_flat, 0, n_reg - 1)]).astype(np.int64)

    lab = gt_flat > 0
    cov = lab & ~unassigned
    stats = {
        "px_labelled": int(lab.sum()),
        "px_unassigned_of_labelled": int((lab & unassigned).sum()),
        # region purity: share of labelled pixels whose GT class equals their region's majority.
        # 1 - this is the geometry cost SAM's boundaries impose, independent of naming.
        "region_purity": float((occ.max(1).sum() / occ.sum()) if occ.sum() else float("nan")),
        "n_regions": n_reg,
        "n_regions_with_gt": int(reg_has.sum()),
        # how often CLIP names a region the way its own pixels are labelled (region-weighted)
        "clip_region_acc": (float((reg_cls[reg_has] == reg_gt[reg_has]).mean())
                            if mode == "sam_clip" and reg_has.any() else float("nan")),
        # the same, weighted by pixels -- large regions matter more to the lift
        "clip_pixel_acc": (float(((cls == gt_flat) & cov).sum() / max(int(cov.sum()), 1))
                           if mode == "sam_clip" else float("nan")),
        "gt_pixel_acc_ceiling": float(((np.where(unassigned, 0, reg_gt[np.clip(s_flat, 0, n_reg - 1)])
                                        == gt_flat) & cov).sum() / max(int(cov.sum()), 1)),
    }
    return torch.from_numpy(cls).to(dev), stats
