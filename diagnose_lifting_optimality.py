"""Is our LIFTING optimal, or does averaging push features off the CLIP manifold?

The linear probe (92%) says the class information survives lifting. That is NOT the same as
saying the lifted feature is well-posed for a cosine comparison against TEXT. NormLift's own
Figure 2 measures a 33.6% semantic-drift rate when linearly interpolating between two ScanNet
class features: the argmax wanders to unrelated classes in the mid-range. So an AVERAGED
feature can stay linearly separable while leaving the CLIP image manifold, and cosine-to-text
is only calibrated ON that manifold. That would explain probe 92% alongside text 46-55%
with the true class at rank 2.0.

If that is what is happening, the fix is on the LIFTING side after all -- and a power diagram
is structurally better placed to exploit it than Gaussians are. Each ray crosses few DISJOINT
cells, so a cell tends to be observed through a consistent SAM mask across views, which means
we can select a single OBSERVED (hence on-manifold) CLIP embedding instead of averaging
several. Gaussians cannot: many overlapping primitives contribute to every pixel, so their
per-primitive feature is a blend by construction.

This streams the views once and, per cell, compares:

  mean        the weighted mean we currently solve for. Off-manifold by construction: a
              convex combination of unit vectors is not a unit vector, and its DIRECTION is
              not an observed CLIP embedding.
  argmax_w    the single highest-weight observed feature. ON-manifold: it IS a CLIP mask
              embedding. This is the cheapest possible "copy, do not average" rule and the
              direct analogue of NormLift's mode-voting, applied across VIEWS rather than
              across spatial neighbours.
  medoid      among the top-K contributing features, the one with the highest total cosine
              similarity to the others. Still on-manifold, but robust to a single bad view
              in a way argmax_w is not.
  oracle_obs  the contributing feature that classifies CORRECTLY, if any exists. This is the
              ceiling for ANY select-one-observed-feature rule and the number that decides
              whether this direction is worth pursuing at all.

Also reported: how many DISTINCT SAM masks contribute to a cell and the weight entropy --
the direct measure of the user's structural claim. Low mask-diversity means selection is
well-defined and the power diagram's disjointness is doing real work.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
import configargparse

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.feature_operator import accumulate_feature_stats_for_views  # noqa: F401
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--topk", type=int, default=6, help="observed features kept per cell")
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    scene = a.scene
    ckpt = f"output/scannet_{scene}_{a.variant}"
    feat_dir = Path(f"artifacts/scannet/{scene}/openclip_features_sam")

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    # Feature filenames follow the SAME sorted-stem convention accumulate_feature_stats_sam.py
    # uses (colmap.py's "all" split preserves sorted-name order). Guessing str(view_index)
    # instead silently loaded ZERO feature maps for almost every view -- the loader returns
    # zeros for a missing file rather than raising -- which produced a diagnostic whose
    # view-consistency was exactly 0.0 and whose mean feature norm was 0.005. Assert the
    # count matches so a mismatch fails loudly instead of quietly zeroing the experiment.
    images_dir = Path(args.data_path) / args.scene / "images"
    image_stems = sorted(p.stem for p in images_dir.iterdir())
    assert len(image_stems) == len(cameras), f"{len(image_stems)} images vs {len(cameras)} cameras"
    P = model.points.shape[0]
    K = a.topk
    D = 512

    # per-cell top-K observed features by weight, plus running mean
    top_w = torch.zeros(P, K, device=device)
    top_f = torch.zeros(P, K, D, device=device)
    top_p = torch.zeros(P, K, device=device)   # mask purity of each kept (cell, view)
    mean_num = torch.zeros(P, D, device=device)
    mean_den = torch.zeros(P, device=device)
    # purity-weighted mean: same accumulation, each view's contribution scaled by how
    # cleanly that cell sat inside ONE SAM mask in that view
    pmean_num = torch.zeros(P, D, device=device)
    pmean_den = torch.zeros(P, device=device)
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    n_used = 0

    print(f"[{scene}] {P} cells, streaming {n_views} views, keeping top-{K} observed features")
    for vi_ in range(n_views):
        cam = cameras[vi_]
        H, W = int(cam.height), int(cam.width)
        stem = image_stems[vi_]
        if not (feat_dir / f"{stem}_f.npy").exists():
            print(f"  [skip] no feature for stem {stem}", flush=True)
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stem, H, W, sam_level=3)
        if float(fmap.abs().max()) == 0.0:
            print(f"  [skip] all-zero feature map for {stem}", flush=True)
            continue
        n_used += 1
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = H * W
        slots_used = slots.reshape(-1)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.arange(npix, device=device).repeat_interleave(64)[keep]
        f_pix = fmap.reshape(-1, D)

        # per-cell weight and weighted feature for THIS view.
        # CHUNKED over nonzeros: `vals[:,None] * f_pix[rows]` materializes an (nnz, 512)
        # tensor, and a 1.2M-pixel view with up to 64 hits/pixel gives nnz ~ 15M -> 30GB.
        # Same trap as the facet-edge gather; 1M-entry chunks keep the peak near 2GB.
        w_cell = torch.zeros(P, device=device).index_add_(0, cols, vals)
        f_cell = torch.zeros(P, D, device=device)
        CH = 1_000_000
        for s0 in range(0, cols.numel(), CH):
            e0 = min(s0 + CH, cols.numel())
            f_cell.index_add_(0, cols[s0:e0], vals[s0:e0, None] * f_pix[rows[s0:e0]])
        seen = w_cell > 0
        f_view = torch.zeros_like(f_cell)
        f_view[seen] = F.normalize(f_cell[seen], dim=-1)   # this view's observed direction

        mean_num += f_cell
        mean_den += w_cell

        # ---- per-(cell, view) SAM mask purity -------------------------------------
        # A cell's feature in THIS view is a weight-average over the pixels whose rays
        # crossed it. If those pixels all lie in ONE SAM mask, the resulting vector is a
        # single mask's CLIP embedding -- on-manifold and unambiguous. If they straddle a
        # mask boundary it is a blend of two objects' embeddings, which is exactly the
        # off-manifold contamination the whole lifting stage suffers from. Purity measures
        # which case this is, per (cell, view), and it is only computable because the power
        # diagram gives EXACT cell membership for every pixel's ray -- a Gaussian method has
        # no disjoint ownership to bin by, so this signal has no 3DGS analogue.
        #
        # purity(cell) = max_m W[cell, m] / sum_{m>=1} W[cell, m], over NON-background masks.
        # Background (id 0) is excluded from both sides: its embedding is the zero row, so it
        # contributes nothing to the feature and counting it would reward cells that mostly
        # saw nothing.
        seg = np.load(feat_dir / f"{stem}_s.npy")
        seg_l3 = torch.from_numpy(seg[3]).to(device).to(torch.long) + 1   # 0 = background
        M = int(seg_l3.max()) + 1
        seg_flat = seg_l3.reshape(-1)
        mask_of_nz = seg_flat[rows]
        fg = mask_of_nz > 0
        Wcm = torch.zeros(P * M, device=device)
        Wcm.index_add_(0, cols[fg] * M + mask_of_nz[fg], vals[fg])
        Wcm = Wcm.view(P, M)
        fg_tot = Wcm.sum(1)
        purity = torch.zeros(P, device=device)
        okp = fg_tot > 0
        purity[okp] = Wcm[okp].max(1).values / fg_tot[okp]
        del Wcm

        pmean_num += f_cell * purity[:, None]
        pmean_den += w_cell * purity

        # insert into the per-cell top-K by weight
        worst = top_w.argmin(dim=1)
        wv = top_w.gather(1, worst[:, None]).squeeze(1)
        better = seen & (w_cell > wv)
        if bool(better.any()):
            idx = torch.where(better)[0]
            slot = worst[idx]
            top_w[idx, slot] = w_cell[idx]
            top_f[idx, slot] = f_view[idx]
            top_p[idx, slot] = purity[idx]
        if (vi_ + 1) % 20 == 0:
            print(f"  view {vi_+1}/{n_views}", flush=True)
        del out_col, out_val, fmap, f_pix
        torch.cuda.empty_cache()

    valid = mean_den > 0
    unit_mean = torch.zeros(P, D, device=device)
    unit_mean[valid] = F.normalize(mean_num[valid], dim=-1)
    # purity-weighted mean: identical accumulation, views scaled by mask purity. Cells that
    # never sat cleanly inside a mask in ANY view fall back to the plain mean rather than
    # being dropped -- the surface-truncation ablation showed that losing coverage costs far
    # more than the contamination it removes (valid-cell coverage 90.2% -> 70.1% cost -2.74
    # mIoU), so no rule here is allowed to shrink the observed set.
    pvalid = pmean_den > 0
    unit_pmean = unit_mean.clone()
    unit_pmean[pvalid] = F.normalize(pmean_num[pvalid], dim=-1)
    # compute the agreement statistic now and release the raw accumulators: on the large
    # scenes every (P, 512) tensor is ~2GB and half a dozen are otherwise live at once
    mean_norm_stat = float(mean_num[valid].norm(dim=-1).div(mean_den[valid]).median())
    del mean_num, pmean_num, pmean_den
    torch.cuda.empty_cache()

    # ---- structural stats: how consistent are the observed features per cell? ----
    nz = (top_w > 0)
    n_obs = nz.sum(1)
    wn = top_w / top_w.sum(1, keepdim=True).clamp_min(1e-9)
    ent = -(wn.clamp_min(1e-9).log() * wn).sum(1)
    # pairwise cosine among a cell's observed features (how consistent the views are).
    #
    # CHUNKED over cells. `top_f[m2]` is a boolean-mask gather that materializes a COPY of
    # (n_selected, K, 512) -- on the large scenes (P ~ 1M after densification) that is 10-12GB
    # and OOM'd a 48GB card, because argmax_f/med_f/top_f/top_p are all live at the same time.
    # This is the same (N, 512) gather trap that cost 37GB on facet edges and 30GB on the
    # lifting gather earlier in this project; the fix is identical, and the arithmetic is
    # unchanged since every cell's Gram matrix is independent of every other cell's.
    CELL_CH = 100_000
    cons = torch.zeros(P, device=device)
    m2 = n_obs >= 2
    m2_idx = torch.where(m2)[0]
    eye = torch.eye(K, device=device)[None]

    def _gram_blocks():
        for s in range(0, m2_idx.numel(), CELL_CH):
            idx = m2_idx[s:s + CELL_CH]
            tf = top_f[idx]                       # (chunk, K, D)
            msk = nz[idx].float()
            yield idx, tf, msk, tf @ tf.transpose(1, 2)

    for idx, tf, msk, G in _gram_blocks():
        pair = msk[:, :, None] * msk[:, None, :] * (1 - eye)
        cons[idx] = (G * pair).sum((1, 2)) / pair.sum((1, 2)).clamp_min(1)
        del tf, G, pair

    # ---- selection rules ----
    argmax_f = top_f.gather(1, top_w.argmax(1)[:, None, None].expand(P, 1, D)).squeeze(1)
    # medoid among observed: highest total similarity to the others
    med_f = argmax_f.clone()
    for idx, tf, msk, G in _gram_blocks():
        tot = (G * (msk[:, :, None] * msk[:, None, :])).sum(2) - 1.0
        tot = tot.masked_fill(nz[idx] == 0, -1e9)
        med_f[idx] = tf.gather(1, tot.argmax(1)[:, None, None].expand(tf.shape[0], 1, D)).squeeze(1)
        del tf, G, tot
    # select the observed view in which this cell sat most cleanly inside one SAM mask,
    # rather than the one that happened to contribute the most rendering weight
    pk = top_p.masked_fill(~nz, -1.0)
    argmax_p = top_f.gather(1, pk.argmax(1)[:, None, None].expand(P, 1, D)).squeeze(1)

    # ---- score each rule ----
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{split}/{scene}", "segment20")
    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    vmask = valid.cpu().numpy()
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vmask, k=64)
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    tids, tnames = [i for i, _ in kept], [n for _, n in kept]
    gt_t = remap_gt_labels(raw_labels, tids)
    n_classes = len(tids) + 1
    text = embed_class_names(tnames, device)

    def score(feat):
        cls = (feat @ text.T).argmax(-1).cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = cls[assigned[owned]] + 1
        _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_t).long(),
                                         torch.from_numpy(pred).long(), n_classes)
        return float(mi), float(ma)

    # oracle over the observed features: is ANY of the top-K correct?
    cell_gt = np.zeros(P, dtype=np.int64)
    cnt = np.zeros((P, n_classes), dtype=np.int64)
    np.add.at(cnt, (assigned[owned], gt_t[owned]), 1)
    cnt[:, 0] = 0
    cell_gt = cnt.argmax(1)
    has_gt = cnt.sum(1) > 0
    sims_all = torch.einsum("pkd,cd->pkc", top_f, text)
    pred_all = sims_all.argmax(-1).cpu().numpy() + 1          # (P, K)
    ok_any = ((pred_all == cell_gt[:, None]) & nz.cpu().numpy()).any(1)
    oracle_frac = float(ok_any[has_gt].mean())
    mean_ok = float((( (unit_mean @ text.T).argmax(-1).cpu().numpy() + 1) == cell_gt)[has_gt].mean())
    amax_ok = float((( (argmax_f @ text.T).argmax(-1).cpu().numpy() + 1) == cell_gt)[has_gt].mean())
    med_ok = float((( (med_f @ text.T).argmax(-1).cpu().numpy() + 1) == cell_gt)[has_gt].mean())

    # ---- LABEL-SPACE voting (N3 "view quorum") --------------------------------------
    # Every rule above aggregates in FEATURE space and classifies once at the end, so a cell
    # whose views disagree gets a blended vector that need not lie near any real mask
    # embedding -- CLIP's manifold is not closed under averaging (NormLift measures 33.6%
    # semantic drift under linear interpolation). Voting classifies each view FIRST, where
    # every vector being scored is a genuine single-view mask embedding, and only then
    # combines -- in a discrete space where "blending" is impossible by construction.
    # sims_all/pred_all above already hold each observed view's own classification, so the
    # vote costs nothing beyond a scatter.
    C = text.shape[0]
    votes_w = torch.zeros(P, C, device=device)
    votes_u = torch.zeros(P, C, device=device)
    votes_p = torch.zeros(P, C, device=device)
    pred_k = sims_all.argmax(-1)                       # (P, K) 0-based class per observed view
    nzf = nz.float()
    for k_ in range(K):
        idx = pred_k[:, k_]
        votes_w.scatter_add_(1, idx[:, None], (top_w[:, k_] * nzf[:, k_])[:, None])
        votes_u.scatter_add_(1, idx[:, None], nzf[:, k_][:, None])
        votes_p.scatter_add_(1, idx[:, None], (top_p[:, k_] * nzf[:, k_])[:, None])

    def score_labels(cls0):
        """Score a per-cell 0-based class assignment (not a feature)."""
        cls = cls0.cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = cls[assigned[owned]] + 1
        _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_t).long(),
                                         torch.from_numpy(pred).long(), n_classes)
        acc = float(((cls + 1) == cell_gt)[has_gt].mean())
        return float(mi), float(ma), acc

    print(f"  views actually contributing features: {n_used}/{n_views}")
    res = {"scene": scene, "cells": int(P), "views": n_views, "views_used": n_used,
           "obs_per_cell_median": float(n_obs.float().median()),
           "weight_entropy_median": float(ent[valid].median()),
           "view_consistency_median": float(cons[m2].median()) if bool(m2.any()) else None,
           "mean_norm_before_renorm": mean_norm_stat,
           "cell_acc_mean": mean_ok, "cell_acc_argmax_w": amax_ok,
           "cell_acc_medoid": med_ok, "cell_acc_oracle_observed": oracle_frac}
    for nm, f in (("mean (current)", unit_mean), ("argmax_w (on-manifold)", argmax_f),
                  ("medoid (on-manifold)", med_f),
                  ("purity-weighted mean", unit_pmean),
                  ("argmax_purity (on-manifold)", argmax_p)):
        mi, ma = score(f)
        res[f"mIoU_{nm}"] = mi
        res[f"mAcc_{nm}"] = ma
    res["cell_acc_purity_mean"] = float(
        (((unit_pmean @ text.T).argmax(-1).cpu().numpy() + 1) == cell_gt)[has_gt].mean())
    res["cell_acc_argmax_purity"] = float(
        (((argmax_p @ text.T).argmax(-1).cpu().numpy() + 1) == cell_gt)[has_gt].mean())
    for nm, v in (("vote_weighted", votes_w), ("vote_unweighted", votes_u),
                  ("vote_purity_weighted", votes_p)):
        mi, ma, acc = score_labels(v.argmax(-1))
        res[f"mIoU_{nm}"] = mi
        res[f"mAcc_{nm}"] = ma
        res[f"cell_acc_{nm}"] = acc
    res["purity_median"] = float(top_p[nz].median()) if bool(nz.any()) else None
    res["purity_frac_below_0.9"] = float((top_p[nz] < 0.9).float().mean()) if bool(nz.any()) else None

    print(f"\n=== {scene}: lifting optimality ===")
    print(f"  observed features per cell (median)   {res['obs_per_cell_median']:.0f} of {K}")
    print(f"  view consistency (mean pairwise cos)  {res['view_consistency_median']}")
    print(f"  ||weighted mean|| / total weight       {res['mean_norm_before_renorm']:.4f}"
          f"   (1.0 = all views agreed exactly)")
    print(f"\n  per-cell accuracy against the cell's majority GT label:")
    print(f"    mean (what we do)                   {mean_ok*100:6.2f}%")
    print(f"    argmax-weight observed feature      {amax_ok*100:6.2f}%")
    print(f"    medoid of observed features         {med_ok*100:6.2f}%")
    print(f"    ORACLE over observed features       {oracle_frac*100:6.2f}%   <- ceiling for "
          f"any select-one rule")
    print(f"    purity-weighted mean                {res['cell_acc_purity_mean']*100:6.2f}%")
    print(f"    argmax-purity observed feature      {res['cell_acc_argmax_purity']*100:6.2f}%")
    print(f"    vote: weight-weighted               {res['cell_acc_vote_weighted']*100:6.2f}%")
    print(f"    vote: unweighted majority           {res['cell_acc_vote_unweighted']*100:6.2f}%")
    print(f"    vote: purity-weighted               {res['cell_acc_vote_purity_weighted']*100:6.2f}%")
    print(f"\n  mask purity of kept (cell,view) pairs: median {res['purity_median']:.4f}, "
          f"{res['purity_frac_below_0.9']*100:.1f}% below 0.9")
    print(f"\n  mIoU:")
    for nm in ("mean (current)", "argmax_w (on-manifold)", "medoid (on-manifold)",
               "purity-weighted mean", "argmax_purity (on-manifold)",
               "vote_weighted", "vote_unweighted", "vote_purity_weighted"):
        base = res["mIoU_mean (current)"]
        d = "" if nm == "mean (current)" else f"   ({(res['mIoU_'+nm]-base)*100:+.2f})"
        print(f"    {nm:<34} {res['mIoU_'+nm]*100:6.2f}{d}")
    if a.output:
        json.dump(res, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
