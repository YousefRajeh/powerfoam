"""NormLift reproduced on OUR 3DGS checkpoints -- a strong Gaussian baseline.

Every Gaussian baseline in this project so far has been either per-primitive argmax on the
Splat Feature Solver's solve, or that solve under OpenGaussian's clustering recipe. Neither
reproduces NormLift, which is the method we are actually trying to beat (35.77/39.62/48.93
mIoU on the 10-scene ScanNet protocol). This script implements NormLift's own pipeline on
our checkpoints so the comparison is against the method, not against a weaker stand-in.

WHY THIS IS FAITHFUL RATHER THAN A RE-DERIVATION
------------------------------------------------
NormLift = (a) l2-normalized semantic back-projection as the per-Gaussian lifting solution,
(b) a norm decomposition into intra-view concentration and inter-view agreement, calibrated
by an effective view count, giving a per-Gaussian reliability score (their Eq 6-8), and
(c) a reliability-guided KNN mode-voting refinement that COPIES a neighbour's feature rather
than averaging, so features stay on the CLIP manifold (their Eq 9-10). It classifies each
Gaussian independently -- no clustering.

(b) and (c) are already implemented and unit-tested in this project, on the foam side:
`AccumulatedFeatureStats.reliability()` and `mode_vote_refine` / `build_knn_csr`. This
script only has to produce the same four accumulators from the Gaussian rasterizer and then
call that same code, so the two representations are scored by ONE implementation of
NormLift rather than two.

The four accumulators drop out of splat-distiller's existing inverse_render loop, which
already returns, per view v and per Gaussian j, the weighted feature sum S_j^v = W_j^v * f_j^v
and the weight sum W_j^v (features are l2-normalized per pixel before back-projection):

    numerator[j]           += S_j^v                  (=> ||numerator|| is NormLift's numerator)
    support[j]             += W_j^v
    intra_sum[j]           += ||S_j^v||              (because W*||f|| == ||W*f||, exactly)
    sum_view_weight_sq[j]  += (W_j^v)^2              (=> n_eff = support^2 / this)

so nothing is approximated: `intra_sum` is the exact quantity the reliability formula wants,
obtained without ever materializing per-view features.

NOTE the eval uses nearest-Gaussian-center correspondence with opacity >= 0.1, matching how
every other Gaussian row in our tables assigns GT points, so the only thing that changes
versus those rows is the LIFTING + REFINEMENT, which is the point.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/rajehyl/splat-distiller")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Imports are deferred into the two stage functions on purpose. `distill` pulls in
# gsplat_ext -> pycolmap and only works under gs_train_venv; the eval pulls in warp /
# fpsample / open_clip and only works under the powerfoam env. Stage 1 writes a small
# tensor file, stage 2 reads it, so the two never share a process.


def accumulate(runner, feature_dim=512):
    from feature_foam_lifting.operator import AccumulatedFeatureStats
    """Stream every training view, accumulating NormLift's four sufficient statistics."""
    P = runner.splats.geometry["means"].shape[0]
    dev = runner.device
    numerator = torch.zeros((P, feature_dim), dtype=torch.float32, device=dev)
    support = torch.zeros(P, dtype=torch.float32, device=dev)
    intra_sum = torch.zeros(P, dtype=torch.float32, device=dev)
    wsq = torch.zeros(P, dtype=torch.float32, device=dev)

    for data in tqdm(runner.trainLoader, desc="NormLift back-projection"):
        camtoworlds = data["camtoworld"].to(dev)
        Ks = data["K"].to(dev)
        pixels = data["image"].to(dev) / 255.0
        H, W = pixels.shape[1:3]
        feats = data["features"].to(dev)
        if feats.shape[-1] < 512:
            pad = torch.zeros(list(feats.shape[:-1]) + [512 - feats.shape[-1]],
                              dtype=feats.dtype, device=dev)
            feats = torch.cat([feats, pad], dim=-1)
        feats = feats[..., :512].permute(0, 3, 1, 2)
        feats = torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear",
                                                align_corners=False).permute(0, 2, 3, 1)
        feats = F.normalize(feats, p=2, dim=-1)

        S, Wv, ids = runner.renderer.inverse_render(
            K=Ks, extrinsic=camtoworlds, width=W, height=H, features=feats)
        S = S[:, :feature_dim]
        numerator[ids] += S
        support[ids] += Wv
        intra_sum[ids] += S.norm(dim=-1)      # == W_j^v * ||f_j^v||, exactly
        wsq[ids] += Wv ** 2
        del S, Wv, ids, feats
        torch.cuda.empty_cache()

    stats = AccumulatedFeatureStats.zeros(P, feature_dim, device=dev)
    stats.numerator, stats.support = numerator, support
    stats.intra_sum, stats.sum_view_weight_sq = intra_sum, wsq
    return stats


def stage_accumulate(a):
    from argparser import DataArgs, DistillArgs
    from distill import Runner
    scene = a.scene
    ckpt = a.ckpt or f"{a.ckpt_root}/{scene}/ckpts/ckpt_29999_rank0.pt"
    # mask_folder is required by the dataclass but unused for SAM+CLIP feature loading
    data_args = DataArgs(dir=f"{a.data_root}/{scene}_colmap", factor=1, test_every=100000,
                         feature_folder=f"{a.feature_root}/{scene}/openclip_features_sam",
                         mask_folder="")
    distill_args = DistillArgs(method="3DGS", ckpt=ckpt, quantize=False, filter=-1, tikhonov=1)
    runner = Runner(data_args, distill_args)
    stats = accumulate(runner)
    opac = runner.splats.geometry["opacities"].detach()
    if float(opac.min()) < 0 or float(opac.max()) > 1:
        opac = torch.sigmoid(opac)
    torch.save({"numerator": stats.numerator.cpu(), "support": stats.support.cpu(),
                "intra_sum": stats.intra_sum.cpu(), "sum_view_weight_sq": stats.sum_view_weight_sq.cpu(),
                "means": runner.splats.geometry["means"].detach().cpu(),
                "opacities": opac.cpu(), "ckpt": ckpt},
               a.stats_path)
    print(f"wrote {a.stats_path}")


def stage_eval(a):
    from feature_foam_lifting.operator import AccumulatedFeatureStats
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           remap_gt_labels, load_scannet_pointcept_gt,
                                           calculate_metrics)
    from point_cloud_query import assign_points_to_nearest_center
    from run_cluster_classify_eval import SCENES
    from run_normlift_refine_eval import mode_vote_refine, build_knn_csr
    from eval_semantic_surface import semantic_surface_metrics

    dev = "cuda"
    d = torch.load(a.stats_path, map_location=dev, weights_only=False)
    P, D = d["numerator"].shape
    stats = AccumulatedFeatureStats.zeros(P, D, device=dev)
    stats.numerator = d["numerator"].to(dev)
    stats.support = d["support"].to(dev)
    stats.intra_sum = d["intra_sum"].to(dev)
    stats.sum_view_weight_sq = d["sum_view_weight_sq"].to(dev)
    means, opac = d["means"].to(dev), d["opacities"].to(dev)
    scene = a.scene

    R = stats.reliability()["reliability"].float()
    valid = stats.support > 0
    unit = torch.zeros_like(stats.numerator)
    unit[valid] = F.normalize(stats.numerator[valid], dim=-1)
    keep = (opac.squeeze() >= a.opacity_threshold) & valid
    R = R * keep
    positions = means.float()
    adjacent, offsets = build_knn_csr(positions, keep, K=a.knn)
    refined = unit
    for _ in range(a.passes):
        refined = mode_vote_refine(refined, R, positions, adjacent, offsets)

    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{split}/{scene}", "segment20")
    assigned = assign_points_to_nearest_center(gt_points, means.cpu().numpy(),
                                               valid=keep.cpu().numpy())
    owned = assigned >= 0
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    out = {}
    for cs in ["opengaussian19", "opengaussian15", "opengaussian10"]:
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = remap_gt_labels(raw_labels, tids)
        text = embed_class_names(tnames, dev)
        for tag, u in (("base", unit), ("refined", refined)):
            cls = (u @ text.T).argmax(-1).cpu().numpy()
            pred = np.zeros(gt_points.shape[0], dtype=np.int64)
            pred[owned] = cls[assigned[owned]] + 1
            ncls = len(tids) + 1
            _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_t).long(),
                                                 torch.from_numpy(pred).long(), ncls)
            m = semantic_surface_metrics(gt_points, gt_t, pred, ncls, tau=a.tau)
            m["mIoU"], m["mAcc"] = float(miou), float(macc)
            out[f"{cs}|{tag}"] = m
            print(f"  {scene} {cs} [{tag}]: mIoU={miou*100:.2f} mAcc={macc*100:.2f} "
                  f"semCD={m['scd']*100:.2f}cm missed={m['n_missed']}", flush=True)
    with open(a.output, "w") as f:
        json.dump({"scene": scene, "method": "normlift", "knn": a.knn,
                   "passes": a.passes, "results": out}, f, indent=2)
    print("wrote", a.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["accumulate", "eval"], required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data-root", default="/home/rajehyl/powerfoam/data/scannet")
    p.add_argument("--feature-root", default="/home/rajehyl/powerfoam/artifacts/scannet")
    p.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    p.add_argument("--ckpt-root", default="/home/rajehyl/gaussian_baseline_scannet")
    p.add_argument("--knn", type=int, default=30, help="NormLift uses KNN-30 neighbours")
    p.add_argument("--passes", type=int, default=1, help="refinement passes (their Eq 9-10)")
    p.add_argument("--opacity-threshold", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    if a.stage == "accumulate":
        stage_accumulate(a)
    else:
        stage_eval(a)


if __name__ == "__main__":
    main()
