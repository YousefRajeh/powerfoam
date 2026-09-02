"""Run the baselines we can actually run, and score them on point AND surface metrics.

WHY ONLY TWO BASELINES. Table~\\ref{tab:3dseg} cites point mIoU/mAcc for nine methods from the
NormLift paper, but no prior work reports surface metrics, so those columns can only be filled for
methods we execute ourselves -- they need per-point predictions, not a published summary. Of the
nine, exactly two are executable here:

  SFS       we hold their lifted features (recon_remote/*/ckpt_features.pt, both arms x 10 scenes)
            and their contrastive relevancy readout, transcribed and verified against
            pre_processing.py::get_relevancy in test_relevancy.py.
  NormLift  post-lifting only: c_i = ||f_i|| * N_eff/(N_eff+1) then confidence-weighted KNN at
            K=36. Their N_eff = (sum_s W_i^s)^2 / sum_s (W_i^s)^2 is exactly the Kish effective
            sample size our accumulator already computes.

The other seven (LangSplat, OpenGaussian, LAGA, THGS, VALA, Occam's LGS, LUDVIG) each need their own
repository, training run and checkpoints. Their point metrics stay as cited; their surface cells stay
dashed. That asymmetry is stated in the table caption rather than hidden.

NORMLIFT NEEDS A PREREQUISITE WE DO NOT HAVE ON SCANNET. N_eff requires per-view weights, i.e. the
accumulator stats. `artifacts/scannet/*/stats_*.pt` exist for foam only; there is no stats_gs_*, so
the NormLift arm is skipped until `--accumulate` has been run. Running NormLift on foam instead
would be the category error recorded in OPEN_ISSUES section M -- it is a Gaussian method.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       load_scannet_pointcept_gt, remap_gt_labels)
from relevancy import embed_negatives, relevancy_scores
from run_spp_gs_eval import mahalanobis_assign

SPLIT = {"scene0645_00": "val"}      # every other scene is in train
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
OUT = "artifacts/scannet/baselines"
NORMLIFT_K = 36          # their knn_vote.py: "K = 36"
NORMLIFT_BETA = 1.0      # their get_confidence.py: "beta = 1"


def point_metrics(pred, gt, n_classes):
    """Per-class IoU/Acc averaged over classes PRESENT in this scene's GT, class 0 ignored.

    Transcribed from NormLift's eval_scannet.py::calculate_metrics so our rows are scored by the
    same rule as the published ones.
    """
    ious, accs = [], []
    valid = gt > 0
    for c in range(1, n_classes):
        g = (gt == c) & valid
        if not g.any():
            continue
        p = (pred == c) & valid
        inter = float((p & g).sum())
        union = float((p | g).sum())
        ious.append(inter / union if union > 0 else 0.0)
        accs.append(inter / float(g.sum()))
    if not ious:
        return 0.0, 0.0
    return 100.0 * float(np.mean(ious)), 100.0 * float(np.mean(accs))


def load_gs(scene, arm, device="cuda"):
    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=device, weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    means = sp["means"].double().cpu().numpy()
    scales = torch.exp(sp["scales"]).double().cpu().numpy()
    q = sp["quats"].double()
    q = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    feat = torch.load(f"recon_remote/{arm}/{scene}/ckpt_features.pt",
                      map_location=device, weights_only=False).float()
    return means, scales, q, feat


def labels_sfs(feat, txt, device):
    """Splat Feature Solver's readout: contrastive relevancy against canonical negatives."""
    neg = embed_negatives(device)
    vm = feat.norm(dim=-1) > 0
    u = torch.zeros_like(feat)
    u[vm] = F.normalize(feat[vm], dim=-1)
    rel = relevancy_scores(u, txt, neg)
    lab = rel.argmax(-1) + 1
    lab[~vm] = 0
    return lab.cpu().numpy()


def labels_normlift(feat, txt, means, stats_path, device):
    """NormLift: confidence-weighted KNN vote over the per-primitive argmax.

    c_i = ||f_i|| * N_eff/(N_eff + beta); labels are then decided by a K=36 neighbour vote weighted
    by c. Returns None when the accumulator stats that supply N_eff are absent.
    """
    if not os.path.exists(stats_path):
        return None
    from feature_foam_lifting.operator import AccumulatedFeatureStats
    st = AccumulatedFeatureStats.load(stats_path, device=device)
    n_eff = st.reliability()["n_eff"].to(device).float()
    del st
    norm = feat.norm(dim=-1)
    conf = norm * (n_eff / (n_eff + NORMLIFT_BETA))
    vm = norm > 0
    u = torch.zeros_like(feat)
    u[vm] = F.normalize(feat[vm], dim=-1)
    base = (u @ txt.T).argmax(-1) + 1
    base[~vm] = 0

    tree = cKDTree(means)
    _, idx = tree.query(means, k=min(NORMLIFT_K, means.shape[0]), workers=-1)
    idx = torch.from_numpy(np.ascontiguousarray(idx)).long().to(device)
    C = txt.shape[0] + 1
    votes = torch.zeros((means.shape[0], C), device=device)
    nb_lab = base[idx]                     # (P, K)
    nb_w = conf[idx]
    votes.scatter_add_(1, nb_lab, nb_w)
    votes[:, 0] = -1.0                     # never vote for "ignore"
    out = votes.argmax(1)
    out[~vm] = 0
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gs_unfroz,gs_froz")
    ap.add_argument("--methods", default="sfs,normlift")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    names = OPENGAUSSIAN_CLASS_SETS[a.class_set]

    for scene in a.scenes.split(","):
        split = SPLIT.get(scene, "train")
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{split}\{scene}", "segment20")
        n2i = {n: i for i, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())
        kept = [(n2i[n], n) for n in names if n in n2i and n2i[n] in present]
        if not kept:
            print(f"[skip] {scene}: no target classes present", flush=True)
            continue
        gt = remap_gt_labels(raw, [i for i, _ in kept])          # 0 = ignore, 1..C
        n_cls = len(kept) + 1
        txt = embed_class_names([n for _, n in kept], device)
        for arm in a.arms.split(","):
            if not os.path.exists(f"recon_remote/{arm}/{scene}/ckpt_features.pt"):
                print(f"[miss] {arm}/{scene}", flush=True)
                continue
            means, scales, quats, feat = load_gs(scene, arm, device)
            assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
            for meth in a.methods.split(","):
                dst = f"{OUT}/{meth}_{arm}_{scene}_{a.class_set}.json"
                if os.path.exists(dst):
                    print(f"[skip] {meth}/{arm}/{scene}", flush=True)
                    continue
                if meth == "sfs":
                    lab = labels_sfs(feat, txt, device)
                else:
                    lab = labels_normlift(feat, txt, means,
                                          f"artifacts/scannet/{scene}/stats_{arm}_ogl3.pt", device)
                    if lab is None:
                        print(f"[blocked] normlift/{arm}/{scene}: no accumulator stats "
                              f"(N_eff unavailable) -- run the GS accumulation first", flush=True)
                        continue
                pred = lab[assigned]
                miou, macc = point_metrics(pred, gt, n_cls)
                idx = GTSurfaceIndex(gt_pts, gt, n_cls)
                sf = semantic_surface_metrics(idx, pred)
                rec = {"scene": scene, "arm": arm, "method": meth, "class_set": a.class_set,
                       "miou": miou, "macc": macc,
                       "scd": sf.get("scd"), "hd95": sf.get("hd95"),
                       "boundary_f1": sf.get("boundary_f1"), "n_missed": sf.get("n_missed")}
                json.dump(rec, open(dst, "w"), indent=1)
                print(f"[ok] {meth:9s} {arm:10s} {scene}  mIoU={miou:5.2f} mAcc={macc:5.2f} "
                      f"SCD={rec['scd'] or 0:.4f} HD95={rec['hd95'] or 0:.4f} "
                      f"BF1={(rec['boundary_f1'] or 0)*100:5.2f}", flush=True)
            del feat
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
