"""SFS and NormLift on ScanNet++ (12 scenes), scored under the same protocol as our own arms.

WHY AN ADAPTER RATHER THAN A REIMPLEMENTATION. run_spp_gs_eval.py scores OUR stack (A_base..J) and
exposes no entry point that takes an arbitrary feature file. Everything needed is nonetheless
already written and verified there: Mahalanobis assignment (Dr.Splat's -- Gaussians overlap and have
unbounded support, so nearest-centre would handicap the baseline), ScanNet++ GT loading with the
benchmark class folding and the per-scene exclusion mask, and score_pred. This file imports those
verbatim and redirects only the path roots, so no scoring math is duplicated on the baseline side.
That is the project's controlling fairness rule: no reimplemented math on either side.

THE TWO METHODS.
  SFS       artifacts/scannetpp_gs/<scene>/solved_weighted_gs_tikh_ogl3.pt IS Splat Feature Solver:
            produced by DistillArgs(method="3DGS", tikhonov=1), i.e. their own distill.py with
            Tikhonov guidance. Read out with their contrastive relevancy (relevancy.py, transcribed
            from pre_processing.py::get_relevancy and checked in test_relevancy.py), not bare cosine.
  NormLift  post-lifting only, on the gs_unfroz solve: confidence c_i = ||f_i|| * Neff/(Neff+1) with
            Neff the Kish effective sample size from the accumulator stats, then the
            confidence-weighted neighbour vote (mode_vote_refine, the project's own).

If the stats file lacks recognisable weight moments the Neff shrinkage cannot be formed; the run
says so loudly and falls back to ||f|| alone rather than silently scoring a different method.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", default=os.path.expanduser("~/spp_gt_semantic"))
    ap.add_argument("--gs-root",
                    default=os.path.expanduser("~/mnt/share/refbench_3dgs_12scenes/output"))
    ap.add_argument("--art", default=os.path.expanduser("~/mnt/share/artifacts/scannetpp_gs"))
    ap.add_argument("--outdir", default="artifacts/spp_baselines")
    ap.add_argument("--methods", default="sfs,normlift")
    ap.add_argument("--scenes", nargs="*", default=None)
    a = ap.parse_args()

    import run_spp_eval as SE
    SE.GT_ROOT = a.gt_root
    import run_spp_gs_eval as GE
    GE.GS = a.gs_root
    GE.ART = a.art

    from run_spp_gs_eval import load_gaussians, mahalanobis_assign
    from run_spp_eval import benchmark_map, load_gt
    from run_overnight import score_pred, SPP
    from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
    from run_normlift_refine_eval import mode_vote_refine
    from normlift_replication import knn_csr_safe
    from relevancy import embed_negatives, relevancy_scores

    dev = "cuda"
    os.makedirs(a.outdir, exist_ok=True)
    top, raw2bench = benchmark_map()
    scenes = a.scenes or list(SPP)
    methods = a.methods.split(",")
    neg = embed_negatives(dev)

    for scene in scenes:
        outp = os.path.join(a.outdir, scene + ".json")
        if os.path.exists(outp):
            print("[skip] " + scene, flush=True)
            continue
        t0 = time.time()
        try:
            means, scales, quats = load_gaussians(scene)
        except Exception as exc:
            print("[miss] {}: {}".format(scene, exc), flush=True)
            continue

        arms = {}
        if "sfs" in methods:
            p = "{}/{}/solved_weighted_gs_tikh_ogl3.pt".format(a.art, scene)
            if os.path.exists(p):
                sv = torch.load(p, map_location="cpu", weights_only=True)
                arms["SFS"] = (sv["primitive_features"].float(), sv["valid_mask"].numpy(),
                               "relevancy")
            else:
                print("  [miss] SFS " + scene, flush=True)
        if "normlift" in methods:
            p = "{}/{}/solved_weighted_gs_unfroz_ogl3.pt".format(a.art, scene)
            sp = "{}/{}/stats_gs_unfroz_ogl3.pt".format(a.art, scene)
            if os.path.exists(p):
                sv = torch.load(p, map_location="cpu", weights_only=True)
                f0 = sv["primitive_features"].float()
                vmn = sv["valid_mask"].numpy()
                # NormLift's reliability R(j) (their Eq. 6-8) comes from the accumulator's own
                # verified implementation -- reconstructing it from raw moment keys here would be
                # exactly the reimplementation this file exists to avoid, and the key names differ
                # between stats versions anyway (this one stores sum_view_weight_sq / intra_sum,
                # not sum_w / sum_w2).
                if os.path.exists(sp):
                    from feature_foam_lifting.operator import AccumulatedFeatureStats
                    rel = AccumulatedFeatureStats.load(sp).reliability()
                    conf = rel["reliability"].float().reshape(-1).cpu()
                    ne = rel.get("n_eff")
                    print("  [normlift] R median {:.4f}{}".format(
                        float(conf.median()),
                        "" if ne is None else "  n_eff median {:.2f}".format(
                            float(ne.float().median()))), flush=True)
                else:
                    conf = f0.norm(dim=-1)
                    print("  [normlift] WARNING stats absent; confidence = ||f|| only, which is "
                          "NOT NormLift -- treat this scene as unscored", flush=True)
                arms["NormLift"] = (f0, vmn, ("refine", conf))
            else:
                print("  [miss] NormLift " + scene, flush=True)
        if not arms:
            continue

        gt_pts, gt_lab0, n_masked = load_gt(scene, top, raw2bench)
        assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
        res = {"scene": scene, "n_masked": int(n_masked), "assignment": "mahalanobis", "arms": {}}

        for tag in sorted(arms):
            feats, vmn, readout = arms[tag]
            P = feats.shape[0]
            if means.shape[0] != P:
                print("  [skip] {} {}: P {} vs {}".format(tag, scene, P, means.shape[0]), flush=True)
                continue
            asg = np.where(vmn[assigned], assigned, -1)
            owned = asg >= 0
            X = feats.to(dev)
            vm = torch.from_numpy(vmn).to(dev)
            u = torch.zeros_like(X)
            u[vm] = F.normalize(X[vm], dim=-1)
            if isinstance(readout, tuple) and readout[0] == "refine":
                pos = torch.from_numpy(means).to(dev).float()
                adj, off = knn_csr_safe(pos, vm, K=30)
                dm = int((off[1:] - off[:-1]).max()) + 1
                u = mode_vote_refine(u, readout[1].to(dev) * vm, pos, adj, off,
                                     chunk=max(256, 200000 // max(dm, 1)))
                del adj, off, pos
            lab_owned = np.where(owned, gt_lab0, -1)
            for K in (100, 50, 20):
                present = sorted(set(np.unique(lab_owned).tolist()) & set(range(K)))
                if not present:
                    continue
                nm = [top[:K][i] for i in present]
                gt_t = torch.from_numpy(remap_gt_labels(lab_owned, present)).long()
                txt = embed_class_names(nm, dev)
                C = len(nm)
                sc = torch.zeros(P, C, device=dev)
                if readout == "relevancy":
                    sc[vm] = relevancy_scores(u[vm], txt, neg)
                else:
                    sc[vm] = u[vm] @ txt.T
                mi, ma = score_pred(sc.argmax(-1).cpu().numpy(), asg, owned, gt_t, C,
                                    gt_pts.shape[0])
                res["arms"].setdefault(tag, {})["spp_top{}".format(K)] = {
                    "mIoU": mi, "mAcc": ma, "n_classes": C, "coverage": float(owned.mean())}
                print("  {} {:9s} top{:<4d} mIoU {:6.2f}  mAcc {:6.2f}  C={}".format(
                    scene, tag, K, mi, ma, C), flush=True)
                del txt, sc
            del X, u, vm
            torch.cuda.empty_cache()
        json.dump(res, open(outp, "w"), indent=1)
        print("[ok] {} {:.0f}s".format(scene, time.time() - t0), flush=True)
    print("SPP_BASELINES DONE")


if __name__ == "__main__":
    from determinism import enable_determinism
    enable_determinism()
    main()
