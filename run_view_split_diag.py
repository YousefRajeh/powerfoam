"""VIEW SPLITTING: how much of our error is view-INCONSISTENT, and how much is systematic?

WHERE THIS COMES FROM. The distractor paper (arXiv 2608.26951) filters transient content by
verification-by-removal: exclude a subset's contribution, re-render, and keep the removal only if it
IMPROVES agreement with evidence that did not generate it. That principle -- test a hypothesis
against held-out evidence -- is exactly what our pseudo-label attempt lacked, and why it failed
(labels drawn from the very features they were meant to correct, gate capture 0.145 vs 0.195).

The clean analogue here is a SPLIT-HALF solve. Accumulate the feature field twice over DISJOINT view
subsets, solve each independently, and compare per-cell predictions:

  * cells whose class FLIPS between halves are view-inconsistent -- their answer depends on which
    views happened to see them, which is the failure mode any consistency method can attack;
  * cells that AGREE across halves are systematically determined -- every view says the same thing,
    so no view-consistency method (theirs, ours, or anyone's) can ever fix them.

WHY IT MATTERS EVEN IF NEGATIVE. This bounds what the entire family can recover. n_eff is ~35
effective views per cell and the errors are stably wrong rather than noisily wrong, so the
prediction is that wrong cells will AGREE across halves. If that holds, view-consistency filtering
is closed for this data and we can say so with a number instead of an argument.

TWO STAGES, TWO INTERPRETERS (same split as run_attribution_diag.py):
    D:/conda/envs/powerfoam/python.exe run_view_split_diag.py --stage accumulate --scene X
    python run_view_split_diag.py --stage analyze --scene X
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

RECON = os.path.join("D:" + os.sep, "Downloads", "spp_results", "full")
FEAT_ROOT = Path("D:" + os.sep) / "Downloads" / "spp_data_1600"


def log(m):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


# ------------------------------------------------------------------ stage 1: accumulate (powerfoam)

def stage_accumulate(scene, device="cuda"):
    import configargparse
    import torch.nn.functional as F
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import accumulate_feature_stats_for_views

    wp.init()
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ck}/model.pt")
    images_dir = Path(args.data_path) / args.scene / "images"
    image_names = sorted(p_.stem for p_ in images_dir.iterdir())
    folder = FEAT_ROOT / scene / "openclip_features_sam_l3"

    def loader(view_id):
        """Identical to accumulate_feature_stats_sam.py: per-pixel SAM-mask CLIP embedding."""
        stem = image_names[view_id]
        fp, sp = folder / f"{stem}_f.npy", folder / f"{stem}_s.npy"
        if not fp.exists():
            return torch.zeros(1066, 1600, 512, device=device)
        feat = torch.from_numpy(np.load(fp)).to(device).float()
        seg = torch.from_numpy(np.load(sp)).to(device).long() + 1
        pad = torch.cat([torch.zeros(1, feat.shape[1], device=device), feat], 0)
        fm = F.embedding(seg, pad).sum(0)
        return fm / (fm.norm(dim=-1, keepdim=True) + 1e-6)

    n = len(dh.cameras)
    # INTERLEAVED, not contiguous: a contiguous split would confound "different views" with
    # "different part of the room", and every cell would simply be unobserved in one half.
    halves = {"A": list(range(0, n, 2)), "B": list(range(1, n, 2))}
    for tag, vids in halves.items():
        log(f"  {scene} half {tag}: {len(vids)} views")
        st = accumulate_feature_stats_for_views(model, dh.cameras, vids, loader, batch_size=1)
        out = f"artifacts/scannetpp/{scene}/stats_half{tag}.pt"
        st.save(out)
        log(f"    saved {out}")


# --------------------------------------------------------------------- stage 2: analyze (gs-view)

def stage_analyze(scene, out_json, device="cuda"):
    import torch.nn.functional as F
    from feature_foam_lifting.operator import (AccumulatedFeatureStats,
                                            solve_geometric_median_from_stats)
    from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
    from point_cloud_query import assign_points_to_power_cells
    from build_true_facet_graph import load_points_radii
    from run_macro_iou_gap import cell_histograms
    from run_overnight import LAM, CSLS_K
    from run_spp_eval import benchmark_map, load_gt, coverage_filter

    art = f"artifacts/scannetpp/{scene}"
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    centers, radii = load_points_radii(ck)
    top, r2b = benchmark_map()

    preds, valids = {}, {}
    for tag in ("A", "B"):
        st = AccumulatedFeatureStats.load(f"{art}/stats_half{tag}.pt")
        # returns (features, valid_mask, meta) -- a tuple, not the dict the saved artifacts use
        fx, vx, _meta = solve_geometric_median_from_stats(st)
        f = fx.to(device).float()
        v = vx.to(device)
        u = torch.zeros_like(f); u[v] = F.normalize(f[v], dim=-1)
        mu = F.normalize(u[v].mean(0, keepdim=True), dim=-1)
        u[v] = F.normalize(u[v] - LAM * mu, dim=-1)
        preds[tag], valids[tag] = u, v
        log(f"  half {tag}: {int(v.sum()):,} valid cells")

    vmn_full = torch.load(f"{art}/solved_geometric_median_nonfrozen_ogl3.pt",
                          map_location="cpu", weights_only=True)["valid_mask"].numpy()
    gt_pts, lab0, _ = load_gt(scene, top, r2b)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn_full, k=64)
    keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn_full, 20.0)
    lab = np.where(keepc, lab0, -1)
    pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
    gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
    H, _ = cell_histograms(assigned, gt_t, len(centers), len(pres))
    txt = embed_class_names([top[:100][i] for i in pres], device)

    both = valids["A"] & valids["B"]
    cls = {}
    for tag in ("A", "B"):
        cv = preds[tag][both] @ txt.T
        rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
        cls[tag] = (cv - 0.5 * rK[None, :]).argmax(1).cpu().numpy()

    agree = cls["A"] == cls["B"]
    idx = torch.nonzero(both).squeeze(1).cpu().numpy()
    has_gt = H.sum(1) > 0
    m = has_gt[idx]
    gtc = H.argmax(1)[idx][m]
    correct = (cls["A"][m] == gtc)
    ag = agree[m]

    log(f"  cells valid in BOTH halves: {int(both.sum()):,}; with GT: {int(m.sum()):,}")
    print(f"\n  split-half agreement overall      : {agree.mean()*100:.1f}%")
    print(f"  agreement on CORRECT cells        : {ag[correct].mean()*100:.1f}%")
    print(f"  agreement on WRONG cells          : {ag[~correct].mean()*100:.1f}%")
    print(f"  accuracy | halves AGREE           : {correct[ag].mean()*100:.1f}%  (n={int(ag.sum()):,})")
    print(f"  accuracy | halves DISAGREE        : {correct[~ag].mean()*100:.1f}%  (n={int((~ag).sum()):,})")
    err_in_disagree = float((~correct & ~ag).sum() / max((~correct).sum(), 1))
    print(f"\n  fraction of ALL errors that are view-inconsistent: {err_in_disagree*100:.1f}%")
    print("  This is the CEILING on what any view-consistency method (incl. arXiv 2608.26951)")
    print("  could recover. The remainder is systematic: every view agrees on the wrong answer.")
    json.dump({"scene": scene, "agree_overall": float(agree.mean()),
               "agree_correct": float(ag[correct].mean()), "agree_wrong": float(ag[~correct].mean()),
               "acc_agree": float(correct[ag].mean()), "acc_disagree": float(correct[~ag].mean()),
               "err_view_inconsistent": err_in_disagree}, open(out_json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["accumulate", "analyze"], required=True)
    p.add_argument("--scene", default="f9f95681fd")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.stage == "accumulate":
        stage_accumulate(a.scene)
    else:
        stage_analyze(a.scene, a.out or f"artifacts/scannetpp/viewsplit_{a.scene}.json")


if __name__ == "__main__":
    main()
