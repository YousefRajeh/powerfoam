"""Full stack with the DERIVED prior correction replacing hand-tuned centering.

The headline stack is: centre features (lam=0.3) -> mode-vote prerefine -> rank-encode -> diffuse on
the true-facet graph -> argmax, giving 41.22 / 43.40 / 50.97.

[[Prior-correction-derived-2026-08-29]] replaced the tuned `lam` at the CLASSIFIER stage with
CSLS (Conneau et al. 2018) + the macro-IoU plug-in decision rule (Nowozin 2014; Koyejo et al. 2014),
beating it by -0.07 / +0.72 / +1.22 under leave-one-out with nothing tuned against labels.

The two do not occupy the same slot, which is the point of this script:
  * centering acts on FEATURES, so it also changes what `mode_vote_refine` sees;
  * CSLS acts on the cells x classes SCORE matrix, which does not exist until after prerefine;
  * the plug-in is a DECISION rule, so it belongs at the very end, after diffusion.

Hence "replace lam with the derived rule" is ambiguous and both readings are measured:
  A_tuned    centred -> prerefine -> rank -> diffuse -> argmax          (the incumbent)
  B_derived  RAW     -> prerefine -> CSLS -> rank -> diffuse -> plugin  (lam fully removed)
  C_hybrid   centred -> prerefine -> CSLS -> rank -> diffuse -> plugin  (lam kept for prerefine only)
  D_nopi     RAW     -> prerefine -> CSLS -> rank -> diffuse -> argmax  (isolates the plug-in)
  E_noCSLS   RAW     -> prerefine -> rank -> diffuse -> plugin          (isolates CSLS)

Diffused rows are convex combinations of distributions and so are themselves distributions, which is
what lets the plug-in consume them directly with no recalibration.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_prior_correction_eval import iou_plugin


def rank_encode(scores, s, device):
    """Fixed distribution shape mapped onto each cell's class ranking (see
    [[simplex-vs-sphere-extension]]): the diffusion benefit comes from the diffused quantity being
    non-negative, bounded and unit-sum, not from its values, so this is invariant to `s`."""
    K = scores.shape[1]
    tmpl = torch.softmax(s * torch.linspace(1.0, -1.0, K, device=device), 0)
    order = scores.argsort(dim=-1, descending=True)
    return torch.zeros_like(scores).scatter_(1, order, tmpl.expand(scores.shape[0], -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--rank-s", type=float, default=200.0)
    p.add_argument("--plugin-s", default="100,200,400")
    p.add_argument("--csls-k", type=int, default=1000)
    p.add_argument("--center-lam", type=float, default=0.3)
    p.add_argument("--arms", default="A,B,C,D,E,F,G",
                   help="F and G attribute C_hybrid's gain between CSLS and the plug-in")
    # TRANSFER TEST. Text-prototype whitening was derived on ScanNet++ ([[Text-prototype-
    # decorrelation-2026-08-31]], +2.12/+2.77 at alpha=0.25, 6/6 scenes). The frozen-constant claim
    # requires the SAME alpha to transfer to ScanNet untouched, exactly as lambda=0.3 and CSLS_K did.
    p.add_argument("--text-white", type=float, default=0.0,
                   help="alpha for text-prototype whitening; 0 = off")
    # Loewdin symmetric orthogonalisation: the PARAMETER-FREE version of the same operation, and
    # therefore the strongest form of the no-tuning claim -- there is no constant to transfer.
    p.add_argument("--text-lowdin", action="store_true")
    p.add_argument("--outdir", default="artifacts/scannet/derived_stack")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    psl = [float(x) for x in a.plugin_s.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        art = f"artifacts/scannet/{scene}"
        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved
        positions = torch.from_numpy(centers).to(device).float()

        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - a.center_lam * mu, dim=-1)

        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0 = adj["adjacent"].to(device).long(); of0 = adj["offsets"].to(device).long()
        sp = next(c for c in [f"{art}/train_stats_sam_{a.variant}{a.suffix}.pt",
                              f"{art}/stats_{a.variant}{a.suffix}.pt"] if os.path.exists(c))
        Rr = AccumulatedFeatureStats.load(sp).reliability()["reliability"].to(device).float() * vm
        Dm = int((of0[1:] - of0[:-1]).max()) + 1

        from run_normlift_refine_eval import mode_vote_refine
        pre = {}
        for tag, u in (("raw", raw), ("cen", cen)):
            pre[tag] = mode_vote_refine(u, Rr, positions, ad0, of0,
                                        chunk=max(256, 200_000 // max(Dm, 1)))
        del raw, cen, Rr
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        keep = vm[src] & vm[dst]; src, dst = src[keep], dst[keep]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
            0, src, torch.ones_like(src))
        del adj, ad0, of0
        torch.cuda.empty_cache()
        print(f"[{scene}] P={P:,} E={src.numel():,}", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{SCENES[scene]}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)
            if a.text_white > 0:
                from run_text_and_pseudo_eval import text_whiten
                text = text_whiten(text, a.text_white)[0]
            elif a.text_lowdin:
                from run_text_and_pseudo_eval import text_lowdin
                text = text_lowdin(text)
            C = len(names)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                print(f"  {cs} [{tag}] {float(miou)*100:.2f}", flush=True)

            cosr = torch.zeros(P, C, device=device); cosr[vm] = pre["raw"][vm] @ text.T
            cosc = torch.zeros(P, C, device=device); cosc[vm] = pre["cen"][vm] @ text.T

            def csls(cm):
                kk = min(a.csls_k, int(vm.sum()))
                r_t = cm[vm].topk(kk, dim=0).values.mean(0)
                out = cm.clone(); out[vm] = cm[vm] - 0.5 * r_t[None, :]
                return out

            def run(cm, tag, plugin):
                p0 = rank_encode(cm, a.rank_s, device); p0[~vm] = 0.0
                x = diffuse(p0, src, dst, deg, a.alpha, a.iters)
                if not plugin:
                    score(x.argmax(-1).cpu().numpy(), tag); return
                xv = x[vm] / x[vm].sum(1, keepdim=True).clamp_min(1e-30)
                for ps in plugin:
                    q = torch.softmax(ps * xv, 1)
                    adj_q, _ = iou_plugin(q, torch.ones(xv.shape[0], device=device))
                    full = torch.zeros(P, C, device=device); full[vm] = adj_q
                    score(full.argmax(-1).cpu().numpy(), f"{tag}_s{ps:g}")
                    del full

            arms = set(a.arms.split(","))
            if "A" in arms: run(cosc, "A_tuned", None)            # the incumbent
            if "B" in arms: run(csls(cosr), "B_derived", psl)     # lam removed entirely
            if "C" in arms: run(csls(cosc), "C_hybrid", psl)      # lam kept for prerefine only
            if "D" in arms: run(csls(cosr), "D_noplugin", None)   # isolates the plug-in
            if "E" in arms: run(cosr, "E_noCSLS", psl)            # isolates CSLS
            # attribution of C_hybrid's gain, holding centering fixed:
            if "F" in arms: run(cosc, "F_cen_plugin", psl)        # centering + plug-in, no CSLS
            if "G" in arms: run(csls(cosc), "G_cen_csls", None)   # centering + CSLS, no plug-in
            del cosr, cosc, text
            torch.cuda.empty_cache()

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
