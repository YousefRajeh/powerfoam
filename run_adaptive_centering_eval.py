"""Foam-derived centering: set lambda per cell from the mixing operator M, not by sweeping.

THE QUESTION
------------
Global feature centering (`f' = normalize(f - lam*mu)`) is the largest single gain found here
(+2.25 mIoU at 19-class, 10 scenes), but `lam = 0.3` was chosen by sweeping ON the evaluation
scenes. That is tuning on test. A rule that derives lambda from measurable geometry would remove the
circularity AND make the correction foam-specific rather than a generic CLIP trick.

THE HYPOTHESIS
--------------
A cell's common-mode share should be CAUSED by ray mixing. Rays through cell j also traverse other
cells, so j's lifted feature accumulates a weighted average of the scene along those rays -- and the
scene average IS the common direction mu. If that is the mechanism, then

    cone share  c_j = cos(f_j, mu)     should rise with impurity   1 - M_jj

where M = D^-1 A^T W^-1 A is the mixing operator (see [[Mixing-operator-framework]]); M_jj is the
fraction of cell j's evidence that is its own. M is computable in closed form ONLY for a foam --
overlapping unbounded Gaussian support admits no exact ray decomposition -- so a lambda derived from
it is a genuinely foam-specific rule, not a hyperparameter.

This script (1) measures whether that correlation exists, then (2) tests the rule it implies:

    lam_j = kappa * c_j                    proportional to the cell's OWN common-mode share
    lam_j = kappa * (1 - M_jj)             proportional to its impurity
    lam_j = kappa * c_j * (1 - M_jj)       both

If the correlation is absent, the mechanism is wrong and a per-cell rule has no basis -- report that
rather than fitting kappa to make it work.

NOTE ON WHY FULL REMOVAL FAILS. Setting lam_j = c_j exactly orthogonalises f_j against mu, and
globally lam = 1.0 collapsed to -7.09. So mu is not purely nuisance: it carries some usable prior
(scene/domain content), and removing all of it destroys that as well as amplifying whatever noise
remains after the subtraction shrinks the norm. Hence kappa < 1 is expected on physical grounds,
not merely empirically.
"""
import argparse
import glob
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
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import HARDEST_FIRST


def spearman(x, y):
    rx = torch.empty_like(x); rx[torch.argsort(x)] = torch.arange(x.numel(), device=x.device).float()
    ry = torch.empty_like(y); ry[torch.argsort(y)] = torch.arange(y.numel(), device=y.device).float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--kappas", default="0.2,0.35,0.5")
    p.add_argument("--global-lams", default="0.3")
    p.add_argument("--max-edges", type=int, default=140_000_000)
    p.add_argument("--outdir", default="artifacts/scannet/adaptive_center")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    kappas = [float(x) for x in a.kappas.split(",")]
    glams = [float(x) for x in a.global_lams.split(",")]

    for scene in a.scenes.split(","):
        out_path = os.path.join(a.outdir, f"{scene}.json")
        if os.path.exists(out_path):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        split = SCENES[scene]
        art = f"artifacts/scannet/{scene}"

        centers, radii = load_points_radii(f"output/scannet_{scene}_{a.variant}")
        solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        del feats, solved

        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)
        cone = (unit @ mu.T).squeeze(1)                       # c_j = cos(f_j, mu)

        # ---- purity M_jj from the gram cache
        cache = torch.load(sorted(glob.glob(f"{art}/gram_cache_*.pt"))[0],
                           map_location="cpu", weights_only=False)
        keys, vals = cache["S_keys"], cache["S_vals"].float()
        support = cache["support"].float().to(device)
        Pc = int(cache["P"]); del cache
        assert Pc == P
        if keys.numel() > a.max_edges:
            from gram_blocks import prune_edges
            keys, vals, _ = prune_edges(keys, vals, P, a.max_edges, verbose=False)
        idx = torch.stack([keys // P, keys % P]).to(device); del keys
        S = torch.sparse_coo_tensor(idx, vals.to(device), (P, P), device=device).coalesce()
        del idx, vals
        rowsum = torch.sparse.mm(S, torch.ones(P, 1, device=device)).squeeze(1)
        Sd = torch.zeros(P, device=device)
        si = S.indices(); sv = S.values(); dm = si[0] == si[1]
        Sd[si[0][dm]] = sv[dm]
        purity = torch.where(rowsum > 0, Sd / rowsum.clamp_min(1e-30), torch.zeros_like(Sd))
        del S

        ok = vm & (support > 0)
        rho_pur = spearman(cone[ok], (1 - purity)[ok])
        rho_sup = spearman(cone[ok], support[ok])
        print(f"[{scene}] cone share mean={float(cone[ok].mean()):.4f}", flush=True)
        print(f"  spearman(cone, impurity 1-M_jj) = {rho_pur:+.4f}", flush=True)
        print(f"  spearman(cone, support)         = {rho_sup:+.4f}", flush=True)

        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                                valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        res = {"scene": scene, "cone_mean": float(cone[ok].mean()),
               "spearman_cone_impurity": rho_pur, "spearman_cone_support": rho_sup, "arms": {}}

        for cs in CLASS_SETS:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                    if name_to_id[n] in present]
            tids = [i for i, _ in kept]; names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, tids)).long()
            text = embed_class_names(names, device)

            def score(cls_np, tag):
                pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(),
                                                     len(tids) + 1)
                res["arms"].setdefault(tag, {})[cs] = {"mIoU": float(miou) * 100,
                                                       "mAcc": float(macc) * 100}
                return float(miou) * 100

            def apply(lam_vec, tag):
                cf = torch.zeros_like(unit)
                cf[vm] = F.normalize(unit[vm] - lam_vec[vm].unsqueeze(1) * mu, dim=-1)
                v = score((cf @ text.T).argmax(-1).cpu().numpy(), tag)
                del cf
                return v

            b = score((unit @ text.T).argmax(-1).cpu().numpy(), "plain")
            print(f"  {cs} [plain] mIoU={b:.2f}", flush=True)
            for lg in glams:
                v = apply(torch.full((P,), lg, device=device), f"global_lam{lg:g}")
                print(f"  {cs} [global_lam{lg:g}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
            for k in kappas:
                for name, vec in (("cone", k * cone),
                                  ("imp", k * (1 - purity)),
                                  ("coneimp", k * cone * (1 - purity))):
                    v = apply(vec.clamp(0, 1), f"{name}_k{k:g}")
                    print(f"  {cs} [{name}_k{k:g}] mIoU={v:.2f} ({v-b:+.2f})", flush=True)
            del text

        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
