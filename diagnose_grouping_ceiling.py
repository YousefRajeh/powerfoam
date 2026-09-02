"""ITEM 3: the EXACT CEILING of the entire grouping/pooling family.

Per-cell purity vs GT is 99.10% and the oracle-partition ceiling is 99.09%, so grouping
cannot add information. The ONLY way a grouping can beat per-cell is if pooling FIXES cells
whose own feature gives the wrong argmax but whose NEIGHBOURHOOD's consensus is right.

This script measures that population directly, and its mirror image (cells that are right on
their own and would be BROKEN by the neighbourhood).

  RESCUE  R = cells with wrong own-argmax but right neighbourhood-majority-argmax
  DAMAGE  D = cells with right own-argmax but wrong neighbourhood-majority-argmax

R is the exact upper bound of what any consensus-pooling grouping can win; D is the exact
lower bound of what it must risk. Reported as cell counts AND as GT-point mass (the metric
reads points), plus the three mIoU arms that bracket it:

  percell        the base
  nbhdvote       apply the neighbourhood majority everywhere (the realistic operator)
  oracle-gate    take the neighbourhood label ONLY where it is right   <- THE CEILING

Neighbourhoods swept over scale: 1-hop / 2-hop on the TRUE FACET GRAPH, and k-means-320
regions (the OpenGaussian codebook size) as the non-local comparison.

Run both on the bare per-cell argmax AND on the best stack (modevote + posterior diffusion),
because headroom that diffusion has already consumed is not available to grouping.

FALSIFIER, pre-registered: if the oracle-gated ceiling is < +2.0 mIoU at 19cls over 10 scenes
on top of modevote+diff, the entire consensus-grouping family is closed and should not be
pursued further.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from run_normlift_refine_eval import mode_vote_refine
from diagnose_scannet_miou import spherical_kmeans

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
HARD_FIRST = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
              "scene0200_00", "scene0097_00", "scene0400_00", "scene0062_00", "scene0000_00"]
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def nbhd_majority(lab, nc, src, dst, hops=1, include_self=False):
    """Majority of predicted labels over the h-hop neighbourhood. lab in 0..nc-1 (0 = none).
    Returns (majority_label, n_voters)."""
    n = lab.shape[0]
    onehot = torch.zeros(n, nc, device=lab.device)
    onehot[torch.arange(n, device=lab.device), lab] = 1.0
    onehot[:, 0] = 0.0                       # unlabelled cells cast no vote
    acc = onehot.clone() if include_self else torch.zeros_like(onehot)
    cur = onehot
    for _ in range(hops):
        nxt = torch.zeros_like(cur).index_add_(0, src, cur[dst])
        acc = acc + nxt
        cur = nxt
    maj = acc.argmax(1)
    maj[acc.max(1).values <= 0] = 0
    return maj, acc.sum(1)


def region_majority(lab, nc, region):
    """Majority predicted label within each region id."""
    n = lab.shape[0]
    nr = int(region.max().item()) + 1
    acc = torch.zeros(nr, nc, device=lab.device)
    acc.index_put_((region, lab), torch.ones(n, device=lab.device), accumulate=True)
    acc[:, 0] = 0.0
    maj = acc.argmax(1)
    maj[acc.max(1).values <= 0] = 0
    return maj[region]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(HARD_FIRST))
    ap.add_argument("--kmeans-k", type=int, default=320)
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    rows = {}        # (arm, cs) -> list of mIoU
    pops = {}        # (base, scope, cs) -> list of dict

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        tf = f"artifacts/scannet/{scene}/adjacency_true_facet.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if not all(os.path.exists(p) for p in (mp, fp, stp, tf, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        n_prim = P.shape[0]
        g = torch.load(tf, map_location="cpu", weights_only=True)
        adjacent = g["adjacent"].long().to(dev)
        offsets = g["offsets"].long().to(dev)
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        refined = mode_vote_refine(unit, R, P, adjacent, offsets)

        # region partition: spherical k-means on the valid features (the codebook analogue)
        vidx = torch.nonzero(vt, as_tuple=True)[0]
        km, _ = spherical_kmeans(unit[vidx], a.kmeans_k, seed=0)
        region = torch.zeros(n_prim, dtype=torch.long, device=dev)
        region[vidx] = km

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            gt_t = torch.from_numpy(gt).long()

            # ---- per-cell GT majority label + owned GT-point mass -------------------
            sc = owned & (gt != 0)
            cid = torch.from_numpy(assign[sc]).to(dev)
            gl = torch.from_numpy(gt[sc]).to(dev)
            cnt = torch.zeros(n_prim, nc, device=dev)
            cnt.index_put_((cid, gl), torch.ones(cid.shape[0], device=dev), accumulate=True)
            mass = cnt.sum(1)
            ymaj = cnt.argmax(1)
            ymaj[mass <= 0] = 0
            scorable = mass > 0            # cells the metric can actually read

            def score(pred_cell):
                pr = np.zeros(len(gt), dtype=np.int64)
                cl = pred_cell.cpu().numpy()
                pr[owned] = cl[assign[owned]]
                _, mi, _, _ = calculate_metrics(gt_t, torch.from_numpy(pr).long(), nc)
                return float(mi) * 100

            for base_tag, u in (("percell", unit), ("modevote+diff", refined)):
                sim = u @ text.T
                p0 = torch.softmax(1000.0 * sim, dim=-1)
                p0[~vt] = 0.0
                pp = diffuse(p0, src, adjacent, deg) if base_tag.endswith("diff") else p0
                yhat = pp.argmax(-1) + 1
                yhat[(pp.sum(-1) <= 0)] = 0     # no feature -> no vote / no prediction
                rows.setdefault((base_tag, cs), []).append(score(yhat))

                for scope, maj in (("1hop", nbhd_majority(yhat, nc, src, adjacent, 1)[0]),
                                   ("2hop", nbhd_majority(yhat, nc, src, adjacent, 2)[0]),
                                   (f"km{a.kmeans_k}", region_majority(yhat, nc, region))):
                    ok_self = (yhat == ymaj) & scorable
                    ok_nb = (maj == ymaj) & scorable
                    rescue = (~ok_self) & ok_nb & scorable
                    damage = ok_self & (~ok_nb) & scorable
                    pops.setdefault((base_tag, scope, cs), []).append(dict(
                        cells=int(scorable.sum()),
                        rescue_cells=int(rescue.sum()), damage_cells=int(damage.sum()),
                        pts=float(mass[scorable].sum()),
                        rescue_pts=float(mass[rescue].sum()),
                        damage_pts=float(mass[damage].sum()),
                        base_ok_pts=float(mass[ok_self].sum()),
                    ))
                    # the two mIoU arms
                    rows.setdefault((f"{base_tag}|{scope}|vote", cs), []).append(score(maj))
                    gated = torch.where(rescue, maj, yhat)
                    rows.setdefault((f"{base_tag}|{scope}|oracle", cs), []).append(score(gated))
        print(f"[{scene}] {(time.time()-t0)/60:.1f} min", flush=True)

    n = len(rows[("percell", CLASS_SETS[0])])
    print(f"\n================ {n} scenes ================")
    print("\n--- POPULATIONS (per-cell scorable cells; pts = GT-point mass) ---")
    print(f"{'base':<14}{'scope':<8}{'cs':<4}{'rescue%c':>10}{'damage%c':>10}"
          f"{'rescue%p':>10}{'damage%p':>10}{'net%p':>9}{'baseacc%':>10}")
    for (b, s, cs), lst in sorted(pops.items()):
        tc = sum(x["cells"] for x in lst); tp = sum(x["pts"] for x in lst)
        rc = sum(x["rescue_cells"] for x in lst); dc = sum(x["damage_cells"] for x in lst)
        rp = sum(x["rescue_pts"] for x in lst); dp = sum(x["damage_pts"] for x in lst)
        ba = sum(x["base_ok_pts"] for x in lst)
        print(f"{b:<14}{s:<8}{cs[11:]:<4}{100*rc/tc:10.2f}{100*dc/tc:10.2f}"
              f"{100*rp/tp:10.2f}{100*dp/tp:10.2f}{100*(rp-dp)/tp:9.2f}{100*ba/tp:10.2f}")

    print("\n--- mIoU ---")
    print(f"{'arm':<32}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS))
    for tag in sorted({k[0] for k in rows}):
        print(f"{tag:<32}" + "".join(
            f"{np.mean(rows[(tag,c)]):9.2f}" if (tag, c) in rows else f"{'--':>9}"
            for c in CLASS_SETS))

    print("\n--- CEILING (oracle-gated minus base), the headroom of consensus grouping ---")
    for b in ("percell", "modevote+diff"):
        for s in ("1hop", "2hop", f"km{a.kmeans_k}"):
            t = f"{b}|{s}|oracle"
            if (t, CLASS_SETS[0]) not in rows:
                continue
            dl = " ".join(f"{np.mean(rows[(t,c)]) - np.mean(rows[(b,c)]):+7.2f}" for c in CLASS_SETS)
            tv = f"{b}|{s}|vote"
            dv = " ".join(f"{np.mean(rows[(tv,c)]) - np.mean(rows[(b,c)]):+7.2f}" for c in CLASS_SETS)
            print(f"  {b:<14}{s:<8} oracle {dl}   |  plain vote {dv}")


if __name__ == "__main__":
    main()
