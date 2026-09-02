"""Stack the OWNERSHIP GATE onto the real best pipeline: mode-vote + diffusion on the TRUE
FACET graph (38.96/41.76/49.51, 10 scenes).

WHAT THE GATE ACTUALLY IS. `run_sam_purity_gate.py` described it as a coverage move. It is not.
Passing `valid=gated` to `assign_points_to_power_cells` deletes the gated cell from the KD-tree,
so its GT points are RE-ASSIGNED to the nearest surviving cell and still receive a prediction.
Coverage stays exactly 100%. This is Theorem 1's ASSIGNMENT lever `a`, and it is a move only a
bounded disjoint partition admits: deleting a subset of sites leaves the power diagram of the
survivors an exact disjoint partition of the same R^3, so the deleted cell's territory is taken
over by its facet neighbours with no overlap and no gap. There is no analogue for a Gaussian
cloud, where deleting a splat leaves a hole in a soup of overlapping supports.

The statistic is GT-free: rank-average of per-cell SAM-mask purity (from the operator A and the
SAM masks the lift already consumes) and NormLift reliability (already in the stats file).

CONTROLS, run in the same pass, because the reassignment ALONE moves mIoU:
  random  -- a uniformly random gate of the SAME size. Its gain is the protocol artifact floor.
  radius  -- drop the largest cells. Tests whether the gate is secretly a size gate.

FALSIFIER, stated before running: the gate increment on top of modevote+diff must be
>= +0.5 mIoU at 19cls over 10 scenes, positive on >= 7/10 scenes, AND must exceed the random
control by >= +0.5 at the same drop fraction. Failing any of the three kills it.
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
from diagnose_scannet_miou import load_foam
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from run_normlift_refine_eval import mode_vote_refine

SCR = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
EPS = 1e-12


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def gate_stat(scene, name, radii, rel):
    d = np.load(os.path.join(SCR, f"rays_{scene}.npz"), allow_pickle=True)
    if name == "sam_purity":
        return d["sam_top"] / np.maximum(d["sam_tot"], EPS)
    if name == "reliability":
        return rel
    if name == "radius":
        return -np.asarray(radii, dtype=np.float64)
    if name == "random":
        return np.random.default_rng(0).random(d["n_rays"].shape[0])
    if "+" in name:
        r = lambda x: np.argsort(np.argsort(x)).astype(np.float64) / max(len(x) - 1.0, 1.0)
        return np.mean([r(gate_stat(scene, p, radii, rel)) for p in name.split("+")], 0)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--gates", default="none,random,radius,sam_purity+reliability")
    ap.add_argument("--drops", default="0.5")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        if not all(os.path.exists(p) for p in (mp, fp, stp,
                                               os.path.join(SCR, f"rays_{scene}.npz"))):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue
        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        n_prim = P.shape[0]
        tf = f"artifacts/scannet/{scene}/adjacency_true_facet.pt"
        g = torch.load(tf, map_location="cpu", weights_only=True)
        adjacent, offsets = g["adjacent"].long().to(dev), g["offsets"].long().to(dev)
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)

        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        refined = mode_vote_refine(unit, R, P, adjacent, offsets)
        rel_np = R.cpu().numpy().astype(np.float64)

        centers, radii = load_foam(f"output/scannet_{scene}_nonfrozen", dev)
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        # assignments, one per (gate, drop)
        assigns = {}
        for gname in [x for x in a.gates.split(",") if x]:
            drops = [0.0] if gname == "none" else [float(x) for x in a.drops.split(",")]
            for dr in drops:
                if gname == "none" or dr == 0.0:
                    gated = valid.copy()
                    tag = "none"
                else:
                    s = gate_stat(scene, gname, radii, rel_np)
                    vi = np.nonzero(valid)[0]
                    order = vi[np.argsort(s[vi], kind="stable")]
                    gated = np.zeros_like(valid)
                    gated[order[int(round(dr * len(order))):]] = True
                    tag = f"{gname}@{dr}"
                if tag in assigns:
                    continue
                assigns[tag] = np.asarray(assign_points_to_power_cells(
                    gt_pts, centers, radii, valid=gated, k=64))

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            for base_tag, u in (("modevote", refined),):
                p0 = torch.softmax(1000.0 * (u @ text.T), dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, src, adjacent, deg)
                for stack_tag, Pm in ((base_tag, p0), (base_tag + "+diff", pd)):
                    cls = Pm.argmax(-1).cpu().numpy() + 1
                    live = (Pm.sum(-1) > 0).cpu().numpy()
                    for tag, asg in assigns.items():
                        owned = asg >= 0
                        sc = owned.copy()
                        sc[owned] = live[asg[owned]]
                        pr = np.zeros(len(gt), dtype=np.int64)
                        pr[sc] = cls[asg[sc]]
                        _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                        torch.from_numpy(pr).long(), nc)
                        res.setdefault((stack_tag, tag, cs), []).append(float(mi) * 100)
        print(f"[{scene}] {(time.time()-t0)/60:.1f} min", flush=True)

    tags = sorted({k[1] for k in res}, key=lambda t: (t != "none", t))
    n = len(next(iter(res.values())))
    print(f"\n=== {n} scenes, TRUE FACET graph ===")
    for stack in ["modevote", "modevote+diff"]:
        if (stack, "none", CLASS_SETS[0]) not in res:
            continue
        print(f"\n--- {stack} ---")
        print(f"{'gate':<28}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS)
              + "   delta vs ungated      pos/n (19cls)")
        base = [np.mean(res[(stack, 'none', c)]) for c in CLASS_SETS]
        for tag in tags:
            if (stack, tag, CLASS_SETS[0]) not in res:
                continue
            v = [np.mean(res[(stack, tag, c)]) for c in CLASS_SETS]
            pos = sum(1 for x, y in zip(res[(stack, tag, CLASS_SETS[0])],
                                        res[(stack, 'none', CLASS_SETS[0])]) if x > y)
            print(f"{tag:<28}" + "".join(f"{x:9.2f}" for x in v) + "   "
                  + " ".join(f"{x-b:+6.2f}" for x, b in zip(v, base))
                  + f"        {pos}/{len(res[(stack, tag, CLASS_SETS[0])])}")


if __name__ == "__main__":
    main()
