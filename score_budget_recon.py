"""Score a higher-point-budget reconstruction end to end: lift -> solve -> assign -> mIoU.

THE QUESTION. Our ScanNet runs gave both foams 3x the GT vertex count, but every published
config pairs RadFoam ~4x above PowerFoam on the same scene (garden 4,194,304 vs 1,200,000;
bonsai 2,097,152 vs 500,000). PowerFoam's own paper states 500k for MipNeRF360 INDOOR and
calls its counts "2x to 4x lower than existing baseline methods". So 3x GT put PowerFoam at
31% of its own indoor default and RadFoam at 7.4% of its own -- and RadFoam is the arm that
underperformed. This scores what the extra budget actually buys in mIoU.

Not just PSNR: a budget can improve reconstruction while leaving semantics flat, and the
LERF-OVS sweep already found final_points effects on (2D) mIoU to be a coin flip -- ramen and
figurines up, teatime and waldo_kitchen down. That was 2D rendered IoU on a different dataset,
so it does not transfer here, but it is a reason not to assume more points help.

EVERY STAGE IS RERUN, because a new point set invalidates all of them: the lifting operator is
per-primitive, the solve is per-primitive, and the GT->cell assignment is a power-diagram
membership query against different sites. Reusing any cached artifact from the 3x run would
silently mismatch.

The paired baseline is the SAME scene at 3x GT with the SAME features and the same scorer, so
the delta is attributable to the budget alone.
"""
import argparse
import os
import subprocess
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
from point_cloud_query import assign_points_to_power_cells

PY = r"D:\conda\envs\powerfoam\python.exe"
DATA = r"D:\Downloads\powerfoam\data\scannet"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def diffuse(p0, adjacent, offsets, n, alpha=0.9, iters=60):
    dev = p0.device
    deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
    src = torch.repeat_interleave(torch.arange(n, device=dev), offsets[1:] - offsets[:-1])
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[adjacent])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def lift_and_solve(run, scene, art):
    """Returns path to the solved features, running the lift only if needed."""
    solved = f"{art}/solved_geometric_median_{run}.pt"
    if os.path.exists(solved):
        print(f"  [cached] {solved}", flush=True)
        return solved
    cfg = f"output/{run}/config.yaml"
    feat = os.path.join(DATA, f"{scene}_colmap", "openclip_features_sam_l3")
    if not (os.path.exists(cfg) and os.path.isdir(feat)):
        print(f"  [miss] config or features for {run}", flush=True)
        return None
    stats = f"{art}/stats_{run}.pt"
    t0 = time.time()
    print(f"  [lift] {run} ...", flush=True)
    r = subprocess.run([PY, "accumulate_feature_stats_sam.py", "--scene", scene,
                        "--config", cfg, "--feature-folder", feat, "--output", stats,
                        "--sam-level", "0"],
                       stdout=open(f"logs_lift_{run}.log", "w"), stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  [FAIL] lift rc={r.returncode} -- see logs_lift_{run}.log", flush=True)
        return None
    subprocess.run([PY, "solve_from_stats.py", "--stats", stats,
                    "--out-template", f"{art}/solved_{{solver}}_{run}.pt",
                    "--solvers", "geometric_median"], capture_output=True, text=True)
    print(f"  [ok] {(time.time()-t0)/60:.1f} min", flush=True)
    return solved if os.path.exists(solved) else None


def score(run, scene, dev):
    art = f"artifacts/scannet/{scene}"
    os.makedirs(art, exist_ok=True)
    mp = f"output/{run}/model.pt"
    if not os.path.exists(mp):
        print(f"[skip] {run}: no model.pt (still training?)", flush=True)
        return None
    m = torch.load(mp, map_location="cpu", weights_only=False)
    n_prim = m["points"].shape[0]
    print(f"[{run}] {n_prim:,} primitives", flush=True)

    solved = lift_and_solve(run, scene, art)
    if not solved:
        return None
    d = torch.load(solved, map_location=dev, weights_only=True)
    feats = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()
    unit = torch.zeros_like(feats)
    vt = torch.from_numpy(valid).to(dev)
    unit[vt] = F.normalize(feats[vt], dim=-1)

    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
    # a new point set means a NEW power-cell membership query; nothing cached is reusable
    cache = f"artifacts/ablation_cache/{scene}_{run}_assign.npy"
    if os.path.exists(cache):
        assign = np.load(cache)
    else:
        centers = m["points"].float().numpy()
        radii = F.softplus(m["radii"].float().squeeze(), beta=100).numpy()
        assign = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, assign)
    owned = assign >= 0

    adjacent = m["adjacency"].long().to(dev)
    offsets = m["adjacency_offsets"].long().to(dev)
    n2i = {n: q for q, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    out = {}
    for cs in CLASS_SETS:
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        gt = remap_gt_labels(raw, [n2i[n] for n in names])
        nc = len(names) + 1
        text = embed_class_names(names, dev)
        sim = unit @ text.T
        p0 = torch.softmax(1000.0 * sim, dim=-1)
        p0[~vt] = 0.0
        pd = diffuse(p0, adjacent, offsets, n_prim)
        for tag, cls, live in (("percell", sim.argmax(-1).cpu().numpy() + 1, valid),
                               ("diffusion", pd.argmax(-1).cpu().numpy() + 1,
                                (pd.sum(-1) > 0).cpu().numpy())):
            sc = owned.copy()
            sc[owned] = live[assign[owned]]
            pred = np.zeros(len(gt), dtype=np.int64)
            pred[sc] = cls[assign[sc]]
            _, mi, _, ma = calculate_metrics(torch.from_numpy(gt).long(),
                                             torch.from_numpy(pred).long(), nc)
            out[(cs, tag)] = (float(mi) * 100, float(ma) * 100, float(sc.mean()) * 100)
    out["n_prim"] = n_prim
    out["valid_frac"] = float(valid.mean()) * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--runs", default="scannet_scene0347_00_nonfrozen,pf_bud500k_scene0347_00,"
                                      "pf_bud1200k_scene0347_00")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}
    for run in a.runs.split(","):
        r = score(run, a.scene, dev)
        if r:
            res[run] = r

    print(f"\n=== {a.scene}: mIoU vs point budget ===")
    print(f"{'run':<34}{'points':>10}{'feat%':>7}"
          + "".join(f"{c[11:]:>9}" for c in CLASS_SETS) + "   (per-cell / diffusion)")
    for tag in ("percell", "diffusion"):
        print(f"-- {tag}")
        base = None
        for run, r in res.items():
            row = "".join(f"{r[(c, tag)][0]:9.2f}" for c in CLASS_SETS)
            d = ""
            if base is None:
                base = [r[(c, tag)][0] for c in CLASS_SETS]
            else:
                d = "  delta " + " ".join(
                    f"{r[(c, tag)][0]-b:+.2f}" for c, b in zip(CLASS_SETS, base))
            print(f"  {run:<32}{r['n_prim']:>10,}{r['valid_frac']:>7.1f}{row}{d}")


if __name__ == "__main__":
    main()
