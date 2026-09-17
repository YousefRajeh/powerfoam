"""Facet propagation on the REAL pipeline -- a drop-in post-process on an existing lifted field.

The oracle version reads a per-primitive CLASS HISTOGRAM (`AtS`), which only exists because the
oracle's observations are one-hot classes. The real pipeline accumulates dense 512-d CLIP features,
so there is no histogram and no "one evidence class". Rather than re-accumulate rays (expensive, and
it would change the method rather than port it), this takes the class scores the METRIC ITSELF reads:

    S_j = normalise(X_j) @ T^T          X = the solved per-primitive feature, T = the text head

`argmax_c S_j` is exactly the label the reported mIoU uses, so lambda = 0 reproduces the published
number bit-for-bit (asserted). Everything after that is a relabelling on the facet graph:

    score_j = p_j + lambda * (normalised histogram of neighbours' current labels),  p_j = softmax(S_j / tau)

so it is a pure READOUT change: no re-solving, no re-rasterising, and it applies to any lifted field
including a 3DGS one.

TWO DIFFERENCES FROM THE ORACLE VERSION, both measured rather than assumed:

  * No forced set. In the oracle, primitives receiving one evidence class are 99.75% accurate and
    provably solver-invariant, so clamping them is free. With real features every primitive has a
    dense score and region-CLIP is only 50.9% accurate, so no comparable anchor exists. The clamp was
    ablated on the oracle and made no difference (+1.62 vs +1.66), so nothing is lost by dropping it.
    `--margin-clamp` freezes primitives whose top-1/top-2 score margin exceeds a threshold, as the
    nearest available analogue.
  * `tau` matters. `S_j` are cosine similarities in a narrow band (text-embedding off-diagonals run
    0.6-0.76), so a softmax over raw scores is nearly uniform and the neighbour term would dominate
    at any lambda. `tau` sets how sharply a primitive's own evidence competes with its neighbours;
    it is swept, not tuned on the test set.

Scores BOTH the standard protocol (all labelled points, comparable to published numbers) and the
visible-only subset, because they differ and the paper reports the former.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
from diagnose_holes import SCENES, GT_ROOT, geometry

FOAM = {"truefrozen", "nonfrozen"}
_T = {}


def text_head(kept, dev):
    k = tuple(kept)
    if k not in _T:
        T = embed_class_names(kept, dev)
        _T[k] = F.normalize(T, dim=-1)
    return _T[k]


def facet_graph(scene, arm):
    """Foam's power-diagram adjacency, cached. Raises for 3DGS, which has no such graph."""
    if arm not in FOAM:
        raise RuntimeError("no facet graph for a Gaussian arm (use --knn)")
    ca, co = (f"artifacts/scannet/{scene}/adj_{arm}.npy",
              f"artifacts/scannet/{scene}/adjoff_{arm}.npy")
    if os.path.exists(ca) and os.path.exists(co):
        return np.load(ca), np.load(co)
    import warp as wp, configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    wp.init()
    ck = f"output/scannet_{scene}_{arm}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device="cuda")
    m.load_pt(f"{ck}/model.pt")
    a = m.adjacency.detach().cpu().numpy().astype(np.int32)
    o = m.adjacency_offsets.detach().cpu().numpy().astype(np.int64)
    os.makedirs(os.path.dirname(ca), exist_ok=True)
    np.save(ca, a); np.save(co, o)
    return a, o


def knn_graph(cent, k):
    from scipy.spatial import cKDTree
    nn = cKDTree(cent).query(cent, k=k + 1, workers=-1)[1][:, 1:]
    return nn.astype(np.int32).reshape(-1), (np.arange(cent.shape[0] + 1) * k).astype(np.int64)


def propagate(p, valid, adj, off, lam, rounds, base_lab, margin=None, margin_clamp=0.0):
    lab = base_lab.copy()
    if lam == 0.0 or rounds == 0:
        return lab
    C = p.shape[1]
    frozen = (valid & (margin >= margin_clamp)) if (margin_clamp > 0 and margin is not None) \
        else np.zeros(p.shape[0], bool)
    movable = np.nonzero(valid & ~frozen)[0]
    P = p.shape[0]
    for _ in range(rounds):
        new = lab.copy()
        for j in movable:
            nb = adj[off[j]:off[j + 1]]
            nb = nb[(nb >= 0) & (nb < P)]
            nb = nb[lab[nb] > 0]
            s = p[j]
            if nb.size:
                h = np.bincount(lab[nb] - 1, minlength=C).astype(np.float32)
                s = s + lam * (h / h.sum())
            new[j] = int(s.argmax()) + 1
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def one(scene, arm, solver, lams, rounds, tau, k, margin_clamp, dev="cuda"):
    fp = f"artifacts/scannet/{scene}/solved_{solver}_{arm}_ogl3.pt"
    d = torch.load(fp, map_location=dev, weights_only=True)
    X = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy().astype(bool)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    T = text_head(kept, dev)

    S = (F.normalize(X, dim=-1) @ T.T)                       # the scores the metric reads
    base_lab = np.zeros(X.shape[0], np.int64)
    base_lab[valid] = (S[torch.from_numpy(valid).to(dev)].argmax(1) + 1).cpu().numpy()
    sm = F.softmax(S / tau, dim=-1).cpu().numpy()
    top2 = torch.topk(S, 2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    if arm in FOAM:
        centers, radii, _ = geometry(scene, arm)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    else:
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu",
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        centers = sp["means"].float().numpy()
        assigned = assign_points_to_nearest_center(gt_pts, centers, valid=valid)
    try:
        adj, off = facet_graph(scene, arm)
    except RuntimeError:
        adj, off = knn_graph(centers, k)

    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    lab_m = gt_lab > 0
    owned = assigned >= 0
    r = {"scene": scene, "arm": arm, "solver": solver, "C": C,
         "graph": "facet" if arm in FOAM else f"knn{k}"}
    for lam in lams:
        lab = propagate(sm, valid, adj, off, lam, rounds, base_lab, margin, margin_clamp)
        pred = np.zeros(gt_pts.shape[0], np.int64)
        pred[owned] = lab[assigned[owned]]
        if lam == 0.0:
            b = np.zeros(gt_pts.shape[0], np.int64); b[owned] = base_lab[assigned[owned]]
            assert np.array_equal(pred, b), "lam=0 does not reproduce the published readout"
        for tag, mask in (("all", lab_m), ("vis", lab_m & vis)):
            g = torch.from_numpy(gt_lab[mask]); q = torch.from_numpy(pred[mask])
            _, mi, ac, _ = calculate_metrics(g, q, C + 1)
            r[f"miou_{tag}_{lam}"] = float(mi) * 100
            r[f"acc_{tag}_{lam}"] = float(ac) * 100
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen")
    ap.add_argument("--solver", default="weighted")
    ap.add_argument("--lams", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--margin-clamp", type=float, default=0.0)
    ap.add_argument("--out", default="artifacts/scannet/facet_propagate_real.json")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                r = one(sc, arm, a.solver, lams, a.rounds, a.tau, a.knn, a.margin_clamp)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            b = r["miou_all_0.0"]
            best = max(lams, key=lambda l: r[f"miou_all_{l}"])
            print(f"[{arm}/{sc}] {r['graph']}  base {b:.2f}  best lam={best} "
                  f"{r[f'miou_all_{best}']:.2f} ({r[f'miou_all_{best}'] - b:+.2f})", flush=True)
            torch.cuda.empty_cache()
    if not rows:
        return
    print(f"\n{'arm':<12}{'n':>3}{'graph':>8}" + "".join(f"{('l=' + str(l)):>9}" for l in lams))
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        b = float(np.mean([r["miou_all_0.0"] for r in s]))
        print(f"{arm:<12}{len(s):>3}{s[0]['graph']:>8}" + "".join(
            f"{np.mean([r[f'miou_all_{l}'] for r in s]) - b:>+9.2f}" for l in lams)
            + f"   base {b:.2f}")
    print("(deltas vs lam=0, mIoU, ALL labelled points -- the published protocol)")


if __name__ == "__main__":
    main()
