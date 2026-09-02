"""Round-trip consistency as a label-free reliability signal for lifted foam features.

THE IDEA, adapted from "Round-Trip Consistency: Bidirectional Diffusion Models Can Predict Their
Own Rollout Errors" (arXiv 2608.00675). That paper composes a learned forward map with a learned
backward map and uses the discrepancy after a round trip as a test-time error estimate requiring no
ground truth. The method there is a latent diffusion model over temporal dynamics and does not
transfer; the PRINCIPLE -- a checkable invariant as a self-supervised error signal -- does.

WHY IT IS COMPUTABLE IN CLOSED FORM HERE. The lift is a least-squares problem A f = b, where A is
the ray-cell traversal operator, and the gram cache already stores S = A^T A and A^T b. So:
    render (3D -> 2D):  b_hat = A f
    re-lift (2D -> 3D): A^T b_hat = A^T A f = S f
One round trip is therefore exactly S f -- no re-rendering, no re-extraction of CLIP features.
The lift actually used in the pipeline is the DIAGONAL approximation f = A^T b / support, so the
round-trip operator that matches the deployed pipeline is

    T = D^{-1} S,     D = diag(support)

and a depth-i round trip is T^i f. T f = f exactly only where a cell's rays are shared with cells
carrying like features; where a cell's support is dominated by rays that also traverse cells of a
DIFFERENT feature -- occlusion boundaries, mixed segments, thin structure -- T f drifts away from f.
That drift is the error signal, and it needs no labels.

WHY THIS IS FOAM-SPECIFIC. S = A^T A with A >= 0 and, for a foam, each ray's support is a set of
DISJOINT ordered segments through a bounded partition. A discrepancy in T f is therefore
attributable to identified neighbouring cells. For Gaussians the analogous A has overlapping,
unbounded support, so many primitive configurations render identically: the round trip is lossy by
construction and a small residual does not imply a correct lift. Disjointness is what makes the
invariant checkable, which is a different property from the convexity arguments tried earlier.

WHAT THIS SCRIPT MEASURES. Whether the residual predicts per-cell correctness -- the same test the
existing reliability measure passed (accuracy rising 0.48 -> 0.87 across its deciles). If
round-trip residual is not at least comparably monotone, the idea is dead and no mIoU run is
warranted. Reported alongside `support` (view count) as the trivial baseline signal, because a new
signal that merely recovers "cells seen by more rays are better" adds nothing.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    classify_primitives,
    embed_class_names,
    load_scannet_pointcept_gt,
    remap_gt_labels,
    OPENGAUSSIAN_CLASS_SETS,
    SCANNET20_CLASS_NAMES,
)
from point_cloud_query import assign_points_to_power_cells


def load_centers_radii(ckpt_dir, device):
    import glob
    p = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(p):
        cands = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if not cands:
            raise SystemExit(f"no checkpoint in {ckpt_dir}")
        p = cands[0]
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("model", "state_dict", "params"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    def find(*names):
        for n in names:
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and k.split(".")[-1] == n:
                    return v
        return None
    c = find("points", "centers", "primal_points", "xyz")
    r = find("radii", "radius", "weights")
    if c is None:
        raise SystemExit(f"could not find centers in {p}; keys={list(sd)[:12]}")
    c = c.float().to(device)
    r = torch.zeros(len(c), device=device) if r is None else r.float().to(device).reshape(-1)
    return c, r


def round_trip(S_idx, S_val, support, f, depths, device, normalize="rowsum"):
    """r_i = 1 - cos(T^i f, f), T the round-trip operator, by repeated sparse matmul.

    normalize="support"  ->  T = diag(1/support) S      (the ORIGINAL, DEFECTIVE operator)
    normalize="rowsum"   ->  T = diag(1/rowsum(S)) S    (row-stochastic; the correction)

    The cache stores S = A^T A, not A^T W^-1 A with W = diag(w_r), w_r = sum_l A[r,l]. Dividing by
    `support` = A^T 1 therefore leaves T sub-stochastic: rowsum(S)_j / support_j is the
    evidence-weighted mean of w_r over the rays through j, measured at median 0.570 (p5 0.114) --
    rays are only ~57% absorbed, so cells whose rays ESCAPE were systematically over-weighted.
    Rescaling each row by its own sum makes T genuinely row-stochastic, so T f = f holds exactly
    when f is constant over each cell's ray support -- which is the property the residual is
    supposed to measure. This is exact when w_r is constant along a cell's rays and the best
    correction available from the cache otherwise (per-ray w_r is not stored)."""
    P, D = f.shape
    S = torch.sparse_coo_tensor(S_idx, S_val, (P, P), device=device).coalesce()
    if normalize == "rowsum":
        denom = torch.sparse.mm(S, torch.ones(P, 1, device=device)).squeeze(1)
    else:
        denom = support
    inv = torch.zeros(P, device=device)
    ok = denom > 0
    inv[ok] = 1.0 / denom[ok]
    f0 = F.normalize(f, dim=-1)
    out, cur = {}, f0
    for i in range(1, max(depths) + 1):
        cur = torch.sparse.mm(S, cur) * inv[:, None]
        cur = F.normalize(cur, dim=-1)
        if i in depths:
            out[i] = (1.0 - (cur * f0).sum(-1)).clamp(min=0).cpu()
    return out


def deciles(signal, correct, valid, n=10, ascending=True):
    """Mean accuracy per decile of `signal`, restricted to `valid` cells."""
    s = signal[valid].numpy()
    c = correct[valid].numpy()
    order = np.argsort(s, kind="stable")
    if not ascending:
        order = order[::-1]
    out = []
    for k, chunk in enumerate(np.array_split(order, n)):
        out.append((k + 1, float(s[chunk].mean()), float(c[chunk].mean()), len(chunk)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    ap.add_argument("--depths", default="1,2,4")
    ap.add_argument("--max-edges", type=int, default=140_000_000)
    ap.add_argument("--normalize", default="rowsum", choices=["rowsum", "support"],
                    help="rowsum = the W^-1-corrected row-stochastic operator (default); "
                         "support = the original defective one, kept to reproduce Results 1-3")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depths = [int(x) for x in a.depths.split(",")]

    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    P = int(cache["P"])
    keys, vals = cache["S_keys"], cache["S_vals"].float()
    support = cache["support"].float()
    print(f"[{a.scene}] P={P:,} edges={keys.numel():,} views={cache['n_views']}", flush=True)

    # keep the diagonal plus the strongest couplings, exactly as the solver does
    if keys.numel() > a.max_edges:
        from gram_blocks import prune_edges
        keys, vals, _ = prune_edges(keys, vals, P, a.max_edges, verbose=True)

    idx = torch.stack([keys // P, keys % P]).to(dev)
    del keys
    S_val = vals.to(dev); del vals

    feats = torch.load(a.features, map_location="cpu", weights_only=False)
    f = feats["primitive_features"].float()
    valid_mask = feats.get("valid_mask", torch.ones(len(f), dtype=torch.bool))
    assert f.shape[0] == P, f"features {f.shape[0]} != cache P {P}"
    f = f.to(dev)

    res = round_trip(idx, S_val, support.to(dev), f, depths, dev, a.normalize)
    print(f"  operator: T = diag(1/{a.normalize}) S")
    del idx, S_val
    torch.cuda.empty_cache()

    # ---- per-cell correctness against ScanNet GT.
    # Mirrors evaluate_point_cloud_miou.py: segment20 ids are already 0..19 with -1 = ignore, and
    # OpenGaussian's convention is to score only classes actually PRESENT in this scene's GT.
    # Pointcept splits these scenes across train/ and val/ (scene0645_00 is in val, the other two
    # in train), so locate the scene rather than assuming a split.
    scene_dir = next((os.path.join(a.gt_root, sp, a.scene) for sp in ("val", "train", "test")
                      if os.path.exists(os.path.join(a.gt_root, sp, a.scene, "coord.npy"))), None)
    if scene_dir is None:
        raise SystemExit(f"no Pointcept GT for {a.scene} under {a.gt_root}")
    gt_pts, gt_raw, all_names = load_scannet_pointcept_gt(scene_dir)
    wanted = OPENGAUSSIAN_CLASS_SETS["opengaussian19"]
    name_to_id = {n: i for i, n in enumerate(SCANNET20_CLASS_NAMES)}
    present = set(np.unique(gt_raw).tolist())
    target_ids = [name_to_id[n] for n in wanted if n in name_to_id and name_to_id[n] in present]
    target_names = [SCANNET20_CLASS_NAMES[i] for i in target_ids]
    print(f"  scoring over {len(target_ids)} classes present in this scene")
    gt_lab = remap_gt_labels(gt_raw, target_ids)

    # assign_points_to_power_cells goes through numpy, so hand it host tensors
    centers, radii = load_centers_radii(a.checkpoint, "cpu")
    assigned = np.asarray(assign_points_to_power_cells(
        torch.from_numpy(gt_pts).float(), centers, radii,
        valid=valid_mask.cpu(), k=64))

    text = embed_class_names(target_names, dev)
    pred = classify_primitives(f, text).cpu().numpy()

    n_cls = len(target_names)
    # remap_gt_labels emits 1..K with 0 = ignore/other; classify_primitives emits 0..K-1.
    # Shift the GT down by one and drop the ignore class so the two conventions line up.
    votes = np.zeros((P, n_cls), dtype=np.int32)
    keep = (assigned >= 0) & (gt_lab > 0)
    np.add.at(votes, (assigned[keep], gt_lab[keep] - 1), 1)
    seen = votes.sum(1) > 0
    cell_gt = votes.argmax(1)
    correct = torch.from_numpy((cell_gt == pred) & seen)
    valid = torch.from_numpy(seen) & (support > 0)
    print(f"  cells with GT: {int(valid.sum()):,}  cell-level acc: {correct[valid].float().mean():.4f}")

    report = {"scene": a.scene, "P": P, "cells_with_gt": int(valid.sum()),
              "cell_accuracy": float(correct[valid].float().mean()), "deciles": {}}
    print("\n  decile |   signal | accuracy | n         (ascending signal)")
    for name, sig, asc in ([(f"roundtrip_d{i}", res[i], True) for i in depths]
                           + [("support(baseline)", support, False)]):
        print(f"  --- {name} ---")
        rows = deciles(sig, correct, valid, ascending=asc)
        for k, sv, acc, n in rows:
            print(f"    {k:>2}   | {sv:8.4f} | {acc:8.4f} | {n:,}")
        report["deciles"][name] = rows
        report[f"spread_{name}"] = rows[0][2] - rows[-1][2]
        print(f"    spread (best decile - worst): {rows[0][2]-rows[-1][2]:+.4f}")

    # ---- Does the round trip carry information INDEPENDENT of support?
    # The raw deciles are confounded: a cell seen by one ray satisfies Tf = f trivially, so a near
    # zero residual can mean "uninformative" rather than "reliable" -- which is exactly what the
    # non-monotone first decile looks like. Conditioning on support removes that confound: if the
    # residual still separates accuracy WITHIN a support band, it is adding information; if the
    # separation collapses, it was only ever re-measuring how often a cell was observed.
    print("\n  === conditional: round-trip d1 spread WITHIN support quintiles ===")
    sup = support[valid].numpy()
    r1 = res[depths[0]][valid].numpy()
    cc = correct[valid].numpy()
    qs = np.quantile(sup, [0.2, 0.4, 0.6, 0.8])
    band = np.digitize(sup, qs)
    report["conditional"] = {}
    for b in range(5):
        m = band == b
        if m.sum() < 500:
            continue
        o = np.argsort(r1[m], kind="stable")
        lo = cc[m][o[: len(o) // 5]].mean()      # lowest-residual fifth
        hi = cc[m][o[-len(o) // 5:]].mean()      # highest-residual fifth
        print(f"    support band {b} (n={int(m.sum()):,}, median sup {np.median(sup[m]):7.1f}): "
              f"acc lo-resid {lo:.4f} | hi-resid {hi:.4f} | separation {lo-hi:+.4f}")
        report["conditional"][f"band{b}"] = {"n": int(m.sum()), "lo": float(lo),
                                             "hi": float(hi), "sep": float(lo - hi)}

    if a.out_json:
        np.savez_compressed(a.out_json.replace(".json", "_signals.npz"),
                            support=support.numpy(), correct=correct.numpy(),
                            valid=valid.numpy(),
                            **{f"r{i}": res[i].numpy() for i in depths})
        with open(a.out_json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  wrote {a.out_json} (+ _signals.npz)")


if __name__ == "__main__":
    main()
