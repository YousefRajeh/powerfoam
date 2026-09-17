"""Cross-surface co-visibility fraction, streamed -- works on any arm, builds no gram cache.

measure_cross_surface_gram.py reads the cached S = A^T A, which exists only for the NONFROZEN arm.
The paper's matched-budget table is the FROZEN arm, so this script recomputes the same quantity
directly from the checkpoint, accumulating only four scalars per view instead of storing edges:

    mass_off      = sum over rays of sum_{t != s} w_t w_s        (all co-visibility mass)
    mass_adjacent = the part of it on power-diagram-adjacent pairs
    mass_diag     = sum_j diag(S)_jj = sum_i sum_j A_ij^2
    sum_D         = sum_j (S 1)_j = sum_i s_i^2

    cross-surface fraction = 1 - mass_adjacent / mass_off

Peak memory is one view's pair expansion, not the whole edge set, so the 266M-edge scenes cost the
same as the small ones. See measure_cross_surface_gram.py for why this fraction is the quantity
that decides whether a surface-block preconditioner is worth building.

INSTRUMENT CHECKS: sum_D must equal mass_diag + mass_off (Lemma 1, in the sum_j (S1)_j form), and
on the nonfrozen arm the result must reproduce the cached-gram number to within the merge
tolerance -- run with --variant nonfrozen to confirm before trusting the frozen number.

Usage:
    python measure_cross_surface_streaming.py --scene scene0070_00 --variant truefrozen
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import configargparse
import numpy as np
import torch
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

MAX_HITS = 64
PAIR_BUDGET = 1 << 26

ADJ_FILE = {"truefrozen": "adjacency_true_facet_frozen.pt",
            "frozen": "adjacency_true_facet_frozen.pt",
            "nonfrozen": "adjacency_nonfrozen.pt"}


def load_adjacency_keys(scene: str, variant: str, P: int, device,
                        adj_file: str | None = None) -> torch.Tensor:
    """Sorted int64 tensor of lo*P + hi over undirected power-diagram edges.

    `adj_file` overrides the per-variant default. The frozen arm only ships the true-facet
    graph, so comparing arms on their defaults compares two different graph constructions --
    use the override to put both arms on the same one before drawing a conclusion.
    """
    path = f"artifacts/scannet/{scene}/{adj_file or ADJ_FILE[variant]}"
    d = torch.load(path, map_location="cpu", weights_only=False)
    Pa = int(d["num_primitives"])
    assert Pa == P, f"{path}: adjacency P={Pa} != model P={P}"
    adj = d["adjacent"].long()
    off = d["offsets"].long()
    deg = off[1:] - off[:-1]
    src = torch.repeat_interleave(torch.arange(deg.numel(), dtype=torch.long), deg)
    assert src.numel() == adj.numel()
    lo = torch.minimum(src, adj)
    hi = torch.maximum(src, adj)
    keys = torch.unique(lo * P + hi).to(device)
    print(f"  adjacency: {os.path.basename(path)}  {keys.numel():,} undirected edges "
          f"(mean degree {2*keys.numel()/max(P,1):.2f})", flush=True)
    return keys


def sanitized_config(path: str) -> str:
    """Return a config path configargparse can actually parse.

    train.py dumps the resolved config into the run's output dir, and newer revisions emit
    optional scalars as literal `null` (e.g. `max_image_width: null`). configargparse feeds that
    string straight to `int()` and dies with
    `argument --max_image_width: invalid int value: 'null'`. The older checkpoints predate that
    key, which is why reading their config.yaml works and reading a fresh run's does not.

    Dropping the null lines restores the argparse default, which is what `null` meant. The
    original file is never modified; a sibling `.sanitized.yaml` is written next to it.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    keep = [ln for ln in lines if not re.match(r"^\s*[A-Za-z_][\w.]*\s*:\s*null\s*$", ln)]
    if len(keep) == len(lines):
        return path
    out = path.replace(".yaml", ".sanitized.yaml")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(keep)
    dropped = [ln.strip() for ln in lines if ln not in keep]
    print(f"  config: dropped {len(dropped)} null-valued key(s) -> {os.path.basename(out)} "
          f"({', '.join(d.split(':')[0] for d in dropped)})", flush=True)
    return out


def gt_distance(scene: str, centers: torch.Tensor, gt_dir: str) -> np.ndarray:
    """Distance from each cell centre to the nearest ground-truth vertex.

    The matched-budget (frozen) arm is built with one primitive per GT vertex, so every one of its
    cells has distance ~0 by construction. The deployed (nonfrozen) arm does not: cells with no GT
    vertex nearby are floaters and near-surface excess, and they are exactly what the matched-budget
    protocol removes. Splitting the co-visibility mass by this distance says how much of the lift's
    error budget that removal hides.
    """
    from scipy.spatial import cKDTree
    coord = np.load(os.path.join(gt_dir, scene, "coord.npy")).astype(np.float64)
    c = centers.detach().float().cpu().numpy().astype(np.float64)
    lo = np.minimum(coord.min(0), c.min(0))
    hi = np.maximum(coord.max(0), c.max(0))
    assert np.all(hi - lo < 100.0), f"{scene}: GT and checkpoint frames look inconsistent"
    d, _ = cKDTree(coord).query(c, k=1, workers=-1)
    print(f"  GT: {len(coord):,} vertices; cell->GT distance "
          f"median {np.median(d):.4f} m, p95 {np.percentile(d, 95):.4f} m", flush=True)
    return d


def view_masses(cols, vals, slots, P, akeys, dgt=None, thresholds=()):
    """Return (mass_off, mass_adjacent, mass_diag, sum_D) for one view's nonzeros.

    When `dgt` is given, also returns per-threshold cross-surface mass on pairs where at least
    one endpoint has no GT vertex within that distance.
    """
    device = cols.device
    slots = slots.long()
    row_start = torch.cumsum(slots, 0) - slots
    n_at = torch.repeat_interleave(slots, slots)
    start_at = torch.repeat_interleave(row_start, slots)
    nnz = cols.numel()

    v64 = vals.double()
    mass_diag = float((v64 * v64).sum())
    # sum_D = sum_i s_i^2 ; s_i is the row sum
    s_row = torch.zeros(slots.numel(), dtype=torch.float64, device=device)
    s_row.index_add_(0, torch.repeat_interleave(
        torch.arange(slots.numel(), device=device), slots), v64)
    sum_D = float((s_row * s_row).sum())
    mass_off = sum_D - mass_diag                       # sum_{t != s} w_t w_s, ordered pairs

    cum = torch.cumsum(n_at, 0)
    mass_adj = 0.0
    unsup = [0.0] * len(thresholds)
    s = 0
    while s < nnz:
        base = cum[s] - n_at[s]
        e = int(torch.searchsorted(cum, base + PAIR_BUDGET).item())
        e = min(max(e, s + 1), nnz)
        n_chunk = n_at[s:e]
        tot = int(n_chunk.sum())
        left = torch.repeat_interleave(torch.arange(s, e, device=device), n_chunk)
        offs = torch.arange(tot, device=device) - torch.repeat_interleave(
            torch.cumsum(n_chunk, 0) - n_chunk, n_chunk)
        right = start_at[left] + offs
        sel = right != left                            # ordered off-diagonal pairs
        left, right = left[sel], right[sel]
        cj, cl = cols[left], cols[right]
        lo = torch.minimum(cj, cl)
        hi = torch.maximum(cj, cl)
        keys = lo * P + hi
        if akeys.numel():
            pos = torch.searchsorted(akeys, keys).clamp_(max=akeys.numel() - 1)
            hit = akeys[pos] == keys
        else:
            hit = torch.zeros_like(keys, dtype=torch.bool)
        w = v64[left] * v64[right]
        mass_adj += float(w[hit].sum())
        if dgt is not None and len(thresholds):
            cross = ~hit                                   # the cross-surface pairs only
            dj, dl = dgt[cj[cross]], dgt[cl[cross]]
            worst = torch.maximum(dj, dl)                  # pair is "unsupported" if EITHER end is
            wc = w[cross]
            for ti, t in enumerate(thresholds):
                unsup[ti] += float(wc[worst > t].sum())
            del cross, dj, dl, worst, wc
        del left, right, offs, cj, cl, lo, hi, keys, hit, w   # `pos` is bound only when akeys is non-empty
        s = e
    return mass_off, mass_adj, mass_diag, sum_D, unsup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0070_00")
    p.add_argument("--variant", default="truefrozen")
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--gt-dir", default=r"D:\Downloads\scannet_pointcept\train",
                   help="if set, also split cross-surface mass by GT support")
    p.add_argument("--no-gt", action="store_true", help="skip the GT-support split")
    p.add_argument("--gt-thresh", default="0.02,0.05,0.10",
                   help="metres; a cell with no GT vertex within t is unsupported at t")
    p.add_argument("--ckpt-dir", default=None,
                   help="explicit output/ dir; overrides the scannet_<scene>_<variant> convention")
    p.add_argument("--no-adjacency", action="store_true",
                   help="compute overlap mass only; for arms with no adjacency graph on disk")
    p.add_argument("--adj-file", default=None,
                   help="override the adjacency graph, e.g. adjacency_true_facet.pt")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    device = "cuda"
    wp.init()
    ckpt = a.ckpt_dir or f"output/scannet_{a.scene}_{a.variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    # parse_known_args, not parse_args: a run's dumped config also carries training-only options
    # that train.py registers outside Params (ckpt_every, resume, ...). They are irrelevant to a
    # read-only measurement, and hard-failing on them would mean chasing each new one by hand.
    args, ignored = parser.parse_known_args(
        ["-c", sanitized_config(f"{ckpt}/config.yaml")])
    if ignored:
        print(f"  config: ignored {len(ignored)} training-only option(s): "
              f"{' '.join(sorted(ignored))}", flush=True)
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")           # do NOT sort_points/resample: index stability
    cameras = dh.cameras
    P = model.points.shape[0]
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    print(f"{a.scene} / {a.variant}: P={P:,}  views={n_views}", flush=True)

    if a.no_adjacency:
        akeys = torch.zeros(0, dtype=torch.long, device=device)
        print("  adjacency: SKIPPED -- overlap mass only, cross-surface undefined", flush=True)
    else:
        akeys = load_adjacency_keys(a.scene, a.variant, P, device, a.adj_file)

    thresholds = [] if a.no_gt else [float(x) for x in a.gt_thresh.split(",")]
    dgt = None
    if thresholds:
        try:
            dgt = torch.from_numpy(gt_distance(a.scene, model.points, a.gt_dir)).to(device)
        except Exception as exc:
            print(f"  [gt] unavailable ({type(exc).__name__}: {exc}); skipping GT split",
                  flush=True)
            dgt, thresholds = None, []

    tot = dict(mass_off=0.0, mass_adj=0.0, mass_diag=0.0, sum_D=0.0)
    unsup_tot = [0.0] * len(thresholds)
    t0 = time.time()
    ar = torch.arange(MAX_HITS, device=device)
    for vi in range(n_views):
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cameras[vi], max_intersections=1024, max_hits_per_pixel=MAX_HITS)
        slots_used = slots.reshape(-1).clamp(max=MAX_HITS)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vv = out_val.reshape(-1)[keep]
        mo, ma, md, sd, un = view_masses(cols, vv, slots_used, P, akeys, dgt, thresholds)
        tot["mass_off"] += mo; tot["mass_adj"] += ma
        tot["mass_diag"] += md; tot["sum_D"] += sd
        for ti in range(len(thresholds)):
            unsup_tot[ti] += un[ti]
        del out_col, out_val, cols, vv, slots_used, keep
        torch.cuda.empty_cache()
        if (vi + 1) % 10 == 0 or vi == n_views - 1:
            f = 1.0 - tot["mass_adj"] / max(tot["mass_off"], 1e-30)
            print(f"  view {vi+1}/{n_views}: cross-surface {f:6.2%}  ({time.time()-t0:.0f}s)",
                  flush=True)

    cross = 1.0 - tot["mass_adj"] / max(tot["mass_off"], 1e-30)
    overlap = tot["mass_off"] / max(tot["sum_D"], 1e-30)
    rel = abs(tot["sum_D"] - (tot["mass_diag"] + tot["mass_off"])) / max(tot["sum_D"], 1e-30)

    print(f"\n=== {a.scene} / {a.variant} ===")
    print(f"  mean per-ray overlap mass o   {overlap:.4f}")
    if akeys.numel():
        print(f"  co-visibility mass, adjacent  {1-cross:.2%}")
        print(f"  co-visibility mass, CROSS     {cross:.2%}")
    else:
        cross = float("nan")
    print(f"  [check] sum_D vs diag+off rel {rel:.2e}"
          f"{'   <-- MISMATCH' if rel > 1e-6 else ''}")
    cross_mass = tot["mass_off"] - tot["mass_adj"]
    gt_split = {}
    for ti, t in enumerate(thresholds):
        frac = unsup_tot[ti] / max(cross_mass, 1e-30)
        gt_split[f"{t:g}"] = frac
        print(f"  of the CROSS mass, >= one endpoint with no GT within {t:.2f} m: {frac:.2%}")

    row = dict(scene=a.scene, variant=a.variant, P=P, n_views=n_views,
               cross_surface_fraction=cross, mean_overlap_mass=overlap,
               consistency_rel_err=rel, cross_unsupported=gt_split, **tot)
    out = a.out or f"artifacts/scannet/{a.scene}/cross_surface_{a.variant}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(row, open(out, "w"), indent=1)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
