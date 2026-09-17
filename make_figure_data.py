"""Freeze every number behind the paper's evidence figures/tables into one JSON.

The figures and the table are two presentations of the same measurement, and which one we want may
change. So this stores the quantities at FINER granularity than any figure draws them:

  * evidence decomposition is per EXACT class count k = 1, 2, 3, ... (the 1/2/>=3 grouping in the
    table is a presentation choice applied afterwards, so the same data can be redrawn as a
    histogram or re-binned without re-running the oracle)
  * histograms store bin EDGES and raw COUNTS, never densities, so bins can be merged and scenes
    pooled by addition
  * the purity panels store n_primitives, n_points and n_correct per bin, so accuracy is
    recomputable at any binning rather than baked in at 25 bins

Raw counts, not fractions, are the unit throughout: fractions cannot be re-pooled across scenes,
counts can. Per-scene rows are kept alongside the pooled totals so any subset can be re-aggregated.

The per-primitive .npz dumps remain the source of truth; this is the reproducible summary layer.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np

ARMS = ["truefrozen", "nonfrozen", "gs_froz", "gs_unfroz"]
FIG_ARMS = ["truefrozen", "gs_froz"]


def evidence_block(rows, arm):
    """Per-exact-k decomposition, pooled by raw point counts (not by averaging fractions)."""
    s = [r for r in rows if r["recon"] == arm]
    if not s:
        return None
    kmax = max(r["C"] for r in s)
    per_k, tot_n = {}, 0
    for k in range(1, kmax + 1):
        nk = sum(r.get(f"b_evk{k}_n_closed", 0) for r in s)
        if nk == 0:
            continue
        # accuracy/mIoU pooled by the points each scene contributed to this k
        acc = sum(r.get(f"b_evk{k}_n_closed", 0) * r.get(f"b_evk{k}_acc_closed", 0.0) for r in s) / nk
        per_k[str(k)] = {
            "n_points": int(nk),
            "accuracy": float(acc),
            "per_scene": {r["scene"]: {"n_points": int(r.get(f"b_evk{k}_n_closed", 0)),
                                       "accuracy": r.get(f"b_evk{k}_acc_closed"),
                                       "miou": r.get(f"b_evk{k}_miou_closed"),
                                       "n_classes": r.get(f"b_evk{k}_ncls_closed")}
                          for r in s if r.get(f"b_evk{k}_n_closed", 0) > 0},
        }
        tot_n += nk
    grouped = {}
    for g, keys in (("1", ["ev1"]), ("2", ["ev2"]), ("3+", ["ev3p"])):
        n = sum(r.get(f"b_{keys[0]}_n_closed", 0) for r in s)
        if n:
            grouped[g] = {
                "n_points": int(n),
                "accuracy": float(sum(r.get(f"b_{keys[0]}_n_closed", 0) * r[f"b_{keys[0]}_acc_closed"]
                                      for r in s) / n),
                "miou_per_scene": {r["scene"]: r.get(f"b_{keys[0]}_miou_closed") for r in s},
                "n_classes_per_scene": {r["scene"]: r.get(f"b_{keys[0]}_ncls_closed") for r in s},
            }
    return {"total_points": int(tot_n), "per_k": per_k, "grouped_for_table": grouped,
            "n_scenes": len(s),
            "overall": {"accuracy": float(np.mean([r["pt_acc_centre_closed"] for r in s])),
                        "miou": float(np.mean([r["pt_miou_centre_closed"] for r in s]))}}


def hist_counts(v, edges):
    return np.histogram(v, bins=edges)[0].astype(np.int64)


def figure_block(stats_dir, arm, purity_bins, rays_bins):
    fs = sorted(glob.glob(os.path.join(stats_dir, f"{arm}_scene*.npz")))
    if not fs:
        return None
    ppr = None
    rpp, pc, pm, npt, ncor, agree = [], [], [], [], [], []
    per_scene = {}
    for f in fs:
        z = np.load(f)
        live = z["live"].astype(bool)
        h = z["hist_prims_per_ray"].astype(np.int64)
        if ppr is None:
            ppr = h.copy()
        else:
            n = max(len(ppr), len(h))
            a = np.zeros(n, np.int64); a[:len(ppr)] = ppr
            b = np.zeros(n, np.int64); b[:len(h)] = h
            ppr = a + b
        rpp.append(z["rays_per_prim"][live]); pc.append(z["purity_count"][live])
        pm.append(z["purity_mass"][live]); npt.append(z["n_points"][live])
        ncor.append(z["n_correct"][live]); agree.append(z["top_class_agree"][live])
        P, nlive, C, nviews = z["meta"]
        per_scene[os.path.basename(f).replace(f"{arm}_", "").replace(".npz", "")] = {
            "P": int(P), "live": int(nlive), "C": int(C), "views": int(nviews)}
    rpp = np.concatenate(rpp); pc = np.concatenate(pc); pm = np.concatenate(pm)
    npt = np.concatenate(npt).astype(np.int64); ncor = np.concatenate(ncor).astype(np.int64)
    agree = np.concatenate(agree).astype(bool)

    pe = np.linspace(0, 1, purity_bins + 1)
    out = {"n_scenes": len(fs), "n_live_primitives": int(rpp.size), "per_scene": per_scene}
    # exact, unbinned: index i holds the number of rays that touched exactly i primitives
    out["prims_per_ray"] = {"counts_by_exact_value": ppr.tolist(),
                            "total_rays": int(ppr.sum()),
                            "mean": float(np.average(np.arange(len(ppr)), weights=ppr))}
    redges = np.unique(np.round(np.logspace(0, np.log10(max(float(rpp.max()), 10.0)),
                                            rays_bins + 1)).astype(np.int64))
    out["rays_per_primitive"] = {
        "bin_edges": redges.tolist(),
        "counts": hist_counts(rpp, redges).tolist(),
        "quantiles": {q: float(np.quantile(rpp, float(q))) for q in ("0.05", "0.25", "0.5",
                                                                    "0.75", "0.95")},
        "mean": float(rpp.mean())}
    for name, v in (("purity_mass", pm), ("purity_count", pc)):
        idx = np.clip(np.digitize(v, pe) - 1, 0, purity_bins - 1)
        out[name] = {
            "bin_edges": pe.tolist(),
            "n_primitives": np.bincount(idx, minlength=purity_bins).astype(np.int64).tolist(),
            "n_points": np.bincount(idx, weights=npt, minlength=purity_bins).astype(np.int64).tolist(),
            "n_correct": np.bincount(idx, weights=ncor, minlength=purity_bins).astype(np.int64).tolist(),
            "median": float(np.median(v))}
    w = npt.astype(float)
    out["top_class_agreement"] = {
        "frac_primitives": float(agree.mean()),
        "frac_scored_points": float(w[agree].sum() / max(w.sum(), 1.0)),
        "corr_count_vs_mass": float(np.corrcoef(pc, pm)[0, 1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/scannet/oracle_buckets_v2.json")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/figure_data.json")
    ap.add_argument("--purity-bins", type=int, default=100)
    ap.add_argument("--rays-bins", type=int, default=200)
    a = ap.parse_args()
    rows = json.load(open(a.rows))
    doc = {"_readme": ("Plot/table source data. Counts are raw so bins can be merged and scenes "
                       "pooled by addition; accuracy is recomputable as n_correct/n_points at any "
                       "binning. evidence.per_k is the exact decomposition; grouped_for_table is "
                       "only a presentation of it. Per-primitive arrays live in the .npz dumps."),
           "protocol": {"labels": "official ScanNet 2D (unfiltered)", "scoring": "visible-only",
                        "ownership": "nearest live centre", "solver": "closed form (Eq. 6)",
                        "scenes": sorted({r["scene"] for r in rows}),
                        "n_scenes": len({r["scene"] for r in rows})},
           "evidence": {}, "figures": {}}
    for arm in ARMS:
        b = evidence_block(rows, arm)
        if b:
            doc["evidence"][arm] = b
    for arm in FIG_ARMS:
        f = figure_block(a.stats, arm, a.purity_bins, a.rays_bins)
        if f:
            doc["figures"][arm] = f
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    for arm in doc["evidence"]:
        e = doc["evidence"][arm]
        ks = sorted(e["per_k"], key=int)
        krange = f"k={ks[0]}..{ks[-1]}" if ks else "k=NONE (rows predate per-k fields)"
        print(f"  {arm:<12} {krange}  {e['total_points']:,} points  "
              f"acc {e['overall']['accuracy']*100:.2f}  mIoU {e['overall']['miou']*100:.2f}")
    if not any(doc["evidence"][k]["per_k"] for k in doc["evidence"]):
        print("  !! no per-k decomposition in these rows -- rerun oracle_projected.py to populate")
    if not doc["figures"]:
        print("  !! no .npz stats found -- rerun with --dump-stats to populate the figures")
    for arm in doc["figures"]:
        f = doc["figures"][arm]
        print(f"  {arm:<12} {f['n_live_primitives']:,} live prims, "
              f"{f['prims_per_ray']['total_rays']:,} rays, "
              f"mean prims/ray {f['prims_per_ray']['mean']:.2f}, "
              f"top-class agree {f['top_class_agreement']['frac_primitives']*100:.1f}%")


if __name__ == "__main__":
    main()
