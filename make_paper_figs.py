"""Paper A figures: how the two representations distribute evidence, and what that buys.

Three panels, PowerFoam (frozen) against 3DGS (frozen), pooled over the ten ScanNet scenes:

  fig1  primitives per ray      -- how many primitives a single camera ray touches
  fig2  rays per primitive      -- how many rays deposit evidence on a single primitive
  fig3  purity                  -- twin panel: the distribution of per-primitive purity, and
                                   accuracy as a function of it

PURITY is computed BOTH ways:
  mass  -- each ray counts alpha*T: (weight of the top class) / (total weight), i.e. `ev_top_share`
  count -- each ray counts 1: (rays carrying the top class) / (total rays reaching the primitive)
MASS IS THE PAPER FIGURE. The readout is W = AtS/D and its argmax is taken over mass, so mass purity
is the quantity that mechanically decides whether a primitive is labelled correctly -- the right
x-axis for a plot whose y-axis is accuracy. Count purity describes the partition but is only
correlated with the outcome, and the two can even disagree about WHICH class is top; that
disagreement rate is printed at the end and bounds how well the count axis could ever work.
The count version is emitted separately as supplementary.

The accuracy curve attributes each scored GT point to the primitive that owns it (nearest live
centre, the same rule the reported mIoU uses), so a bin's accuracy is the accuracy of the points
owned by primitives in that purity bin -- point-weighted, not primitive-weighted. Bins holding
fewer than --min-bin points are dropped rather than plotted as noise.
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("truefrozen", "PowerFoam", "#1f77b4"), ("gs_froz", "3DGS", "#d62728")]


def load(d, arm):
    """Pool every scene for one arm. Returns dict of concatenated per-primitive arrays."""
    fs = sorted(glob.glob(os.path.join(d, f"{arm}_scene*.npz")))
    if not fs:
        raise FileNotFoundError(f"no npz for arm {arm} in {d}")
    out = {k: [] for k in ("rays_per_prim", "purity_count", "purity_mass",
                           "n_evidence_rays", "n_points", "n_correct", "top_class_agree")}
    hist = None
    for f in fs:
        z = np.load(f)
        live = z["live"].astype(bool)
        for k in out:
            out[k].append(z[k][live])
        h = z["hist_prims_per_ray"]
        hist = h.copy() if hist is None else hist + h[:len(hist)] if len(h) >= len(hist) else hist
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["hist_prims_per_ray"] = hist
    res["n_scenes"] = len(fs)
    return res


def fig_prims_per_ray(data, out, xmax):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for arm, lbl, col in ARMS:
        h = data[arm]["hist_prims_per_ray"].astype(float)
        x = np.arange(len(h))
        m = x <= xmax
        tot = h.sum()
        ax.bar(x[m], h[m] / tot, width=0.9, alpha=0.55, color=col,
               label=f"{lbl} (mean {np.average(x, weights=h):.2f})")
    ax.set_xlabel("primitives per ray")
    ax.set_ylabel("fraction of rays")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Primitives contributing to one ray")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


def fig_rays_per_prim(data, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    lo, hi = 1, max(data[a]["rays_per_prim"].max() for a, _, _ in ARMS)
    bins = np.logspace(0, np.log10(max(hi, 10)), 60)
    for arm, lbl, col in ARMS:
        v = data[arm]["rays_per_prim"].astype(float)
        v = v[v >= lo]
        ax.hist(v, bins=bins, alpha=0.55, color=col, density=True,
                label=f"{lbl} (median {np.median(v):,.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("rays per primitive")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title("Rays depositing evidence on one primitive")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


def fig_purity(data, out, key, what, xlab, nbins, min_bin):
    """Distribution of per-primitive purity, and the accuracy of the points those primitives own."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6), sharex=True)
    edges = np.linspace(0, 1, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for arm, lbl, col in ARMS:
        d = data[arm]
        p = d[key]
        ax1.hist(p, bins=edges, alpha=0.55, color=col, density=True,
                 label=f"{lbl} (median {np.median(p):.3f})")
        idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
        npt = np.bincount(idx, weights=d["n_points"], minlength=nbins)
        ncor = np.bincount(idx, weights=d["n_correct"], minlength=nbins)
        keep = npt >= min_bin
        ax2.plot(ctr[keep], 100.0 * ncor[keep] / npt[keep], "o-", color=col, ms=3.5, label=lbl)
        drop = int((~keep).sum())
        if drop:
            print(f"    {lbl} [{what}]: {drop}/{nbins} bins dropped (<{min_bin} points)")
    ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(f"Per-primitive purity ({what})")
    ax2.set_xlabel(f"purity  =  {xlab}")
    ax2.set_ylabel("accuracy of owned points (%)")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--outdir", default="artifacts/figs")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--min-bin", type=int, default=500)
    ap.add_argument("--ppr-xmax", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = {arm: load(a.stats, arm) for arm, _, _ in ARMS}
    for arm, lbl, _ in ARMS:
        d = data[arm]
        print(f"{lbl}: {d['n_scenes']} scenes, {len(d['rays_per_prim']):,} live primitives")
    fig_prims_per_ray(data, os.path.join(a.outdir, "fig_prims_per_ray.pdf"), a.ppr_xmax)
    fig_rays_per_prim(data, os.path.join(a.outdir, "fig_rays_per_prim.pdf"))
    # mass is the paper figure (the solver's argmax follows mass); count is supplementary
    fig_purity(data, os.path.join(a.outdir, "fig_purity.pdf"), "purity_mass",
               "evidence mass", "mass of top class / total mass", a.bins, a.min_bin)
    fig_purity(data, os.path.join(a.outdir, "fig_purity_count_supp.pdf"), "purity_count",
               "ray counts", "rays of top class / total rays", a.bins, a.min_bin)

    # Do the two definitions agree on which class is top? Where they disagree, a count-purity
    # x-axis cannot explain the prediction, because the argmax follows mass.
    print()
    print("count-purity vs mass-purity:")
    for arm, lbl, _ in ARMS:
        d = data[arm]
        ag = d["top_class_agree"].astype(bool)
        w = d["n_points"].astype(float)
        r = np.corrcoef(d["purity_count"], d["purity_mass"])[0, 1]
        print(f"  {lbl:<10} same top class: {ag.mean() * 100:.2f}% of primitives, "
              f"{(w[ag].sum() / max(w.sum(), 1)) * 100:.2f}% of scored points   "
              f"| corr(count, mass) = {r:.3f}   "
              f"| median count {np.median(d['purity_count']):.3f} vs mass {np.median(d['purity_mass']):.3f}")


if __name__ == "__main__":
    main()
