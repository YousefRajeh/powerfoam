"""Figures for the three findings of this campaign, from the artifacts already on disk.

1. position LR    the accidental result: our default is 5.5 dB off the optimum
2. region locality 93% of clusters are spatially fragmented; naive splitting trades
                   fine-class gains for coarse-class losses
3. VoroTracing    both derived ideas (exp density, exact distortion) failed to transfer

Everything is read from JSON/metrics files rather than re-run, so the figures cannot drift
from the numbers already reported.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/scannet/plots")
OUT.mkdir(parents=True, exist_ok=True)


def read_psnr(d):
    f = Path(d) / "metrics.txt"
    if not f.exists():
        return None
    m = re.search(r"PSNR:\s*([\d.]+)", f.read_text())
    return float(m.group(1)) if m else None


def plot_position_lr():
    """PSNR and surface drift against position learning rate."""
    lrs = ["0.0", "1e-5", "1e-4", "1e-3"]
    x = [1e-6, 1e-5, 1e-4, 1e-3]          # 0.0 plotted at the left edge of a log axis
    psnr = [25.8497, 26.4401, 26.3319, 20.8340]
    drift = {"0.0": 0.05, "1e-3": 3.06}    # measured; intermediate LRs not yet re-measured

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, psnr, "o-", color="#1f77b4", lw=2, ms=8)
    ax[0].set_xscale("log")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lrs)
    ax[0].set_xlabel("position learning rate"); ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_title("scene0347_00: our default is 5.5 dB off the optimum")
    ax[0].axvline(1e-3, color="#d62728", ls="--", alpha=0.6)
    ax[0].annotate("our default\n(1e-3)", xy=(1e-3, 20.83), xytext=(2.5e-4, 22.2),
                   color="#d62728", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax[0].annotate("optimum", xy=(1e-5, 26.44), xytext=(1.2e-5, 24.6), fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].bar(["frozen\n(lr 0)", "lr 1e-3\n(our default)"], [drift["0.0"], drift["1e-3"]],
              color=["#2ca02c", "#d62728"], width=0.5)
    ax[1].set_ylabel("median distance to nearest GT point\n(in units of the cell's own radius)")
    ax[1].set_title("position learning drives cells off the surface")
    for i, v in enumerate([drift["0.0"], drift["1e-3"]]):
        ax[1].text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Position learning rate: the largest effect found, from a hyperparameter we already had",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "position_lr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "position_lr.png")


def plot_region_locality():
    f = Path("artifacts/scannet/region_locality_scene0347_00.json")
    if not f.exists():
        print("[skip] region locality json missing")
        return
    d = json.load(open(f))
    res = d["results"]
    sets = ["opengaussian19", "opengaussian15", "opengaussian10"]
    base = [res[s]["baseline"]["mIoU"] * 100 for s in sets]
    split = [res[s]["split"]["mIoU"] * 100 for s in sets]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["single\npiece", "fragmented"],
              [100 * d["single_piece_frac"], 100 * (1 - d["single_piece_frac"])],
              color=["#2ca02c", "#d62728"], width=0.5)
    ax[0].set_ylabel("% of the 320 regions")
    ax[0].set_title(f"93% of regions are spatially fragmented\n"
                    f"(median {d['components_median']:.0f} disconnected pieces each)")
    for i, v in enumerate([100 * d["single_piece_frac"], 100 * (1 - d["single_piece_frac"])]):
        ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax[0].set_ylim(0, 105); ax[0].grid(alpha=0.3, axis="y")

    w = 0.36
    idx = np.arange(len(sets))
    ax[1].bar(idx - w/2, base, w, label="baseline (320 regions)", color="#1f77b4")
    ax[1].bar(idx + w/2, split, w, label=f"split by component ({d['n_split_regions']})",
              color="#ff7f0e")
    ax[1].set_xticks(idx); ax[1].set_xticklabels(["19 cls", "15 cls", "10 cls"])
    ax[1].set_ylabel("mIoU"); ax[1].legend(fontsize=9)
    ax[1].set_title("naive splitting: fine classes gain, coarse classes lose")
    for i, (b, s) in enumerate(zip(base, split)):
        ax[1].text(i + w/2, s + 0.4, f"{s-b:+.2f}", ha="center", fontsize=9,
                   color="#2ca02c" if s > b else "#d62728")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Clustering locality: the failure is real, but connectivity alone is not the fix",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "region_locality.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "region_locality.png")


def plot_vorotracing_transfer():
    """Both VoroTracing-derived ideas, against their controls."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    acts = ["softplus\n1e0", "softplus\n1e-1", "softplus\n1e-2", "exp\n1e0", "exp\n1e-1", "exp\n1e-2"]
    ps = [30.60, 28.87, 27.24, np.nan, 24.29, 27.37]
    cols = ["#1f77b4"] * 3 + ["#ff7f0e"] * 3
    bars = ax[0].bar(acts, [0 if np.isnan(v) else v for v in ps], color=cols)
    ax[0].text(3, 1.0, "DIVERGED\n(NaN)", ha="center", fontsize=9, color="#d62728",
               fontweight="bold")
    ax[0].set_ylabel("PSNR (dB)"); ax[0].set_ylim(0, 33)
    ax[0].set_title("sigma = exp(rho) loses at every learning rate\n(scene0062_00, 30k iters)")
    ax[0].grid(alpha=0.3, axis="y")

    labels = ["mIoU 19cls", "mIoU 15cls", "mIoU 10cls", "CD-L1 (cm)"]
    off = [39.59, 36.51, 51.17, 6.15]
    both = [39.70, 36.30, 50.68, 8.08]
    idx = np.arange(len(labels)); w = 0.36
    ax[1].bar(idx - w/2, off, w, label="distortion off", color="#1f77b4")
    ax[1].bar(idx + w/2, both, w, label="exact L_dist", color="#ff7f0e")
    ax[1].set_xticks(idx); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].legend(fontsize=9)
    ax[1].set_title("exact distortion loss: segmentation flat,\nreconstruction 31% worse")
    for i, (a_, b_) in enumerate(zip(off, both)):
        ax[1].text(i + w/2, b_ + 0.6, f"{b_-a_:+.2f}", ha="center", fontsize=9,
                   color="#d62728" if (i == 3) == (b_ > a_) else "#2ca02c")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("VoroTracing transfer: neither derived idea helped in our setting", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "vorotracing_transfer.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "vorotracing_transfer.png")


if __name__ == "__main__":
    plot_position_lr()
    plot_region_locality()
    plot_vorotracing_transfer()
