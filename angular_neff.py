"""How angularly independent are a cell's views? ScanNet vs LERF-OVS.

WHY. Multi-view lifting implicitly assumes a cell's observations are several pieces of evidence. If
the camera only ever sees that cell from one direction, they are one piece of evidence repeated --
they share the same occluder, the same mask boundary, the same grazing angle -- and averaging or
taking a median over them cannot recover anything. Measured on ScanNet scene0062_00: 6 views per
cell but an angular effective count of 1.01, i.e. 83% of the votes are redundant. That single number
explains why consensus errors dominate there and why robust aggregation has so little to work with.

The natural question is whether that is a property of lifting or of the CAPTURE. ScanNet is a
handheld sweep past surfaces; LERF-OVS orbits its objects. If LERF's angular effective count is
genuinely larger, then angular-diversity weighting -- discounting a cluster of near-identical views
so a far-side view is not outvoted by redundant neighbours -- becomes a real lever there, and the
same idea is simply inapplicable to ScanNet.

CLOSED FORM. With unit viewing directions u_v for the n views of a cell, the standard
equicorrelation effective sample size n/(1+(n-1)rho) with rho the mean pairwise cosine reduces to

    n_eff  =  n^2 / || sum_v u_v ||^2

because sum over ALL ordered pairs of u_i . u_j is exactly ||sum u||^2. So the whole statistic needs
one running 3-vector and one counter per cell -- no pairwise matrix, no per-cell loop.
n_eff = 1 means every view is from the same direction; n_eff = n means mutually orthogonal.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCANNET = ["scene0062_00", "scene0347_00", "scene0000_00"]
LERF = ["figurines", "ramen", "teatime", "waldo_kitchen"]


def run(ckpt_dir, label, weight_floor, max_views):
    import configargparse
    import warp as wp

    from build_true_facet_graph import load_points_radii
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    wp.init()
    pr = configargparse.ArgParser(); add_group(pr, Params)
    pr.add_argument("-c", "--config", is_config_file=True)
    cargs = pr.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs); dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ckpt_dir}/model.pt")
    P = model.points.shape[0]
    cc, _ = load_points_radii(ckpt_dir)
    centres = np.asarray(cc, np.float64)

    S = np.zeros((P, 3))
    n = np.zeros(P)
    ids = range(len(dh.cameras)) if max_views == 0 else range(min(max_views, len(dh.cameras)))
    for k in ids:
        cam = dh.cameras[k]
        op = export_operator_for_views(model, [cam], [k])
        cols = op.col_indices.cpu().numpy()
        vals = op.values.cpu().numpy().astype(np.float64)
        tot = np.bincount(cols, weights=vals, minlength=P)
        seen = tot > weight_floor
        if not seen.any():
            continue
        eye = cam.eye.detach().cpu().numpy().astype(np.float64).reshape(3)
        d = eye[None, :] - centres[seen]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        S[seen] += d
        n[seen] += 1.0
    m = n >= 2
    neff = np.ones(P)
    neff[m] = n[m] ** 2 / np.maximum((S[m] ** 2).sum(1), 1e-12)
    neff = np.clip(neff, 1.0, None)
    # mean pairwise angle implied by the resultant: rho = (||S||^2 - n) / (n(n-1))
    rho = np.zeros(P)
    rho[m] = ((S[m] ** 2).sum(1) - n[m]) / np.maximum(n[m] * (n[m] - 1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(rho[m], -1, 1)))
    print(f"{label:>22} {len(ids):>5} {int(m.sum()):>9,} {n[m].mean():>7.2f} "
          f"{neff[m].mean():>8.2f} {np.median(neff[m]):>8.2f} "
          f"{100*(1-neff[m].mean()/max(n[m].mean(),1e-9)):>8.1f}% {ang.mean():>9.1f}")
    return {"label": label, "views": len(ids), "cells": int(m.sum()),
            "n_mean": float(n[m].mean()), "neff_mean": float(neff[m].mean()),
            "neff_median": float(np.median(neff[m])), "ang_deg": float(ang.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-floor", type=float, default=1e-3)
    ap.add_argument("--max-views", type=int, default=60,
                    help="cap per scene for speed; 0 = all")
    ap.add_argument("--scannet", nargs="*", default=SCANNET)
    ap.add_argument("--lerf", nargs="*", default=LERF)
    a = ap.parse_args()
    from determinism import enable_determinism
    enable_determinism()

    print(f"{'scene':>22} {'views':>5} {'cells':>9} {'n/cell':>7} "
          f"{'n_eff':>8} {'median':>8} {'redund':>9} {'mean ang':>9}")
    out = []
    for s in a.scannet:
        ck = f"output/scannet_{s}_truefrozen"
        if os.path.isdir(ck):
            out.append(run(ck, f"ScanNet {s}", a.weight_floor, a.max_views))
    for s in a.lerf:
        ck = f"output/lerf_ovs_{s}"
        if os.path.isdir(ck):
            out.append(run(ck, f"LERF {s}", a.weight_floor, a.max_views))

    sn = [o for o in out if o["label"].startswith("ScanNet")]
    lf = [o for o in out if o["label"].startswith("LERF")]
    if sn and lf:
        print(f"\nScanNet mean n_eff {np.mean([o['neff_mean'] for o in sn]):.2f}  "
              f"(mean pairwise angle {np.mean([o['ang_deg'] for o in sn]):.1f} deg)")
        print(f"LERF    mean n_eff {np.mean([o['neff_mean'] for o in lf]):.2f}  "
              f"(mean pairwise angle {np.mean([o['ang_deg'] for o in lf]):.1f} deg)")
        r = np.mean([o['neff_mean'] for o in lf]) / max(np.mean([o['neff_mean'] for o in sn]), 1e-9)
        print(f"-> LERF cells carry {r:.2f}x the angular independence of ScanNet cells")


if __name__ == "__main__":
    main()
