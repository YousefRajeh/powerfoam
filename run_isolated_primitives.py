"""Fraction of primitives on which the closed-form lift is EXACTLY optimal.

THE THEOREM THIS MEASURES. Build the ray-sharing graph G on primitives: j ~ l if some ray hits
both. A^T A is block-diagonal with respect to G's connected components, so the normal equations
decouple by component. For a component that is a SINGLE primitive -- no ray touching j touches any
other primitive -- row-stochasticity forces A_ij = 1 on those rays, the objective restricted to j is
sum_i ||x_j - B_i||^2, and its minimiser is the plain mean, which is exactly the row-sum estimate:

        x'_j = sum_i A_ij B_i / sum_i A_ij = mean_i B_i = x_hat_j        (exact, not bounded)

So on the ISOLATED set the closed form is not an approximation with a (1+beta) or (1+gamma) gap --
it is the least-squares optimum. Verified on synthetic block-diagonal operators in
test_beta_gamma.py ("no shared rays -> gamma == 0 for any k").

WHY THIS IS THE NUMBER, not frac(k_i == 1). frac(k_i == 1) counts RAYS with a single contributor
and is what Table 1 / Figure 3 report. Isolation is a strictly stronger, PER-PRIMITIVE condition:
EVERY ray touching j must be single-hit. One shared ray couples j to its neighbour and the identity
is lost. So isolated_frac <= frac(k_i == 1) always, and the gap between them is the point.

CHEAP: no CG solve. Only the operator is needed, so this runs in minutes where compute_beta.py
takes an hour on the Gaussian arm.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import

import configargparse
import numpy as np
import torch

from camera_bridge import K_from_ray_dirs
from configs import Params, add_group
from data_loader import DataHandler

OUT = "artifacts/scannet/isolated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--arm", default="pf_truefrozen",
                    choices=("pf_truefrozen", "pf_nonfrozen", "gs_froz", "gs_unfroz"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    recon = "nonfrozen" if a.arm.startswith("gs_") else a.arm[3:]
    cfg = f"output/scannet_{a.scene}_{recon}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()

    if a.arm.startswith("pf_"):
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{a.scene}_{recon}/model.pt")
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{a.arm}/{a.scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].cuda(), sp["quats"].cuda()
        gs_ = torch.exp(sp["scales"].cuda())
        go = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)

    rows, cols, vals, offs = [], [], [], 0
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if a.arm.startswith("pf_"):
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            P = op.num_primitives
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().cuda()
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.cuda(), W, H,
                                                    max_hits_per_pixel=a.cap,
                                                    transmittance_floor=1e-3)
            P = gm.shape[0]
        rows.append(ri.to(torch.int64) + offs)
        cols.append(ci.to(torch.int64))
        vals.append(vv.float())
        offs += H * W
        del ri, ci, vv

    row, col, val = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    del rows, cols, vals
    R, nnz = offs, val.numel()

    k = torch.bincount(row, minlength=R)                    # contributors per ray
    touched = torch.zeros(P, dtype=torch.bool, device=dev)
    touched[col] = True
    n_touched = int(touched.sum())

    # A primitive is ISOLATED iff every ray touching it is single-hit, i.e. max_i k_i == 1 over its
    # rays. scatter_reduce with amax gives that in one pass over the nonzeros.
    maxk = torch.zeros(P, dtype=torch.int64, device=dev)
    maxk.scatter_reduce_(0, col, k[row], reduce="amax", include_self=True)
    isolated = touched & (maxk == 1)
    n_iso = int(isolated.sum())

    support = torch.zeros(P, device=dev).index_add_(0, col, val)   # render mass per primitive
    tot_sup = float(support.sum())
    iso_sup = float(support[isolated].sum())

    # Ray-side comparison: the statistic Table 1 already reports.
    live = k > 0
    frac_k1 = float((k[live] == 1).float().mean())
    # Rays all of whose contributors are isolated -- these rays are fully explained exactly.
    ray_iso = torch.zeros(R, dtype=torch.bool, device=dev)
    ray_iso[row[isolated[col]]] = True

    res = {"scene": a.scene, "arm": a.arm, "views": a.views, "P": int(P),
           "P_touched": n_touched, "nnz": int(nnz), "rays_live": int(live.sum()),
           "frac_k1_rays": frac_k1,
           "n_isolated": n_iso,
           "frac_isolated_of_touched": n_iso / max(n_touched, 1),
           "frac_isolated_of_all_P": n_iso / max(P, 1),
           "frac_support_isolated": iso_sup / max(tot_sup, 1e-12),
           "mean_k": float(k[live].float().mean()),
           "max_k": int(k.max())}
    json.dump(res, open(f"{OUT}/{a.arm}_{a.scene}.json", "w"), indent=1)
    print(f"[{a.arm}/{a.scene}] P={P:,} touched={n_touched:,} nnz={nnz:,}", flush=True)
    print(f"  frac(k_i==1) over rays      = {frac_k1*100:.2f}%   (Table 1's statistic)", flush=True)
    print(f"  ISOLATED primitives         = {n_iso:,} / {n_touched:,} touched "
          f"= {res['frac_isolated_of_touched']*100:.2f}%", flush=True)
    print(f"  share of render mass        = {res['frac_support_isolated']*100:.2f}%", flush=True)
    print(f"  mean k = {res['mean_k']:.2f}   max k = {res['max_k']}", flush=True)


if __name__ == "__main__":
    main()
