"""Overlap mass `mean_o` for the arms whose beta runs predate the field, without re-solving.

`o_i = (sum_j A_ij)^2 - sum_j A_ij^2` is an operator-only quantity: it needs the rasteriser's
weights and nothing else -- no CG, no residuals, no ground truth. `compute_beta.py` computes it, but
only as a side effect of a 300-iteration CG that costs hours on the 3DGS arm (509M nonzeros on
scene0070). Every 3DGS scene except scene0000 was measured before `sum_o` was added, so the
cross-arm claim currently rests on n=1.

This streams one view at a time and accumulates two scalars, so it is minutes rather than hours.
It reproduces `compute_beta.py` exactly:

    sel = linspace(0, ncam-1, views)          same view subset
    cap = 512, transmittance_floor = 1e-3     same operator
    mean_o = sum_i o_i / sum_i (row sum)      per unit ray mass, NOT per ray

VALIDATION IS NOT OPTIONAL: scene0000 already has `mean_o` from the full path, so this script
recomputes it first and refuses to continue if it disagrees. A fast path that silently disagrees
with the slow one would poison the only claim the bound still supports.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from camera_bridge import K_from_ray_dirs

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
REF = {"scene0000_00": 0.8275798390263883}      # from the full compute_beta.py path


def one(scene, arm, views, cap, dev="cuda"):
    from gsplat_baseline.export_gsplat_operator import export_view_operator
    cfg = f"output/scannet_{scene}_truefrozen/config.yaml"      # cameras only; arm-independent
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, views).astype(int).tolist()

    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=dev, weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
    gs_ = torch.exp(sp["scales"].to(dev))
    go = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
    gc = torch.zeros((gm.shape[0], 1), device=dev)

    sum_o = 0.0
    sum_mass = 0.0
    nnz = 0
    o_chunks = []
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        K, _ = K_from_ray_dirs(cam)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        vm = torch.linalg.inv(c2w).float().to(dev)
        ri, _, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.to(dev), W, H,
                                               max_hits_per_pixel=cap, transmittance_floor=1e-3)
        r = ri.to(torch.int64).to(dev); v = vv.float().to(dev)
        nnz += v.numel()
        R = H * W
        rs = torch.zeros(R, device=dev).index_add_(0, r, v)
        rq = torch.zeros(R, device=dev).index_add_(0, r, v * v)
        o = (rs * rs - rq).clamp_min(0)
        sum_o += float(o.sum()); sum_mass += float(rs.sum())
        o_chunks.append(o.float().cpu())
        del ri, vv, r, v, rs, rq, o
        torch.cuda.empty_cache()
    o_all = torch.cat(o_chunks)
    q = [float(torch.quantile(o_all, x)) for x in (0.5, 0.9, 0.99)]
    del gm, gq, gs_, go, gc, ck, sp
    torch.cuda.empty_cache()
    return {"scene": scene, "arm": arm, "views": len(sel), "cap": cap, "nnz": int(nnz),
            "sum_o": sum_o, "mean_o": sum_o / max(sum_mass, 1e-30),
            "o_p50": q[0], "o_p90": q[1], "o_p99": q[2], "rays": int(o_all.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arm", default="gs_froz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--tol", type=float, default=2e-3, help="relative tolerance on the scene0000 check")
    ap.add_argument("--out", default="artifacts/scannet/mean_o_backfill.json")
    a = ap.parse_args()
    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out))
    done = {(r["arm"], r["scene"]) for r in rows}
    checked = False
    for sc in a.scenes.split(","):
        if (a.arm, sc) in done:
            print(f"[{a.arm}/{sc}] cached", flush=True); continue
        try:
            r = one(sc, a.arm, a.views, a.cap)
        except Exception as e:
            print(f"[{a.arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        if sc in REF:
            ref = REF[sc]; rel = abs(r["mean_o"] - ref) / ref
            ok = rel < a.tol
            print(f"[CHECK] {sc}: fast {r['mean_o']:.6f} vs compute_beta {ref:.6f} "
                  f"rel {rel:.2e} -> {'AGREES' if ok else '*** DISAGREES ***'}", flush=True)
            if not ok:
                print("ABORTING: the fast path does not reproduce the reference.", flush=True)
                return
            checked = True
        rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{a.arm}/{sc}] views {r['views']} nnz {r['nnz']:,} "
              f"mean_o {r['mean_o']:.4f}  o p50/p90/p99 "
              f"{r['o_p50']:.3f}/{r['o_p90']:.3f}/{r['o_p99']:.3f}", flush=True)
    if not checked:
        print("\nWARNING: scene0000 was not recomputed, so the fast path is UNVALIDATED in this run.")
    v = [r["mean_o"] for r in rows if r["arm"] == a.arm]
    if v:
        print(f"\n{a.arm}: n={len(v)}  mean_o {np.mean(v):.4f}  range {min(v):.4f}-{max(v):.4f}")


if __name__ == "__main__":
    main()
