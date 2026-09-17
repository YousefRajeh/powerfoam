"""ReLaGS's Maximum Weight Pruning quantity, measured on foam and 3DGS.

`max_weight_pruning.py` + `forward.cu` define it exactly:

    w = alpha * T                                          # == our operator value A_ij
    atomicMaxFloatContrib(&max_contribution[id], w)        # running MAX over pixels
    setMaxContribution: torch.max(stored, this_view)       # then MAX over views
    keep iff  max_contribution > tau_contrib = 5e-4        # run_scannet.sh CONTRIB=0.0005

so  omega_j = max over ALL rays of ALL views of A_ij,  and low-omega primitives are DELETED.

WHY THIS IS MEASURED SEPARATELY. Our `solve_view_geomed.py` claimed to subsume MWP by using the
per-view MASS  m_jv = sum_{i in v} A_ij  as a weight. That is a SUM, and MWP is a MAX. They select
nearly opposite populations: a primitive touched by ten thousand rays at 1e-6 each has large total
mass but tiny peak contribution -- MWP deletes it, the mass weighting promotes it. The subsumption
claim was wrong, so the quantity has to be measured directly.

THE PREDICTION WORTH TESTING. A foam ray touches ~2.5 primitives, so a primitive that is hit is
usually THE contributor and omega_j should sit near 1. A 3DGS ray touches ~33, so most Gaussians
never dominate any pixel and omega_j should be small for many of them. If that holds, MWP is a
repair for a 3DGS-specific pathology and foam has little to gain -- which is a claim about the
representations, not about tuning.
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
from diagnose_holes import SCENES

FOAM = {"truefrozen", "nonfrozen"}
TAU_CONTRIB = 5e-4          # run_scannet.sh: CONTRIB=0.0005


def omega_max(scene, arm, dev="cuda", cap=64, tfloor=1e-3, max_views=0):
    """omega_j = max over every ray of every view of A_ij, exactly as MWP defines it."""
    cfg = f"output/scannet_{scene}_{arm if arm in FOAM else 'truefrozen'}/config.yaml"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    sel = list(range(len(dh.cameras)))
    if max_views:
        sel = np.linspace(0, len(sel) - 1, max_views).astype(int).tolist()

    if arm in FOAM:
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{scene}_{arm}/model.pt")
        P = model.points.shape[0]
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location=dev, weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].to(dev), sp["quats"].to(dev)
        gs_ = torch.exp(sp["scales"].to(dev))
        go = torch.sigmoid(sp["opacities"].to(dev).reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)
        P = gm.shape[0]

    om = torch.zeros(P, device=dev)
    mass = torch.zeros(P, device=dev)
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        if arm in FOAM:
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                           max_intersections=4096, transmittance_threshold=tfloor)
            ci, vv = op.col_indices, op.values
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            _, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.to(dev), W, H,
                                                   max_hits_per_pixel=cap,
                                                   transmittance_floor=tfloor)
        c_ = ci.to(torch.int64).to(dev); v_ = vv.float().to(dev)
        om.scatter_reduce_(0, c_, v_, reduce="amax")
        mass.index_add_(0, c_, v_)
        del ci, vv, c_, v_
        torch.cuda.empty_cache()
    return om.cpu().numpy(), mass.cpu().numpy(), P, len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--out", default="artifacts/scannet/max_contribution.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                om, mass, P, nv = omega_max(sc, arm, max_views=a.max_views)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            hit = om > 0
            r = {"arm": arm, "scene": sc, "P": int(P), "views": nv,
                 "hit_frac": float(hit.mean()),
                 "omega_p50": float(np.median(om[hit])) if hit.any() else float("nan"),
                 "omega_p10": float(np.quantile(om[hit], 0.1)) if hit.any() else float("nan"),
                 "pruned_frac_of_hit": float((om[hit] <= TAU_CONTRIB).mean()) if hit.any() else 0.0,
                 "pruned_frac_of_all": float((om <= TAU_CONTRIB).mean()),
                 # how much total evidence mass sits on the primitives MWP would delete: if this is
                 # large, pruning is discarding signal, not noise
                 "mass_frac_pruned": float(mass[om <= TAU_CONTRIB].sum() / max(mass.sum(), 1e-30))}
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] hit {r['hit_frac']:.1%}  omega p50 {r['omega_p50']:.4f} "
                  f"p10 {r['omega_p10']:.5f}  MWP would prune {r['pruned_frac_of_hit']:.2%} of hit "
                  f"({r['mass_frac_pruned']:.2%} of mass)", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'n':>3}{'omega p50':>11}{'omega p10':>12}{'MWP prunes':>12}{'of mass':>10}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{arm:<12}{len(s):>3}{f('omega_p50'):>11.4f}{f('omega_p10'):>12.5f}"
              f"{f('pruned_frac_of_hit'):>11.2%}{f('mass_frac_pruned'):>10.2%}")
    print(f"\n(tau_contrib = {TAU_CONTRIB}, from ReLaGS run_scannet.sh CONTRIB=0.0005)")


if __name__ == "__main__":
    main()
