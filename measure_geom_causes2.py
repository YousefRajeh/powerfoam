"""Surface vs straddle, using each renderer's ACTUAL occupancy test.

The first version of this measurement used foam's power radius alone and the Gaussians' 1-sigma
axis. Both are wrong, in opposite directions, so its surface/straddle split is retracted.

FOAM. `foam_solid.py` reproduces the render kernels: the solid a primitive actually renders is

    R_i = Ball(c_i, r_i)  n  (radical half-spaces)  n  {x : (x - c_i).n_i <= h}

Because a point's owner here IS its power cell, the radical half-spaces hold for the owner by
definition, so containment reduces to the ball AND the dipole. `h = 0` (the centre-crossing
convention `ablation_opacity_dipole.py` already uses; the kernel's displaced height only moves the
plane, not its existence). Ignoring the dipole -- as the first version did -- overstates coverage,
and RENDERER_SPEC 9 already measured 19.2% (frozen) / 41.2% (unfrozen) of GT points sitting on the
EMPTY side of their owner's dipole.

3DGS. From `ProjectionEWA3DGSFused.cu:164-178` and `RasterizeToPixels3DGSFwd.cu:145-149`:

    alpha = opac * exp(-sigma),  sigma = 0.5 * d_M^2,   keep iff alpha >= ALPHA_THRESHOLD = 1/255
      =>  d_M <= sqrt(2 * ln(255 * opac)),   and the projection caps the box at 3.33 sigma

so the extent is OPACITY-AWARE, not a fixed 3 sigma: a Gaussian at the 1/255 opacity floor has zero
reach, one at opacity 1 reaches 3.33 sigma. Using 1 sigma (the first version) understated it ~3x.

d_M is computed in 3D from the checkpoint's quaternion+scale, which is the same covariance the
projection derives its 2D form from.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT, geometry
from point_cloud_query import assign_points_to_power_cells

FOAM = {"truefrozen", "nonfrozen"}
ALPHA_THRESHOLD = 1.0 / 255.0
SIGMA_CAP = 3.33


def foam_normals(scene, arm):
    """Dipole plane normals, cached -- loading the model is the slow part."""
    cache = f"artifacts/scannet/{scene}/normals_{arm}.npy"
    if os.path.exists(cache):
        return np.load(cache)
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
    n = m.get_normals().detach().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, n)
    return n


def gs_mahalanobis(pts, cent, quats, scales, own):
    """d_M(p, owner(p)) in 3D, from the checkpoint's quaternion + log-scale."""
    q = quats[own]
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    d = (pts - cent[own])[:, None, :]                 # (N,1,3)
    loc = (d @ R).squeeze(1)                          # into the Gaussian's frame
    s = scales[own].clip(1e-9)
    return np.sqrt(((loc / s) ** 2).sum(-1))


def one(scene, arm, stats):
    z = np.load(os.path.join(stats, f"{arm}_{scene}.npz"))
    live = z["live"].astype(bool)
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    vis = np.load(os.path.join("artifacts", "scannet", scene, "gt_visible.npy"))
    m = (gl > 0) & vis
    P, g = pts[m].astype(np.float32), gl[m]
    idx = np.nonzero(live)[0]

    if arm in FOAM:
        cent, rad, _ = geometry(scene, arm)
        cent = cent.astype(np.float32); rad = rad.astype(np.float32)
        nrm = foam_normals(scene, arm)
        # owner = the power cell, so the radical half-spaces hold for it by construction
        a = assign_points_to_power_cells(P, cent, rad, valid=live, k=64)
        ok = a >= 0
        own = np.where(ok, a, 0)
        dv = P - cent[own]
        dist = np.linalg.norm(dv, axis=1)
        n = nrm[own]
        n = n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
        in_ball = dist <= rad[own]
        in_dipole = (dv * n).sum(1) <= 0.0
        covered = ok & in_ball & in_dipole
        extra = {"frac_in_ball": float((ok & in_ball).mean()),
                 "frac_in_dipole": float((ok & in_dipole).mean())}
    else:
        ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu",
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        cent = sp["means"].float().numpy()
        quats = sp["quats"].float().numpy()
        scales = np.exp(sp["scales"].float().numpy())
        opac = torch.sigmoid(sp["opacities"].float().reshape(-1)).numpy()
        own = idx[cKDTree(cent[live]).query(P, k=1, workers=-1)[1]]
        dM = gs_mahalanobis(P, cent, quats, scales, own)
        reach = np.minimum(SIGMA_CAP,
                           np.sqrt(np.maximum(2.0 * np.log(opac[own] / ALPHA_THRESHOLD), 0.0)))
        covered = dM <= reach
        extra = {"median_reach_sigma": float(np.median(reach)),
                 "frac_opac_below_floor": float((opac[own] <= ALPHA_THRESHOLD).mean())}

    # ceiling loss under the SAME ownership used for coverage
    K = int(g.max())
    occ = np.zeros((cent.shape[0], K + 1), np.int64)
    np.add.at(occ, (own, g), 1)
    maj = occ[:, 1:].argmax(1) + 1
    wrong = maj[own] != g
    n_pts = P.shape[0]
    r = {"scene": scene, "arm": arm, "n_scored": n_pts,
         "loss_total": float(wrong.mean()),
         "loss_surface": float((wrong & ~covered).sum()) / n_pts,
         "loss_straddle": float((wrong & covered).sum()) / n_pts,
         "frac_covered": float(covered.mean())}
    r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--stats", default="artifacts/scannet/plotstats")
    ap.add_argument("--out", default="artifacts/scannet/geom_causes2.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                r = one(sc, arm, a.stats)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] loss {r['loss_total']:.4f} = surface {r['loss_surface']:.4f} + "
                  f"straddle {r['loss_straddle']:.4f}   covered {r['frac_covered']:.1%}", flush=True)
    if not rows:
        return
    print(f"\n{'arm':<12}{'loss':>9}{'SURFACE':>10}{'STRADDLE':>10}{'covered':>10}{'detail':>34}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s if k in r]))
        det = (f"ball {f('frac_in_ball'):.1%} dipole {f('frac_in_dipole'):.1%}"
               if arm in FOAM else
               f"reach {f('median_reach_sigma'):.2f}sig")
        print(f"{arm:<12}{f('loss_total'):>9.4f}{f('loss_surface'):>10.4f}{f('loss_straddle'):>10.4f}"
              f"{f('frac_covered'):>9.1%}{det:>34}")


if __name__ == "__main__":
    main()
