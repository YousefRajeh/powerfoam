"""Opacity-cull fairness pass: all three arms, with and without OpenGaussian's <0.1 exclusion.

THE PROBLEM. OpenGaussian's protocol is Mahalanobis assignment AND an opacity<0.1 cull. We applied
the assignment but not the cull, and applied no analogue to foam -- so the arms were scored under
different rules and the Gaussian arm's win over foam (27.81 vs 26.59) was not trustworthy.

THE CALIBRATION (exact, not a tuned number).
  Gaussian: opacity is stored as a logit, so alpha = sigmoid(o_raw) >= 0.1  <=>  o_raw >= ln(1/9)
            = -2.19722.
  Foam:     carries an unbounded volumetric DENSITY sigma (softplus, beta=10), not a 0-1 opacity, so
            0.1 does not port. The renderer's per-primitive alpha is alpha = 1 - exp(-sigma*l), so
                1 - exp(-sigma*l) = 0.1   =>   sigma_min = -ln(0.9)/l = 0.105361/l
            with l the characteristic traversal length through the cell. l is taken as the scene's
            own median primitive spacing -- already computed for the coverage filter -- so the
            threshold is scene-adaptive and introduces no new constant.

This gives both representations the same PHYSICAL criterion (a primitive contributing <0.1 alpha
across its own width) rather than the same numeric value, which is the fair comparison.

BOTH WAYS ARE RUN. Culling EXCLUDES GT points from scoring, so it is not neutral -- it removes hard
cases and can only be interpreted against the uncalled number.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

ALPHA_MIN = 0.1
LOGIT_MIN = float(np.log(ALPHA_MIN / (1 - ALPHA_MIN)))      # -2.19722, Gaussian side
SIGMA_NUM = float(-np.log(1 - ALPHA_MIN))                    # 0.105361, foam numerator


def mean_chord_length(volumes):
    """Mean chord of a convex body is 4V/S (classical). For the effective sphere of the cell's own
    volume that is (4/3)*(3V/4pi)^(1/3) -- the EXPECTED distance a random ray travels through this
    cell, which is exactly the length that turns density into the alpha the renderer sees.

    This replaces the nearest-neighbour distance used earlier: NN distance is a spacing between
    centres, not a traversal length, and it made the arms incomparable because RadFoam's 4xSfM
    packs many more, much smaller cells into the same volume (kept 29-40% vs PowerFoam's 79-95%).
    Volume is the right invariant -- a cell that occupies little space cannot absorb much light
    regardless of how its neighbours are spaced."""
    v = np.maximum(np.asarray(volumes, dtype=np.float64), 1e-18)
    return (4.0 / 3.0) * np.cbrt(3.0 * v / (4.0 * np.pi))


def per_cell_length(centers):
    """PER-CELL traversal extent l_j = distance to the nearest other centre.

    A scene-global median was wrong: OpenGaussian culls on a PER-PRIMITIVE peak opacity, and one
    l per scene judges every cell against the same threshold. It also made the arms
    incomparable -- RadFoam's 4xSfM packs far more, far smaller cells into the same volume than
    PowerFoam, so a shared median mapped to a much harsher density cut (kept 34-48% vs 84-97%).
    The nearest-neighbour distance is each cell's own local scale, so sigma_j >= 0.105361/l_j
    asks the same physical question of every primitive: does it reach alpha 0.1 across its OWN
    width."""
    c = np.asarray(centers)
    nn, _ = cKDTree(c).query(c, k=2, workers=-1)
    return np.maximum(nn[:, 1], 1e-9)


def gaussian_keep(scene, gs_root):
    p = os.path.join(gs_root, f"refbench-{scene}", "point_cloud", "iteration_30000",
                     "scene_point_cloud.ply")
    o = np.asarray(PlyData.read(p)["vertex"]["opacity"]).astype(np.float64)
    return o >= LOGIT_MIN, float((o >= LOGIT_MIN).mean())


def radfoam_keep(ckpt, centers, valid):
    from radfoam_adapter import radfoam_density
    sig = radfoam_density(ckpt).astype(np.float64)
    thr = SIGMA_NUM / per_cell_length(centers)
    k = sig >= thr
    return k, float(k.mean()), float(np.median(thr))


def powerfoam_keep(ckpt_dir, centers, valid):
    """PowerFoam stores the same unbounded density; activation mirrors RadFoamScene."""
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=False)
    raw = None
    for k in ("density", "densities", "_density"):
        if isinstance(sd, dict) and k in sd and torch.is_tensor(sd[k]):
            raw = sd[k].float().reshape(-1); break
    if raw is None and isinstance(sd, dict):
        for v in sd.values():
            if torch.is_tensor(v) and v.ndim <= 2 and v.numel() == len(centers):
                raw = v.float().reshape(-1); break
    if raw is None:
        return None, None, None
    sig = torch.nn.functional.softplus(raw, beta=10).numpy().astype(np.float64)
    thr = SIGMA_NUM / per_cell_length(centers)
    k = sig >= thr
    return k, float(k.mean()), float(np.median(thr))


if __name__ == "__main__":
    from build_true_facet_graph import load_points_radii
    GS = r"D:\Downloads\refbench_3dgs_12scenes\output"
    RECON = r"D:\Downloads\spp_results\full"
    SPP = ["0d2ee665be", "3864514494", "27dd4da69e", "c50d2d1d42", "578511c8a9", "5942004064",
           "f9f95681fd", "d755b3d9d8", "3db0a1c8f3", "9071e139d9", "e7af285f7d", "09c1414f1b"]
    print(f"alpha>={ALPHA_MIN}  ->  gaussian logit >= {LOGIT_MIN:.5f} | "
          f"foam sigma >= {SIGMA_NUM:.6f}/spacing\n")
    print(f"{'scene':<13}{'gs kept':>9}{'pf kept':>9}{'pf thr':>9}{'rf kept':>9}{'rf thr':>9}")
    out = {}
    for s in SPP:
        row = {}
        try:
            _, gk = gaussian_keep(s, GS); row["gs"] = gk
        except Exception as e:
            row["gs"] = None
        try:
            ck = os.path.join(RECON, f"spp_pf_unfroz_{s}")
            c, _ = load_points_radii(ck)
            sv = torch.load(f"artifacts/scannetpp/{s}/solved_geometric_median_nonfrozen_ogl3.pt",
                            map_location="cpu", weights_only=True)
            vm = sv["valid_mask"].numpy()
            _, pk, pt = powerfoam_keep(ck, c, vm); row["pf"], row["pf_thr"] = pk, pt
        except Exception as e:
            row["pf"] = row["pf_thr"] = None
        try:
            rck = os.path.join(RECON, f"spp_rf_unfroz4x_{s}", "model.pt")
            if not os.path.exists(rck):
                rck = os.path.join(RECON, f"spp_rf_unfroz_{s}", "model.pt")
            from radfoam_adapter import load_radfoam_foam
            rc, _ = load_radfoam_foam(rck)
            _, rk, rt = radfoam_keep(rck, rc, np.ones(len(rc), dtype=bool))
            row["rf"], row["rf_thr"] = rk, rt
        except Exception as e:
            row["rf"] = row["rf_thr"] = None
        out[s] = row
        f = lambda v, w=9, p=3: (f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}")
        print(f"{s:<13}{f(row.get('gs'))}{f(row.get('pf'))}{f(row.get('pf_thr'))}"
              f"{f(row.get('rf'))}{f(row.get('rf_thr'))}")
    json.dump(out, open("artifacts/scannetpp/cull_fractions.json", "w"), indent=1)
