"""Does a SIZE / DISTANCE term explain the oracle shortfall where overlap mass fails?

A30 refuted `mean_o` as a predictor: gs_unfroz carries the highest overlap of any arm (0.846) and
is nearly the most accurate, while gs_froz at the same overlap (0.839) is 4.7 mIoU worse. And the
sign of the overlap correlation flips between frozen and unfrozen arms. So the shortfall is not
"how much do rays mix primitives" but something about WHICH primitives get mixed.

THE HYPOTHESIS UNDER TEST. A primitive can only be assigned one label. If its spatial support is
large compared to the distance between class boundaries in the scene, it necessarily straddles two
classes and NO solver can recover a correct label for it -- that is an irreducible floor set by
geometry, not by the lift. The natural dimensionless form is

    kappa_j = r_j / d_bnd        r_j    = primitive's spatial extent
                                 d_bnd  = local distance to the nearest DIFFERENT-class GT point

which is scale-free, comparable across representations, and (unlike mean_o) can move in opposite
directions for foam and 3DGS under the same intervention -- which is what the data demands, since
unfreezing helps 3DGS (+4.7) and hurts foam (-9.3).

WHAT r_j IS, PER REPRESENTATION. The power cell radius for foam; for 3DGS the Gaussians have no
hard support, so the extent is taken as the geometric mean of the three scales (the sigma of an
equivalent-volume isotropic Gaussian). Both are reported alongside the nearest-neighbour spacing
between primitive centres, which is representation-agnostic and needs no such convention -- if the
conclusion only holds for one definition of extent it is a definition artifact and is reported so.

ALSO MEASURED: straddle fraction, the share of primitives whose support actually contains GT points
of more than one class. This is the direct consequence the ratio is a proxy for, it needs no
threshold, and it is the quantity that bounds achievable accuracy from below.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center
from diagnose_holes import SCENES, GT_ROOT, geometry

FOAM = {"truefrozen", "nonfrozen"}


def arm_geometry(scene, arm, dev="cuda"):
    """centers and a per-primitive spatial extent, in world units, for either representation."""
    if arm in FOAM:
        centers, radii, _ = geometry(scene, arm)
        return np.asarray(centers), np.asarray(radii).reshape(-1)
    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cpu", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    centers = sp["means"].float().numpy()
    # equivalent-volume isotropic sigma: (s1 s2 s3)^(1/3), computed in log space for stability
    scales = torch.exp(sp["scales"].float())
    r = torch.exp(torch.log(scales.clamp_min(1e-12)).mean(-1)).numpy()
    return centers, r


def one(scene, arm, class_set, dev="cuda"):
    from scipy.spatial import cKDTree
    centers, r = arm_geometry(scene, arm)
    P = centers.shape[0]

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    lab_m = gt_lab > 0
    pts, lab = gt_pts[lab_m], gt_lab[lab_m]

    # d_bnd: for each labelled GT point, distance to the nearest GT point of a DIFFERENT class.
    # Computed per class against the complement, so it is exact rather than a k-NN approximation.
    d_bnd = np.full(len(pts), np.inf, np.float64)
    for c in np.unique(lab):
        m = lab == c
        if m.all():
            continue
        d, _ = cKDTree(pts[~m]).query(pts[m], k=1, workers=-1)
        d_bnd[m] = d

    # each primitive inherits the boundary scale of its nearest labelled GT point
    tree = cKDTree(pts)
    dc, nn = tree.query(centers, k=1, workers=-1)
    kappa = r / np.maximum(d_bnd[nn], 1e-9)

    # nearest-neighbour spacing between primitive centres (no extent convention involved)
    dnn, _ = cKDTree(centers).query(centers, k=2, workers=-1)
    spacing = dnn[:, 1]

    # STRADDLE: does a primitive's own support hold GT points of more than one class?
    if arm in FOAM:
        _, radii, _ = geometry(scene, arm)
        assigned = assign_points_to_power_cells(pts, centers, np.asarray(radii).reshape(-1),
                                                valid=None, k=64)
    else:
        assigned = assign_points_to_nearest_center(pts, centers, valid=None)
    C = len(kept)
    votes = np.zeros((P, C + 1), np.int32)
    ok = assigned >= 0
    np.add.at(votes, (assigned[ok], lab[ok]), 1)
    tot = votes[:, 1:].sum(1)
    has = tot > 0
    top = votes[:, 1:].max(1)
    purity = np.where(has, top / np.maximum(tot, 1), np.nan)
    straddle = np.where(has, (votes[:, 1:] > 0).sum(1) > 1, False)

    # within the primitives that hold GT, the achievable ceiling if every one took its majority
    ceiling = float(top[has].sum() / max(tot[has].sum(), 1))
    return dict(scene=scene, arm=arm, P=int(P),
                r_p50=float(np.median(r)), r_mean=float(r.mean()),
                spacing_p50=float(np.median(spacing)),
                d_bnd_p50=float(np.median(d_bnd[np.isfinite(d_bnd)])),
                kappa_p50=float(np.median(kappa)), kappa_mean=float(kappa.mean()),
                kappa_gt1=float((kappa > 1).mean()),
                straddle_frac=float(straddle[has].mean()),
                purity_mean=float(np.nanmean(purity[has])),
                ceiling=ceiling, gt_cover=float(has.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--oracle", default="artifacts/scannet/oracle_stream.json")
    ap.add_argument("--out", default="artifacts/scannet/scale_vs_boundary.json")
    a = ap.parse_args()

    rows = []
    if os.path.exists(a.out):
        try:
            rows = json.load(open(a.out))
        except Exception:
            rows = []
    done = {(r["arm"], r["scene"]) for r in rows}
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            if (arm, sc) in done:
                continue
            try:
                r = one(sc, arm, a.class_set)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] P {r['P']:>9,} r50 {r['r_p50']:.4f} spacing {r['spacing_p50']:.4f} "
                  f"d_bnd {r['d_bnd_p50']:.4f} kappa50 {r['kappa_p50']:.3f} "
                  f"straddle {r['straddle_frac']:.1%} ceiling {r['ceiling']:.4f}", flush=True)

    print(f"\n{'arm':<12}{'P':>10}{'r_p50':>9}{'spacing':>9}{'d_bnd':>8}{'kappa50':>9}"
          f"{'k>1':>7}{'straddle':>10}{'purity':>8}{'ceiling':>9}")
    for arm in a.arms.split(","):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{arm:<12}{f('P'):>10,.0f}{f('r_p50'):>9.4f}{f('spacing_p50'):>9.4f}"
              f"{f('d_bnd_p50'):>8.4f}{f('kappa_p50'):>9.3f}{f('kappa_gt1'):>7.1%}"
              f"{f('straddle_frac'):>10.1%}{f('purity_mean'):>8.4f}{f('ceiling'):>9.4f}")

    # correlate every candidate term against the measured oracle shortfall
    if os.path.exists(a.oracle):
        orc = {(x["recon"], x["scene"]): x for x in json.load(open(a.oracle))}
        from scipy.stats import spearmanr, pearsonr
        pair = [(r, orc[(r["arm"], r["scene"])]) for r in rows if (r["arm"], r["scene"]) in orc]
        if len(pair) > 3:
            y = [1 - o["miou_tikhonov"] for _, o in pair]
            print(f"\npredictors of Eq.18 oracle shortfall, n={len(pair)} scene-arms:")
            for nm in ("kappa_p50", "kappa_mean", "kappa_gt1", "straddle_frac", "purity_mean",
                       "ceiling", "r_p50", "spacing_p50", "P"):
                v = [r[nm] for r, _ in pair]
                print(f"  {nm:<14} spearman {spearmanr(v, y).statistic:+.3f}  "
                      f"pearson {pearsonr(v, y)[0]:+.3f}")
            for nm in ("mean_o", "mean_o_cross"):
                v = [o[nm] for _, o in pair]
                print(f"  {nm+' (A30)':<14} spearman {spearmanr(v, y).statistic:+.3f}  "
                      f"pearson {pearsonr(v, y)[0]:+.3f}")
            arms = [x for x in a.arms.split(",") if any(r["arm"] == x for r, _ in pair)]
            print(f"\n  ARM MEANS (n={len(arms)}):")
            for nm in ("kappa_p50", "straddle_frac", "ceiling", "mean_o", "mean_o_cross"):
                src = (lambda r, o: r[nm]) if nm in rows[0] else (lambda r, o: o[nm])
                xv = [np.mean([src(r, o) for r, o in pair if r["arm"] == x]) for x in arms]
                yv = [np.mean([1 - o["miou_tikhonov"] for r, o in pair if r["arm"] == x]) for x in arms]
                print(f"    {nm:<14} pearson {pearsonr(xv, yv)[0]:+.3f}   "
                      f"{dict(zip(arms, [round(v, 4) for v in xv]))}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
