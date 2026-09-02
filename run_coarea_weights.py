"""Choose the spherical power weights by MINIMISING EXACT CO-AREA, not by matching capacity.

WHERE THIS COMES FROM. `argmax_c <f,t_c>` is the nearest-site Voronoi query on the CLIP sphere
(because both are unit, ||f-t_c||^2 = 2 - 2<f,t_c>). Adding per-class weights makes it a POWER
diagram, `argmax_c [<f,t_c> + w_c/2]`, and partial centering is exactly that with
w_c = -2 lam colmean_c. The open question is what CRITERION should pick w.

CAPACITY IS THE WRONG CRITERION -- measured. Semi-discrete OT (Aurenhammer-Hoffmann-Aronov)
gives the unique w matching prescribed capacities, solved to convergence. Uniform-cell capacity
scored 24.86 and uniform-volume 27.65 against partial centering's 38.58 (19cls, 10 scenes), and
the GT-capacity ORACLE was the WORST arm at 13.25 -- so the failure is capacity matching itself,
not the choice of target. Consistent with two prior refutations of marginal matching (Sinkhorn
prior matching, and the mass-conserving TPFA flow, where diffusion moved class mass AWAY from
GT while scoring better). IoU rewards OVERLAP, not totals.

THE CRITERION PROPOSED HERE. Pick the weights that minimise the PERIMETER of the induced 3D
segmentation, subject to a bounded displacement of the spherical diagram's faces:

    min_{||w||_inf <= delta}  sum_{(i,j) facets} A_ij * [ yhat_i != yhat_j ]

Three reasons this is the right shape:
  * EXACT. For a bounded disjoint partition the perimeter of a region IS the sum of the shared
    facet areas on its boundary -- the discrete co-area formula holds with equality. A Gaussian
    cloud has no facets and no perimeter, so this objective is undefined there, not merely
    worse. This is the same quantity that made co-area TV the only positive foam arm (+0.14/
    +0.47/+1.00) while uniform-weight TV was destructive (-5.72).
  * ADMISSIBLE. Uses only geometry and features. No GT, no class-specific supervision, nothing
    fit to the target labels.
  * DISPLACEMENT-BOUNDED. delta caps how far each face of the spherical diagram may move -- the
    text-side analogue of the dipole's displacement field. delta = 0 recovers plain argmax, so
    the baseline is nested inside the family.

DEGENERACY, and why delta handles it. Unconstrained perimeter minimisation collapses to one
class everywhere (perimeter 0). The delta constraint is what makes the problem well posed; it
is a hard constraint, not a penalty, so the trade-off is explicit rather than tuned.

OPTIMISATION. w has only C <= 19 coordinates and the objective is piecewise constant in w, so
gradient methods do not apply. Coordinate descent over a grid within [-delta, delta] is exact
enough and cheap: each evaluation is one argmax plus one edge-difference count.

FALSIFIER, stated before running: the co-area-optimal weights must beat partial centering at
its validated lambda by >= +0.3 mIoU at 19cls. Note the honest risk -- minimising boundary area
may simply produce large smooth regions that score well on perimeter and badly on IoU, which is
precisely the failure mode the delta sweep will expose (perimeter falls monotonically in delta
by construction; mIoU need not).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SP = (r"C:\Users\rajehyl\AppData\Local\Temp\claude"
      r"\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad")
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
LAMBDA = {"opengaussian19": 0.5, "opengaussian15": 0.5, "opengaussian10": 0.4}


def perimeter(sim, w, i, j, area, vt):
    """Exact co-area: total shared-facet area whose two cells carry different labels."""
    cls = (sim + w[None, :] / 2).argmax(-1)
    diff = (cls[i] != cls[j]) & vt[i] & vt[j]
    return (area * diff.float()).sum()


def minimise_coarea(sim, i, j, area, vt, delta, steps=9, sweeps=3):
    """Coordinate descent on w within the box ||w||_inf <= delta.

    The objective is piecewise constant in w, so this grids each coordinate in turn and keeps
    the best. Cheap: C * steps * sweeps evaluations, each one argmax + one edge count.
    """
    C = sim.shape[1]
    w = torch.zeros(C, device=sim.device)
    best = perimeter(sim, w, i, j, area, vt)
    grid = torch.linspace(-delta, delta, steps, device=sim.device)
    for _ in range(sweeps):
        for c in range(C):
            keep, cur = w[c].item(), best
            for g in grid:
                w[c] = g
                p = perimeter(sim, w, i, j, area, vt)
                if p < cur:
                    cur, keep = p, g.item()
            w[c] = keep
            best = cur
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SPLIT))
    ap.add_argument("--deltas", default="0.01,0.03,0.1,0.3")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        af = os.path.join(SP, f"area_{scene}_pf_nonfroz.npz")
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if not all(os.path.exists(p) for p in (af, fp, apth)):
            print(f"[skip] {scene}: missing artifact", flush=True)
            continue

        A = np.load(af)
        bnd = ~A["unbounded"].astype(bool)
        ei = torch.from_numpy(A["i"][bnd].astype(np.int64)).to(dev)
        ej = torch.from_numpy(A["j"][bnd].astype(np.int64)).to(dev)
        area = torch.from_numpy(A["area"][bnd].astype(np.float32)).to(dev)

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            sim = unit @ text.T

            def score(tag, w, extra=""):
                cls = (sim + w[None, :] / 2).argmax(-1).cpu().numpy() + 1
                sc = owned.copy()
                sc[owned] = valid[assign[owned]]
                pred = np.zeros(len(gt), dtype=np.int64)
                pred[sc] = cls[assign[sc]]
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                per = perimeter(sim, w, ei, ej, area, vt).item()
                res.setdefault((tag, cs), []).append((float(mi) * 100, per))
                return float(mi) * 100, per

            zero = torch.zeros(len(names), device=dev)
            m0, p0 = score("argmax (delta=0)", zero)
            lam = LAMBDA[cs]
            wc = -2.0 * lam * sim[vt].mean(0)
            mc, pc = score("partial centering", wc)

            for delta in [float(x) for x in a.deltas.split(",")]:
                w, _ = minimise_coarea(sim, ei, ej, area, vt, delta)
                mm, pp = score(f"co-area min (delta={delta})", w)
                if cs == "opengaussian19":
                    print(f"  [{scene}] delta={delta:<5} perimeter {p0:.1f} -> {pp:.1f} "
                          f"({100*(pp-p0)/max(p0,1e-9):+.1f}%)  mIoU {m0:.2f} -> {mm:.2f}",
                          flush=True)
            if cs == "opengaussian19":
                print(f"  [{scene}] centering: perimeter {pc:.1f} "
                      f"({100*(pc-p0)/max(p0,1e-9):+.1f}%)  mIoU {mc:.2f}", flush=True)
        print(f"[{scene}] done", flush=True)

    n = len(next(iter(res.values())))
    print(f"\n=== {n} scenes ===")
    print(f"{'arm':<28}" + "".join(f"{c[11:]:>10}" for c in CLASS_SETS)
          + f"{'perim m^2':>11}   delta vs centering")
    base = {c: np.mean([r[0] for r in res[("partial centering", c)]]) for c in CLASS_SETS}
    for tag in sorted(set(k[0] for k in res)):
        row = "".join(f"{np.mean([r[0] for r in res[(tag,c)]]):10.2f}" for c in CLASS_SETS)
        per = np.mean([r[1] for r in res[(tag, CLASS_SETS[0])]])
        dl = "  " + " ".join(f"{np.mean([r[0] for r in res[(tag,c)]])-base[c]:+.2f}"
                             for c in CLASS_SETS)
        print(f"{tag:<28}{row}{per:11.1f}{dl}")
    print("\nIf perimeter falls monotonically while mIoU does not, the criterion is measuring "
          "something real but not what IoU rewards -- report that plainly.")


if __name__ == "__main__":
    main()
