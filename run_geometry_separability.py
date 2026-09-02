"""Do the classes CLIP cannot separate differ GEOMETRICALLY? Foam knows the answer exactly.

THE MOTIVATION. Every probe today says geometry is not the bottleneck and features are: oracle
ceiling 91.92 vs 26.59 achieved; facet purity +10.6 available vs +1.7 realised; c_intra boundary
AUC 0.72 but +0.02 when acted on. And the per-class breakdown shows WHICH classes fail --

    kitchen counter 0.00   kitchen cabinet 3.17   refrigerator 2.34   wall 32.57
    microwave      87.24   trash can      72.88   ceiling     76.03   door  59.52

The failures are large flat surfaces; the successes are compact objects with distinctive
silhouettes. A SAM crop of a counter and of the cabinet beneath it are both "beige planar region in
a kitchen" -- CLIP has no way to separate them, and no 3D smoothing can recover a distinction the
2D encoder never made.

But those classes are geometrically UNAMBIGUOUS: a counter is horizontal at ~0.9 m, a cabinet is
vertical, a floor is horizontal at 0, a ceiling horizontal at the top. PowerFoam gives per cell,
exactly: the dipole NORMAL (from the stored quaternion, `scene.py::get_normals`), position, and
size. Right now all of that drives SMOOTHING and none of it drives the DECISION.

WHY THIS IS NOT THE ALREADY-REFUTED IDEA. [[Geometric-bias-corrections-2026-08-29]] found that any
correction CONDITIONED on geometry correlates with content and removes signal -- that was geometry
as a bias term SUBTRACTED from features. This is geometry as classification EVIDENCE, a different
question, and it is aimed exactly at the classes that fail.

WHAT IS MEASURED. For each GT point, its owning cell's verticality |n_z| (1 = horizontal surface,
0 = vertical) and height above the scene floor. Then, per class pair among the failing/succeeding
classes: how separable are they on those two numbers alone, versus on the CLIP cosine?
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names
from build_true_facet_graph import load_points_radii
from point_cloud_query import assign_points_to_power_cells
from run_overnight import RECON, LAM
from run_spp_eval import benchmark_map, load_gt, coverage_filter

SCENES = ["f9f95681fd", "c50d2d1d42", "3864514494", "5942004064"]
PAIRS = [("kitchen counter", "kitchen cabinet"), ("kitchen counter", "wall"),
         ("kitchen cabinet", "wall"), ("floor", "ceiling"), ("wall", "floor"),
         ("refrigerator", "kitchen cabinet"), ("table", "floor")]


def normals_from_quat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def auc(pos, neg):
    """Rank-based AUC of a 1-D score, symmetrised so 0.5 = no information."""
    if len(pos) < 10 or len(neg) < 10:
        return None
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1
    a = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a)


def main():
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    agg = {}
    for scene in SCENES:
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"artifacts/scannetpp/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            continue
        centers, radii = load_points_radii(ck)
        centers = np.asarray(centers, dtype=np.float64)
        sd = torch.load(os.path.join(ck, "model.pt"), map_location="cpu", weights_only=False)
        nrm = normals_from_quat(sd["quaternions"].float().numpy().astype(np.float64))
        del sd
        sv = torch.load(sp, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vm = sv["valid_mask"].to(device); vmn = sv["valid_mask"].cpu().numpy()
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)
        cen = unit.clone(); cen[vm] = F.normalize(unit[vm] - LAM * mu, dim=-1)
        del feats, sv

        gt, lab0, _ = load_gt(scene, top, r2b)
        a = assign_points_to_power_cells(gt, centers, radii, valid=vmn, k=64)
        keepc, _, _ = coverage_filter(gt, a, centers, vmn, 20.0)
        ok = (a >= 0) & keepc & (lab0 >= 0)
        own, lab, pts = a[ok], lab0[ok], gt[ok]
        # geometry per point, from its owning cell
        vert = np.abs(nrm[own][:, 2])                 # 1 = horizontal surface, 0 = vertical
        floor_z = np.percentile(pts[:, 2], 1.0)
        height = pts[:, 2] - floor_z
        name_of = {i: n for i, n in enumerate(top[:100])}
        present = {name_of[c]: c for c in np.unique(lab) if c in name_of}

        for A, B in PAIRS:
            if A not in present or B not in present:
                continue
            ia, ib = present[A], present[B]
            ma, mb = lab == ia, lab == ib
            if ma.sum() < 50 or mb.sum() < 50:
                continue
            txt = embed_class_names([A, B], device)
            cos = (cen[torch.from_numpy(own).to(device)] @ txt.T)
            margin = (cos[:, 0] - cos[:, 1]).cpu().numpy()   # CLIP's own discriminative axis
            row = {"n_a": int(ma.sum()), "n_b": int(mb.sum()),
                   "auc_vert": auc(vert[ma], vert[mb]),
                   "auc_height": auc(height[ma], height[mb]),
                   "auc_clip": auc(margin[ma], margin[mb]),
                   "med_vert_a": float(np.median(vert[ma])), "med_vert_b": float(np.median(vert[mb])),
                   "med_h_a": float(np.median(height[ma])), "med_h_b": float(np.median(height[mb]))}
            agg.setdefault(f"{A} vs {B}", []).append(row)
            del txt, cos
        del unit, cen
        torch.cuda.empty_cache()

    print(f"{'class pair':<34}{'n':>7}{'AUC vert':>10}{'AUC height':>12}{'AUC CLIP':>10}"
          f"   medians (vert / height)")
    for k, rows in agg.items():
        f = lambda key: np.mean([r[key] for r in rows if r[key] is not None]) \
            if any(r[key] is not None for r in rows) else float("nan")
        n = int(np.mean([r["n_a"] + r["n_b"] for r in rows]))
        print(f"{k:<34}{n:>7,}{f('auc_vert'):>10.3f}{f('auc_height'):>12.3f}{f('auc_clip'):>10.3f}"
              f"   {f('med_vert_a'):.2f}/{f('med_vert_b'):.2f}  "
              f"{f('med_h_a'):.2f}m/{f('med_h_b'):.2f}m")
    json.dump({k: v for k, v in agg.items()},
              open("artifacts/scannetpp/geometry_separability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
