"""Point mIoU and surface metrics for the RadFoam arms, on RadFoam's own terms.

RadFoam is a VORONOI foam, so two things differ from our power-diagram pipeline and both are
forced by the representation rather than chosen:

  * ownership is NEAREST CENTRE, not `argmin ||x-c||^2 - r^2` -- there are no radii to weight with.
    (A Voronoi diagram is the power diagram with all radii equal, so the radical plane degenerates
    to the perpendicular bisector; `foam_isosurface` with radii = 0 is therefore exactly the
    Voronoi-face extractor and needs no special case.)
  * there is no dipole and no detail sites, so the surface cannot be a within-cell interface. It is
    the interface BETWEEN adjacent cells that straddle a density level -- which is precisely the
    construction PowerFoam's Sec. 3.3 describes as its predecessor's, and which it replaced because
    it "necessitates the explicit placement of zero-density points".

OPACITY. A RadFoam cell is unbounded, so `alpha` needs a length scale. We use the cell's own mean
distance to its Voronoi neighbours -- the local chord a ray would actually traverse -- giving
alpha = 1 - exp(-sigma * L). That is the same quantity the power-diagram arms use (there L = 2r),
so the opacity mask means the same thing on both sides.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")
# (kind, checkpoint template, solved-features file). `kind` decides ownership: a Voronoi foam has
# no radii so points go to the NEAREST CENTRE, while a power diagram uses argmin ||x-c||^2 - r^2.
# Both arms of both methods run through the identical protocol below so the comparison is
# apples-to-apples by construction rather than by assuming older stored numbers are compatible.
ARMS = {
    "rf_froz":       ("radfoam",   "recon_remote/rf_froz/{s}/model.pt",   "solved_gm_rf_match_ogl3.pt"),
    "rf_unfroz":     ("radfoam",   "recon_remote/rf_unfroz/{s}/model.pt", "solved_gm_rf_unfroz_ogl3.pt"),
    "pf_truefrozen": ("powerfoam", "output/scannet_{s}_truefrozen",       "solved_geometric_median_truefrozen_ogl3.pt"),
    "pf_nonfrozen":  ("powerfoam", "output/scannet_{s}_nonfrozen",        "solved_geometric_median_nonfrozen_ogl3.pt"),
}


def local_len(centers, adjacency, offsets):
    """Mean distance to Voronoi neighbours: the chord length that turns sigma into an alpha."""
    deg = np.diff(offsets)
    out = np.zeros(len(centers))
    for i in range(len(centers)):
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        if len(nb):
            out[i] = float(np.linalg.norm(centers[nb] - centers[i], axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--surface", action="store_true", help="also extract and score a surface")
    ap.add_argument("--out", default="artifacts/radfoam_eval.json")
    a = ap.parse_args()

    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from point_cloud_query import assign_points_to_nearest_center

    enable_determinism()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in rows}

    for scene in a.scenes:
        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
        nc = len(names) + 1
        text = embed_class_names(names, "cuda")

        for arm in a.arms:
            if (arm, scene) in done:
                continue
            kind, ckt, fn = ARMS[arm]
            ck = ckt.format(s=scene)
            fp = f"artifacts/scannet/{scene}/{fn}"
            exists = os.path.isdir(ck) if kind == "powerfoam" else os.path.exists(ck)
            if not (exists and os.path.exists(fp)):
                print(f"[miss] {arm}/{scene}", flush=True)
                continue
            if kind == "radfoam":
                sd = torch.load(ck, map_location="cpu", weights_only=False)
                c = sd["xyz"].float().numpy().astype(np.float64)
                radii = None
            else:
                from build_true_facet_graph import load_points_radii
                import torch.nn.functional as _F
                cc, rr = load_points_radii(ck)
                c = np.asarray(cc, dtype=np.float64); radii = np.asarray(rr, dtype=np.float64)
                sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
                sd = {"density": _F.softplus(sd["density"].float(), beta=100),
                      "adjacency": sd["adjacency"], "adjacency_offsets": sd["adjacency_offsets"]}
            sigma = sd["density"].float().numpy().reshape(-1)
            adj = sd["adjacency"].numpy().astype(np.int64)
            off = sd["adjacency_offsets"].numpy().astype(np.int64)
            d = torch.load(fp, map_location="cpu", weights_only=True)
            feats = d["primitive_features"].float()
            vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(len(c), bool)
            assert len(c) == len(feats), (len(c), len(feats))

            # the chord a ray traverses: 2r for a bounded power cell, the mean neighbour
            # distance for an unbounded Voronoi cell. Same physical quantity either way.
            L = (2.0 * radii) if radii is not None else local_len(c, adj, off)
            alpha = 1.0 - np.exp(-np.maximum(sigma, 0.0) * L)

            # published protocol: bare arg-max, opacity mask, classes present in the scene
            pcls = classify_primitives(feats.cuda(), text).cpu().numpy() + 1
            pcls[~vm] = 0
            pcls[alpha < a.alpha] = 0

            if radii is None:
                owner = np.asarray(assign_points_to_nearest_center(gt_pts, c))
            else:
                from point_cloud_query import assign_points_to_power_cells
                owner = np.asarray(assign_points_to_power_cells(gt_pts, c, radii,
                                                               valid=None, k=8))
            pred = np.where(owner >= 0, pcls[np.clip(owner, 0, len(c) - 1)], 0)

            ious, accs = [], []
            for k in range(1, nc):
                g = gt_lab == k
                if not g.any():
                    continue
                p = pred == k
                inter = float((g & p).sum())
                union = float((g | p).sum())
                ious.append(inter / union if union else 0.0)
                accs.append(inter / float(g.sum()))
            rec = {"arm": arm, "scene": scene, "n_prims": int(len(c)),
                   "frac_opaque": float((alpha >= a.alpha).mean()),
                   "n_classes": len(ious),
                   "mIoU": float(np.mean(ious)), "mAcc": float(np.mean(accs))}

            if a.surface:
                from foam_exact_surface import foam_isosurface
                idx = MeshSurfaceIndex(scene, gt_lab, nc)
                zero_r = np.zeros(len(c)) if radii is None else radii
                occ = np.where(pcls > 0, alpha, -1.0)
                pts, cls, areas = foam_isosurface(c, zero_r, occ, pcls, adj, off,
                                                  a.alpha, V.min(0), V.max(0))
                if len(pts):
                    m = semantic_surface_metrics_mesh(idx, pts, cls)
                    rec.update({"area_m2": float(sum(areas.values())),
                                "gt_area_m2": float(mesh.get_surface_area())})
                    rec.update({k: float(m[k]) for k in ("scd", "hd95", "boundary_f1", "n_missed")
                                if isinstance(m.get(k), (int, float))})

            rows.append(rec)
            json.dump(rows, open(a.out, "w"), indent=1)
            s = (f"  SCD {100*rec['scd']:6.2f}cm BF1 {rec['boundary_f1']:.3f}"
                 if "scd" in rec else "")
            print(f"{arm:11} {scene:14} {len(c):>9,} prims  opaque {rec['frac_opaque']:5.1%}  "
                  f"mIoU {rec['mIoU']:.4f}  mAcc {rec['mAcc']:.4f}{s}", flush=True)

    for arm in a.arms:
        g = [r for r in rows if r["arm"] == arm]
        if g:
            print(f"\n{arm}: {len(g)} scenes  mIoU {np.mean([x['mIoU'] for x in g]):.4f}  "
                  f"mAcc {np.mean([x['mAcc'] for x in g]):.4f}"
                  + (f"  SCD {100*np.mean([x['scd'] for x in g if 'scd' in x]):.2f}cm"
                     if any('scd' in x for x in g) else ""))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
