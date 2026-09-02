"""ScanNet++ surface metrics under plain per-cell argmax, against ScanNet++'s own labelled mesh.

Fills tab:3dseg_spp's surface columns. Same rule as everywhere else in the paper: each cell's own
feature, cosine against the raw class names, argmax -- no clustering, no codebook, no refinement.
A class's predicted region is the set of GT points whose OWN cell argmaxes to that class.

WHY THIS NEEDS ITS OWN MESH CODE. `mesh_surface.MeshSurfaceIndex` is hardcoded to ScanNet's layout
(`scenes10_points3d/{scene}/points3d.ply`, labels from Pointcept `segment20.npy`, vertex order
identical). ScanNet++ stores geometry and annotation differently:

    {scene}/scans/mesh_aligned_0.05.ply     the mesh (vertices AND faces)
    {scene}/scans/segments_anno.json        segGroups: label + member vertex ids
    metadata/semantic_benchmark/top100.txt  the ordered class list; label id = row index
    metadata/semantic_benchmark/map_benchmark.csv   raw label -> benchmark class
    {scene}/scans/mesh_aligned_0.05_mask.txt        vertices the dataset excludes

so the labels come from folding `segments_anno` through `map_benchmark`, exactly as
`run_spp_eval.load_gt` does -- which is reused here rather than reimplemented, so the GT is
byte-identical to the mIoU numbers already in the database. Unlabelled/excluded vertices are -1
there and become 0 (ignore) here, matching the scoring convention.

FACE LABELS follow mesh_surface's rule: majority over a face's non-zero vertex labels, requiring at
least two of three to agree; a face with three different classes is dropped as an ambiguous
boundary rather than assigned arbitrarily.

CORRESPONDENCE is each representation's own query -- exact power-cell membership for the foams,
Mahalanobis for Gaussians -- which is the same asymmetry already documented for the ScanNet++ mIoU
table and is not introduced here.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from evaluate_point_cloud_miou import embed_class_names
from mesh_surface import TAU, _sample_mesh_uniform, face_labels
from run_grid_search import SCENES
from run_spp_eval import GT_ROOT, benchmark_map, load_gt

ART = "artifacts/scannetpp_gs"
GT_SAMPLES_PER_M2 = 2500
MIN_SAMPLES = 500


def spp_mesh(scene):
    p = os.path.join(GT_ROOT, scene, "scans", "mesh_aligned_0.05.ply")
    m = o3d.io.read_triangle_mesh(p)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def surface_metrics(V, T, vert_lab, n_classes, pred_pts, pred_cls, seed=0):
    """Per class: exact point-to-triangle for pred->GT, area-uniform samples for GT->pred."""
    fl = face_labels(T, vert_lab)
    per, live = {}, []
    for c in range(1, n_classes):
        sel = fl == c
        pm = pred_cls == c
        if not sel.any() or not pm.any():
            if sel.any():
                per[c] = {"missed": True}
            continue
        sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                        o3d.utility.Vector3iVector(T[sel]))
        sub.remove_unreferenced_vertices()
        area = float(sub.get_surface_area())
        if area <= 0:
            continue
        rs = o3d.t.geometry.RaycastingScene()
        rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sub))
        ppts = pred_pts[pm]
        d_p2g = rs.compute_distance(o3d.core.Tensor(np.ascontiguousarray(ppts, dtype=np.float32),
                                                    dtype=o3d.core.Dtype.Float32)).numpy()
        n = max(MIN_SAMPLES, int(area * GT_SAMPLES_PER_M2))
        gsamp = _sample_mesh_uniform(np.asarray(sub.vertices), np.asarray(sub.triangles), n, seed)
        d_g2p, _ = cKDTree(ppts).query(gsamp, k=1, workers=-1)
        prec, rec = float((d_p2g <= TAU).mean()), float((d_g2p <= TAU).mean())
        m = {"missed": False, "area_m2": area, "n_pred": int(pm.sum()),
             "mae_pred2gt": float(d_p2g.mean()), "mae_gt2pred": float(d_g2p.mean()),
             "scd": float((d_p2g.mean() + d_g2p.mean()) / 2),
             "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
             "boundary_f1": float(2 * prec * rec / max(prec + rec, 1e-9))}
        per[c] = m
        live.append(m)
    if not live:
        return {"n_scored": 0, "n_missed": sum(1 for v in per.values() if v.get("missed"))}
    agg = {k: float(np.mean([m[k] for m in live]))
           for k in ("mae_pred2gt", "mae_gt2pred", "scd", "hd95", "boundary_f1")}
    agg["n_scored"] = len(live)
    agg["n_missed"] = sum(1 for v in per.values() if v.get("missed"))
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recon", default="gs_unfroz")
    ap.add_argument("--solver", default="weighted")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--class-size", type=int, default=100)
    ap.add_argument("--out", default="artifacts/scannetpp/spp_surface.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    top, r2b = benchmark_map()
    rows = []
    for scene in SCENES[:a.scenes]:
        fp = f"{ART}/{scene}/solved_{a.solver}_{a.recon}_ogl3.pt"
        if not os.path.exists(fp):
            print(f"  [miss] {scene}: {fp}", flush=True)
            continue
        gt_pts, lab0, _ = load_gt(scene, top, r2b)
        V, T = spp_mesh(scene)
        if len(V) != len(lab0):
            print(f"  [skip] {scene}: mesh {len(V)} verts vs labels {len(lab0)}", flush=True)
            continue

        sv = torch.load(fp, map_location=dev, weights_only=True)
        feats = sv["primitive_features"].float().to(dev)
        vm = sv["valid_mask"].to(dev)
        unit = torch.zeros_like(feats)
        unit[vm] = F.normalize(feats[vm], dim=-1)

        from run_spp_gs_eval import load_gaussians, mahalanobis_assign
        means, scales, quats = load_gaussians(scene)
        assigned = mahalanobis_assign(np.asarray(gt_pts, dtype=np.float64), means, scales, quats)
        vmn = vm.cpu().numpy()
        assigned = np.where((assigned >= 0) & vmn[np.clip(assigned, 0, None)], assigned, -1)
        owned = assigned >= 0

        present = sorted({c for c in np.unique(lab0).tolist() if 0 <= c < a.class_size})
        names = [top[c] for c in present]
        text = embed_class_names(names, dev)
        cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1        # PLAIN per-cell argmax

        remap = {c: i + 1 for i, c in enumerate(present)}
        vert_lab = np.array([remap.get(int(c), 0) for c in lab0], dtype=np.int64)
        pred = np.zeros(len(gt_pts), dtype=np.int64)
        pred[owned] = cls[assigned[owned]]

        sm = surface_metrics(V, T, vert_lab, len(present) + 1,
                             np.asarray(gt_pts, dtype=np.float64), pred)
        sm.update({"scene": scene, "recon": a.recon, "n_classes": len(present)})
        rows.append(sm)
        print(f"  {scene}  C={len(present):3d}  scd={sm.get('scd', float('nan'))*100:6.2f}cm  "
              f"bf1={sm.get('boundary_f1', float('nan'))*100:5.2f}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1, default=float)
    if rows:
        for k in ("mae_pred2gt", "mae_gt2pred", "scd", "hd95", "boundary_f1"):
            v = [r[k] for r in rows if k in r]
            if v:
                print(f"MEAN {k:14s} {np.mean(v):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
