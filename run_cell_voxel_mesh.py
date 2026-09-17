"""The simplest extraction: voxelise cell OWNERSHIP, then marching cubes per class.

No dipole cut, no CSG of ball and half-space, no thresholds beyond opacity. For every voxel, find
the cell that owns it (the power diagram's own membership test, `argmin ||x-c_i||^2 - r_i^2`), take
that cell's class, and march the indicator of each class. The bounded power diagram means a cell
only owns points inside its own ball, so `max_dist=r` is the whole of the boundedness.

This is worth having as the baseline extraction precisely because it is the least clever one: it
uses only what the representation asserts about ownership, and nothing about what the renderer does
with that ownership. Everything more elaborate has to beat it.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--bound-sphere", action="store_true", default=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="artifacts/cell_voxel_mesh.json")
    ap.add_argument("--outdir", default="artifacts/surface_viz/cellvox")
    a = ap.parse_args()

    import open3d as o3d
    from skimage import measure

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from mesh_surface import (MeshSurfaceIndex, _sample_mesh_uniform, load_mesh,
                              semantic_surface_metrics_mesh)
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import foam_volume, grid_from_bbox

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    scene = a.scene
    ck = f"output/scannet_{scene}_{a.arm}"
    c, r = load_points_radii(ck)
    c = np.asarray(c, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
    sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
    alpha = 1.0 - np.exp(-sigma * 2.0 * r)

    mesh = load_mesh(scene)
    V = np.asarray(mesh.vertices)
    _, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    nc = len(names) + 1
    idx = MeshSurfaceIndex(scene, gt_lab, nc)

    d = torch.load(f"artifacts/scannet/{scene}/{a.solved}", map_location="cpu", weights_only=True)
    text = embed_class_names(names, "cuda")
    neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
    pcls = classify_with_rejection(d["primitive_features"].float().cuda(), text, neg,
                                   a.margin).cpu().numpy()
    pcls[~d["valid_mask"].numpy()] = 0
    if a.oracle:
        from oracle_labels import oracle_labels
        pcls, ost = oracle_labels(c, r, V, gt_lab, nc)
        print(f"[oracle] {ost['n_cells_with_gt']:,} cells own GT, purity {ost['vote_purity']:.3f}",
              flush=True)

    keep = (alpha >= a.alpha) & (pcls > 0)
    cls_of = np.where(keep, pcls, 0)
    lo, dims = grid_from_bbox(V.min(0), V.max(0), a.voxel)
    print(f"grid {dims} = {np.prod(dims)/1e6:.1f}M voxels @ {a.voxel*100:.0f}cm, "
          f"{int(keep.sum()):,}/{len(c):,} cells", flush=True)
    _, cvol = foam_volume(c, r, alpha, cls_of, lo, dims, a.voxel,
                          max_dist=r if a.bound_sphere else None)
    print(f"occupied voxels {(cvol > 0).sum():,} ({100*(cvol > 0).mean():.2f}% of grid)", flush=True)

    VS, FS, CS = [], [], []
    off = 0
    for k, nm in enumerate(names, start=1):
        vol = (cvol == k).astype(np.float32)
        if vol.max() == 0:
            print(f"  {nm:<16} empty", flush=True)
            continue
        vv, ff, _, _ = measure.marching_cubes(vol, level=0.5)
        vw = lo + (vv + 0.5) * a.voxel
        VS.append(vw); FS.append(ff + off); CS.append(np.full(len(ff), k, dtype=np.int64))
        off += len(vw)
        t = vw[ff]
        ar = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                        axis=1).sum())
        print(f"  {nm:<16} {len(ff):>8,} tris  {ar:6.2f} m2", flush=True)
    MV = np.concatenate(VS); MF = np.concatenate(FS); MC = np.concatenate(CS)

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(MV),
                                  o3d.utility.Vector3iVector(MF))
    m.compute_vertex_normals()
    tag = "oracle" if a.oracle else "predicted"
    p = f"{a.outdir}/{scene}_cellvox_{tag}.ply"
    o3d.io.write_triangle_mesh(p, m)

    pts, cls = [], []
    for cid in np.unique(MC):
        Fc = MF[MC == cid]
        t = MV[Fc]
        ar = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                        axis=1).sum())
        if ar <= 0:
            continue
        s = _sample_mesh_uniform(MV, Fc, max(500, int(ar * 2500)), 0)
        pts.append(s); cls.append(np.full(len(s), cid, dtype=np.int64))
    mm = semantic_surface_metrics_mesh(idx, np.concatenate(pts), np.concatenate(cls))
    t = MV[MF]
    area = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                      axis=1).sum())
    rec = {"scene": scene, "labels": tag, "voxel": a.voxel, "alpha": a.alpha,
           "n_tris": int(len(MF)), "area_m2": area,
           "gt_area_m2": float(mesh.get_surface_area())}
    rec.update({k: float(mm[k]) for k in ("scd", "hd95", "boundary_f1", "n_missed")
                if isinstance(mm.get(k), (int, float))})
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    rows.append(rec); json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\ntotal {len(MF):,} tris  area {area:.1f} m2 (GT {rec['gt_area_m2']:.1f})")
    print(f"SCD {100*rec['scd']:.2f}cm  HD95 {100*rec['hd95']:.2f}cm  "
          f"BF1 {rec['boundary_f1']:.3f}  missed {rec.get('n_missed', 0):.0f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
