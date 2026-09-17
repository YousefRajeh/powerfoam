"""Compare the two surfaces the foam can give you, and score both against the GT mesh.

    (a) TSDF   -- powerfoam/mesh_extractor.py, the AUTHORS' own extractor (upstream
                  theialab/powerfoam): render depth+normal from every view, fuse into a TSDF,
                  march it. Renderer-native, and the same recipe applies to any method that
                  renders depth, so it is the fair cross-representation control.
    (b) EXACT  -- foam_exact_surface.py: the isosurface as power-diagram faces in closed form.
                  A power diagram partitions space into convex polyhedra, so at a density level
                  the surface IS the union of faces separating occupied from unoccupied cells --
                  planar polygons, no grid, no interpolation. Verified against closed-form cases
                  to 2.96e-16 (tests/test_foam_exact_surface.py).

WHY BOTH. `ResearchVault/Methods/Radfoam-Reuse-Inventory.md` notes that mesh_extractor "sidesteps"
the true cell polytopes by fusing rendered depth. If the two surfaces agree, that validates the
exact extractor against the renderer the model was actually trained through; where they disagree
quantifies what the fusion costs. Neither alone answers that.

THE ISOVALUE PROBLEM, and how it is handled. TSDF fusion has no density threshold -- it takes the
median-depth surface the renderer produces (`depth_quantiles=0.5`). The exact extractor needs a
sigma level, and there is no a-priori value that corresponds to "where the renderer put the
surface". So sigma_iso is SWEPT and the sweep is reported: the value that best matches TSDF is an
output of this comparison, not an input to it.

Reported per surface: area, and accuracy/completeness against the GT mesh (exact point-to-triangle
for pred->GT, area-uniform mesh samples for GT->pred), plus the two-way chamfer BETWEEN the two
surfaces.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def gt_mesh_samples(scene, samples_per_m2=2500, seed=0):
    """Area-uniform samples of the whole GT mesh, plus a raycasting scene for exact distances."""
    import open3d as o3d
    from mesh_surface import load_mesh, _sample_mesh_uniform
    mesh = load_mesh(scene)
    V, T = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    area = float(mesh.get_surface_area())
    pts = _sample_mesh_uniform(V, T, max(50_000, int(area * samples_per_m2)), seed)
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return pts, rs, area, (V.min(0), V.max(0))


def dist_to_mesh(rs, q):
    import open3d as o3d
    t = o3d.core.Tensor(np.ascontiguousarray(q, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    return rs.compute_distance(t).numpy().astype(np.float64)


def score(name, pred, gt_pts, rs, tau=0.02):
    """Accuracy (pred->GT surface, exact) and completeness (GT samples->pred)."""
    if len(pred) == 0:
        return {"surface": name, "n_pred": 0, "empty": True}
    d_p2g = dist_to_mesh(rs, pred)
    d_g2p, _ = cKDTree(pred).query(gt_pts, k=1, workers=-1)
    return {
        "surface": name, "n_pred": int(len(pred)), "empty": False,
        "acc_mean": float(d_p2g.mean()), "acc_median": float(np.median(d_p2g)),
        "comp_mean": float(d_g2p.mean()), "comp_median": float(np.median(d_g2p)),
        "chamfer": float((d_p2g.mean() + d_g2p.mean()) / 2),
        "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
        "f1@2cm": float(2 * (d_p2g <= tau).mean() * (d_g2p <= tau).mean()
                        / max((d_p2g <= tau).mean() + (d_g2p <= tau).mean(), 1e-9)),
    }


def tsdf_surface(scene, arm, voxel_size, sdf_trunc, depth_trunc, max_cluster, split="all",
                 samples_per_m2=2500):
    """Run the AUTHORS' MeshExtractor and return area-uniform samples of the fused mesh."""
    import configargparse
    from configs import Params, add_group
    from mesh_surface import _sample_mesh_uniform
    from powerfoam.mesh_extractor import MeshExtractor

    cfg = f"output/scannet_{scene}_{arm}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    for name, typ, dflt in (("mesh_name", str, "mesh_cmp"), ("depth_trunc", float, depth_trunc),
                            ("voxel_size", float, voxel_size), ("sdf_trunc", float, sdf_trunc),
                            ("mesh_res", int, 1024), ("max_cluster", int, max_cluster)):
        p.add_argument(f"--{name}", type=typ, default=dflt)
    p.add_argument("--unbounded", action="store_true")
    args = p.parse_args(["-c", cfg])
    ex = MeshExtractor(args, cfg)
    if split != "test":
        # MeshExtractor.__init__ loads only the TEST split. Fusing every view instead is a strict
        # improvement in coverage: TSDF completeness is limited by how much of the surface was
        # ever observed, and the test split is a small subset of the trajectory.
        ex.data.reload(split, downsample=args.downsample[-1])
    pkg = ex.render_views()
    mesh = ex.fuse_to_mesh(pkg)
    V, T = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(T) == 0:
        return np.zeros((0, 3)), 0.0
    area = float(mesh.get_surface_area())
    return _sample_mesh_uniform(V, T, max(50_000, int(area * samples_per_m2)), 0), area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--arms", nargs="*", default=["truefrozen"])
    ap.add_argument("--sigma-iso", nargs="*", type=float, default=[10.0, 34.7, 100.0, 300.0])
    ap.add_argument("--voxel-size", type=float, default=0.01)
    ap.add_argument("--sdf-trunc", type=float, default=0.04)
    ap.add_argument("--depth-trunc", type=float, default=6.0)
    ap.add_argument("--max-cluster", type=int, default=50)
    ap.add_argument("--out", default="artifacts/foam_surface_compare.json")
    ap.add_argument("--skip-tsdf", action="store_true")
    ap.add_argument("--split", default="all", choices=("all", "train", "test"),
                    help="views fused by the TSDF extractor; upstream uses test only")
    ap.add_argument("--alpha-min", nargs="*", type=float, default=[0.1, 0.5],
                    help="opacity threshold for a primitive to contribute a dipole patch")
    ap.add_argument("--skip-iso", action="store_true")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from foam_exact_surface import foam_dipole_surface, foam_isosurface

    rows = []
    for scene in a.scenes:
        gt_pts, rs, gt_area, (blo, bhi) = gt_mesh_samples(scene)
        print(f"\n=== {scene}: GT mesh area {gt_area:.2f} m^2, {len(gt_pts):,} samples ===",
              flush=True)
        for arm in a.arms:
            ck = f"output/scannet_{scene}_{arm}"
            if not os.path.isdir(ck):
                print(f"[miss] {ck}", flush=True)
                continue
            c, r = load_points_radii(ck)
            sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
            sigma = F.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
            adj = sd["adjacency"].numpy().astype(np.int64)
            off = sd["adjacency_offsets"].numpy().astype(np.int64)
            pc = np.ones(len(c), dtype=np.int64)

            tsdf_pts = np.zeros((0, 3)); tsdf_area = 0.0
            if not a.skip_tsdf:
                try:
                    tsdf_pts, tsdf_area = tsdf_surface(scene, arm, a.voxel_size, a.sdf_trunc,
                                                       a.depth_trunc, a.max_cluster, a.split)
                    m = score(f"tsdf[{a.split}]", tsdf_pts, gt_pts, rs)
                    m.update(scene=scene, arm=arm, area_m2=tsdf_area, sigma_iso=None,
                             split=a.split)
                    rows.append(m)
                    print(f"  TSDF        area {tsdf_area:7.2f}  acc {m['acc_mean']*100:6.2f}cm  "
                          f"comp {m['comp_mean']*100:6.2f}cm  chamfer {m['chamfer']*100:6.2f}cm  "
                          f"F1 {m['f1@2cm']:.3f}", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  TSDF FAILED: {type(e).__name__}: {e}", flush=True)

            # DIPOLE SURFACE: the surface the renderer actually uses.
            nrm = None
            if "quaternions" in sd:
                q = sd["quaternions"].float()
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                nrm = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z),
                                   2 * (x * z - w * y)], dim=-1).numpy()
            if nrm is not None:
                th = sd["texel_height"].float()
                hgt = (th.mean(dim=1) * torch.as_tensor(r).float()).numpy()
                for am in a.alpha_min:
                    alpha = 1.0 - np.exp(-sigma * 2.0 * r)
                    pts, cls, areas = foam_dipole_surface(c, r, nrm, hgt, alpha, pc, adj, off,
                                                          blo, bhi, alpha_min=am)
                    area = float(sum(areas.values()))
                    m = score(f"dipole@a{am:g}", pts, gt_pts, rs)
                    m.update(scene=scene, arm=arm, area_m2=area, alpha_min=am,
                             live_prims=int((alpha >= am).sum()), n_cells=int(len(c)))
                    if len(tsdf_pts) and len(pts):
                        d1, _ = cKDTree(tsdf_pts).query(pts, k=1, workers=-1)
                        d2, _ = cKDTree(pts).query(tsdf_pts, k=1, workers=-1)
                        m["chamfer_to_tsdf"] = float((d1.mean() + d2.mean()) / 2)
                    rows.append(m)
                    print(f"  dipole a>={am:<4g} area {area:7.2f}  acc {m['acc_mean']*100:6.2f}cm"
                          f"  comp {m['comp_mean']*100:6.2f}cm  chamfer {m['chamfer']*100:6.2f}cm"
                          f"  F1 {m['f1@2cm']:.3f}"
                          + (f"  d(TSDF) {m['chamfer_to_tsdf']*100:.2f}cm"
                             if "chamfer_to_tsdf" in m else ""), flush=True)

            for s_iso in ([] if a.skip_iso else a.sigma_iso):
                pts, cls, areas = foam_isosurface(c, r, sigma, pc, adj, off, s_iso, blo, bhi)
                area = float(sum(areas.values()))
                m = score(f"exact@{s_iso:g}", pts, gt_pts, rs)
                m.update(scene=scene, arm=arm, area_m2=area, sigma_iso=s_iso,
                         occupied_cells=int((sigma >= s_iso).sum()), n_cells=int(len(c)))
                if len(tsdf_pts) and len(pts):
                    d1, _ = cKDTree(tsdf_pts).query(pts, k=1, workers=-1)
                    d2, _ = cKDTree(pts).query(tsdf_pts, k=1, workers=-1)
                    m["chamfer_to_tsdf"] = float((d1.mean() + d2.mean()) / 2)
                rows.append(m)
                if m.get("empty"):
                    print(f"  exact s={s_iso:<7g} EMPTY", flush=True)
                else:
                    print(f"  exact s={s_iso:<7g} area {area:7.2f}  "
                          f"acc {m['acc_mean']*100:6.2f}cm  comp {m['comp_mean']*100:6.2f}cm  "
                          f"chamfer {m['chamfer']*100:6.2f}cm  F1 {m['f1@2cm']:.3f}"
                          + (f"  d(TSDF) {m['chamfer_to_tsdf']*100:.2f}cm"
                             if "chamfer_to_tsdf" in m else ""), flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
