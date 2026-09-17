"""Per-class surface metrics for a GAUSSIAN method, from its RENDERED surface -- same protocol as
run_foam_visible_surface.py, so the two are directly comparable.

WHAT MAKES THIS LIKE-FOR-LIKE. The foam's surface points come from back-projecting `front_t_surf`,
i.e. the depth its renderer reports. The Gaussian equivalent is the depth ITS renderer reports, so
this back-projects gsplat's expected depth (`render_mode="RGB+ED"`) for the same cameras. Neither
side gets a reconstruction step (no TSDF, no marching cubes, no isovalue) and neither is scored on
GT geometry -- both are scored on the surface each method would actually draw.

LABELS. A Gaussian render has no "front primitive index" the way the foam rasteriser does, so each
back-projected point takes the class of the nearest Gaussian CENTRE. That is the same
nearest-centre convention the point-level protocol uses for Gaussian arms, and for OpenGaussian the
class itself comes from its own codebook: leaf_ind -> per-leaf pooled CLIP -> cosine argmax, with
leaves seen by fewer than two views zeroed (their eval_scannet.py:140) and therefore left
UNPREDICTED rather than defaulting to the first class.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401  must precede gsplat

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = r"D:\Downloads\scannet_pointcept"
OG_ART = r"D:\Downloads\powerfoam\artifacts\opengaussian_r2"


def load_og(scene, device):
    """OpenGaussian's saved ply + codebook. Returns geometry and per-Gaussian leaf index."""
    from plyfile import PlyData
    ply = PlyData.read(os.path.join(OG_ART, scene, "point_cloud.ply"))["vertex"].data
    g = lambda k: np.ascontiguousarray(ply[k], dtype=np.float32)
    means = np.stack([g("x"), g("y"), g("z")], -1)
    scales = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], -1))
    quats = np.stack([g(f"rot_{i}") for i in range(4)], -1)
    opac = 1.0 / (1.0 + np.exp(-g("opacity")))
    d = np.load(os.path.join(OG_ART, scene, "cluster_lang.npz"))
    return (torch.as_tensor(means, device=device), torch.as_tensor(quats, device=device),
            torch.as_tensor(scales, device=device), torch.as_tensor(opac, device=device), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0062_00"])
    ap.add_argument("--class-sets", default="opengaussian19")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="artifacts/gs_rendered_surface.json")
    a = ap.parse_args()

    import configargparse
    from gsplat import rasterization

    from camera_bridge import K_from_ray_dirs
    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh

    enable_determinism()
    dev = "cuda"
    rows = []

    for scene in a.scenes:
        if not os.path.isdir(os.path.join(OG_ART, scene)):
            print(f"[miss] {scene}", flush=True)
            continue
        print(f"\n=== {scene} / opengaussian ===", flush=True)
        means, quats, scales, opac, cb = load_og(scene, dev)

        # cameras: identical to the ones the foam was rendered from
        cfg = f"output/scannet_{scene}_truefrozen/config.yaml"
        p = configargparse.ArgParser()
        add_group(p, Params)
        p.add_argument("-c", "--config", is_config_file=True)
        args = p.parse_args(["-c", cfg])
        dh = DataHandler(args)
        dh.reload("all", downsample=args.downsample[-1])
        cams = dh.cameras if a.max_views is None else dh.cameras[:a.max_views]

        pts = []
        colors = torch.zeros((means.shape[0], 3), device=dev)
        for i, cam in enumerate(cams):
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[i].double()
            vm = torch.linalg.inv(c2w).float().to(dev)
            W, H = int(cam.width), int(cam.height)
            with torch.no_grad():
                out, alphas, _ = rasterization(
                    means, quats, scales, opac, colors, vm[None], K.to(dev)[None], W, H,
                    render_mode="RGB+ED")
            depth = out[0, ..., -1]                       # expected depth
            al = alphas[0, ..., 0]
            rm = cam.ray_maps
            if rm is None and getattr(cam, "cam_ray_dirs", None) is not None:
                rm = cam._build_ray_maps_from_basis()
            if rm is None:
                rm = cam._build_pinhole_ray_maps()
            rm = rm.to(dev)
            o, d = rm[..., :3], rm[..., 3:]
            hit = (depth > 0) & torch.isfinite(depth) & (al > 0.5)
            m = torch.zeros_like(hit)
            m[::a.stride, ::a.stride] = True
            hit &= m
            if bool(hit.any()):
                pts.append((o[hit] + depth[hit].unsqueeze(-1) * d[hit]).cpu().numpy())
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(cams)} views, "
                      f"{sum(len(x) for x in pts):,} surface points", flush=True)
        surf = np.concatenate(pts) if pts else np.zeros((0, 3))
        print(f"  {len(surf):,} rendered surface points", flush=True)

        # nearest-Gaussian lookup for labelling
        tree = cKDTree(means.cpu().numpy())
        _, near = tree.query(surf, k=1, workers=-1)

        leaf_ind = np.clip(cb["leaf_ind"], None, 319)
        occu = cb["occu_count"]
        leaf_feat = torch.from_numpy(cb["leaf_feat"]).float().to(dev)
        leaf_feat[torch.from_numpy(occu).to(dev) < 2] *= 0.0

        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        gt_pts, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        if len(V) != len(raw):
            print(f"  [MISALIGNED] mesh {len(V):,} vs labels {len(raw):,}", flush=True)
            continue
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())

        for cs in a.class_sets.split(","):
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = F.normalize(embed_class_names(names, dev), dim=1, p=2)
            lf = F.normalize(leaf_feat, dim=1, p=2)
            max_id = torch.argmax(text @ lf.T, dim=0)
            max_id[torch.from_numpy(occu).to(dev) < 2] = -1     # zeroed leaf -> no prediction
            gcls = (max_id.cpu().numpy()[leaf_ind] + 1)         # per-Gaussian, 0 = unpredicted

            cls = gcls[near]
            keep = cls > 0
            pp, cc = surf[keep], cls[keep]
            print(f"  {cs}: {len(pp):,} surface points carry a class "
                  f"({100*keep.mean():.1f}% of rendered), {len(names)} classes present", flush=True)
            idx = MeshSurfaceIndex(scene, gt_lab, nc)
            m = semantic_surface_metrics_mesh(idx, pp, cc)
            rec = {"scene": scene, "method": "opengaussian", "class_set": cs,
                   "n_pred": int(len(pp)), "stride": a.stride,
                   "gt_area_m2": float(mesh.get_surface_area())}
            rec.update({k: float(m[k]) for k in
                        ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "n_missed")
                        if isinstance(m.get(k), (int, float))})
            rows.append(rec)
            print(f"  -> SCD {100*rec['scd']:.2f}cm  HD95 {100*rec['hd95']:.2f}cm  "
                  f"BF1 {rec['boundary_f1']:.3f}  acc {100*rec['mae_pred2gt']:.2f}cm  "
                  f"comp {100*rec['mae_gt2pred']:.2f}cm  missed {rec.get('n_missed', 0):.0f}",
                  flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
