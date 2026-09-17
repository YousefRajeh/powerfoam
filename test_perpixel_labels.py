"""Per-pixel GT labels: SPLATTED POINTS vs RAYCAST MESH. Which one is trustworthy?

The projected oracle needs, for each pixel, the class of the surface ACTUALLY VISIBLE there. Two
ways to get it from ScanNet:

  SPLAT  project the labelled GT points, give each a disc whose pixel radius is its world spacing
         at that depth, and z-buffer. Needs no mesh. But it approximates a surface by discs, so it
         can leak background through gaps in a foreground surface and can bleed a foreground label
         outward past the true silhouette.

  MESH   cast one ray per pixel against the triangulated surface and take the hit triangle's
         nearest vertex by barycentric coordinate. Occlusion is exact, coverage is whatever the
         mesh actually covers.

`scenes10_points3d/<scene>/points3d.ply` has 72,007 vertices for scene0097 and the GT point cloud
has 72,007 points at a maximum separation of 0.0000 m -- the mesh IS the labelled cloud, with
connectivity. So vertex labels are exact and the two methods differ ONLY in how they resolve
visibility, which is exactly the thing under test.

This script does not assume the mesh is right either: it reports where they disagree and what the
disagreement looks like, so the choice is made on measurement rather than on which sounds better.
"""
from __future__ import annotations
import argparse, glob, os, sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import GT_ROOT

MESH_ROOT = r"D:\Downloads\scenes10_points3d"


def splat_labels(pts_w, cls, vm, K, H, W, spacing, dev, rmax=10):
    """Z-buffered disc splat. Returns (H*W,) int64 class, 0 = unlabelled."""
    BIG = 1 << 62
    R = vm[:3, :3]; t = vm[:3, 3]
    pc = pts_w @ R.T + t
    z = pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    front = z > 1e-6
    u = fx * pc[:, 0] / z.clamp_min(1e-6) + cx
    v = fy * pc[:, 1] / z.clamp_min(1e-6) + cy
    rad = (0.5 * fx * spacing / z.clamp_min(1e-6)).round().clamp(1, rmax)
    ui, vi = u.round().long(), v.round().long()
    zq = (z * 10000.0).clamp(0, 1 << 40).long()
    key0 = zq * 64 + (cls - 1)
    buf = torch.full((H * W,), BIG, dtype=torch.long, device=dev)
    for du in range(-rmax, rmax + 1):
        for dv in range(-rmax, rmax + 1):
            m = front & (rad >= max(abs(du), abs(dv)))
            if not bool(m.any()):
                continue
            uu = ui[m] + du; vv = vi[m] + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not bool(ok.any()):
                continue
            buf.scatter_reduce_(0, vv[ok] * W + uu[ok], key0[m][ok],
                                reduce="amin", include_self=True)
    hit = buf < BIG
    out = torch.zeros(H * W, dtype=torch.long, device=dev)
    out[hit] = (buf[hit] % 64) + 1
    depth = torch.full((H * W,), float("inf"), device=dev)
    depth[hit] = (buf[hit] // 64).float() / 10000.0
    return out, depth


def mesh_labels(scene_rc, tri_vidx, vert_cls, cam, c2w, H, W, dev):
    """Exact raycast. Returns (H*W,) int64 class (0 = ray missed the mesh) and depth."""
    import open3d as o3d
    d_cam = cam.cam_ray_dirs.reshape(-1, 3).double().cpu().numpy()
    Rw = c2w[:3, :3].numpy()
    dirs = d_cam @ Rw.T
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    orig = np.broadcast_to(c2w[:3, 3].numpy(), dirs.shape)
    rays = o3d.core.Tensor(np.ascontiguousarray(
        np.concatenate([orig, dirs], 1), dtype=np.float32))
    ans = scene_rc.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    pid = ans["primitive_ids"].numpy()
    uv = ans["primitive_uvs"].numpy()
    hit = np.isfinite(t_hit)
    out = np.zeros(H * W, np.int64)
    if hit.any():
        # barycentric: w0 = 1-u-v on the triangle's first vertex, then u, v
        b = np.stack([1.0 - uv[hit, 0] - uv[hit, 1], uv[hit, 0], uv[hit, 1]], 1)
        vsel = tri_vidx[pid[hit]][np.arange(b.shape[0]), b.argmax(1)]
        out[hit] = vert_cls[vsel]
    depth = np.where(hit, t_hit, np.inf)
    return (torch.from_numpy(out).to(dev),
            torch.from_numpy(depth.astype(np.float32)).to(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0097_00")
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--rmax", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import open3d as o3d
    from camera_bridge import K_from_ray_dirs

    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{a.scene}_{a.recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", a.scene)) if os.path.isdir(q)][0]
    gt_pts, raw, names = load_scannet_pointcept_gt(cand, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in pres]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])

    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, a.scene, "points3d.ply"))
    V = np.asarray(mesh.vertices); tri = np.asarray(mesh.triangles)
    if V.shape[0] != gt_pts.shape[0]:
        raise SystemExit(f"mesh V {V.shape[0]} != GT {gt_pts.shape[0]}; label transfer needed")
    dmax = float(np.abs(V - gt_pts).max())
    print(f"mesh vertices identical to GT points to {dmax:.2e} m "
          f"({V.shape[0]:,} verts, {tri.shape[0]:,} faces)")
    vert_cls = gt_lab.astype(np.int64)

    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    lab_m = gt_lab > 0
    pts_t = torch.from_numpy(np.ascontiguousarray(gt_pts[lab_m])).float().to(dev)
    cls_t = torch.from_numpy(gt_lab[lab_m].astype(np.int64)).to(dev)
    from scipy.spatial import cKDTree
    dnn, _ = cKDTree(gt_pts[lab_m]).query(gt_pts[lab_m], k=2, workers=-1)
    spacing = float(np.median(dnn[:, 1]))
    print(f"GT median point spacing {spacing:.4f} m")

    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()
    tot = dict(mesh=0, splat=0, both=0, agree=0, npx=0, sbleed=0, sleak=0, dsum=0.0, dn=0)
    print(f"\n{'view':>5}{'mesh cov':>10}{'splat cov':>11}{'both':>8}{'agree':>9}"
          f"{'splat nearer':>14}{'splat farther':>15}")
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        K, _ = K_from_ray_dirs(cam)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        vm = torch.linalg.inv(c2w).float().to(dev)
        sl, sd = splat_labels(pts_t, cls_t, vm, K.to(dev), H, W, spacing, dev, a.rmax)
        ml, md = mesh_labels(scene_rc, tri, vert_cls, cam, c2w.float(), H, W, dev)
        npx = H * W
        mh, sh = ml > 0, sl > 0
        both = mh & sh
        agree = (ml == sl) & both
        # where both hit, is the splat in front of or behind the true surface?
        dd = (sd - md)[both]
        near = int((dd < -2 * spacing).sum()); far = int((dd > 2 * spacing).sum())
        tot['mesh'] += int(mh.sum()); tot['splat'] += int(sh.sum())
        tot['both'] += int(both.sum()); tot['agree'] += int(agree.sum()); tot['npx'] += npx
        tot['sbleed'] += near; tot['sleak'] += far
        tot['dsum'] += float(dd.abs().sum()); tot['dn'] += int(both.sum())
        print(f"{vi:>5}{int(mh.sum())/npx:>10.1%}{int(sh.sum())/npx:>11.1%}"
              f"{int(both.sum())/npx:>8.1%}{int(agree.sum())/max(int(both.sum()),1):>9.2%}"
              f"{near/max(int(both.sum()),1):>14.2%}{far/max(int(both.sum()),1):>15.2%}")

    print(f"\n=== {a.scene}, {a.views} views ===")
    print(f"  mesh coverage        {tot['mesh']/tot['npx']:.1%}")
    print(f"  splat coverage       {tot['splat']/tot['npx']:.1%}")
    print(f"  both labelled        {tot['both']/tot['npx']:.1%}")
    print(f"  LABEL AGREEMENT      {tot['agree']/max(tot['both'],1):.2%}")
    print(f"  splat depth |err|    {tot['dsum']/max(tot['dn'],1):.4f} m "
          f"(spacing {spacing:.4f})")
    print(f"  splat in FRONT of surface (bleed past silhouette) {tot['sbleed']/max(tot['both'],1):.2%}")
    print(f"  splat BEHIND surface (leaked through a gap)       {tot['sleak']/max(tot['both'],1):.2%}")


if __name__ == "__main__":
    main()
