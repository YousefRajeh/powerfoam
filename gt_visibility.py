"""Which labelled GT points were ever visible? Parameter-free, via incident-triangle hits.

WHY NOT A DISTANCE TOLERANCE. A GT point lies ON the mesh, so the ray toward it hits its own surface
at t ~= dist and every point self-occludes unless slack is allowed. The obvious fix is
`t_hit >= dist - TOL`, but TOL is then a free parameter and the answer moves with it -- measured, the
visible fraction runs 87.4% (TOL=5mm) to 95.8% (TOL=10cm) on scene0070, a 7-point swing. A scoring
set should not depend on that.

THE PARAMETER-FREE TEST. A GT point IS a mesh vertex, and the raycast returns which triangle was hit.
So ask the exact question instead of a proxy one:

    ray toward vertex v hits a triangle INCIDENT TO v   -> that is v's own surface -> VISIBLE
    ray hits any other triangle (necessarily in front)  -> something occludes it   -> NOT visible

No tolerance, no depth comparison. A point is visible if it passes in ANY view.

WHY IT MATTERS. A labelled point that no ray ever reached deposited evidence on nothing, so scoring
it charges the lift for the capture's view coverage rather than for the lift. The mask depends only
on the GT mesh and the cameras -- never on a reconstruction -- so the identical points are removed
for every arm.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from diagnose_scannet_miou import load_scannet_pointcept_gt
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
from diagnose_holes import GT_ROOT, SCENES
from oracle_projected import MESH_ROOT
from camera_bridge import K_from_ray_dirs


def incident_csr(tri, nverts):
    """vertex -> the triangles containing it, as CSR (offsets, values)."""
    v = tri.reshape(-1)
    t = np.repeat(np.arange(tri.shape[0], dtype=np.int64), 3)
    order = np.argsort(v, kind="stable")
    v, t = v[order], t[order]
    counts = np.bincount(v, minlength=nverts)
    off = np.zeros(nverts + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    return off, t


def one(scene, dev="cuda"):
    import open3d as o3d
    d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)][0]
    pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
    n2i = {n: i for i, n in enumerate(names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
    gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
    lab = gl > 0
    idx = np.nonzero(lab)[0]                      # global vertex ids of the labelled points
    P = pts[idx].astype(np.float64)

    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, scene, "points3d.ply"))
    tri = np.asarray(mesh.triangles)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    off, inc = incident_csr(tri, pts.shape[0])

    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_truefrozen/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])

    in_fr = np.zeros(P.shape[0], bool)
    vis = np.zeros(P.shape[0], bool)
    for vi in range(len(dh.cameras)):
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        _, info = K_from_ray_dirs(cam)
        c2w = np.eye(4); c2w[:3, :4] = dh.c2ws[vi].double().numpy()
        R, eye = c2w[:3, :3], c2w[:3, 3]
        pc = (P - eye) @ R
        z = pc[:, 2]
        zz = np.where(z > 1e-6, z, 1.0)
        u = info["fx"] * pc[:, 0] / zz + info["cx"]
        v = info["fy"] * pc[:, 1] / zz + info["cy"]
        fr = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        in_fr |= fr
        cand = np.nonzero(fr & ~vis)[0]            # skip points already proven visible
        if cand.size == 0:
            continue
        dirs = P[cand] - eye
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        o = np.broadcast_to(eye, dirs.shape)
        ans = rc.cast_rays(o3d.core.Tensor(np.ascontiguousarray(
            np.concatenate([o, dirs], 1), dtype=np.float32)))
        t = ans["t_hit"].numpy()
        pid = ans["primitive_ids"].numpy().astype(np.int64)
        ok = ~np.isfinite(t)                       # a miss cannot be an occlusion
        hit = np.isfinite(t)
        if hit.any():
            gv = idx[cand[hit]]                    # the vertex each ray was aimed at
            ph = pid[hit]
            # is the hit triangle incident to that vertex? CSR membership, vectorised per degree
            found = np.zeros(gv.shape[0], bool)
            deg = off[gv + 1] - off[gv]
            for k in range(int(deg.max()) if deg.size else 0):
                m = k < deg
                if not m.any():
                    break
                found[m] |= inc[off[gv[m]] + k] == ph[m]
            ok[hit] = found
        vis[cand] |= ok
    return dict(scene=scene, labelled=int(P.shape[0]),
                in_frustum=float(in_fr.mean()), visible=float(vis.mean()),
                out_of_frustum=float(1 - in_fr.mean()),
                occluded=float((in_fr & ~vis).mean())), idx, vis, pts.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--save", action="store_true", help="write artifacts/scannet/<scene>/gt_visible.npy")
    a = ap.parse_args()
    print(f"{'scene':<14}{'labelled':>10}{'in frustum':>12}{'VISIBLE':>10}"
          f"{'out of frustum':>16}{'occluded':>10}")
    T = dict(lab=0, vis=0, fr=0)
    for sc in a.scenes.split(","):
        try:
            r, idx, vis, nall = one(sc)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if a.save:
            full = np.zeros(nall, bool); full[idx] = vis
            outd = os.path.join("artifacts", "scannet", sc)
            os.makedirs(outd, exist_ok=True)
            np.save(os.path.join(outd, "gt_visible.npy"), full)
        T['lab'] += r['labelled']; T['vis'] += int(r['visible'] * r['labelled'])
        T['fr'] += int(r['in_frustum'] * r['labelled'])
        print(f"{sc:<14}{r['labelled']:>10,}{r['in_frustum']:>12.2%}{r['visible']:>10.2%}"
              f"{r['out_of_frustum']:>16.2%}{r['occluded']:>10.2%}", flush=True)
    if T['lab']:
        print(f"\nTOTAL labelled {T['lab']:,}   in frustum {T['fr']/T['lab']:.2%}   "
              f"VISIBLE {T['vis']/T['lab']:.2%}   never seen {1-T['vis']/T['lab']:.2%}")


if __name__ == "__main__":
    main()
