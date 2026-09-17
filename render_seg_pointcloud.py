"""GT point cloud coloured by ground truth, by prediction, and by error -- no rasteriser involved.

WHY THIS EXISTS. Every panel so far went through PowerFoam's rasteriser, so any oddity in a figure
is ambiguous between "the labels are wrong" and "the render is wrong" -- and one such rendering bug
was already found and fixed here. This script bypasses the renderer completely: it assigns the
official GT points to power cells, reads each cell's predicted class, and plots the POINTS. What you
see is the label field itself.

It also prints the mIoU of exactly what it draws, so the picture and the number cannot disagree.

ASSIGNMENT is `assign_points_to_power_cells` -- the same function `evaluate_point_cloud_miou` uses
(argmin over primitives of ||x-c||^2 - r^2), not a nearest-centre approximation. PowerFoam needs no
frozen-point trick because its power diagram partitions space exactly.

PROJECTION uses the same TorchCamera as the image panels (c2w_rot / eye), so a point-cloud view at
view V lines up with the rendered panel at view V and the two can be read side by side.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_ply_xyz(path):
    """Minimal binary-little-endian PLY vertex reader for x/y/z.

    Written here rather than importing render_seg_gsplat_ply because that module pulls in gsplat,
    which only builds in the gs-view env -- this script runs in powerfoam.
    """
    with open(path, "rb") as f:
        fmt, props, count = None, [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property") and count:
                parts = line.split()
                if len(parts) == 3:
                    props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise SystemExit("unsupported ply format %r" % fmt)
        np_of = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1",
                 "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        dt = np.dtype([(nm, np_of[ty]) for ty, nm in props])
        arr = np.frombuffer(f.read(count * dt.itemsize), dtype=dt, count=count)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def project(points, cam):
    """World points -> pixel coords + depth, using the render camera's own pose/intrinsics."""
    R = cam.c2w_rot.detach().cpu().numpy().astype(np.float64)
    eye = cam.eye.detach().cpu().numpy().astype(np.float64)
    K = cam.intrinsics_matrix()
    K = K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)
    cam_pts = (points - eye) @ R          # world -> camera (R is camera->world, so R^T applied right)
    z = cam_pts[:, 2].copy()
    # PowerFoam looks down -z in camera space for these scenes; flip so depth is positive forward.
    if np.median(z) < 0:
        cam_pts = cam_pts * np.array([1.0, -1.0, -1.0])
        z = cam_pts[:, 2].copy()
    ok = z > 1e-6
    u = np.full(len(points), -1e9)
    v = np.full(len(points), -1e9)
    u[ok] = K[0, 0] * cam_pts[ok, 0] / z[ok] + cam.width / 2.0
    v[ok] = K[1, 1] * cam_pts[ok, 1] / z[ok] + cam.height / 2.0
    return u, v, z, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--gt-dir", required=True, help="Pointcept scene dir with coord.npy/segment20.npy")
    ap.add_argument("--class-names", required=True)
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--point-size", type=float, default=0.0, help="0 = auto from point count")
    # TOP-DOWN BY DEFAULT. A camera view shows only what that frame sees -- on scene0000_00 view 140
    # that was 9,214 of 81,369 points, far too sparse to read an error map from. Top-down shows the
    # whole room at once, which is the point of looking at the cloud instead of a render.
    ap.add_argument("--mode", default="topdown", choices=["topdown", "camera"])
    ap.add_argument("--pooled", action="store_true")
    # 3DGS column. Gaussians overlap and own no region of space, so the standard convention is
    # nearest-centre by Euclidean distance -- assign_points_to_nearest_center. Foam keeps its exact
    # power-cell query. Each representation is read with its own natural correspondence; only the
    # correspondence differs, the scoring afterwards is identical.
    ap.add_argument("--gs-ply", default=None, help="3DGS .ply to add a comparison column")
    ap.add_argument("--gs-features", default=None, help="solved features for the 3DGS ply")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seg_palette as P
    from point_cloud_query import assign_points_to_power_cells
    from render_seg_powerfoam import build_scene
    P.enable_determinism()

    names = [s.strip() for s in a.class_names.split(",") if s.strip()]
    model, dh, args = build_scene(a.ckpt, a.split)

    sv = torch.load(a.features, map_location="cpu", weights_only=True)
    feats, vm = sv["primitive_features"], sv["valid_mask"].numpy().astype(bool)
    col, cls, meta = P.primitive_colours(feats, vm, names, device="cuda", pooled=a.pooled)
    cls = cls.numpy()

    gt_pts = np.load(os.path.join(a.gt_dir, "coord.npy")).astype(np.float64)
    gt_lab = np.load(os.path.join(a.gt_dir, "segment20.npy")).reshape(-1)

    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    pred = np.full(len(gt_pts), -1, dtype=np.int64)
    pred[owned] = cls[assigned[owned]]

    # GT ids are indices into the scene's own class list; map to OUR vocabulary by name so the two
    # colourings are directly comparable rather than sharing indices by accident.
    import re
    ALL = eval(re.search(r"SCANNET20_CLASS_NAMES\s*=\s*(\[.*?\])",
                         open("evaluate_point_cloud_miou.py").read(), re.S).group(1))
    name_to_ours = {nm: i for i, nm in enumerate(names)}
    gt_ours = np.full(len(gt_pts), -1, dtype=np.int64)
    for gid in np.unique(gt_lab[gt_lab >= 0]):
        if gid < len(ALL) and ALL[gid] in name_to_ours:
            gt_ours[gt_lab == gid] = name_to_ours[ALL[gid]]

    # mIoU over classes present in GT -- the same present-classes-only convention the harness uses.
    scored = gt_ours >= 0
    ious = []
    for c in np.unique(gt_ours[scored]):
        inter = np.sum((pred == c) & (gt_ours == c))
        union = np.sum(((pred == c) | (gt_ours == c)) & scored)
        if union:
            ious.append(inter / union)
    miou = float(np.mean(ious)) if ious else float("nan")
    correct = (pred == gt_ours) & scored
    print("  points %d | owned %.1f%% | scored %d | mIoU %.4f | point acc %.4f"
          % (len(gt_pts), 100.0 * owned.mean(), int(scored.sum()), miou,
             correct.sum() / max(1, scored.sum())))

    if a.mode == "camera":
        cam = dh.cameras[a.view]
        u, v, z, ok = project(gt_pts, cam)
        inframe = ok & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        order = np.argsort(-z)                   # far to near: painter's algorithm
        order = order[inframe[order]]
        extent = (0, cam.width, cam.height, 0)
    else:
        # Orthographic top-down on the two widest axes, drawn bottom-up so upper surfaces win.
        span = gt_pts.max(0) - gt_pts.min(0)
        up = int(np.argmin(span))                # the thin axis is height
        ax0, ax1 = [i for i in range(3) if i != up]
        u, v = gt_pts[:, ax0], gt_pts[:, ax1]
        order = np.argsort(gt_pts[:, up])
        extent = (u.min(), u.max(), v.min(), v.max())

    def cols_for(lab):
        c = np.full((len(lab), 3), 0.82)
        for i, nm in enumerate(names):
            m = lab == i
            if not m.any():
                continue
            rgb = meta["hue_slots"].get(nm) or meta.get("background_hues", {}).get(nm) or P.GREY
            c[m] = np.asarray(rgb, dtype=float)
        return c

    # ---- optional 3DGS column -------------------------------------------------------------------
    gs_pred = gs_correct = None
    if a.gs_ply and a.gs_features:
        from point_cloud_query import assign_points_to_nearest_center
        gcent = read_ply_xyz(a.gs_ply)
        gsv = torch.load(a.gs_features, map_location="cpu", weights_only=True)
        if isinstance(gsv, dict):
            gfeat, gvm = gsv["primitive_features"], gsv["valid_mask"].numpy().astype(bool)
        else:
            gfeat = gsv; gvm = (gsv.norm(dim=-1) > 0).numpy()
        if gfeat.shape[0] != gcent.shape[0]:
            raise SystemExit("3DGS rows %d != ply vertices %d" % (gfeat.shape[0], gcent.shape[0]))
        _, gcls, _ = P.primitive_colours(gfeat, gvm, names, device="cuda", pooled=a.pooled)
        gcls = gcls.numpy()
        gassign = assign_points_to_nearest_center(gt_pts, gcent, valid=gvm)
        gs_pred = np.full(len(gt_pts), -1, dtype=np.int64)
        gowned = gassign >= 0
        gs_pred[gowned] = gcls[gassign[gowned]]
        gs_correct = (gs_pred == gt_ours) & scored
        gi = []
        for c in np.unique(gt_ours[scored]):
            inter = np.sum((gs_pred == c) & (gt_ours == c))
            union = np.sum(((gs_pred == c) | (gt_ours == c)) & scored)
            if union:
                gi.append(inter / union)
        print("  [3dgs] mIoU %.4f | point acc %.4f | owned %.1f%%"
              % (float(np.mean(gi)) if gi else float("nan"),
                 gs_correct.sum() / max(1, scored.sum()), 100.0 * gowned.mean()))
        both = (correct & gs_correct & scored).sum()
        only_f = (correct & ~gs_correct & scored).sum()
        only_g = (~correct & gs_correct & scored).sum()
        neither = (~correct & ~gs_correct & scored).sum()
        s = max(1, int(scored.sum()))
        print("  [bridge] both %.1f%% | foam-only %.1f%% | 3dgs-only %.1f%% | neither %.1f%%"
              % (100.0*both/s, 100.0*only_f/s, 100.0*only_g/s, 100.0*neither/s))
        print("  [bridge] oracle-of-two %.1f%% vs foam %.1f%% -> %.1f pts recoverable by fixing "
              "only what 3DGS already gets right"
              % (100.0*(both+only_f+only_g)/s, 100.0*correct.sum()/s, 100.0*only_g/s))

    err = np.full((len(gt_pts), 3), 0.85)
    err[scored & correct] = (0.72, 0.78, 0.72)   # correct: muted green
    err[scored & ~correct] = (0.85, 0.15, 0.15)  # wrong: red
    err[~scored] = (0.93, 0.93, 0.93)            # not scored (class absent from vocabulary)

    panels = [("ground truth", cols_for(gt_ours)),
              ("foam prediction", cols_for(pred)),
              ("foam errors (red = wrong, %.1f%% correct)"
               % (100.0 * correct.sum() / max(1, scored.sum())), err)]
    if gs_pred is not None:
        gerr = np.full((len(gt_pts), 3), 0.85)
        gerr[scored & gs_correct] = (0.72, 0.78, 0.72)
        gerr[scored & ~gs_correct] = (0.85, 0.15, 0.15)
        gerr[~scored] = (0.93, 0.93, 0.93)
        # WHERE THE TWO DIFFER is the actionable panel: blue is what foam already wins and 3DGS
        # does not, orange is the headroom foam could recover from a representation that is
        # otherwise weaker overall.
        agree = np.full((len(gt_pts), 3), 0.93)
        agree[scored & correct & gs_correct] = (0.80, 0.84, 0.80)
        agree[scored & correct & ~gs_correct] = (0.15, 0.45, 0.80)
        agree[scored & ~correct & gs_correct] = (0.95, 0.55, 0.10)
        agree[scored & ~correct & ~gs_correct] = (0.30, 0.30, 0.30)
        panels += [("3DGS errors (%.1f%% correct)"
                    % (100.0 * gs_correct.sum() / max(1, scored.sum())), gerr),
                   ("blue = foam only, orange = 3DGS only, dark = neither", agree)]
    ps = a.point_size or max(0.6, min(6.0, 90000.0 / max(1, len(order))))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0 * len(panels), 7.2), dpi=170)
    axes = np.atleast_1d(axes)
    for ax, (title, c) in zip(axes, panels):
        ax.scatter(u[order], v[order], c=c[order], s=ps, marker=".", linewidths=0)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)
    fig.suptitle("%s  view %d   mIoU %.3f  (%d GT points, %.1f%% owned)"
                 % (os.path.basename(a.gt_dir), a.view, miou, len(gt_pts), 100.0 * owned.mean()),
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out, "| %d points in frame" % len(order))


if __name__ == "__main__":
    main()
