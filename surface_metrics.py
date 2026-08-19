"""Extended surface metrics, with DELIBERATELY PRECISE NAMING.

Naming rule enforced here: a metric is either
  (a) a DISTANCE, reported in metres/cm, named MAE / RMSE / median / Hausdorff -- never
      "accuracy", which is dimensionless; or
  (b) a FRACTION in [0,1] produced by a decision criterion (a distance threshold, or voxel
      occupancy), named precision / recall / F1 / IoU / Dice, always with the criterion
      stated.
The MVS literature (DTU, and 2DGS/GOF/TrimGS following it) calls the mean rec->GT distance
"accuracy" and the mean GT->rec distance "completeness". Those are mean absolute errors, not
accuracies -- a fraction cannot be 1.33cm. We report them as MAE and note the legacy names
only where needed to line up with published tables.

Two families, kept distinct:

* **Distances** (mae_rec2gt / mae_gt2rec / chamfer_l1 / rmse / median / HD / HD95):
  nearest-neighbour distances between sampled reconstruction points and GT points.
  HD is the raw max distance -- extremely outlier-sensitive, one stray triangle sets it --
  so HD95 (95th percentile) is reported alongside and is the number to trust.

* **Fractions with a criterion**: threshold-based precision/recall/F1 (criterion: nearest
  neighbour within tau) and voxel-occupancy IoU / Dice / precision / recall (criterion:
  voxel contains surface). both sides are voxelized on a
  shared grid and compared as occupancy sets. NOTE the ScanNet GT here is a POINT CLOUD
  (Pointcept `coord.npy`), not a watertight mesh, so this is **surface**-occupancy, not
  solid-volume occupancy: a voxel counts as occupied if the surface passes through it.
  Both sides are treated identically, so the comparison is fair, but it must not be read
  as a volumetric overlap.

  Dice and F1 are mathematically identical for binary sets (both = 2|A∩B|/(|A|+|B|)); both
  names are printed because both get asked for, not because they differ.

Voxelization is sensitive to sampling density on the reconstruction side: too few sampled
points and occupied voxels are missed, deflating recall. Sampling is therefore scaled to
the grid, and the effective sample density is reported so the number can be sanity-checked.
"""
import numpy as np
import open3d as o3d


def voxel_keys(pts, voxel, origin):
    idx = np.floor((pts - origin) / voxel).astype(np.int64)
    # pack to a single int64 key; ranges are tiny for room-scale scenes
    m = idx.min(0)
    idx = idx - m
    dims = idx.max(0) + 1
    return idx[:, 0] * (dims[1] * dims[2]) + idx[:, 1] * dims[2] + idx[:, 2], m, dims


def voxel_occupancy_metrics(rec_pts, gt_pts, voxel):
    """Surface-occupancy IoU / Dice / precision / recall on a shared voxel grid."""
    origin = np.minimum(rec_pts.min(0), gt_pts.min(0)) - voxel
    allp = np.vstack([rec_pts, gt_pts])
    idx = np.floor((allp - origin) / voxel).astype(np.int64)
    dims = idx.max(0) + 1
    keys = idx[:, 0] * (dims[1] * dims[2]) + idx[:, 1] * dims[2] + idx[:, 2]
    kr = np.unique(keys[:len(rec_pts)])
    kg = np.unique(keys[len(rec_pts):])
    inter = np.intersect1d(kr, kg, assume_unique=True).size
    union = kr.size + kg.size - inter
    prec = inter / max(kr.size, 1)
    rec = inter / max(kg.size, 1)
    dice = 2 * inter / max(kr.size + kg.size, 1)
    return {"voxel": voxel, "iou": inter / max(union, 1), "dice": dice, "f1": dice,
            "precision": prec, "recall": rec,
            "n_vox_rec": int(kr.size), "n_vox_gt": int(kg.size), "n_vox_inter": int(inter)}


def full_surface_metrics(rec_pts, gt_pts, thresh=0.05, voxels=(0.02, 0.05)):
    rec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec_pts))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    d1 = np.asarray(rec.compute_point_cloud_distance(gt))   # rec -> gt (accuracy)
    d2 = np.asarray(gt.compute_point_cloud_distance(rec))   # gt -> rec (completeness)

    prec = float((d1 < thresh).mean())
    recl = float((d2 < thresh).mean())
    out = {
        # --- distances (metres). d1 = rec->GT, d2 = GT->rec ---
        "mae_rec2gt": float(d1.mean()),          # MVS papers call this "accuracy"
        "mae_gt2rec": float(d2.mean()),          # MVS papers call this "completeness"
        "chamfer_l1": float((d1.mean() + d2.mean()) / 2),
        "rmse_rec2gt": float(np.sqrt((d1 ** 2).mean())),
        "rmse_gt2rec": float(np.sqrt((d2 ** 2).mean())),
        "median_rec2gt": float(np.median(d1)), "median_gt2rec": float(np.median(d2)),
        "hd": float(max(d1.max(), d2.max())),
        "hd95": float(max(np.percentile(d1, 95), np.percentile(d2, 95))),
        "hd95_rec2gt": float(np.percentile(d1, 95)),
        "hd95_gt2rec": float(np.percentile(d2, 95)),
        # --- fractions in [0,1]; criterion = nearest neighbour within `thresh` ---
        "threshold_m": thresh,
        "precision@tau": prec, "recall@tau": recl,
        "f1@tau": float(2 * prec * recl / max(prec + recl, 1e-9)),
        "n_rec": int(len(rec_pts)), "n_gt": int(len(gt_pts)),
    }
    out["voxel_metrics"] = [voxel_occupancy_metrics(rec_pts, gt_pts, v) for v in voxels]
    return out


def print_metrics(tag, m):
    tau = m.get("threshold_m", 0.05) * 100
    print(f"[{tag}]")
    print(f"  distances (cm): MAE rec->GT={m['mae_rec2gt']*100:6.2f}  MAE GT->rec={m['mae_gt2rec']*100:6.2f}  "
          f"CD-L1={m['chamfer_l1']*100:6.2f}  RMSE={m['rmse_rec2gt']*100:6.2f}/{m['rmse_gt2rec']*100:.2f}")
    print(f"  hausdorff (cm): HD={m['hd']*100:7.2f}  HD95={m['hd95']*100:6.2f} "
          f"(rec->gt {m['hd95_rec2gt']*100:.2f}, gt->rec {m['hd95_gt2rec']*100:.2f})")
    print(f"  fractions (criterion: NN within {tau:.0f}cm): "
          f"precision={m['precision@tau']:.3f} recall={m['recall@tau']:.3f} F1={m['f1@tau']:.3f}")
    for v in m["voxel_metrics"]:
        print(f"  voxel {v['voxel']*100:.0f}cm: IoU={v['iou']:.4f} Dice/F1={v['dice']:.4f} "
              f"prec={v['precision']:.4f} rec={v['recall']:.4f} "
              f"(rec {v['n_vox_rec']}, gt {v['n_vox_gt']}, inter {v['n_vox_inter']})")


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--gt", default=None, help="path to coord.npy (sparse vertices)")
    p.add_argument("--gt-mesh", default=None,
                   help="path to the ScanNet points3d.ply MESH. Strongly preferred over --gt: "
                        "the mesh is densely sampled so GT fills its own surface voxels. With "
                        "sparse vertices (~3cm spacing) voxel IoU measures GT sparsity, not "
                        "reconstruction error -- measured on scene0000_00, GT occupies 80k "
                        "voxels at 2cm as vertices vs 434k when sampled from the mesh.")
    p.add_argument("--gt-sample", type=int, default=2_000_000)
    p.add_argument("--tag", default=None)
    p.add_argument("--n-sample", type=int, default=2_000_000)
    p.add_argument("--voxels", default="0.02,0.05,0.10")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    rec = np.asarray(mesh.sample_points_uniformly(number_of_points=a.n_sample).points)
    if a.gt_mesh:
        gtm = o3d.io.read_triangle_mesh(a.gt_mesh)
        gt = np.asarray(gtm.sample_points_uniformly(a.gt_sample).points).astype(np.float64)
    else:
        gt = np.load(a.gt).astype(np.float64)
    m = full_surface_metrics(rec, gt, voxels=[float(x) for x in a.voxels.split(",")])
    print_metrics(a.tag or Path(a.mesh).stem, m)
    if a.output:
        json.dump(m, open(a.output, "w"), indent=2)
        print(f"wrote {a.output}")
