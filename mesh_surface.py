"""Semantic surface metrics against ScanNet's LABELLED MESH, not its vertex cloud.

WHY THIS REPLACES `ablation_surface.GTSurfaceIndex`. That index represents class c by the SUBSET OF
GT VERTICES labelled c, and it represents the prediction the same way, so both sides of every
distance are discrete point sets. Two consequences, and both distorted the dipole-surface results:

  1. `pred -> GT_c` measured distance to the nearest labelled VERTEX rather than to the class's
     surface. Bounded below by the vertex spacing (median 1.26 cm on scene0347_00).
  2. `GT_c -> pred` is mean-distance-to-nearest, which falls monotonically as the predicted set
     gets denser -- for free, with no improvement in placement. This is what invalidated the first
     dipole-surface comparison: the extracted surface carried 5-50x more points than the reference,
     and the apparent completeness gain inverted once matched.

Using the mesh fixes both. `pred -> GT_c` becomes exact point-to-triangle distance to the class-c
SURFACE, independent of how many points either side has. `GT_c -> pred` samples that surface
uniformly BY AREA at a density WE choose, so the reference density is controlled rather than
inherited from however finely ScanNet happened to tessellate.

Measured honestly, before writing this: switching the reference from vertices to faces changes the
raw numbers very little (8.81 -> 8.77 cm median on 2.46M dipole samples), because ScanNet's
tessellation is already fine relative to the errors involved. The reason to use the mesh is not a
better number -- it is that the metric no longer has a floor set by vertex spacing, and no longer
rewards emitting more points.

THE MESH. `D:\\Downloads\\scenes10_points3d\\{scene}\\points3d.ply`, VCGLIB-generated (ScanNet's
vh_clean_2 pipeline). Its vertices are BIT-IDENTICAL to Pointcept's `coord.npy` in the same order
(verified: max |V - G| = 0.0), which is what lets `segment20.npy` labels attach to mesh vertices
and hence to faces.

FACE LABELS, and the rule for disagreement. A face carries three vertex labels. We take the
majority among its NON-ZERO labels and require at least two of the three to agree; a face whose
vertices give three different classes, or fewer than two labelled vertices, is DROPPED as an
ambiguous boundary face rather than assigned arbitrarily. Label 0 is the ignore label throughout
(OpenGaussian's convention), so a vertex deleted by the opacity mask simply stops voting -- which
is how the mask, defined per point, carries over to a surface representation.

MISSED CLASSES ARE COUNTED, NOT DROPPED, exactly as in ablation_surface: if a class is never
predicted its distances are undefined, and excluding it silently would reward a method for
predicting nothing. Such classes are reported in `n_missed`, which must be read next to the means.
"""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

MESH_ROOT = r"D:\Downloads\scenes10_points3d"
TAU = 0.02              # 2 cm boundary criterion, same as ablation_surface
GT_SAMPLES_PER_M2 = 2500        # ~2 cm spacing; ScanNet's own vertices are ~1360 /m^2
MIN_SAMPLES_PER_CLASS = 500


def load_mesh(scene: str):
    p = os.path.join(MESH_ROOT, scene, "points3d.ply")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no ScanNet mesh for {scene} at {p}")
    return o3d.io.read_triangle_mesh(p)


def _sample_mesh_uniform(V: np.ndarray, T: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Area-uniform points on a triangle mesh, seeded and version-independent.

    Open3D's `sample_points_uniformly` is the obvious call, but its signature varies across builds
    (this one takes no `seed`), and an unseeded reference set would make the GT->pred direction
    non-reproducible run to run. Doing it here is ten lines and removes both problems.

    Uniform over AREA, not over triangles: pick each triangle with probability proportional to its
    area, then a uniform barycentric point inside it. The sqrt in `u` is what makes the barycentric
    draw uniform over the triangle rather than clustered toward one vertex.
    """
    a = V[T[:, 1]] - V[T[:, 0]]
    b = V[T[:, 2]] - V[T[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    tot = areas.sum()
    if tot <= 0 or len(T) == 0:
        return V[:0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), size=n, p=areas / tot)
    u = np.sqrt(rng.random(n))[:, None]
    v = rng.random(n)[:, None]
    # (1-u) A + u(1-v) B + u v C
    return (1 - u) * V[T[idx, 0]] + u * (1 - v) * V[T[idx, 1]] + u * v * V[T[idx, 2]]


def face_labels(tris: np.ndarray, vert_labels: np.ndarray) -> np.ndarray:
    """Majority label per face over non-zero vertex labels; 0 where ambiguous or unlabelled."""
    L = vert_labels[tris]                                  # (F, 3)
    out = np.zeros(len(tris), dtype=np.int64)
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    # at least two agreeing and non-zero
    ab, ac, bc = (a == b) & (a != 0), (a == c) & (a != 0), (b == c) & (b != 0)
    out[bc] = b[bc]
    out[ac] = a[ac]
    out[ab] = a[ab]                                        # a==b wins ties by construction
    return out


class MeshSurfaceIndex:
    """Per-class mesh surface for one (scene, class-set). Build once, reuse across methods.

    Mirrors `ablation_surface.GTSurfaceIndex`: `.classes()` lists the classes present, and
    `semantic_surface_metrics_mesh(index, pred_points, pred_cls)` scores a prediction against it.
    """

    def __init__(self, scene: str, vert_labels: np.ndarray, n_classes: int,
                 samples_per_m2: float = GT_SAMPLES_PER_M2, seed: int = 0):
        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        Tri = np.asarray(mesh.triangles)
        if len(V) != len(vert_labels):
            raise ValueError(f"{scene}: mesh has {len(V)} vertices, labels have {len(vert_labels)}")
        self.scene = scene
        self.n_classes = n_classes
        self.scenes = {}        # class -> RaycastingScene (for exact pred->surface distance)
        self.samples = {}       # class -> (N,3) area-uniform samples (for surface->pred)
        self.trees = {}         # class -> KD-tree over those samples
        self.area = {}

        fl = face_labels(Tri, vert_labels)
        for c in range(1, n_classes):
            sel = fl == c
            if not sel.any():
                continue                       # absent from this scene: not scored, per protocol
            sub = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(V),
                o3d.utility.Vector3iVector(Tri[sel]))
            sub.remove_unreferenced_vertices()
            area = float(sub.get_surface_area())
            if area <= 0:
                continue
            rs = o3d.t.geometry.RaycastingScene()
            rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sub))
            n = max(MIN_SAMPLES_PER_CLASS, int(area * samples_per_m2))
            pts = _sample_mesh_uniform(np.asarray(sub.vertices),
                                       np.asarray(sub.triangles), n, seed)
            self.scenes[c] = rs
            self.samples[c] = pts
            self.trees[c] = cKDTree(pts)
            self.area[c] = area

    def classes(self):
        return sorted(self.scenes)

    def distance_to_class(self, c: int, query: np.ndarray) -> np.ndarray:
        """Exact point-to-triangle distance from `query` to the class-c surface."""
        q = o3d.core.Tensor(np.ascontiguousarray(query, dtype=np.float32),
                            dtype=o3d.core.Dtype.Float32)
        return self.scenes[c].compute_distance(q).numpy().astype(np.float64)


def semantic_surface_metrics_mesh(index: MeshSurfaceIndex, pred_pts: np.ndarray,
                                  pred_cls: np.ndarray, tau: float = TAU):
    """Same metric definitions as ablation_surface, with the mesh as the reference.

    `pred_pts` (N,3) predicted surface points, `pred_cls` (N,) their 1-based class ids.

    mae_pred2gt(c)  mean over PRED_c of exact distance to the class-c SURFACE
    mae_gt2pred(c)  mean over area-uniform samples OF that surface of distance to PRED_c
    scd(c)          mean of the two
    hd95(c)         max of the two 95th percentiles
    boundary_f1(c)  F1 under "within tau of the other side"
    """
    per_class = {}
    for c in index.classes():
        pm = pred_cls == c
        n_pred = int(pm.sum())
        if n_pred == 0:
            per_class[c] = {"n_pred": 0, "missed": True, "area_m2": index.area[c]}
            continue
        ppts = pred_pts[pm]
        d_p2g = index.distance_to_class(c, ppts)                 # exact, density-independent
        d_g2p, _ = cKDTree(ppts).query(index.samples[c], k=1, workers=-1)
        prec, rec = float((d_p2g <= tau).mean()), float((d_g2p <= tau).mean())
        per_class[c] = {
            "n_pred": n_pred, "missed": False, "area_m2": index.area[c],
            "n_gt_samples": int(len(index.samples[c])),
            "mae_pred2gt": float(d_p2g.mean()), "mae_gt2pred": float(d_g2p.mean()),
            "scd": float((d_p2g.mean() + d_g2p.mean()) / 2),
            "hd95": float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))),
            "boundary_precision": prec, "boundary_recall": rec,
            "boundary_f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        }
    live = [m for m in per_class.values() if not m["missed"]]
    n_missed = sum(1 for m in per_class.values() if m["missed"])
    if not live:
        return {"n_classes_present": len(per_class), "n_missed": n_missed, "n_scored": 0}
    agg = {k: float(np.mean([m[k] for m in live]))
           for k in ("mae_pred2gt", "mae_gt2pred", "scd", "hd95",
                     "boundary_precision", "boundary_recall", "boundary_f1")}
    agg.update({"n_classes_present": len(per_class), "n_missed": n_missed,
                "n_scored": len(live), "tau": tau, "per_class": per_class})
    return agg
