"""Extract the boundary of the UNION of PowerFoam's rendered solids, as a triangle mesh.

WHAT THE RENDERER DRAWS (verified in tests/test_foam_solid.py against both kernels): each primitive
contributes a convex solid

    R_i = Ball(p_i, r_i)  n  (n_j radical half-spaces)  n  {x : n_i.(x - p_i) <= h}

and opacity is `1 - exp(-sigma_i * |ray n R_i|)`. A camera therefore sees the boundary of the union
of the occupied solids -- NOT a per-cell dipole facet, which is what every earlier extractor here
produced and why those meshes looked like disconnected slivers.

THE CONSTRUCTION. For each occupied cell, intersect its half-spaces into a convex polytope, take its
faces, and keep only the faces that lie on the boundary of the occupied union:

  * a face on the radical plane with neighbour j is INTERIOR if j is also occupied -- the two solids
    meet exactly there (tests confirm: no point belongs to both, points on either side belong to
    exactly one), so the union is connected across it and the face must be dropped;
  * a face on the radical plane with an EMPTY or transparent neighbour is exposed -- keep it;
  * the dipole face is the matter/void interface inside the cell -- always keep;
  * sphere-cap faces are exposed by construction -- keep. Where bounding spheres do not overlap, the
    solids are genuinely separated by space belonging to neither, and `L_connect` in the paper
    actively minimises overlap, so some separation is the trained state of the model, not an
    extraction artifact.

The sphere is approximated by `n_sphere_dirs` tangent half-spaces (a circumscribed polytope), which
slightly OVER-estimates the cap area -- an error in the unflattering direction.

The dipole plane is taken at its base height h = 0 here. The soft-Voronoi displacement makes the
true interface non-planar, so it cannot enter a convex polytope directly; displacing the extracted
dipole faces afterwards is the natural refinement and is left to the caller.
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

_FACE_NEIGHBOUR, _FACE_DIPOLE, _FACE_SPHERE = 0, 1, 2


def _sphere_dirs(n):
    """`n` roughly uniform directions on S^2 (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)


def cell_polytope(i, centers, radii, normals, neigh_i, sphere_dirs, height=0.0):
    """Convex polytope of R_i. Returns (verts, faces, kind, nbr) or None if empty/degenerate.

    `kind[f]` is one of _FACE_NEIGHBOUR / _FACE_DIPOLE / _FACE_SPHERE and `nbr[f]` is the
    neighbour index for _FACE_NEIGHBOUR faces (-1 otherwise).
    """
    ci, ri = centers[i], radii[i]
    n = normals[i] / max(np.linalg.norm(normals[i]), 1e-12)

    A, b, kind, nbr = [], [], [], []
    for j in neigh_i:
        j = int(j)
        if j == i:
            continue
        cj, rj = centers[j], radii[j]
        A.append(2.0 * (cj - ci))
        b.append((cj @ cj - rj * rj) - (ci @ ci - ri * ri))
        kind.append(_FACE_NEIGHBOUR)
        nbr.append(j)
    A.append(n); b.append(n @ ci + height); kind.append(_FACE_DIPOLE); nbr.append(-1)
    for u in sphere_dirs:
        A.append(u); b.append(u @ ci + ri); kind.append(_FACE_SPHERE); nbr.append(-1)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    kind = np.asarray(kind); nbr = np.asarray(nbr)

    # a strictly interior point: the site, nudged off the dipole plane into the dense half
    x0 = ci - n * (1e-3 * ri)
    slack = b - A @ x0
    if slack.min() <= 1e-12:
        return None                                   # site not strictly inside -> skip

    try:
        hs = HalfspaceIntersection(np.hstack([A, -b[:, None]]), x0)
        pts = hs.intersections
        if len(pts) < 4:
            return None
        hull = ConvexHull(pts)
    except Exception:
        return None

    verts = hull.points
    # GROUP SIMPLICES BY THE FACE THEY LIE ON. ConvexHull triangulates each planar face into many
    # simplices; every one of them is coplanar and belongs to a single polytope face. Merging them
    # back into one convex polygon and fan-triangulating gives (n-2) triangles instead of however
    # many the hull emitted -- exact, no geometry changes, just far fewer triangles. The sphere
    # approximation is the main beneficiary: it contributes up to `len(sphere_dirs)` faces per cell.
    Anorm = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    groups = {}
    for simplex in hull.simplices:
        cen = verts[simplex].mean(axis=0)
        tight = np.abs(A @ cen - b) / Anorm
        s = int(np.argmin(tight))
        if tight[s] > 1e-6:
            continue                                   # interior simplex, not on any face
        groups.setdefault(s, set()).update(int(v) for v in simplex)

    faces, fkind, fnbr = [], [], []
    for s, vset in groups.items():
        idx = np.fromiter(vset, dtype=np.int64)
        if len(idx) < 3:
            continue
        P = verts[idx]
        n_face = A[s] / Anorm[s]
        u = np.cross(n_face, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-8:
            u = np.cross(n_face, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n_face, u)
        d = P - P.mean(axis=0)
        order = np.argsort(np.arctan2(d @ v, d @ u))    # order around the face centroid
        ring = idx[order]
        for t in range(1, len(ring) - 1):               # fan-triangulate the convex polygon
            faces.append((ring[0], ring[t], ring[t + 1]))
            fkind.append(kind[s]); fnbr.append(nbr[s])
    if not faces:
        return None
    return verts, np.asarray(faces), np.asarray(fkind), np.asarray(fnbr)


def union_boundary(centers, radii, normals, adjacency, offsets, occupied, prim_class,
                   n_sphere_dirs=48, height=None, keep_sphere_caps=True, progress=None,
                   weld_tol=1e-6):
    """Triangle mesh of the boundary of the union of the occupied solids.

    Returns (verts (V,3), faces (F,3), face_class (F,), stats). Faces on a radical plane shared with
    another OCCUPIED cell are dropped as interior, which is what makes the result connected rather
    than a pile of disjoint patches.
    """
    dirs = _sphere_dirs(n_sphere_dirs)
    deg = np.diff(offsets)
    V, F, C = [], [], []
    off = 0
    n_int = n_ext = n_dip = n_sph = n_skip = n_step = 0
    live = np.nonzero(occupied)[0]
    for count, i in enumerate(live):
        if progress and count % progress == 0:
            print(f"    {count:,}/{len(live):,} cells", flush=True)
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        h = 0.0 if height is None else float(height[i])
        out = cell_polytope(int(i), centers, radii, normals, nb, dirs, height=h)
        if out is None:
            n_skip += 1
            continue
        verts, faces, kind, nbr = out
        keep = np.ones(len(faces), dtype=bool)
        is_nbr = kind == _FACE_NEIGHBOUR
        # A SHARED RADICAL FACE IS INTERIOR ONLY WHERE MATTER LIES ON BOTH SIDES.
        # Cell i's matter fills its cell below its own dipole plane at height h_i; cell j's fills
        # below h_j, and the two heights differ. On the face they share there is therefore a band
        # between h_i and h_j where one side is matter and the other is void -- a vertical step that
        # is genuinely part of the boundary. Dropping the whole shared face leaves that band open,
        # which is exactly the crack that cost 16% of ray coverage (93.2% -> 83.7%) when sphere caps
        # were removed. Here a shared face is kept where it lies OUTSIDE the neighbour's matter,
        # i.e. on the neighbour's void side n_j.(x - p_j) > h_j, and dropped elsewhere. These step
        # walls are what stitch per-cell dipole faces into a continuous surface -- no caps, no
        # dilation, no threshold.
        shared = is_nbr & (nbr >= 0)
        if shared.any():
            occ_nbr = occupied[np.clip(nbr, 0, len(occupied) - 1)]
            for fi in np.nonzero(shared & occ_nbr)[0]:
                j = int(nbr[fi])
                nj = normals[j] / max(np.linalg.norm(normals[j]), 1e-12)
                hj = 0.0 if height is None else float(height[j])
                # the face triangle is interior iff its centroid sits inside j's matter
                cenf = verts[faces[fi]].mean(axis=0)
                if (cenf - centers[j]) @ nj <= hj:
                    keep[fi] = False
                    n_int += 1
                else:
                    n_step += 1
        n_ext += int((keep & is_nbr).sum())
        n_dip += int((keep & (kind == _FACE_DIPOLE)).sum())
        if not keep_sphere_caps:
            keep &= kind != _FACE_SPHERE
        else:
            n_sph += int((keep & (kind == _FACE_SPHERE)).sum())
        if not keep.any():
            continue
        f = faces[keep]
        V.append(verts)
        F.append(f + off)
        C.append(np.full(len(f), int(prim_class[i]), dtype=np.int64))
        off += len(verts)

    if not F:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=np.int64), {"n_cells": 0})
    V = np.concatenate(V); F = np.concatenate(F); C = np.concatenate(C)
    n_raw_v = len(V)
    if weld_tol > 0:
        # WELD. Each cell's polytope is built independently, so a vertex shared by several cells is
        # duplicated once per cell and the mesh is only geometrically connected, not topologically.
        # Quantising to `weld_tol` and merging makes adjacent cells share vertices, which is what
        # makes the result a mesh a downstream tool can treat as one surface.
        key = np.round(V / weld_tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        V = V[first]
        F = inv[F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F, C = F[keep], C[keep]
    tri = V[F]
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                      axis=1).sum())
    stats = {"n_cells": int(len(live)), "n_skipped": n_skip,
             "n_verts": int(len(V)), "n_verts_before_weld": int(n_raw_v), "n_faces": int(len(F)),
             "n_interior_dropped": n_int, "n_exposed_neighbour": n_ext,
             "n_dipole": n_dip, "n_sphere": n_sph, "n_step": n_step, "area_m2": area}
    return V, F, C, stats
