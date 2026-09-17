"""The foam's isosurface, computed EXACTLY -- no voxels, no marching cubes, no discretisation.

WHY THIS EXISTS. Voxelising a foam and running marching cubes measures the grid, not the foam.
Measured on an analytic sphere: with a SMOOTH field (what a Gaussian mixture gives) marching cubes
recovers the area to -0.1% at h = 2 cm, but with a PIECEWISE-CONSTANT field (what a foam density
is) it overestimates by +9.8% and the error does not shrink with resolution -- 7.7% at 4 cm, 9.8%
at 2 cm, 8.9% at 1 cm -- because a step function gives the interpolator nothing to interpolate and
every crossing snaps to a voxel midpoint. That is a bias against the foam introduced purely by the
extractor, and it is unnecessary, because the foam's isosurface is available in closed form.

THE CONSTRUCTION. A power diagram partitions space into convex polyhedra. Fix a density level
sigma_iso and call a cell occupied when sigma_i >= sigma_iso. The isosurface is then exactly the
union of the FACES separating an occupied cell from an unoccupied one -- there is no interpolation
to do, because the density is constant within each cell and jumps across the face.

Each such face lies in the radical plane of its two sites. From
    ||x - c_i||^2 - r_i^2  =  ||x - c_j||^2 - r_j^2
the plane is linear:
    2 (c_j - c_i) . x  =  (||c_j||^2 - r_j^2) - (||c_i||^2 - r_i^2)
When r_i = r_j this is the perpendicular bisector; a larger r_i pushes the plane AWAY from site i,
which is what makes the cell of a larger-radius site bigger.

The face's extent is that plane clipped by cell i's other half-spaces (one per adjacency neighbour)
and by the scene box. Clipping a convex polygon by half-planes is exact, so the polygon, its area
and uniform samples on it are exact.

COST. Only faces on the occupancy boundary are built, not all faces: an interior face between two
occupied cells is not part of the surface. That is typically a small fraction of the diagram.
"""
import numpy as np


DISC_SIDES = 16       # half-planes used to approximate the compact-support disc


def radical_plane(ci, cj, ri, rj):
    """Outward normal and offset of the plane separating cells i and j, as n.x = d.

    Returned normal points from i toward j, so `n.x <= d` is the half-space belonging to cell i.
    """
    n = 2.0 * (cj - ci)
    d = (cj @ cj - rj * rj) - (ci @ ci - ri * ri)
    return n, d


def _plane_basis(n):
    """Orthonormal (u, v) spanning the plane with normal n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _clip_polygon_2d(poly, a, b, c):
    """Sutherland-Hodgman clip of a convex polygon by the half-plane a*x + b*y <= c."""
    if len(poly) == 0:
        return poly
    out = []
    m = len(poly)
    for k in range(m):
        p, q = poly[k], poly[(k + 1) % m]
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


def face_polygon(i, j, centers, radii, neigh_i, box_lo, box_hi, extent=None):
    """Exact polygon of the power-diagram face between cells i and j, in world coordinates.

    `neigh_i` are cell i's adjacency neighbours (j may be among them; it is skipped). The face is
    the radical plane of (i, j) clipped by i's half-spaces against every other neighbour, then by
    the scene box.
    """
    ci, cj = centers[i], centers[j]
    n, d = radical_plane(ci, cj, radii[i], radii[j])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.zeros((0, 3))
    origin = n * (d / (nn * nn))                   # point on the plane closest to the world origin
    u, v = _plane_basis(n)

    if extent is None:
        extent = float(np.linalg.norm(box_hi - box_lo))
    poly = np.array([[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]])

    for k in neigh_i:
        if k == j or k == i:
            continue
        nk, dk = radical_plane(ci, centers[k], radii[i], radii[k])
        # nk.x <= dk in the plane's coordinates: x = origin + s*u + t*v
        a, b = nk @ u, nk @ v
        c = dk - nk @ origin
        if abs(a) < 1e-15 and abs(b) < 1e-15:
            if c < 0:
                return np.zeros((0, 3))            # infeasible: face is empty
            continue
        poly = _clip_polygon_2d(poly, a, b, c)
        if len(poly) == 0:
            return np.zeros((0, 3))

    for axis in range(3):                          # clip to the scene box
        e = np.zeros(3); e[axis] = 1.0
        for sgn, lim in ((1.0, box_hi[axis]), (-1.0, -box_lo[axis])):
            a, b = sgn * (e @ u), sgn * (e @ v)
            c = lim - sgn * (e @ origin)
            if abs(a) < 1e-15 and abs(b) < 1e-15:
                if c < 0:
                    return np.zeros((0, 3))
                continue
            poly = _clip_polygon_2d(poly, a, b, c)
            if len(poly) == 0:
                return np.zeros((0, 3))

    return origin + poly[:, :1] * u + poly[:, 1:] * v


def polygon_area(pts3):
    """Area of a planar convex polygon given in order."""
    if len(pts3) < 3:
        return 0.0
    p0 = pts3[0]
    return float(0.5 * np.linalg.norm(
        np.cross(pts3[1:-1] - p0, pts3[2:] - p0).sum(axis=0)))


def sample_polygon(pts3, n, rng):
    """n uniform samples on a convex polygon, by fan triangulation weighted by triangle area."""
    if len(pts3) < 3 or n <= 0:
        return np.zeros((0, 3))
    p0 = pts3[0]
    tri = np.stack([np.broadcast_to(p0, (len(pts3) - 2, 3)), pts3[1:-1], pts3[2:]], axis=1)
    ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tot = ar.sum()
    if tot <= 0:
        return np.zeros((0, 3))
    pick = rng.choice(len(ar), size=n, p=ar / tot)
    r1, r2 = rng.random(n), rng.random(n)
    sq = np.sqrt(r1)
    w = np.stack([1 - sq, sq * (1 - r2), sq * r2], axis=1)
    t = tri[pick]
    return (w[:, :, None] * t).sum(axis=1)


def foam_isosurface(centers, radii, density, prim_class, adjacency, offsets, sigma_iso,
                    box_lo, box_hi, samples_per_m2=2500, min_per_class=500, seed=0):
    """Exact isosurface of the foam at `sigma_iso`.

    Returns (points (N,3), class (N,), area_by_class). A face is on the surface when exactly one of
    its two cells is occupied; the face takes the OCCUPIED cell's class, since that is the material
    whose boundary it is.
    """
    rng = np.random.default_rng(seed)
    occ = density >= sigma_iso
    deg = np.diff(offsets)
    faces = []                                     # (i_occupied, j_other)
    for i in np.nonzero(occ)[0]:
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        for j in nb[~occ[nb]]:
            faces.append((i, int(j)))

    by_class = {}
    for i, j in faces:
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        poly = face_polygon(i, j, centers, radii, nb, box_lo, box_hi)
        a = polygon_area(poly)
        if a <= 0:
            continue
        by_class.setdefault(int(prim_class[i]), []).append((poly, a))

    pts, cls, areas = [], [], {}
    for c, items in by_class.items():
        tot = float(sum(a for _, a in items))
        if tot <= 0:
            continue
        areas[c] = tot
        n_target = max(min_per_class, int(tot * samples_per_m2))
        for poly, a in items:
            k = int(round(n_target * a / tot))
            if k <= 0:
                continue
            s = sample_polygon(poly, k, rng)
            if len(s):
                pts.append(s)
                cls.append(np.full(len(s), c, dtype=np.int64))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), {}
    return np.concatenate(pts), np.concatenate(cls), areas


def dipole_polygon(i, centers, radii, normals, heights, neigh_i, box_lo, box_hi, max_radius=None):
    """The primitive's OWN surface patch: its displaced dipole plane clipped by its power cell.

    This is the surface the renderer actually uses. rasterize.py builds it as
        n . x = n . p + h                       (ray_plane_intersect, rendering_math.py:132)
    with `n` the primitive's normal (scene.get_normals, the quaternion's first axis), `p` its
    centre and `h` a texel-interpolated displacement (`texel_height * radii`); a ray is clipped to
    it via `t_surf` rather than being allowed to cross the whole cell. Restricting that plane to
    the cell it belongs to gives a bounded planar patch, and the union of patches over primitives
    the renderer would actually stop at IS the rendered surface -- with no grid and no isovalue.

    `max_radius` bounds the patch to a disc of that radius about the cell centre, giving the foam
    the COMPACT SUPPORT a Gaussian has by construction. Without it a cell facing open air owns its
    plane until the scene's bounding box stops it: measured on scene0062_00 the median patch is
    13.6 cm^2 but the 99th percentile is 6.6 m^2 and the largest single patch is 8.9 m^2 -- half the
    room's entire GT surface -- so the top 5% of cells contribute 55% of all claimed area. That tail,
    not a broad error, is what turned the extracted semantic surface into volumetric fog. The disc is
    realised as `disc_sides` half-planes at distance `max_radius`, i.e. a CIRCUMSCRIBED polygon whose
    area exceeds the disc's by n*tan(pi/n)/pi (1.4% at the default 16 sides) -- an over-estimate, so
    the cap never flatters the method.
    """
    # AN EMPTY NEIGHBOUR LIST MEANS AN EMPTY CELL, NOT AN UNBOUNDED ONE. Unlike a Voronoi
    # diagram, a power diagram admits BURIED sites: if r_i is small enough relative to its
    # neighbours' radii, the half-spaces of those neighbours cover it completely and cell i is
    # empty, so the site never appears in the regular triangulation and has no facets at all.
    # Measured on scene0062_00/truefrozen: 10,473 of 51,610 sites (20.3%) are buried. Treating
    # their missing neighbours as "nothing to clip against" handed each of them the whole scene
    # box -- mean patch 6.19 m^2, max 9.69 m^2 against an 18 m^2 room, versus 16.6 cm^2 mean and
    # 0.04 m^2 MAX for sites with a real cell. Every oversized patch in this extractor came from
    # here, not from the foam being space-filling.
    if len(neigh_i) == 0:
        return np.zeros((0, 3))

    ci = centers[i]
    n = normals[i]
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.zeros((0, 3))
    n = n / nn
    origin = ci + n * heights[i]                    # a point on the displaced plane
    u, v = _plane_basis(n)
    extent = float(np.linalg.norm(box_hi - box_lo))
    poly = np.array([[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]])

    for k in neigh_i:
        if k == i:
            continue
        nk, dk = radical_plane(ci, centers[k], radii[i], radii[k])
        a, b = nk @ u, nk @ v
        c = dk - nk @ origin
        if abs(a) < 1e-15 and abs(b) < 1e-15:
            if c < 0:
                return np.zeros((0, 3))
            continue
        poly = _clip_polygon_2d(poly, a, b, c)
        if len(poly) == 0:
            return np.zeros((0, 3))

    if max_radius is not None and np.isfinite(max_radius):
        # the centre projects to the plane origin, so the disc is centred at (0, 0) in (u, v)
        for t in np.linspace(0.0, 2.0 * np.pi, DISC_SIDES, endpoint=False):
            poly = _clip_polygon_2d(poly, np.cos(t), np.sin(t), float(max_radius))
            if len(poly) == 0:
                return np.zeros((0, 3))

    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        for sgn, lim in ((1.0, box_hi[axis]), (-1.0, -box_lo[axis])):
            a, b = sgn * (e @ u), sgn * (e @ v)
            c = lim - sgn * (e @ origin)
            if abs(a) < 1e-15 and abs(b) < 1e-15:
                if c < 0:
                    return np.zeros((0, 3))
                continue
            poly = _clip_polygon_2d(poly, a, b, c)
            if len(poly) == 0:
                return np.zeros((0, 3))
    return origin + poly[:, :1] * u + poly[:, 1:] * v


def foam_dipole_surface(centers, radii, normals, heights, alpha, prim_class, adjacency, offsets,
                        box_lo, box_hi, alpha_min=0.1, samples_per_m2=2500, min_per_class=500,
                        seed=0, max_radius=None):
    """Rendered surface of the foam: dipole patches of primitives opaque enough to stop a ray.

    `alpha` is the per-primitive opacity the renderer would accumulate crossing that primitive,
    1 - exp(-sigma * L); `alpha_min` is the same 0.1 threshold used everywhere else in this project.
    Unlike a density ISOSURFACE, this does not turn every occupied/unoccupied cell boundary into
    surface -- a foam tiles all of space, so that construction emitted 2800 m^2 against an 18 m^2
    scene. Here each primitive contributes at most its own patch, so the total tracks the scene.
    """
    rng = np.random.default_rng(seed)
    deg = np.diff(offsets)
    live = np.nonzero(alpha >= alpha_min)[0]
    by_class = {}
    for i in live:
        nb = adjacency[offsets[i]:offsets[i] + deg[i]]
        mr = (max_radius[i] if isinstance(max_radius, np.ndarray) else max_radius)
        poly = dipole_polygon(i, centers, radii, normals, heights, nb, box_lo, box_hi, mr)
        a = polygon_area(poly)
        if a <= 0:
            continue
        by_class.setdefault(int(prim_class[i]), []).append((poly, a))

    pts, cls, areas = [], [], {}
    for c, items in by_class.items():
        tot = float(sum(a for _, a in items))
        if tot <= 0:
            continue
        areas[c] = tot
        n_target = max(min_per_class, int(tot * samples_per_m2))
        for poly, a in items:
            k = int(round(n_target * a / tot))
            if k <= 0:
                continue
            sm = sample_polygon(poly, k, rng)
            if len(sm):
                pts.append(sm)
                cls.append(np.full(len(sm), c, dtype=np.int64))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), {}
    return np.concatenate(pts), np.concatenate(cls), areas


# The renderer's soft-Voronoi temperature (rasterize.py:64, `temp = wp.constant(10.0)`).
SV_TEMP = 10.0


def soft_voronoi_height(pts, sites_w, heights_w, radius, temp=SV_TEMP):
    """The displacement field of Eq. (3), evaluated exactly as the renderer evaluates it.

    `plane_intersection_fwd_local` (rendering_math/rasterize) weights each detail site by
        w_i = exp(-temp * ||x - s_i||^2 / radius^2)
    using the 3-D distance from the BASE-plane intersection to the world-space site, and returns
    the weighted mean of the site heights. Site coordinates are stored in units of radius in the
    (tangent, bitangent) frame and heights in units of radius (scene.py:397-410), so both are
    converted to world units by the caller.

    Averaging the heights instead -- which is the temp -> 0 limit of this softmax -- collapses the
    displaced surface to a flat plane and discards the high-frequency geometry the detail sites
    exist to carry. That is what an earlier version of this module did.
    """
    d2 = ((pts[:, None, :] - sites_w[None, :, :]) ** 2).sum(-1) / (radius * radius)
    w = np.exp(-temp * d2)
    return (w * heights_w[None, :]).sum(-1) / np.maximum(w.sum(-1), 1e-20)


def _subdivide(tri, n):
    """Split each triangle into n^2 similar triangles, so a curved displacement is resolved."""
    out = []
    for a, b, c in tri:
        for i in range(n):
            for j in range(n - i):
                p = lambda u, v: a + (b - a) * (u / n) + (c - a) * (v / n)
                out.append((p(i, j), p(i + 1, j), p(i, j + 1)))
                if j < n - i - 1:
                    out.append((p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)))
    return np.array(out)


def displaced_patch(poly, centre, normal, sites_w, heights_w, radius, subdiv=3,
                    temp=SV_TEMP, bound_radius=None):
    """Fan-triangulate a base dipole polygon and push every vertex along the normal by Eq. (3).

    Returns (triangles (T,3,3), area). The area is measured on the DISPLACED triangles, so a
    corrugated patch correctly reports more area than its flat footprint.
    """
    if len(poly) < 3:
        return np.zeros((0, 3, 3)), 0.0
    fan = np.array([(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)])
    tri = _subdivide(fan, subdiv) if subdiv > 1 else fan
    flat = tri.reshape(-1, 3)
    h = soft_voronoi_height(flat, sites_w, heights_w, radius, temp)
    tri = (flat + normal[None, :] * h[:, None]).reshape(-1, 3, 3)
    if bound_radius is not None:
        # THE CELL IS BOUNDED BY ITS OWN SPHERE. Power Foam's cells are the power cell INTERSECTED
        # with Ball(p_i, r_i) -- that is the "bounded" in bounded power diagram, and the radius is
        # a learned parameter used both as the power weight and as this bound (paper Sec. 3.2).
        # The base face passes through the centre, so face n Ball is a disc of radius r_i; after
        # displacement the surface leaves that plane, so membership is tested per vertex against
        # the ball itself. A triangle is kept only if all three displaced vertices are inside,
        # which under-counts slightly at the rim rather than over-counting.
        inside = (np.linalg.norm(tri - centre[None, None, :], axis=2) <= bound_radius).all(axis=1)
        tri = tri[inside]
        if len(tri) == 0:
            return np.zeros((0, 3, 3)), 0.0
    a = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                   axis=1).sum())
    return tri, a


def sample_triangles(tri, n, rng):
    """n area-uniform samples over a triangle soup."""
    if len(tri) == 0 or n <= 0:
        return np.zeros((0, 3))
    ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tot = ar.sum()
    if tot <= 0:
        return np.zeros((0, 3))
    pick = rng.choice(len(ar), size=n, p=ar / tot)
    r1, r2 = rng.random(n), rng.random(n)
    sq = np.sqrt(r1)
    w = np.stack([1 - sq, sq * (1 - r2), sq * r2], axis=1)
    t = tri[pick]
    return (w[:, :, None] * t).sum(axis=1)
