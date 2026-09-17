"""The SOLID each PowerFoam primitive renders, reproduced exactly from the render kernels.

WHY THIS MODULE EXISTS. Every surface extraction in this repo up to now took the primitive's dipole
FACE -- the displaced plane clipped to the cell -- and called that the surface. Reading the kernels
(`rasterize.py::benchmark_kernel`, `raytrace.py`) shows that is not what is rendered. Neither kernel
ever produces a surface patch. Both compute a SEGMENT LENGTH and convert it to opacity:

    hit, t_near, t_far = ray_sphere_intersect(o, d, c_i, r_i)     # start: the sphere chord
    for j in adjacency(i):                                        # clip by radical planes
        t_face, dp = ray_pface_intersect_diff(...)
        t_far  = min(t_face, t_far)  if dp >= 0 else t_far
        t_near = max(t_face, t_near) if dp <  0 else t_near
    t_surf, dp = ray_plane_intersect(o, d, c_i, n_i, h(x_bar))    # clip by the dipole plane
    t_far  = min(t_surf, t_far)  if dp >= 0 else t_far
    t_near = max(t_surf, t_near) if dp <  0 else t_near
    alpha  = 1 - exp(-sigma_i * (t_far - t_near))

Each of those clips is a half-space test, so the rendered support of primitive i is the convex solid

    R_i = Ball(p_i, r_i)  n  (n_j radical half-spaces)  n  {x : n_i.(x - p_i) <= d(x)}

and `t_far - t_near` is the length of the ray's chord through R_i. What a camera sees is therefore
the boundary of the UNION of these solids -- and adjacent solids share their radical-plane faces, so
that union is connected wherever cells are adjacent. A per-cell dipole facet cannot be connected to
anything, which is why extractions built on it look like floating slivers no matter how they are
bounded or which adjacency graph they use.

Note on the dipole half-space direction, which the kernel expresses indirectly. With
`dp = dot(n_i, dir)`: when dp >= 0 the segment's FAR end is cut at t_surf, so the kept part is
before the ray crosses the plane along +n; when dp < 0 the NEAR end is cut, so the kept part is
after crossing against -n. Both cases keep `n_i.(x - p_i) <= h`. This is asserted in the tests.
"""
import numpy as np

from foam_exact_surface import SV_TEMP, soft_voronoi_height


def cell_halfspaces(i, centers, radii, neigh_i):
    """Radical-plane half-spaces of cell i as (A, b) with `A x <= b`, one row per neighbour.

    Uses the same radical plane the kernel does: `2(c_j - c_i).x <= (|c_j|^2 - r_j^2) - (|c_i|^2 - r_i^2)`.
    """
    ci, ri = centers[i], radii[i]
    rows, rhs = [], []
    for j in neigh_i:
        j = int(j)
        if j == i:
            continue
        cj, rj = centers[j], radii[j]
        rows.append(2.0 * (cj - ci))
        rhs.append((cj @ cj - rj * rj) - (ci @ ci - ri * ri))
    if not rows:
        return np.zeros((0, 3)), np.zeros(0)
    return np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)


def dipole_height_at(x, centre, sites_w, heights_w, radius, temp=SV_TEMP):
    """Displacement d(x) of Eq. 3 evaluated at world points x (the kernel evaluates it at the
    BASE-plane intersection, so callers must pass that point, not the displaced one)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return soft_voronoi_height(x, sites_w, heights_w, radius, temp)


def ray_segment(i, ray_o, ray_d, centers, radii, normals, neigh_i,
                sites_w=None, heights_w=None, use_dipole=True):
    """Reproduce the kernel's clipping. Returns (hit, t_near, t_far); length = t_far - t_near."""
    ci, ri, n = centers[i], radii[i], normals[i]
    n = n / max(np.linalg.norm(n), 1e-12)

    oc = ray_o - ci
    qb = 2.0 * (oc @ ray_d)
    qc = oc @ oc - ri * ri
    disc = qb * qb - 4.0 * qc
    if disc < 0:
        return False, 0.0, 0.0
    s = np.sqrt(disc)
    t_near, t_far = (-qb - s) / 2.0, (-qb + s) / 2.0
    if t_near < 0.0 and t_far < 0.0:
        return False, 0.0, 0.0
    t_near = max(t_near, 0.0)

    A, b = cell_halfspaces(i, centers, radii, neigh_i)
    for k in range(len(b)):
        dp = ray_d @ A[k]
        if abs(dp) < 1e-15:
            if (ray_o @ A[k]) > b[k]:
                return False, 0.0, 0.0
            continue
        t_face = (b[k] - ray_o @ A[k]) / dp
        if dp >= 0.0:
            t_far = min(t_face, t_far)
        else:
            t_near = max(t_face, t_near)
        if t_near > t_far:
            return False, 0.0, 0.0

    if use_dipole:
        dp = n @ ray_d
        if abs(dp) < 1e-15:
            if (ray_o - ci) @ n > 0.0:
                return False, 0.0, 0.0
        else:
            # base-plane intersection first (h = 0), exactly as plane_intersection_fwd_local does
            t0 = ((ci - ray_o) @ n) / dp
            t_query = t_near if dp >= 0.0 else max(t_near, t0)
            h = 0.0
            if sites_w is not None:
                h = float(dipole_height_at(ray_o + t_query * ray_d, ci,
                                           sites_w, heights_w, ri)[0])
            t_surf = ((ci - ray_o) @ n + h) / dp
            if dp >= 0.0:
                t_far = min(t_surf, t_far)
            else:
                t_near = max(t_surf, t_near)
    if t_near > t_far:
        return False, 0.0, 0.0
    return True, float(t_near), float(t_far)


def contains(i, x, centers, radii, normals, neigh_i, height=0.0):
    """Membership in R_i, independent of any ray. Used to cross-check `ray_segment`."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ci, ri, n = centers[i], radii[i], normals[i]
    n = n / max(np.linalg.norm(n), 1e-12)
    ok = np.linalg.norm(x - ci, axis=1) <= ri + 1e-12
    A, b = cell_halfspaces(i, centers, radii, neigh_i)
    if len(b):
        ok &= ((x @ A.T) <= b[None, :] + 1e-12).all(axis=1)
    ok &= ((x - ci) @ n) <= height + 1e-12
    return ok
