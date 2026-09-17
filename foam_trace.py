"""Trace the foam's surface as an ATLAS OF CHARTS, instead of sampling a voxel grid.

THE OBSERVATION THAT MAKES THIS WORK. Inside one cell the surface is a HEIGHT FIELD OVER A FLAT
DISC. In the cell's own frame (tangent t, bitangent b, normal n) a surface point is

    x(u, v) = p + u t + v b + h(u, v) n,        u^2 + v^2 <= r^2

so each cell is an honest 2-D coordinate patch and the foam is an atlas of them. Projecting onto the
surface is therefore CLOSED FORM -- choose (u, v), evaluate h, you are exactly on it. A general
implicit surface needs Newton iterations for that; here it is one evaluation, which is what makes
tracing cheaper and sharper than marching a grid.

h is the soft Voronoi of Eq. 3. The detail sites are stored in units of radius in the (t, b) frame,
and the kernel evaluates the blend at the BASE-plane intersection, so all the distances live in the
plane and reduce to

    h(u, v) = sum_k w_k d_k / sum_k w_k,    w_k = exp(-tau ((u - u_k)^2 + (v - v_k)^2) / r^2)

which also gives the derivatives in closed form, hence exact analytic normals:

    x_u = t + h_u n,  x_v = b + h_v n   =>   N = x_u x x_v = n - h_u t - h_v b

WHY TRACE RATHER THAN VOXELISE. The grid version cost 92 M voxels for a 3 m room, quantised every
detail-site bump onto a 1 cm lattice, and needed welding afterwards to become a mesh. Tracing is
resolution-free, evaluates the learned geometry exactly, carries analytic normals, and produces
connectivity as it goes -- inside a chart the sample lattice IS the triangulation.

WHAT IT DOES NOT DO. It follows the same geometry, so it does not smooth anything: the exposed
surface is largely spherical cap and is bumpy at cell radius (~1.3 cm measured), and substituting
ground-truth normals changed no metric. This produces a sharper, lighter mesh of the same surface.
"""
import numpy as np

from foam_exact_surface import SV_TEMP


def chart_frames(quats):
    """(n, t, b) per cell, matching scene.get_normals / get_tangents, with b = n x t."""
    import torch
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    t = torch.stack([2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)], -1)
    n = (n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)).numpy().astype(np.float64)
    t = (t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)).numpy().astype(np.float64)
    t = t - (t * n).sum(-1, keepdims=True) * n            # re-orthogonalise
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(n, t)
    return n, t, b


def height_uv(u, v, sites_uv, heights, radius, temp=SV_TEMP, grad=False):
    """h(u, v) and optionally (dh/du, dh/dv). `sites_uv` are the detail sites in the same (u, v)
    coordinates (already multiplied by the radius)."""
    du = u[:, None] - sites_uv[None, :, 0]
    dv = v[:, None] - sites_uv[None, :, 1]
    w = np.exp(-temp * (du * du + dv * dv) / (radius * radius))
    W = np.maximum(w.sum(-1), 1e-20)
    h = (w * heights[None, :]).sum(-1) / W
    if not grad:
        return h
    k = -2.0 * temp / (radius * radius)
    dw_du = w * (k * du)
    dw_dv = w * (k * dv)
    hu = (dw_du * (heights[None, :] - h[:, None])).sum(-1) / W
    hv = (dw_dv * (heights[None, :] - h[:, None])).sum(-1) / W
    return h, hu, hv


def chart_lattice(radius, delta):
    """Regular (u, v) lattice covering the chart disc, plus the (nu, nv) shape for connectivity."""
    m = int(np.ceil(radius / delta))
    g = (np.arange(-m, m + 1)) * delta
    U, V = np.meshgrid(g, g, indexing="ij")
    return U, V, len(g)


def sample_chart(i, centers, radii, N, T, B, sites_uv, heights_w, delta, temp=SV_TEMP):
    """Points, analytic normals and the lattice mask for one chart. Returns None if degenerate."""
    r = radii[i]
    U, V, k = chart_lattice(r, delta)
    u = U.reshape(-1); v = V.reshape(-1)
    inside = (u * u + v * v) <= r * r
    if inside.sum() < 3:
        return None
    h, hu, hv = height_uv(u, v, sites_uv[i], heights_w[i], r, temp, grad=True)
    P = (centers[i][None, :] + u[:, None] * T[i][None, :] + v[:, None] * B[i][None, :]
         + h[:, None] * N[i][None, :])
    # N_surf = n - h_u t - h_v b   (exact, from the chart parameterisation)
    Ns = (N[i][None, :] - hu[:, None] * T[i][None, :] - hv[:, None] * B[i][None, :])
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
    # the height field can leave the ball; the cell is bounded, so drop those
    inside &= np.linalg.norm(P - centers[i][None, :], axis=1) <= r
    return P, Ns, inside, k


def exposed(P, i, neigh, centers, radii, N, T, B, sites_uv, heights_w, temp=SV_TEMP, tol=0.0):
    """True where a point of chart i is on the UNION's boundary, i.e. buried in no other cell.

    A point x lies inside cell j's solid when s_j(x) > 0 with
        s_j(x) = min(r_j - ||x - p_j||, h_j(proj_j x) - n_j.(x - p_j))
    so the chart's surface is exposed exactly where every neighbour's s_j is <= 0. This is the
    hand-off test: where it flips, the surface passes from this chart to that neighbour's.
    """
    keep = np.ones(len(P), dtype=bool)
    for j in neigh:
        j = int(j)
        if j == i:
            continue
        d = P - centers[j][None, :]
        rad = radii[j] - np.linalg.norm(d, axis=1)
        m = rad > tol                                  # only points inside j's ball can be buried
        if not m.any():
            continue
        t_ = d[m] @ N[j]
        uu = d[m] @ T[j]; vv = d[m] @ B[j]
        hj = height_uv(uu, vv, sites_uv[j], heights_w[j], radii[j], temp)
        sj = np.minimum(rad[m], hj - t_)
        idx = np.nonzero(m)[0]
        keep[idx[sj > tol]] = False
    return keep


def trace_surface(centers, radii, quats, sites_uv, heights_w, prim_class, adjacency, offsets,
                  live, delta_frac=0.25, temp=SV_TEMP, weld=True, progress=None):
    """Walk every chart, keep the exposed part, triangulate the lattice, weld across charts.

    `delta_frac` sets the step as a fraction of each cell's radius, so sampling density follows the
    representation rather than a global grid.
    """
    Nn, Tt, Bb = chart_frames(quats)
    VP, VN, FF, FC = [], [], [], []
    off = 0
    n_charts = 0
    for count, i in enumerate(live):
        i = int(i)
        if progress and count % progress == 0:
            print(f"    {count:,}/{len(live):,} charts", flush=True)
        out = sample_chart(i, centers, radii, Nn, Tt, Bb, sites_uv, heights_w,
                           delta_frac * radii[i], temp)
        if out is None:
            continue
        P, Ns, inside, k = out
        nb = adjacency[offsets[i]:offsets[i + 1]]
        inside[inside] &= exposed(P[inside], i, nb, centers, radii, Nn, Tt, Bb,
                                  sites_uv, heights_w, temp)
        if inside.sum() < 3:
            continue
        ok = inside.reshape(k, k)
        # EMIT EACH TRIANGLE ON ITS OWN 3 CORNERS, not on all 4 corners of a quad. The exposure test
        # leaves each chart a thin, fragmented sliver -- most of a cell's interface is legitimately
        # buried in its neighbours -- so when the surviving region is only a few samples wide almost
        # every sample is rim and a 4-corner rule emits almost nothing: measured 195,089 triangles
        # from 1,081,170 vertices (0.18 tri/vertex against a healthy ~2) and 3.3 m^2 against an
        # 18 m^2 GT. Splitting the rule per triangle recovers the half-quads along every sliver edge.
        a = np.arange(k * k).reshape(k, k)
        i00 = a[:-1, :-1]; i10 = a[1:, :-1]; i01 = a[:-1, 1:]; i11 = a[1:, 1:]
        o00 = ok[:-1, :-1]; o10 = ok[1:, :-1]; o01 = ok[:-1, 1:]; o11 = ok[1:, 1:]
        tA = o00 & o10 & o11
        tB = o00 & o11 & o01
        # the other diagonal, for quads where exactly the opposite pair of corners survives
        tC = o00 & o10 & o01
        tD = o10 & o11 & o01
        parts = []
        if tA.any():
            parts.append(np.stack([i00[tA], i10[tA], i11[tA]], 1))
        if tB.any():
            parts.append(np.stack([i00[tB], i11[tB], i01[tB]], 1))
        use_c = tC & ~tA & ~tB
        if use_c.any():
            parts.append(np.stack([i00[use_c], i10[use_c], i01[use_c]], 1))
        use_d = tD & ~tA & ~tB
        if use_d.any():
            parts.append(np.stack([i10[use_d], i11[use_d], i01[use_d]], 1))
        if not parts:
            continue
        f = np.concatenate(parts, 0)
        VP.append(P); VN.append(Ns)
        FF.append(f + off); FC.append(np.full(len(f), int(prim_class[i]), dtype=np.int64))
        off += len(P)
        n_charts += 1

    if not FF:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=np.int64), {"n_charts": 0})
    V = np.concatenate(VP); Nv = np.concatenate(VN)
    F = np.concatenate(FF); C = np.concatenate(FC)
    stats = {"n_charts": n_charts, "n_verts_raw": int(len(V)), "n_faces": int(len(F))}
    if weld:
        # charts meet along the hand-off curve but their lattices do not coincide there; welding at
        # half the local step joins them into one surface instead of leaving a hairline seam
        tolw = 0.5 * delta_frac * float(np.median(radii[live]))
        key = np.round(V / tolw).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        V = V[first]; Nv = Nv[first]; F = inv[F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F, C = F[keep], C[keep]
        stats["weld_tol_m"] = float(tolw)
    tri = V[F]
    stats["n_verts"] = int(len(V))
    stats["n_faces_final"] = int(len(F))
    stats["area_m2"] = float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    return V, Nv, F, C, stats
