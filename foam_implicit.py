"""The foam's matter as a single implicit function, so its surface is one connected manifold.

THE CONSTRUCTION. Each cell defines a solid by an INTERSECTION -- inside its bounding ball and below
its displaced dipole interface:

    s_i(x) = min( r_i - ||x - p_i|| ,  h_i(x) - n_i.(x - p_i) )

and the matter is the UNION of those solids, so with intersection = min and union = max:

    f(x) = max_i s_i(x),      surface = { x : f(x) = 0 }

`h_i` is the soft-Voronoi displacement of Eq. 3, evaluated -- as the render kernel evaluates it -- at
the projection of x onto the cell's BASE dipole plane. The interface is therefore a curved manifold,
not a plane: the detail sites push it up and down locally, which is the whole point of decoupling
geometry from appearance. Only in the zero-displacement case does a lone cell reduce to a half-ball.

WHY THIS SUCCEEDS WHERE PER-CELL FACE EXTRACTION FAILED. Building each cell's polytope separately
and gluing the faces gives, in order of severity:
  * sphere caps as first-class geometry (85% of all faces) -- the extracted object was a pile of
    balls, because a cell is only cut where a neighbour's ball actually overlaps it;
  * steps: neighbouring cells' dipole heights differ, so their faces meet at the shared radical
    plane at different heights, leaving a band open (measured: 16% of ray coverage lost);
  * per-cell normal noise, since nothing in training couples a cell's normal to its neighbours'.
Under `max`, two overlapping cells' displaced interfaces simply INTERPENETRATE and the zero-set
passes continuously from one to the other along their intersection curve. No caps, no steps, no
stitching heuristic, no threshold.

ON THE RADICAL PLANES. The render kernels do clip each cell by its neighbours' radical planes, but
only to partition the ray so density is not double-counted while integrating -- it is bookkeeping for
the integral, not a statement about where matter is. `clip_cells=True` adds those half-spaces to the
min so both variants can be compared against ground truth rather than argued about.
"""
import numpy as np

from foam_exact_surface import SV_TEMP


def implicit_volume(centers, radii, normals, sites_w, heights_w, prim_class, lo, dims, h,
                    live=None, clip_cells=False, adjacency=None, offsets=None, temp=SV_TEMP,
                    progress=None, beta=None):
    """Evaluate f = max_i s_i on a grid. Returns (f, winner) with winner = argmax cell (-1 empty).

    Only voxels inside a cell's own ball can be affected by it, so each cell touches a small AABB
    and the whole thing is a scatter-max rather than a dense evaluation.
    """
    nx, ny, nz = (int(d) for d in dims)
    f = np.full((nx, ny, nz), -1e9, dtype=np.float32)
    win = np.full((nx, ny, nz), -1, dtype=np.int32)
    live = range(len(centers)) if live is None else live
    deg = None if offsets is None else np.diff(offsets)

    for count, i in enumerate(live):
        i = int(i)
        if progress and count % progress == 0:
            print(f"    {count:,} cells", flush=True)
        ci, ri = centers[i], radii[i]
        ni = normals[i] / max(np.linalg.norm(normals[i]), 1e-12)

        i0 = np.maximum(np.floor((ci - ri - lo) / h).astype(np.int64), 0)
        i1 = np.minimum(np.ceil((ci + ri - lo) / h).astype(np.int64) + 1,
                        np.array([nx, ny, nz]))
        if np.any(i1 <= i0):
            continue
        gx = (np.arange(i0[0], i1[0]) + 0.5) * h + lo[0]
        gy = (np.arange(i0[1], i1[1]) + 0.5) * h + lo[1]
        gz = (np.arange(i0[2], i1[2]) + 0.5) * h + lo[2]
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        P = np.stack([X, Y, Z], -1).reshape(-1, 3)

        u = P - ci
        d_ball = ri - np.linalg.norm(u, axis=1)
        t = u @ ni                                   # signed height above the base plane
        base = P - t[:, None] * ni[None, :]          # projection onto the base plane
        w = np.exp(-temp * (((base[:, None, :] - sites_w[i][None, :, :]) ** 2).sum(-1))
                   / (ri * ri))
        hh = (w * heights_w[i][None, :]).sum(-1) / np.maximum(w.sum(-1), 1e-20)
        s = np.minimum(d_ball, hh - t)

        if clip_cells and deg is not None:
            for j in adjacency[offsets[i]:offsets[i] + deg[i]]:
                j = int(j)
                if j == i:
                    continue
                A = 2.0 * (centers[j] - ci)
                b = (centers[j] @ centers[j] - radii[j] ** 2) - (ci @ ci - ri * ri)
                nA = max(np.linalg.norm(A), 1e-12)
                s = np.minimum(s, (b - P @ A) / nA)

        sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
        cur = f[sl].reshape(-1)
        better = s > cur
        if beta is not None:
            # SMOOTH UNION (metaball blend). A hard `max` keeps every cell's spherical cap as its
            # own bulge, so the envelope of a packing of half-balls is bumpy at radius scale no
            # matter how the dipole planes are oriented -- substituting ground-truth normals changed
            # nothing. log-sum-exp fuses overlapping caps into a single surface instead:
            #     f = (1/beta) log sum_i exp(beta * s_i)
            # accumulated pairwise so it stays a running blend, and evaluated in the shifted form
            # so the exponential never overflows. beta -> inf recovers the hard union.
            m_ = np.maximum(cur, s)
            f[sl] = (m_ + np.log1p(np.exp(-np.abs(cur - s) * beta)) / beta
                     ).astype(np.float32).reshape(f[sl].shape)
        else:
            if better.any():
                cur[better] = s[better].astype(np.float32)
                f[sl] = cur.reshape(f[sl].shape)
        if better.any():
            wv = win[sl].reshape(-1)
            wv[better] = prim_class[i] if prim_class is not None else i
            win[sl] = wv.reshape(win[sl].shape)
    return f, win


def implicit_mesh(f, win, lo, h, level=0.0):
    """Marching cubes on f at `level`. Returns (verts, faces, face_class)."""
    from skimage import measure
    from mesh_surface import face_labels

    if float(f.max()) < level or float(f.min()) > level:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    verts, faces, _, _ = measure.marching_cubes(f, level=level)
    vw = lo + (verts + 0.5) * h
    shape = np.array(f.shape, dtype=np.int64)
    # a surface vertex sits between an inside and an outside voxel; take the class of the INSIDE
    # one, i.e. of the bracketing voxel with the larger f, exactly as the voxel extractor does
    lo_i = np.clip(np.floor(verts).astype(np.int64), 0, shape - 1)
    hi_i = np.clip(np.ceil(verts).astype(np.int64), 0, shape - 1)
    best = np.full(len(verts), -np.inf)
    vcls = np.zeros(len(verts), dtype=np.int64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = np.where(dx, hi_i[:, 0], lo_i[:, 0])
                iy = np.where(dy, hi_i[:, 1], lo_i[:, 1])
                iz = np.where(dz, hi_i[:, 2], lo_i[:, 2])
                fv = f[ix, iy, iz]
                take = fv > best
                best[take] = fv[take]
                vcls[take] = win[ix[take], iy[take], iz[take]]
    return vw, faces, face_labels(faces, vcls)
