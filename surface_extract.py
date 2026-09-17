"""Extract the surface each method ACTUALLY produces, for the volumetric surface metric.

WHY THIS EXISTS. `ablation_surface` / `mesh_surface` score a RELABELLING of the ground-truth vertex
cloud: the predicted region of class c is `gt_points[pred == c]`, so both sides of the comparison
live on GT geometry and the representation enters only through the point->primitive assignment.
Nothing about the Gaussians' or the foam's own geometry is ever measured. This module produces the
other thing: the surface each representation would actually hand you, so the metric becomes a
statement about geometry rather than about labelling.

THE COMMON CONSTRUCTION, so that no method is advantaged by its extractor. Every method is
voxelised on the SAME grid (default 2 cm, matching the metric's tau) over the GT mesh's bounding
box, and every method defines occupancy by the same rule -- its own density field crossing an
isovalue -- expressed as the alpha a ray accumulates crossing one voxel:

    alpha = 1 - exp(-sigma * h)   >=   iso          h = voxel size

The predicted SURFACE is then the MARCHING-CUBES isosurface of that field at the same level, not
the boundary shell of the occupied voxels. Both would give a surface, but a shell quantises every
predicted point to a voxel centre, which puts a systematic ~h/2 offset into every distance and
rewards whichever method happens to sit near cell centres; marching cubes interpolates the crossing
and places vertices sub-voxel. It also makes the two sides of the metric the SAME construction --
a labelled triangle mesh sampled uniformly by area -- so the predicted and reference geometry are
treated identically rather than one being a point cloud and the other a mesh.

Classes are carried through the mesh the same way the GT side does it: each isosurface vertex takes
the winning class of the voxel it falls in, each face takes its vertices' majority label
(`mesh_surface.face_labels`), and the predicted region of class c is an area-uniform sample of the
faces labelled c (`mesh_surface._sample_mesh_uniform`). Sampling by area, not per triangle, keeps a
finely tessellated region from dominating.

WHAT DIFFERS PER REPRESENTATION, which is the honest part:
  foam       occupancy and class are EXACT. Every voxel centre lies in exactly one power cell
             (argmin ||x-c_i||^2 - r_i^2, the traversal kernel's own formula), so the class is that
             cell's class and sigma is that cell's density -- no interpolation, no isovalue
             ambiguity in the class assignment. This is the sense in which the foam "already knows
             where its surface is".
  gaussians  density at a voxel is the sum of alpha_i * exp(-0.5 * mahalanobis^2) over Gaussians
             covering it, and the class is that of the single largest contributor at that voxel
             (winner-take-all). Summing per class instead would need a (voxels x classes) buffer;
             the winner is what a renderer would show and costs two arrays.

THE ISOVALUE IS A CHOICE AND IS SWEPT. There is no canonical isovalue that means the same thing for
a Gaussian mixture and a piecewise-constant foam density, so a single value would embed a decision
favouring one representation. `--iso` accepts several and every score is reported per isovalue; a
conclusion that survives the sweep is a conclusion about the representations, and one that does not
is a conclusion about the threshold.
"""
import numpy as np
import torch


def grid_from_bbox(lo, hi, h, margin=0.05):
    lo = np.asarray(lo, dtype=np.float64) - margin
    hi = np.asarray(hi, dtype=np.float64) + margin
    dims = np.maximum(np.ceil((hi - lo) / h).astype(np.int64), 1)
    return lo, dims


def isosurface_samples(dens, cls_vol, lo, h, iso, samples_per_m2=2500, min_per_class=500,
                       seed=0):
    """Marching-cubes isosurface of a density volume -> area-uniform samples with class labels.

    `dens` and `cls_vol` are (nx, ny, nz). Returns (points (N,3), class (N,)) plus the per-class
    surface area, so a caller can report how much surface each method actually produced.

    The vertex class is read from the voxel the vertex falls in, the face label is the majority of
    its three vertices (identical to mesh_surface.face_labels on the GT side), and each class is
    sampled proportionally to its own area.
    """
    import numpy as np
    from skimage import measure
    from mesh_surface import face_labels, _sample_mesh_uniform

    if float(dens.max()) < iso:                 # nothing crosses the level
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), {}
    verts, faces, _, _ = measure.marching_cubes(dens, level=iso)
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), {}

    # marching_cubes returns vertices in VOXEL index space; the volume samples cell CENTRES, so a
    # vertex at index v sits at lo + (v + 0.5) * h in world space.
    vw = lo + (verts + 0.5) * h
    # A VERTEX MUST TAKE ITS CLASS FROM THE OCCUPIED SIDE. Every isosurface vertex lies BETWEEN an
    # occupied and an empty voxel, so rounding to the nearest index picks the empty one about half
    # the time -- and an empty voxel carries class 0, which face_labels then treats as unlabelled
    # and drops. On a hand-checked single-cell cube this discarded the entire surface (area 0
    # against an exact 1.5 m^2). So among the eight voxels bracketing the vertex, take the one with
    # the highest density, i.e. the material whose boundary this is.
    shape = np.array(dens.shape, dtype=np.int64)
    lo_i = np.clip(np.floor(verts).astype(np.int64), 0, shape - 1)
    hi_i = np.clip(np.ceil(verts).astype(np.int64), 0, shape - 1)
    best_d = np.full(len(verts), -np.inf)
    vcls = np.zeros(len(verts), dtype=np.int64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = np.where(dx, hi_i[:, 0], lo_i[:, 0])
                iy = np.where(dy, hi_i[:, 1], lo_i[:, 1])
                iz = np.where(dz, hi_i[:, 2], lo_i[:, 2])
                dv = dens[ix, iy, iz]
                take = dv > best_d
                best_d[take] = dv[take]
                vcls[take] = cls_vol[ix[take], iy[take], iz[take]]
    fl = face_labels(faces, vcls)

    pts, cls, areas = [], [], {}
    for c in np.unique(fl):
        if c <= 0:
            continue
        sel = fl == c
        F = faces[sel]
        tri = vw[F]
        area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                   tri[:, 2] - tri[:, 0]), axis=1).sum())
        if area <= 0:
            continue
        nsamp = max(min_per_class, int(area * samples_per_m2))
        p = _sample_mesh_uniform(vw, F, nsamp, seed)
        pts.append(p)
        cls.append(np.full(len(p), c, dtype=np.int64))
        areas[int(c)] = area
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), {}
    return np.concatenate(pts), np.concatenate(cls), areas


def foam_volume(centers, radii, density, prim_class, lo, dims, h, chunk=2_000_000,
                max_dist=None):
    """Exact power-cell voxelisation -> (density volume, class volume).

    Every voxel centre lies in exactly one power cell, so both volumes are exact: no interpolation
    and no isovalue enters the CLASS assignment, only the occupancy decision made later.

    `max_dist` (per-cell, in metres) gives the voxel field the SAME compact support the polygon
    extractor gets from its patch cap: a voxel counts as occupied only if it lies within that
    distance of its owning cell's centre. Without it a power diagram tessellates ALL of space, so
    every voxel in the room is owned by some cell and inherits its opacity -- measured occupancy
    mean 0.636 on scene0062_00, i.e. a solid blob whose isosurface is the room's outside (102-191
    m^2 against an 18 m^2 GT). Any comparison against a Gaussian field, whose support is compact by
    construction, must apply this or it is comparing space-filling to compact and calling the
    difference an extractor effect.
    """
    from point_cloud_query import assign_points_to_power_cells
    nx, ny, nz = [int(d) for d in dims]
    dens = np.zeros(nx * ny * nz, dtype=np.float32)
    cvol = np.zeros(nx * ny * nz, dtype=np.int64)
    total = nx * ny * nz
    for s0 in range(0, total, chunk):
        e = min(s0 + chunk, total)
        f = np.arange(s0, e, dtype=np.int64)
        ix, rem = np.divmod(f, ny * nz)
        iy, iz = np.divmod(rem, nz)
        pts = lo + (np.stack([ix, iy, iz], -1) + 0.5) * h
        owner = np.asarray(assign_points_to_power_cells(pts, centers, radii, valid=None, k=8))
        ok = owner >= 0
        if max_dist is not None and ok.any():
            o = owner[ok]
            far = np.linalg.norm(pts[ok] - centers[o], axis=1) > max_dist[o]
            keep = np.nonzero(ok)[0]
            ok[keep[far]] = False
        if not ok.any():
            continue
        dens[f[ok]] = density[owner[ok]]
        cvol[f[ok]] = prim_class[owner[ok]]
    return dens.reshape(nx, ny, nz), cvol.reshape(nx, ny, nz)


def gaussian_volume(means, scales, quats, opacities, prim_class, lo, dims, h,
                    device="cuda", k_sigma=2.0, max_pairs=40_000_000):
    """Splat Gaussians into the grid -> (density volume, winning-class volume).

    VECTORISED OVER GAUSSIANS, not looped. A Python loop over primitives is unusable here: the
    frozen arms have ~10^5 Gaussians and the unfrozen ones ~2.4x10^6, i.e. hours per scene. Instead
    Gaussians are grouped by the SIZE of their voxel footprint, and each group is splatted in one
    batched scatter -- every Gaussian in a group covers the same number of voxels, so their local
    offset grids are identical and can be broadcast.

    The class of a voxel is that of its single largest contributor (winner-take-all), resolved with
    a scatter-reduce on amax followed by an equality match, which is order-independent -- unlike
    accumulating "last writer wins", which would depend on primitive ordering.
    """
    nx, ny, nz = [int(d) for d in dims]
    dens = torch.zeros(nx * ny * nz, device=device)
    best = torch.zeros(nx * ny * nz, device=device)

    m = torch.as_tensor(means, device=device).float()
    sc = torch.as_tensor(scales, device=device).float()
    q = torch.as_tensor(quats, device=device).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    al = torch.as_tensor(opacities, device=device).float().reshape(-1)
    pc = torch.as_tensor(prim_class, device=device).long()
    lo_t = torch.as_tensor(lo, device=device).float()

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)
    ext = k_sigma * (sc.unsqueeze(-2) * R.abs()).sum(-1)

    c0 = torch.floor((m - ext - lo_t) / h).long()
    c1 = torch.ceil((m + ext - lo_t) / h).long()
    hi = torch.tensor([nx, ny, nz], device=device)
    c0 = c0.clamp(min=torch.zeros(3, device=device, dtype=torch.long), max=hi)
    c1 = c1.clamp(min=torch.zeros(3, device=device, dtype=torch.long), max=hi)
    span = (c1 - c0).clamp_min(0)
    live = (span > 0).all(-1)

    key = span[:, 0] * 1_000_000 + span[:, 1] * 1000 + span[:, 2]
    for k in torch.unique(key[live]):
        sel = torch.nonzero(live & (key == k), as_tuple=True)[0]
        sx, sy, sz = [int(v) for v in span[sel[0]]]
        npix = sx * sy * sz
        if npix == 0:
            continue
        step = max(1, int(max_pairs // max(npix, 1)))
        og = torch.stack(torch.meshgrid(torch.arange(sx, device=device),
                                        torch.arange(sy, device=device),
                                        torch.arange(sz, device=device),
                                        indexing="ij"), -1).reshape(-1, 3)
        for b0 in range(0, sel.numel(), step):
            g = sel[b0:b0 + step]
            idx = c0[g].unsqueeze(1) + og.unsqueeze(0)                 # (G, npix, 3)
            p = lo_t + (idx.float() + 0.5) * h
            d = torch.einsum("gpc,gcd->gpd", p - m[g].unsqueeze(1), R[g])
            md = ((d / sc[g].clamp_min(1e-8).unsqueeze(1)) ** 2).sum(-1)
            val = (al[g].unsqueeze(1) * torch.exp(-0.5 * md)).reshape(-1)
            lin = ((idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]).reshape(-1)
            dens.index_add_(0, lin, val)
            best.scatter_reduce_(0, lin, val, reduce="amax")
            del idx, p, d, md, val, lin
    # resolve the winning class: re-walk and keep the primitive whose contribution equals the max
    cls = torch.zeros(nx * ny * nz, dtype=torch.long, device=device)
    for k in torch.unique(key[live]):
        sel = torch.nonzero(live & (key == k), as_tuple=True)[0]
        sx, sy, sz = [int(v) for v in span[sel[0]]]
        npix = sx * sy * sz
        if npix == 0:
            continue
        step = max(1, int(max_pairs // max(npix, 1)))
        og = torch.stack(torch.meshgrid(torch.arange(sx, device=device),
                                        torch.arange(sy, device=device),
                                        torch.arange(sz, device=device),
                                        indexing="ij"), -1).reshape(-1, 3)
        for b0 in range(0, sel.numel(), step):
            g = sel[b0:b0 + step]
            idx = c0[g].unsqueeze(1) + og.unsqueeze(0)
            p = lo_t + (idx.float() + 0.5) * h
            d = torch.einsum("gpc,gcd->gpd", p - m[g].unsqueeze(1), R[g])
            md = ((d / sc[g].clamp_min(1e-8).unsqueeze(1)) ** 2).sum(-1)
            val = (al[g].unsqueeze(1) * torch.exp(-0.5 * md)).reshape(-1)
            lin = ((idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]).reshape(-1)
            wins = val >= best[lin] - 1e-12
            if bool(wins.any()):
                cls[lin[wins]] = pc[g].unsqueeze(1).expand(-1, npix).reshape(-1)[wins]
            del idx, p, d, md, val, lin, wins
    return (dens.reshape(nx, ny, nz).cpu().numpy(),
            cls.reshape(nx, ny, nz).cpu().numpy())
