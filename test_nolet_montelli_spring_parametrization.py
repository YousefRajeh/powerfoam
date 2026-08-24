"""
test_nolet_montelli_spring_parametrization.py

Runnable reimplementation of Nolet & Montelli (2005), "Optimal parametrization of
tomographic models", GJI 161(2):365-372, on a 2-D synthetic straight-ray tomography
problem with deliberately non-uniform coverage INCLUDING a genuinely uncovered void.

What is implemented (locators refer to the paper):
  * resolving length ell(r) from the COLUMN SUM of the tomographic matrix via a
    logarithmic relation  (Section 4, p.369: "using a logarithmic relationship
    between the column sum of the tomographic matrix and the resolving length at
    the location of that parameter")
  * spring energy E = sum_i sum_{j in N_i} (L_ij - ell_ij)^2 / ell_ij^2   (eq. 3)
    with analytic gradient (eq. 8), natural neighbours = Delaunay edges (Section 2)
  * node budget from equilateral-simplex volume argument (eqs 4-5), 2-D analogue
  * conjugate-gradient inner loop / re-Delaunay outer loop (Section 3)
  * linear barycentric interpolation over simplices (eqs 14-16)

Measurements requested:
  (a) does node density track ray coverage?
  (b) does rank(A) INCREASE after reparametrization, or does only N fall?
  (c) reconstruction error in covered vs uncovered regions, reported SEPARATELY.

CPU-only, small, ~30 s. No GPU, no network.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# geometry: unit square, with a VOID box that no ray is allowed to enter
# ----------------------------------------------------------------------------
VOID = (0.05, 0.45, 0.60, 0.95)   # xmin, xmax, ymin, ymax  -> exactly zero coverage
# a "high resolution" target zone where we deliberately pack extra rays
DENSE = (0.55, 0.85, 0.10, 0.40)


def in_box(p, box):
    x0, x1, y0, y1 = box
    return (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)


def seg_hits_box(a, b, box, nsamp=200):
    t = np.linspace(0, 1, nsamp)[:, None]
    p = a[None, :] * (1 - t) + b[None, :] * t
    return bool(in_box(p, box).any())


def make_rays(n_bg=900, n_dense=700):
    """Random chords of the unit square, rejecting any that enters VOID.
    Plus a bundle concentrated on DENSE to create strongly non-uniform coverage."""
    rays = []
    while len(rays) < n_bg:
        a = RNG.random(2)
        b = RNG.random(2)
        # push endpoints to the boundary-ish by extending, then clip to square
        if np.linalg.norm(a - b) < 0.35:
            continue
        if seg_hits_box(a, b, VOID):
            continue
        rays.append((a, b))
    while len(rays) < n_bg + n_dense:
        # rays that pass through the DENSE box
        c = np.array([RNG.uniform(DENSE[0], DENSE[1]), RNG.uniform(DENSE[2], DENSE[3])])
        th = RNG.uniform(0, np.pi)
        d = np.array([np.cos(th), np.sin(th)])
        a = c - 0.30 * d
        b = c + 0.30 * d
        if np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(b > 1):
            continue
        if seg_hits_box(a, b, VOID):
            continue
        rays.append((a, b))
    return [(np.asarray(a), np.asarray(b)) for a, b in rays]


# ----------------------------------------------------------------------------
# forward operator on a node parametrization with barycentric interpolation
# (paper eqs 14-16).  Ray integral -> sum over quadrature points of h_k * ds
# ----------------------------------------------------------------------------
def build_A_nodes(nodes, tri, rays, nq=160):
    N = len(nodes)
    rows, cols, vals = [], [], []
    for r, (a, b) in enumerate(rays):
        L = np.linalg.norm(b - a)
        ds = L / nq
        t = (np.arange(nq) + 0.5) / nq
        pts = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
        simp = tri.find_simplex(pts)
        ok = simp >= 0
        if not ok.any():
            continue
        simp = simp[ok]
        pts_ok = pts[ok]
        # barycentric coordinates
        X = tri.transform[simp, :2]
        off = pts_ok - tri.transform[simp, 2]
        bary2 = np.einsum('ijk,ik->ij', X, off)
        bary = np.column_stack([bary2, 1 - bary2.sum(axis=1)])
        vi = tri.simplices[simp]
        acc = {}
        for k in range(3):
            for idx, w in zip(vi[:, k], bary[:, k]):
                acc[idx] = acc.get(idx, 0.0) + w * ds
        for idx, w in acc.items():
            rows.append(r); cols.append(idx); vals.append(w)
    return csr_matrix((vals, (rows, cols)), shape=(len(rays), N))


def build_A_pixels(nx, rays, nq=160):
    """Column sums of this fine pixel matrix give the coverage field (Section 4)."""
    N = nx * nx
    rows, cols, vals = [], [], []
    for r, (a, b) in enumerate(rays):
        L = np.linalg.norm(b - a)
        ds = L / nq
        t = (np.arange(nq) + 0.5) / nq
        pts = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
        ix = np.clip((pts[:, 0] * nx).astype(int), 0, nx - 1)
        iy = np.clip((pts[:, 1] * nx).astype(int), 0, nx - 1)
        flat = iy * nx + ix
        u, c = np.unique(flat, return_counts=True)
        for idx, cc in zip(u, c):
            rows.append(r); cols.append(idx); vals.append(cc * ds)
    return csr_matrix((vals, (rows, cols)), shape=(len(rays), N))


# ----------------------------------------------------------------------------
# resolving length field: LOGARITHMIC in the column sum (Section 4)
# ----------------------------------------------------------------------------
def resolving_length_field(colsum, nx, ell_min=0.022, ell_max=0.075):
    """ell = ell_min * (cmax/c)^p, i.e. log ell linear in log c, clipped to
    [ell_min, ell_max].  Zero-coverage cells get ell_max BY CLIPPING -- the
    unclipped value is +infinity.  This is exactly the paper's situation: Fig. 8
    caption states the target resolution field is capped ('0 km to 1500 km')."""
    c = colsum.reshape(nx, nx)
    cmax = c.max()
    with np.errstate(divide='ignore'):
        ell = ell_min * (cmax / np.maximum(c, 1e-12)) ** 0.35
    ell_raw = ell.copy()
    ell = np.clip(ell, ell_min, ell_max)
    return ell, ell_raw, (c == 0)


def sample_field(field, pts, nx):
    ix = np.clip((pts[:, 0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((pts[:, 1] * nx).astype(int), 0, nx - 1)
    return field[iy, ix]


# ----------------------------------------------------------------------------
# node budget, paper eqs (4)-(5), 2-D analogue:
#   equilateral triangle area = sqrt(3)/4 ell^2 ;  N_tri = V / mean(area)
#   for a 2-D triangulation N_nodes ~ N_tri / 2
# ----------------------------------------------------------------------------
def node_budget(ell_field):
    mean_area = (np.sqrt(3) / 4.0) * np.mean(ell_field ** 2)
    n_tri = 1.0 / mean_area
    return int(round(n_tri / 2.0))


# ----------------------------------------------------------------------------
# spring energy (eq 3) and gradient (eq 8) over Delaunay edges
# ----------------------------------------------------------------------------
def edges_of(tri):
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def energy_and_grad(xfree, fixed_pts, free_idx, all_n, edges, ell_nodes):
    P = np.empty((all_n, 2))
    P[free_idx] = xfree.reshape(-1, 2)
    mask = np.ones(all_n, bool); mask[free_idx] = False
    P[mask] = fixed_pts
    i, j = edges[:, 0], edges[:, 1]
    d = P[i] - P[j]
    L = np.linalg.norm(d, axis=1)
    L = np.maximum(L, 1e-12)
    ell = 0.5 * (ell_nodes[i] + ell_nodes[j])
    # eq (3): each unordered edge appears twice in the double sum -> factor 2
    E = 2.0 * np.sum((L - ell) ** 2 / ell ** 2)
    # eq (8)
    coef = (4.0 * (1.0 - ell / L) / ell ** 2)[:, None]
    G = np.zeros((all_n, 2))
    np.add.at(G, i, coef * d)
    np.add.at(G, j, -coef * d)
    return E, G[free_idx].ravel()


def optimize_nodes(P, boundary_mask, ell_field, nx, outer=10, inner=400, verbose=True):
    P = P.copy()
    free_idx = np.where(~boundary_mask)[0]
    fixed_pts = P[boundary_mask]
    hist = []
    for it in range(outer):
        tri = Delaunay(P)
        edges = edges_of(tri)
        ell_nodes = sample_field(ell_field, P, nx)   # recomputed each outer loop
        f = lambda x: energy_and_grad(x, fixed_pts, free_idx, len(P), edges, ell_nodes)
        E0 = f(P[free_idx].ravel())[0]
        res = minimize(f, P[free_idx].ravel(), jac=True, method='CG',
                       options={'maxiter': inner, 'gtol': 1e-8})
        P[free_idx] = np.clip(res.x.reshape(-1, 2), 1e-4, 1 - 1e-4)
        hist.append((E0, res.fun))
        if verbose:
            print(f"    outer {it:2d}: E {E0:10.1f} -> {res.fun:10.1f} "
                  f"(inner CG decrease {100*(1-res.fun/E0):5.1f}%)")
    return P, hist


# ----------------------------------------------------------------------------
def truth(pts):
    """Smooth background + one anomaly inside the covered region + one anomaly
    that sits entirely inside the VOID."""
    x, y = pts[:, 0], pts[:, 1]
    m = 0.3 * np.sin(3 * np.pi * x) * np.cos(2 * np.pi * y)
    m += 1.0 * np.exp(-((x - 0.70) ** 2 + (y - 0.25) ** 2) / (2 * 0.05 ** 2))
    m += 1.0 * np.exp(-((x - 0.25) ** 2 + (y - 0.78) ** 2) / (2 * 0.06 ** 2))
    return m


def hull_ring(n):
    t = np.linspace(0, 1, n, endpoint=False)
    pts = []
    for u in t:
        pts += [[u, 0.0], [1.0, u], [1.0 - u, 1.0], [0.0, 1.0 - u]]
    return np.array(pts)


def make_grid_nodes(n_target):
    k = int(round(np.sqrt(n_target)))
    g = (np.arange(k) + 0.5) / k
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel()])


def make_grid_nodes_exact(n_target):
    """Jittered-free uniform grid with node count as close to n_target as possible."""
    k = max(2, int(round(np.sqrt(n_target))))
    g = (np.arange(k) + 0.5) / k
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel()])


def main():
    nx = 100
    rays = make_rays(n_bg=260, n_dense=180)
    print(f"[setup] rays: {len(rays)}   fine grid: {nx}x{nx}")

    Apix = build_A_pixels(nx, rays)
    colsum = np.asarray(Apix.sum(axis=0)).ravel()
    ell_field, ell_raw, dead2d = resolving_length_field(colsum, nx)
    frac_dead_pix = dead2d.mean()
    print(f"[coverage] fine pixels with EXACTLY zero coverage: "
          f"{dead2d.sum()}/{nx*nx} = {100*frac_dead_pix:.2f}%")
    print(f"[ell] unclipped resolving length on zero-coverage pixels: "
          f"{'inf' if not np.isfinite(ell_raw[dead2d]).all() else ell_raw[dead2d].max():.4g}"
          f"   (clipped to ell_max={ell_field.max():.3f})")
    print(f"[ell] ell range on covered pixels: "
          f"{ell_field[~dead2d].min():.4f} .. {ell_field[~dead2d].max():.4f}")

    N_opt = node_budget(ell_field)
    print(f"[budget] eq(4)-(5) 2-D analogue -> N = {N_opt} nodes")

    # ---- starting configuration: paper Section 3 recommends a tessellation that
    # is already within sqrt(2) of the target spacing.  We seed by rejection
    # sampling with local spacing ell, plus a fixed convex-hull ring.
    ring = hull_ring(max(8, int(round(np.sqrt(N_opt)))))
    interior = []
    tries = 0
    while len(interior) < N_opt - len(ring) and tries < 400000:
        tries += 1
        p = RNG.random(2) * 0.98 + 0.01
        l = sample_field(ell_field, p[None, :], nx)[0]
        if interior:
            dmin = np.min(np.linalg.norm(np.array(interior) - p, axis=1))
            if dmin < 0.75 * l:
                continue
        interior.append(p)
    P0 = np.vstack([ring, np.array(interior)])
    bmask = np.zeros(len(P0), bool); bmask[:len(ring)] = True
    print(f"[start] seeded {len(P0)} nodes ({len(ring)} fixed on hull)")

    print("[spring] conjugate-gradient inner loop / re-Delaunay outer loop (Sec. 3):")
    P1, hist = optimize_nodes(P0, bmask, ell_field, nx)
    print(f"[spring] E: {hist[0][0]:.1f} (initial) -> {hist[-1][1]:.1f} (final) "
          f"= {100*hist[-1][1]/hist[0][0]:.1f}% of start")

    # xi = L_ij / ell_ij quality statistic (paper Section 4)
    for tag, P in (("start", P0), ("optimized", P1)):
        tri = Delaunay(P); E = edges_of(tri)
        L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        ell = 0.5 * (sample_field(ell_field, P[E[:, 0]], nx) +
                     sample_field(ell_field, P[E[:, 1]], nx))
        xi = L / ell
        print(f"[xi] {tag:9s} mean={xi.mean():.3f} sd={xi.std():.3f} "
              f"min={xi.min():.3f} max={xi.max():.3f}")

    # ---- (a) does node density track coverage? --------------------------------
    def density_stats(P, tag):
        covered = ~sample_field(dead2d, P, nx)
        n_void = int(sample_field(in_box_field(nx), P, nx).sum())
        in_dense = ((P[:, 0] > DENSE[0]) & (P[:, 0] < DENSE[1]) &
                    (P[:, 1] > DENSE[2]) & (P[:, 1] < DENSE[3])).sum()
        area_dense = (DENSE[1] - DENSE[0]) * (DENSE[3] - DENSE[2])
        area_void = (VOID[1] - VOID[0]) * (VOID[3] - VOID[2])
        print(f"[density] {tag:9s} N={len(P):4d} | dense-zone nodes={in_dense:4d} "
              f"(rel.density {in_dense/len(P)/area_dense:.2f}x) | "
              f"VOID nodes={n_void:4d} (rel.density {n_void/len(P)/area_void:.2f}x)")
        return in_dense, n_void

    density_stats(P0, "start")
    density_stats(P1, "optimized")

    # ---- (b) THE DECISIVE MEASUREMENT: rank(A) --------------------------------
    def rank_of(P):
        tri = Delaunay(P)
        A = build_A_nodes(P, tri, rays).toarray()
        tol = max(A.shape) * np.finfo(float).eps * np.linalg.norm(A, 2)
        rk = np.linalg.matrix_rank(A, tol=max(tol, 1e-10))
        cs = A.sum(axis=0)
        return A, tri, rk, int((np.abs(cs) <= 1e-12).sum())

    print("\n=== (b) RANK MEASUREMENT (decisive) ===")
    print(f"  M = {len(rays)} rays  ->  rank(A) can never exceed {len(rays)}")
    Apix_d = Apix.toarray()
    rk_pix = np.linalg.matrix_rank(Apix_d, tol=1e-10)
    print(f"  {'fine pixel grid':20s} N={nx*nx:5d}  rank(A)={rk_pix:5d}  "
          f"dead unknowns={int((colsum<=1e-12).sum()):5d} "
          f"({100*(colsum<=1e-12).mean():5.2f}%)  nullity={nx*nx-rk_pix}")
    grids = [("uniform grid (=N)", make_grid_nodes_exact(len(P1))),
             ("Nolet start", P0),
             ("Nolet OPTIMIZED", P1)]
    for tag, P in grids:
        A, tri, rk, ndead = rank_of(P)
        print(f"  {tag:20s} N={len(P):5d}  rank(A)={rk:5d}  "
              f"dead unknowns={ndead:5d} ({100*ndead/len(P):5.2f}%)  "
              f"nullity={len(P)-rk}")

    # ---- (c) reconstruction error, covered vs uncovered, SEPARATELY ----------
    print("\n=== (c) RECONSTRUCTION ERROR, covered vs VOID (reported separately) ===")
    # evaluation points on a fine grid
    g = (np.arange(nx) + 0.5) / nx
    GX, GY = np.meshgrid(g, g)
    EV = np.column_stack([GX.ravel(), GY.ravel()])
    m_true = truth(EV)
    void_mask = in_box(EV, VOID)
    # synthetic data from the TRUE continuous model along each ray
    d_obs = np.array([ray_integral_true(a, b) for a, b in rays])
    for tag, P in (("uniform grid", make_grid_nodes_exact(len(P1))),
                   ("Nolet start", P0),
                   ("Nolet OPTIM", P1)):
        tri = Delaunay(P)
        A = build_A_nodes(P, tri, rays)
        sol = lsqr(A, d_obs, damp=1e-3, iter_lim=800)[0]
        simp = tri.find_simplex(EV)
        rec = interp_nodes(tri, sol, EV, simp)
        err = rec - m_true
        rms_cov = np.sqrt(np.mean(err[~void_mask] ** 2))
        rms_void = np.sqrt(np.mean(err[void_mask] ** 2))
        rms_truth_void = np.sqrt(np.mean(m_true[void_mask] ** 2))
        print(f"  {tag:14s} N={len(P):4d}  RMS covered={rms_cov:.4f}   "
              f"RMS VOID={rms_void:.4f}   (RMS of truth in VOID={rms_truth_void:.4f})")
    print(f"  {'PREDICT-ZERO':14s} N=   0  RMS covered={np.sqrt(np.mean(m_true[~void_mask]**2)):.4f}   "
          f"RMS VOID={np.sqrt(np.mean(m_true[void_mask]**2)):.4f}   <- baseline: recover nothing")

    # ---- (d) is the rank gain NEW INFORMATION, or basis realignment? ---------
    print("\n=== (d) IS THE RANK GAIN NEW INFORMATION? ===")
    print(f"  Intrinsic rank of the DATA (fine 100x100 pixel operator) = {rk_pix}")
    print("  Every node parametrization is A_fine @ H for an interpolation matrix H,")
    print("  so rank(A_nodes) <= min(rank(A_fine), N). The data-side rank is IDENTICAL")
    print("  for all grids below; only the alignment of span(H) with row(A_fine) moves.")
    for tag, P in grids:
        A, tri, rk, ndead = rank_of(P)
        print(f"  {tag:20s} rank/N = {rk}/{len(P)} = {100*rk/len(P):5.2f}% estimable  "
              f"| rank/rank(A_fine) = {100*rk/rk_pix:5.2f}% of available information")

    # ---- (e) diagnostic mode: does ell add anything over the column sum? -----
    print("\n=== (e) DIAGNOSTIC MODE: information content of ell vs column sum ===")
    cs_pix = colsum.reshape(nx, nx)
    fin = cs_pix > 0
    order_ell = np.argsort(ell_field[fin].ravel())
    order_cs = np.argsort(-cs_pix[fin].ravel())
    agree = np.mean(order_ell == order_cs)
    print(f"  Section 4 defines ell as a monotone (logarithmic) function of the column")
    print(f"  sum of A. Rank correlation of ell with -colsum on covered pixels: "
          f"{np.corrcoef(np.argsort(np.argsort(ell_field[fin])), np.argsort(np.argsort(-cs_pix[fin])))[0,1]:.6f}")
    print(f"  Identical orderings: {100*agree:.2f}%  ->  ell is a RELABELLING of the")
    print(f"  column sum; as a diagnostic it carries NO information beyond colsum(A).")


def in_box_field(nx):
    g = (np.arange(nx) + 0.5) / nx
    GX, GY = np.meshgrid(g, g)
    P = np.dstack([GX, GY])
    return in_box(P, VOID)


def interp_nodes(tri, vals, pts, simp):
    out = np.zeros(len(pts))
    ok = simp >= 0
    s = simp[ok]
    X = tri.transform[s, :2]
    off = pts[ok] - tri.transform[s, 2]
    b2 = np.einsum('ijk,ik->ij', X, off)
    b = np.column_stack([b2, 1 - b2.sum(axis=1)])
    out[ok] = np.einsum('ij,ij->i', b, vals[tri.simplices[s]])
    return out


def ray_integral_true(a, b, nq=400):
    L = np.linalg.norm(b - a)
    t = (np.arange(nq) + 0.5) / nq
    pts = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
    return truth(pts).sum() * L / nq


if __name__ == "__main__":
    main()
