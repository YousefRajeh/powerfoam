"""
test_bohm2000_regridding.py

Runnable reimplementation of the regridding criterion of
  Boehm, Galuppo & Vesnaver, "3D adaptive tomography using Delaunay triangles
  and Voronoi polygons", Geophysical Prospecting 48(4):723-744, 2000.

Criterion (their Appendix, p.743):
    A = U W V^T
    null-space energy of pixel i:  e_i = sum_{j : w_j < tau} v_ij^2
    local reliability             r_i = 1 - e_i
Their Fig.3 block diagram (p.727), right-hand loop: REMOVE reference points
where local reliability is too low, ADD reference points elsewhere.  Removing a
reference point regenerates the Voronoi diagram, so neighbouring cells grow to
absorb the removed cell's area (this is a *regeneration*, not the pairwise
merge of Vesnaver 1996).

DECISIVE QUESTION: after regridding, does rank(A) INCREASE (new information
about poorly covered regions) or does only the unknown count FALL (the method
merely stops asking questions the data cannot answer)?

CPU only, no GPU, seconds to run.
"""

import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(20000)

# ---------------------------------------------------------------- geometry ---
NSITE0 = 400          # initial reference points (Voronoi generators)
NSAMP = 800           # samples per ray for segment-length accumulation
EVAL = 160            # evaluation raster


HIDDEN = False   # if True, place an extra anomaly INSIDE the unilluminated strip


def true_slowness(xy):
    """Smooth background + a compact anomaly (cf. their Fig.4 model)."""
    x, y = xy[:, 0], xy[:, 1]
    s = 0.30 + 0.10 * y
    r = np.hypot(x - 0.45, y - 0.42)
    s = s + 0.22 * np.exp(-(r / 0.14) ** 2)
    if HIDDEN:
        rh = np.hypot(x - 0.50, y - 0.90)
        s = s + 0.25 * np.exp(-(rh / 0.12) ** 2)
    return s


def make_rays():
    """Sources on the left wall, receivers on the right wall, plus a surface
    line.  Coverage is deliberately non-uniform: NOTHING illuminates the top
    strip y > 0.78 or the bottom-right corner, so genuinely dead cells exist."""
    rays = []
    src_y = np.linspace(0.05, 0.70, 14)
    rec_y = np.linspace(0.05, 0.70, 14)
    for sy in src_y:
        for ry in rec_y:
            rays.append(((0.0, sy), (1.0, ry)))
    # a few sub-vertical rays, left half only -> right side stays sparse
    for sx in np.linspace(0.05, 0.45, 6):
        for rx in np.linspace(0.05, 0.45, 6):
            rays.append(((sx, 0.0), (rx, 0.72)))
    return np.array(rays, dtype=float)


RAYS = make_rays()


def build_A(sites, rays=RAYS, nsamp=NSAMP):
    """Tomographic matrix: A[k,i] = length of ray k inside Voronoi cell i.
    Cells are defined implicitly by nearest-generator assignment (exactly the
    definition on their p.725), so no polygon clipping is needed."""
    tree = cKDTree(sites)
    n = len(sites)
    A = np.zeros((len(rays), n))
    t = (np.arange(nsamp) + 0.5) / nsamp
    for k, (p0, p1) in enumerate(rays):
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)
        L = np.linalg.norm(p1 - p0)
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        _, idx = tree.query(pts)
        np.add.at(A[k], idx, L / nsamp)
    return A


def null_space_energy(A, tau_rel=1e-3):
    """e_i = sum_{w_j < tau} v_ij^2 ; tau = tau_rel * w_max  (their eq. A3)."""
    _, w, Vt = np.linalg.svd(A, full_matrices=True)
    n = Vt.shape[0]
    wfull = np.zeros(n)
    wfull[: len(w)] = w
    tau = tau_rel * w.max()
    mask = wfull < tau
    e = (Vt[mask] ** 2).sum(axis=0)
    return e, w, tau


def numerical_rank(A, tol_rel=1e-10):
    w = np.linalg.svd(A, compute_uv=False)
    return int((w > tol_rel * w.max()).sum()), w


# ------------------------------------------------------- evaluation masks ----
gx = (np.arange(EVAL) + 0.5) / EVAL
GX, GY = np.meshgrid(gx, gx, indexing="xy")
EPTS = np.column_stack([GX.ravel(), GY.ravel()])
ETRUE = true_slowness(EPTS)


def coverage_density():
    """Ray-path density on the evaluation raster.  FIXED across all grids --
    the rays never move, so the well/poor/dead partition is grid-independent."""
    dens = np.zeros(len(EPTS))
    tree = cKDTree(EPTS)
    t = (np.arange(NSAMP) + 0.5) / NSAMP
    for p0, p1 in RAYS:
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        _, idx = tree.query(pts)
        np.add.at(dens, idx, 1.0)
    return dens


DENS = coverage_density()
M_DEAD = DENS == 0
nz = DENS[~M_DEAD]
q = np.quantile(nz, 0.25)
M_POOR = (~M_DEAD) & (DENS <= q)
M_WELL = (~M_DEAD) & (DENS > q)
# region of the hidden anomaly (entirely inside the unilluminated strip)
M_HIDDEN = np.hypot(EPTS[:, 0] - 0.50, EPTS[:, 1] - 0.90) < 0.18
assert M_DEAD[M_HIDDEN].all(), "hidden-anomaly region must be fully dead"


def evaluate(sites, u):
    """Rasterize the piecewise-constant estimate and report RMSE per region."""
    tree = cKDTree(sites)
    _, idx = tree.query(EPTS)
    est = u[idx]
    err = est - ETRUE
    out = {}
    for name, m in (("well", M_WELL), ("poor", M_POOR), ("dead", M_DEAD),
                    ("hidn", M_HIDDEN)):
        out[name] = float(np.sqrt((err[m] ** 2).mean()))
    out["all"] = float(np.sqrt((err ** 2).mean()))
    return out


def dead_cells(A, tol=1e-12):
    return int((np.abs(A).sum(axis=0) <= tol).sum())


def solve(A, d):
    """Damped/minimum-norm least squares (pinv = truncated SVD, their
    'small singular values can be zeroed', p.724)."""
    return np.linalg.pinv(A, rcond=1e-10) @ d


# ------------------------------------------------------------------ run ------
def main():
    sites = RNG.random((NSITE0, 2))
    # data are generated once from the TRUE field on a very fine raster, so the
    # data vector never changes as the grid changes.
    fine = cKDTree(EPTS)
    t = (np.arange(NSAMP) + 0.5) / NSAMP
    d = np.zeros(len(RAYS))
    for k, (p0, p1) in enumerate(RAYS):
        L = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
        pts = p0[None, :] + t[:, None] * (np.asarray(p1) - np.asarray(p0))[None, :]
        _, idx = fine.query(pts)
        d[k] = ETRUE[idx].sum() * L / NSAMP

    print(f"rays = {len(RAYS)}   eval raster = {EVAL}x{EVAL}")
    print(f"raster fractions: dead={M_DEAD.mean():.4f} "
          f"poor={M_POOR.mean():.4f} well={M_WELL.mean():.4f}")
    print()
    hdr = (f"{'it':>3} {'ncells':>7} {'rank(A)':>8} {'rank/ncell':>11} "
           f"{'deadcell':>9} {'rmse_well':>10} {'rmse_poor':>10} "
           f"{'rmse_dead':>10} {'rmse_all':>9}")
    print(hdr)
    print("-" * len(hdr))

    hist = []
    for it in range(7):
        A = build_A(sites)
        r, w = numerical_rank(A)
        u = solve(A, d)
        ev = evaluate(sites, u)
        dc = dead_cells(A)
        hist.append((it, len(sites), r, dc, ev))
        print(f"{it:>3} {len(sites):>7} {r:>8} {r/len(sites):>11.4f} "
              f"{dc:>9} {ev['well']:>10.5f} {ev['poor']:>10.5f} "
              f"{ev['dead']:>10.5f} {ev['all']:>9.5f}")

        if it == 6:
            break
        # --- Boehm et al. regridding, right-hand loop of their Fig.3 ---------
        e, _, _ = null_space_energy(A)
        rel = 1.0 - e
        keep = rel >= 0.5            # remove unreliable reference points
        if keep.sum() < 12:
            break
        if keep.all():
            print("   (no point fell below the reliability threshold; stop)")
            break
        sites = sites[keep]

    print()
    r0, rN = hist[0][2], hist[-1][2]
    n0, nN = hist[0][1], hist[-1][1]
    print(f"DECISIVE: rank(A) {r0} -> {rN}  (delta {rN - r0});  "
          f"unknowns {n0} -> {nN}  (delta {nN - n0})")
    print(f"          rank as fraction of unknowns: "
          f"{r0/n0:.4f} -> {rN/nN:.4f}")
    e0, eN = hist[0][4], hist[-1][4]
    for k in ("well", "poor", "dead", "all"):
        print(f"          rmse[{k:>4}] {e0[k]:.5f} -> {eN[k]:.5f}  "
              f"({100*(eN[k]-e0[k])/e0[k]:+.1f}%)")

    # ---- control: does the rank ceiling come from the rays, not the grid? ---
    print()
    print("CONTROL: rank of A on independent random grids of varying size")
    for n in (60, 120, 250, 400, 800):
        s = RNG.random((n, 2))
        Ac = build_A(s)
        rc, _ = numerical_rank(Ac)
        print(f"   ncells={n:>4}  rank={rc:>4}  deadcells={dead_cells(Ac):>4}")
    print(f"   (number of rays = {len(RAYS)} is the hard ceiling on rank)")

    full_loop(d)
    baseline(d)


# ---------------------------------------------------------------------------
def baseline(d):
    """Is regridding doing anything a trivial nearest-LIVE-cell fill on the
    ORIGINAL grid does not already do?  Vesnaver (1996, p.148) calls cell
    merging 'a nearest-neighbour interpolation'; this quantifies that."""
    print()
    print("BASELINE CONTROL: original 400-cell grid, no regridding, dead cells "
          "filled from nearest LIVE cell")
    ring = M_DEAD & (~M_HIDDEN)

    def contrast(sites, u, tag):
        tree = cKDTree(sites)
        _, idx = tree.query(EPTS)
        est = u[idx]
        ct = est[M_HIDDEN].mean() - est[ring].mean()
        tt = ETRUE[M_HIDDEN].mean() - ETRUE[ring].mean()
        ev = evaluate(sites, u)
        print(f"   {tag:<34} well={ev['well']:.5f} poor={ev['poor']:.5f} "
              f"dead={ev['dead']:.5f} hidn={ev['hidn']:.5f}  "
              f"contrast est={ct:+.4f} / true={tt:+.4f} "
              f"(recovered {ct/tt:6.2f})")

    rng = np.random.default_rng(7)
    sites0 = rng.random((NSITE0, 2))
    A0 = build_A(sites0)
    u0 = solve(A0, d)
    contrast(sites0, u0, "no regridding, min-norm")

    # interpolation-only control: keep the ORIGINAL grid, but overwrite every
    # unreliable cell with the value of its nearest RELIABLE cell.
    e, _, _ = null_space_energy(A0)
    rel = 1.0 - e
    good = rel >= 0.5
    uf = u0.copy()
    tr = cKDTree(sites0[good])
    _, j = tr.query(sites0[~good])
    uf[~good] = u0[good][j]
    contrast(sites0, uf, "no regridding + nearest-reliable")

    # Boehm et al. regridding, thr=0.50, 1 iteration (best row of the sweep)
    keep = good
    new = sites0[keep]
    hi = np.argsort(-rel[keep])[: max(1, int(0.25 * len(new)))]
    tr = cKDTree(new)
    _, nb = tr.query(new[hi], k=2)
    sites1 = np.vstack([new, 0.5 * (new[hi] + new[nb[:, 1]])])
    A1 = build_A(sites1)
    contrast(sites1, solve(A1, d), "Boehm regridding (thr=0.50, 1 it)")
    r1, _ = numerical_rank(A1)
    r0, _ = numerical_rank(A0)
    print(f"   rank: {r0} (400 cells) -> {r1} ({len(sites1)} cells)")
    ablation(sites0, sites1)


# ---------------------------------------------------------------------------
def ablation(sites0, sites1):
    """AIRTIGHT TEST.  The 'contrast' above is confounded by the background
    gradient.  Instead: run the SAME pipeline on the field WITH and WITHOUT the
    hidden anomaly and ask whether the estimate inside the hidden region
    differs at all.  If it does not, zero information about that region was
    recovered -- the apparent improvement is interpolation, nothing else.
    Note the grid itself is field-independent (null-space energy depends only
    on A, i.e. on the raypath geometry), so sites0/sites1 are common to both."""
    print()
    print("AIRTIGHT ABLATION: same grids, field WITH vs WITHOUT hidden anomaly")

    def data_for(hidden):
        global HIDDEN
        HIDDEN = hidden
        et = true_slowness(EPTS)
        fine = cKDTree(EPTS)
        t = (np.arange(NSAMP) + 0.5) / NSAMP
        dd = np.zeros(len(RAYS))
        for k, (p0, p1) in enumerate(RAYS):
            p0 = np.asarray(p0); p1 = np.asarray(p1)
            L = np.linalg.norm(p1 - p0)
            pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            _, idx = fine.query(pts)
            dd[k] = et[idx].sum() * L / NSAMP
        return dd, et

    d_on, et_on = data_for(True)
    d_off, et_off = data_for(False)
    print(f"   traveltime data change: max|d_on - d_off| = "
          f"{np.abs(d_on - d_off).max():.3e}  (rays never touch the anomaly)")
    print(f"   true field change inside hidden region: "
          f"{(et_on - et_off)[M_HIDDEN].mean():+.4f}")

    for tag, s in (("original 400-cell grid", sites0),
                   ("Boehm-regridded grid  ", sites1)):
        A = build_A(s)
        tree = cKDTree(s)
        _, idx = tree.query(EPTS)
        e_on = solve(A, d_on)[idx]
        e_off = solve(A, d_off)[idx]
        diff = np.abs(e_on - e_off)[M_HIDDEN].mean()
        print(f"   {tag}: mean|est_on - est_off| inside hidden region "
              f"= {diff:.3e}")
    print("   -> compare each against the true change printed above.  Note the")
    print("      anomaly is a Gaussian, so a small tail does reach the rays;")
    print("      max|d_on-d_off| above is that leakage, and it is the ONLY")
    print("      real information either grid can be responding to.")


# ---------------------------------------------------------------------------
def full_loop(d, thresholds=(0.2, 0.5, 0.8, 0.95)):
    """Both halves of their Fig.3 right-hand loop: REMOVE generators where the
    local reliability 1-e_i is below threshold, and ADD generators (splitting
    the cell towards its most reliable neighbour) where reliability is high.
    Swept over the reliability threshold."""
    print()
    print("FULL Fig.3 LOOP (remove low-reliability AND add high-reliability "
          "points), threshold sweep")
    hdr = (f"{'thr':>5} {'it':>3} {'ncells':>7} {'rank':>5} {'rk/nc':>7} "
           f"{'dead':>5} {'rmse_well':>10} {'rmse_poor':>10} {'rmse_dead':>10}"
           f" {'rmse_hidn':>10}")
    print(hdr)
    print("-" * len(hdr))
    for thr in thresholds:
        rng = np.random.default_rng(7)
        sites = rng.random((NSITE0, 2))
        for it in range(5):
            A = build_A(sites)
            r, _ = numerical_rank(A)
            u = solve(A, d)
            ev = evaluate(sites, u)
            print(f"{thr:>5.2f} {it:>3} {len(sites):>7} {r:>5} "
                  f"{r/len(sites):>7.4f} {dead_cells(A):>5} "
                  f"{ev['well']:>10.5f} {ev['poor']:>10.5f} {ev['dead']:>10.5f}"
                  f" {ev['hidn']:>10.5f}")
            if it == 4:
                break
            e, _, _ = null_space_energy(A)
            rel = 1.0 - e
            keep = rel >= thr
            if keep.sum() < 12:
                break
            new = sites[keep]
            # add: split the most reliable cells by inserting a generator
            # halfway to the nearest kept neighbour (their "add new ones")
            hi = np.argsort(-rel[keep])[: max(1, int(0.25 * len(new)))]
            if len(new) > 2:
                tr = cKDTree(new)
                _, nb = tr.query(new[hi], k=2)
                extra = 0.5 * (new[hi] + new[nb[:, 1]])
                new = np.vstack([new, extra])
            sites = new
        print()


if __name__ == "__main__":
    import sys
    if "--hidden" in sys.argv:
        HIDDEN = True
        ETRUE = true_slowness(EPTS)
        print("*** HIDDEN-ANOMALY VARIANT: an anomaly of amplitude 0.25 sits "
              "at (0.50,0.90), entirely inside the unilluminated strip ***\n")
    main()
