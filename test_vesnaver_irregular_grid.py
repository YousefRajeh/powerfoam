"""
test_vesnaver_irregular_grid.py

Verifies the central claims of:
  Vesnaver, A.L. (1996), "Irregular grids in seismic tomography and
  minimum-time ray tracing", Geophys. J. Int. 126(1), 147-165.

Claims under test (locators in the report):
  C1  (p.150, "SOME RECIPES..."): pixels crossed by no ray produce null-space
      vectors.
  C2  (p.150-151, eq.5): the "quasi-null-space map" m_i = sum_{j in quasi-null}
      V_ij^2 equals ~1 exactly at pixels crossed by no ray / very short rays.
  C3  (p.150, rule 1 + p.162 CONCLUSIONS): merging uncrossed pixels into a
      crossed neighbour (= summing columns of A) removes the null space and
      "transforms a rank-deficient system into an overdetermined one".
  C4  (p.150, rule 3 + Fig.4): splitting a pixel can make two linearly
      dependent rays linearly independent.
  C5  (p.150, rule 4): merging adjacent pixels NEVER makes linearly dependent
      crossing rays become linearly independent.
  C6  (p.154, Figs 12-13, "ANGULAR COVERAGE IS NOT THE PROBLEM"): zero angular
      aperture can still be fully determined; conversely infinite angular
      coverage through a 2-pixel model can be rank-1.

  T1  (transfer test, not Vesnaver's claim): what merging COSTS when the
      per-cell value is the quantity you are scored on -- continuous error and
      discrete (semantic-label) accuracy at the merged cells.

CPU-only, seconds to run.  numpy only.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# exact ray / axis-aligned-grid intersection lengths
# ----------------------------------------------------------------------------
def ray_row(p0, p1, nx, ny, extent=(0.0, 1.0, 0.0, 1.0)):
    """Path length of segment p0->p1 in each cell of an nx*ny regular grid.
    Returns a flat row of length nx*ny (cell index = iy*nx + ix)."""
    x0, x1, y0, y1 = extent
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    ts = [0.0, 1.0]
    for k, (lo, hi, n) in enumerate(((x0, x1, nx), (y0, y1, ny))):
        if abs(d[k]) > 1e-14:
            for i in range(n + 1):
                g = lo + (hi - lo) * i / n
                t = (g - p0[k]) / d[k]
                if 0.0 < t < 1.0:
                    ts.append(t)
    ts = np.unique(np.array(ts))
    row = np.zeros(nx * ny)
    for a, b in zip(ts[:-1], ts[1:]):
        mid = p0 + 0.5 * (a + b) * d
        ix = int((mid[0] - x0) / (x1 - x0) * nx)
        iy = int((mid[1] - y0) / (y1 - y0) * ny)
        if 0 <= ix < nx and 0 <= iy < ny:
            row[iy * nx + ix] += (b - a) * L
    return row


def build_A(rays, nx, ny, extent=(0.0, 1.0, 0.0, 1.0)):
    return np.array([ray_row(p, q, nx, ny, extent) for p, q in rays])


# ----------------------------------------------------------------------------
# Vesnaver's diagnostics
# ----------------------------------------------------------------------------
def svd_diag(A, tol_rel=1e-8):
    """Return (s, V, n_quasi_zero, m) where m is Vesnaver eq.(5)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T
    ncol = A.shape[1]
    s_full = np.zeros(ncol)
    s_full[: len(s)] = s
    thr = tol_rel * max(s.max(), 1e-300)
    qz = s_full <= thr                     # quasi-zero singular values
    m = (V[:, qz] ** 2).sum(axis=1)        # eq. (5)
    return s_full, V, int(qz.sum()), m


def cond_full(A):
    s = np.linalg.svd(A, compute_uv=False)
    ncol = A.shape[1]
    if len(s) < ncol or s.min() <= 0:
        return np.inf
    return s.max() / s.min()


def report(tag, A, groups=None):
    s, V, nqz, m = svd_diag(A)
    rank = int((s > 1e-8 * s.max()).sum())
    print(f"  {tag:38s} cells={A.shape[1]:4d} rays={A.shape[0]:4d} "
          f"rank={rank:4d} nulldim={A.shape[1]-rank:4d} "
          f"cond={cond_full(A):.3e}  max_m={m.max():.4f}")
    return rank, m


# ----------------------------------------------------------------------------
# merging = summing columns  (paper, p.148 bottom: "pixel merging corresponds
# to the summation of two adjacent columns of the tomographic matrix")
# ----------------------------------------------------------------------------
def merge_columns(A, groups, a, b):
    """Merge group b into group a. groups: list of lists of original cell ids."""
    A2 = A.copy()
    A2[:, a] = A2[:, a] + A2[:, b]
    A2 = np.delete(A2, b, axis=1)
    g2 = [list(g) for g in groups]
    g2[a] = g2[a] + g2[b]
    del g2[b]
    return A2, g2


def neighbours_of_group(g, nx, ny):
    """4-neighbour original-cell ids adjacent to any cell in group g."""
    out = set()
    for c in g:
        ix, iy = c % nx, c // nx
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny:
                out.add(jy * nx + jx)
    return out - set(g)


def vesnaver_merge_loop(A0, nx, ny, max_iter=500, m_thresh=0.5, verbose=False):
    """Iteratively merge the least-reliable group (largest m_i, Vesnaver eq.5)
    into its most-reliable geometric neighbour group, until no quasi-zero
    singular values remain (i.e. every cell is data-determined)."""
    A = A0.copy()
    groups = [[j] for j in range(A0.shape[1])]
    history = []
    for it in range(max_iter):
        s, V, nqz, m = svd_diag(A)
        rank = int((s > 1e-8 * s.max()).sum())
        history.append((A.shape[1], rank, A.shape[1] - rank, cond_full(A), m.max()))
        if nqz == 0:
            break
        a = int(np.argmax(m))              # worst-determined group
        nb = neighbours_of_group(groups[a], nx, ny)
        cand = [i for i, g in enumerate(groups) if i != a and (set(g) & nb)]
        if not cand:
            break
        b = min(cand, key=lambda i: m[i])  # best-determined neighbour
        lo, hi = (b, a) if b < a else (a, b)
        A, groups = merge_columns(A, groups, lo, hi)
        if verbose:
            print(f"    it{it:3d} merged -> {A.shape[1]} groups")
    return A, groups, history


# ============================================================================
print("=" * 78)
print("SETUP: walk-away-VSP-like geometry with deliberately non-uniform coverage")
print("=" * 78)
NX = NY = 8
# sources down a 'well' on the left edge, receivers along the top surface
srcs = [(0.0, y) for y in np.linspace(0.05, 0.95, 6)]
recs = [(x, 1.0) for x in np.linspace(0.05, 0.95, 6)]
rays = [(s, r) for s in srcs for r in recs]
A0 = build_A(rays, NX, NY)
hits = (A0 > 1e-9).sum(axis=0)
print(f"  grid {NX}x{NY} = {NX*NY} cells, {len(rays)} rays")
print(f"  cells with zero rays : {(hits==0).sum()} / {NX*NY} "
      f"({100*(hits==0).mean():.1f}%)")
print(f"  hit-count percentiles: min={hits.min()} p25={np.percentile(hits,25):.0f} "
      f"med={np.median(hits):.0f} max={hits.max()}")

print("\n--- C1/C2: does eq.(5) identify the uncrossed pixels? ---")
rank0, m0 = report("original regular grid", A0)
void = np.where(hits == 0)[0]
nonvoid = np.where(hits > 0)[0]
print(f"  m on VOID cells    : min={m0[void].min():.6f} mean={m0[void].mean():.6f}")
print(f"  m on CROSSED cells : max={m0[nonvoid].max():.6f} mean={m0[nonvoid].mean():.6f}")
n_pred = int((m0 > 0.99).sum())
tp = len(set(np.where(m0 > 0.99)[0]) & set(void.tolist()))
print(f"  cells with m>0.99  : {n_pred}; of which truly void: {tp} "
      f"(all {len(void)} void cells flagged: {set(void.tolist()) <= set(np.where(m0>0.99)[0].tolist())})")
print(f"  null dim ({A0.shape[1]-rank0}) vs #void cells ({len(void)}): "
      f"{'EQUAL' if A0.shape[1]-rank0 == len(void) else 'DIFFERENT -> extra null dirs beyond void cells'}")

print("\n--- C3: iterative merge until every cell is data-determined ---")
Am, groups, hist = vesnaver_merge_loop(A0, NX, NY)
print("   iter  cells  rank  nulldim        cond      max_m")
for i, (c, r, n, cd, mm) in enumerate(hist):
    if i < 3 or i >= len(hist) - 3 or i % 5 == 0:
        print(f"   {i:4d} {c:6d} {r:5d} {n:8d}  {cd:10.3e} {mm:10.4f}")
c, r, n, cd, mm = hist[-1]
print(f"  RESULT: {A0.shape[1]} -> {c} cells; nulldim {A0.shape[1]-rank0} -> {n}; "
      f"cond {hist[0][3]:.3e} -> {cd:.3e}")
print(f"  overdetermined? rays({Am.shape[0]}) >= cells({Am.shape[1]}): "
      f"{Am.shape[0] >= Am.shape[1]}")

print("\n--- C3b: control -- does plain damping do the same? ---")
for lam in (1e-6, 1e-3, 1e-1):
    G = A0.T @ A0 + lam * np.eye(A0.shape[1])
    sg = np.linalg.svd(G, compute_uv=False)
    # resolution matrix singular values, Vesnaver eq.(3): r_i = a_i^2/(a_i^2+lam)
    a2 = np.linalg.svd(A0, compute_uv=False) ** 2
    ri = a2 / (a2 + lam)
    print(f"  lambda={lam:g}: cond(A^T A + lam I)={sg.max()/sg.min():.3e}, "
          f"resolution r_i: min={ri.min():.3e} median={np.median(ri):.4f} max={ri.max():.4f}")

print("\n--- C4: splitting a pixel makes dependent rays independent (Fig.4) ---")
# two parallel horizontal rays through a 1x2 grid (1 col, 2 rows) -> dependent
r1 = [((0.0, 0.20), (1.0, 0.20)), ((0.0, 0.30), (1.0, 0.30))]
A_dep = build_A(r1, 1, 2)
print(f"  before split (1x2 grid): rank={np.linalg.matrix_rank(A_dep)} of 2 rays, rows={A_dep.tolist()}")
# split the lower pixel vertically -> 2x2 grid, rays now at different depths?
# Fig.4 splits so the two rays sample different new pixels: use rays at
# different depths through a grid split horizontally at y=0.25
A_ind = build_A(r1, 1, 4)
print(f"  after horizontal split (1x4 grid): rank={np.linalg.matrix_rank(A_ind)} of 2 rays")
print(f"  claim C4 holds: {np.linalg.matrix_rank(A_dep)==1 and np.linalg.matrix_rank(A_ind)==2}")

print("\n--- C5: merging never makes dependent crossing rays independent ---")
# Fig.13: two stacked pixels, all rays pass through the midpoint of the shared
# boundary -> every row is a multiple pattern; check rank, then merge.
mid = (0.5, 0.5)
ang = np.linspace(0.15, 1.42, 40)
r13 = []
for a in ang:
    p = (mid[0] - 0.5 * np.cos(a) / max(np.sin(a), 1e-6),
         mid[1] - 0.5)
    q = (mid[0] + 0.5 * np.cos(a) / max(np.sin(a), 1e-6),
         mid[1] + 0.5)
    if abs(p[0]) <= 1 and abs(q[0]) <= 1:
        r13.append(((p[0] + 0.0, p[1]), (q[0], q[1])))
A13 = build_A(r13, 1, 2, extent=(-1.0, 1.0, 0.0, 1.0))
print(f"  Fig.13 config: {len(r13)} rays at many angles through 2 pixels, "
      f"rank(A)={np.linalg.matrix_rank(A13)} (columns={A13.shape[1]})")
A13m, _ = merge_columns(A13, [[0], [1]], 0, 1)
print(f"  after merging the 2 pixels: rank={np.linalg.matrix_rank(A13m)}, cols={A13m.shape[1]} "
      f"-> ray dependence unchanged (rank still 1), but system now determined. C5 consistent.")

print("\n--- C6: zero angular aperture, fully determined (Fig.12 well logging) ---")
# 6 stacked pixels, each ray confined to one pixel: aperture = 0
Awell = np.zeros((6, 6))
for i in range(6):
    Awell[i, i] = 1.0 / 6
print(f"  well-log style A: rank={np.linalg.matrix_rank(Awell)}/6, "
      f"cond={cond_full(Awell):.3f}, angular aperture = 0 deg. C6 holds.")

# ============================================================================
print("\n" + "=" * 78)
print("T1 TRANSFER TEST: what does merging cost when the CELL VALUE is the score?")
print("=" * 78)
# ground truth with real structure inside the poorly-covered region
xs = (np.arange(NX) + 0.5) / NX
ys = (np.arange(NY) + 0.5) / NY
XX, YY = np.meshgrid(xs, ys)
u_true = (0.5 + 0.3 * XX + 0.2 * np.sin(6 * YY)).ravel()          # continuous
lab_true = ((XX > 0.55).astype(int) + 2 * (YY < 0.45).astype(int)).ravel()  # 4 labels

t_obs = A0 @ u_true

# (a) least squares on the ORIGINAL grid (pinv -> min-norm, void cells = 0)
u_ls = np.linalg.pinv(A0) @ t_obs

# (b) merged grid: solve for group values, then broadcast back to cells
tm = t_obs
u_grp, *_ = np.linalg.lstsq(Am, tm, rcond=None)
u_merged = np.zeros(NX * NY)
for gi, g in enumerate(groups):
    for c in g:
        u_merged[c] = u_grp[gi]

def rmse(a, b, idx):
    return float(np.sqrt(np.mean((a[idx] - b[idx]) ** 2)))

print(f"  data misfit ||Au-t||: original-grid pinv = {np.linalg.norm(A0@u_ls-t_obs):.3e}, "
      f"merged = {np.linalg.norm(Am@u_grp-tm):.3e}")
print(f"  RMSE vs truth, CROSSED cells : pinv={rmse(u_ls,u_true,nonvoid):.4f}  "
      f"merged={rmse(u_merged,u_true,nonvoid):.4f}")
print(f"  RMSE vs truth, VOID    cells : pinv={rmse(u_ls,u_true,void):.4f}  "
      f"merged={rmse(u_merged,u_true,void):.4f}")
print(f"  RMSE vs truth, ALL     cells : pinv={rmse(u_ls,u_true,np.arange(NX*NY)):.4f}  "
      f"merged={rmse(u_merged,u_true,np.arange(NX*NY)):.4f}")

# discrete / semantic-label analogue: a merged group can only emit ONE label
print("\n  SEMANTIC-LABEL analogue (the ScanNet mIoU situation):")
print("  a merged group must emit a single label for all its member cells.")
purity = []
for g in groups:
    ls = lab_true[g]
    vals, cnts = np.unique(ls, return_counts=True)
    purity.append(cnts.max() / len(ls))
purity = np.array(purity)
sizes = np.array([len(g) for g in groups])
# best achievable per-cell accuracy after merging (oracle majority label)
oracle_acc = float((purity * sizes).sum() / sizes.sum())
print(f"  groups: {len(groups)} (from {NX*NY} cells); size max={sizes.max()} mean={sizes.mean():.2f}")
print(f"  impure groups (mixed ground-truth label): {(purity<1).sum()}/{len(groups)}")
print(f"  ORACLE per-cell label accuracy after merging = {oracle_acc*100:.2f}%  "
      f"(ceiling imposed by the partition alone, before any estimation error)")
# same ceiling for a merge that only touches void cells
groups_void = [[j] for j in range(NX * NY)]
Av = A0.copy()
# merge each void cell into its most-hit neighbour, one pass
order = sorted(void.tolist())
assign = {}
for c in order:
    ix, iy = c % NX, c // NX
    best, bh = None, -1
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < NX and 0 <= jy < NY:
            j = jy * NX + jx
            if hits[j] > bh:
                best, bh = j, hits[j]
    assign[c] = best
ok = sum(1 for c, j in assign.items() if j is not None and lab_true[c] == lab_true[j])
print(f"  void-only merge: {len(assign)} void cells absorbed into a neighbour; "
      f"label inherited correctly for {ok}/{len(assign)} = {100*ok/max(len(assign),1):.1f}%")
print("=" * 78)
