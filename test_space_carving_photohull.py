"""
test_space_carving_photohull.py

Tests whether Kutulakos & Seitz (1999) "A Theory of Shape by Space Carving"
photo-hull machinery predicts per-cell identifiability in OUR setting:
known geometry, unknown per-cell value, occluded substochastic forward operator
    b_r = sum_j A_rj f_j ,   A_rj = alpha_j * prod_{k<j on ray r} (1-alpha_k)

Two forward models on the SAME synthetic 2D scene:
  (K&S model)  hard surface: ray returns value of first opaque cell  -> Space Carving
  (our model)  soft transmittance mixing over ALL cells              -> least squares

Measured:
  1. # cells never touched by any ray (ray mass == 0)
  2. whether the carving output distinguishes never-seen cells from seen-and-carved
  3. Spearman rho of per-cell |error| against:
        - ray mass (A^T 1)_j                       <-- BASELINE TO BEAT
        - photo-hull membership (binary)
        - K&S visibility count |Vis_{V*}(v)|       <-- graded, best-case for K&S
        - raw seen-count (# rays with A_rj > 0)
CPU-only, no GPU, no writes outside this file's own stdout.
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

RNG = np.random.default_rng(0)
G = 64                      # grid GxG
TOL = 1e-6

# ----------------------------------------------------------------------------
# 1. Scene: opaque occupancy with a concave notch + a separate occluding wall
#    that shadows a pocket of free space and part of the block.
# ----------------------------------------------------------------------------
def build_scene():
    occ = np.zeros((G, G), bool)
    occ[18:46, 18:46] = True          # main block
    occ[26:38, 18:34] = False         # deep concave notch opening downward (-y)
    occ[6:10, 10:54] = True           # wall slab in front of the notch mouth
    occ[48:52, 24:40] = True          # second slab
    return occ

def build_values(occ):
    """Ground-truth per-cell scalar 'feature' (stand-in for a CLIP channel)."""
    yy, xx = np.mgrid[0:G, 0:G]
    f = (np.sin(xx / 7.0) * np.cos(yy / 5.0)
         + 0.5 * np.sin((xx + yy) / 4.0)
         + 0.3 * RNG.standard_normal((G, G)))
    return f

# ----------------------------------------------------------------------------
# 2. Cameras on a circle, fan of rays each.  DDA march over the grid.
# ----------------------------------------------------------------------------
def cameras(n_cam=12, n_ray=140, radius=54.0):
    cams = []
    c = (G - 1) / 2.0
    for i in range(n_cam):
        a = 2 * np.pi * i / n_cam
        o = np.array([c + radius * np.cos(a), c + radius * np.sin(a)])
        look = np.array([c, c]) - o
        look /= np.linalg.norm(look)
        perp = np.array([-look[1], look[0]])
        fov = 0.62                                  # half-angle, radians
        dirs = []
        for t in np.linspace(-fov, fov, n_ray):
            d = look * np.cos(t) + perp * np.sin(t)
            dirs.append(d / np.linalg.norm(d))
        cams.append((o, np.array(dirs)))
    return cams

def march(o, d, max_steps=400, step=0.5):
    """Return ordered list of unique cell indices the ray passes through."""
    cells = []
    last = -1
    p = o.copy()
    for _ in range(max_steps):
        p = p + d * step
        ix, iy = int(np.floor(p[0])), int(np.floor(p[1]))
        if ix < 0 or iy < 0 or ix >= G or iy >= G:
            if cells:
                break            # left the volume after entering
            else:
                continue         # not yet entered
        idx = iy * G + ix
        if idx != last:
            cells.append(idx)
            last = idx
    return cells

# ----------------------------------------------------------------------------
# 3. Our forward operator A (soft transmittance, known geometry)
# ----------------------------------------------------------------------------
ALPHA_OCC, ALPHA_FREE = 0.90, 0.020

def build_A(cams, occ):
    alpha = np.where(occ.ravel(), ALPHA_OCC, ALPHA_FREE)
    rows, cols, vals, ray_paths = [], [], [], []
    r = 0
    for o, dirs in cams:
        for d in dirs:
            path = march(o, d)
            if not path:
                continue
            T = 1.0
            for j in path:
                w = alpha[j] * T
                if w > 1e-5:
                    rows.append(r); cols.append(j); vals.append(w)
                T *= (1.0 - alpha[j])
                if T < 1e-4:
                    break
            ray_paths.append(path)
            r += 1
    A = csr_matrix((vals, (rows, cols)), shape=(r, G * G))
    return A, ray_paths

# ----------------------------------------------------------------------------
# 4. K&S hard-surface rendering + Space Carving Algorithm (Sec. 4, p.5)
# ----------------------------------------------------------------------------
BG = np.nan   # background label

def render_hard(cams, occ, f):
    """Pixel = value of first opaque cell hit; NaN if ray misses the object."""
    imgs = []
    for o, dirs in cams:
        px = []
        for d in dirs:
            v = BG
            for j in march(o, d):
                if occ.ravel()[j]:
                    v = f.ravel()[j]
                    break
            px.append(v)
        imgs.append(np.array(px))
    return imgs

def visibility_of(V, cams, imgs):
    """For current volume V (bool GxG), for each camera/ray find the FIRST cell
    of V hit.  Returns dict cell -> list of (cam, pixel_value)."""
    vis = {}
    Vf = V.ravel()
    for ci, (o, dirs) in enumerate(cams):
        img = imgs[ci]
        for k, d in enumerate(dirs):
            for j in march(o, d):
                if Vf[j]:
                    vis.setdefault(j, []).append((ci, img[k]))
                    break
    return vis

def space_carve(cams, imgs, thresh=0.12, max_iter=60):
    """Step 1-3 of the Space Carving Algorithm.  Lambertian consist_K:
    a voxel is photo-consistent iff (a) it projects to no background pixel and
    (b) std of the observed colors <= thresh.  A voxel with EMPTY Vis_V(v)
    is trivially consistent (consist_K is never invoked) -- exactly as the
    algorithm specifies."""
    V = np.ones((G, G), bool)
    for it in range(max_iter):
        vis = visibility_of(V, cams, imgs)
        carve = []
        for j, obs in vis.items():
            cols = np.array([c for _, c in obs])
            if np.isnan(cols).any():           # Def. 1 (1): background pixel
                carve.append(j); continue
            if len(cols) >= 2 and cols.std() > thresh:   # consist_K, K=2
                carve.append(j)
        if not carve:
            break
        Vf = V.ravel(); Vf[carve] = False; V = Vf.reshape(G, G)
    return V, it, vis

# ----------------------------------------------------------------------------
def main():
    occ = build_scene()
    f = build_values(occ)
    cams = cameras()

    A, ray_paths = build_A(cams, occ)
    ray_mass = np.asarray(A.sum(axis=0)).ravel()          # (A^T 1)_j
    seen_count = np.asarray((A > 0).sum(axis=0)).ravel()
    b = A @ f.ravel()

    n_cells = G * G
    dead = ray_mass <= TOL
    print("=" * 74)
    print("SCENE: %dx%d = %d cells, %d rays from %d cameras, %.1f cells/ray"
          % (G, G, n_cells, A.shape[0], len(cams), A.nnz / A.shape[0]))
    print("occupied cells: %d (%.1f%%)" % (occ.sum(), 100 * occ.mean()))
    print("-" * 74)
    print("[Q1] cells NEVER touched by any ray (ray mass == 0): %d (%.2f%%)"
          % (dead.sum(), 100 * dead.mean()))

    # ---- recover f from b (min-norm least squares) ----
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=4000)
    fhat = sol[0]
    err = np.abs(fhat - f.ravel())
    print("     relative residual ||Af-b||/||b|| = %.3e" % (sol[3] / np.linalg.norm(b)))
    print("     mean |err| live cells %.4f  |  dead cells %.4f"
          % (err[~dead].mean(), err[dead].mean() if dead.any() else float('nan')))

    # ---- Space Carving ----
    imgs = render_hard(cams, occ, f)
    Vstar, iters, vis_star = space_carve(cams, imgs)
    hull = Vstar.ravel()
    kscount = np.zeros(n_cells)
    for j, obs in vis_star.items():
        kscount[j] = len(obs)

    print("-" * 74)
    print("[SPACE CARVING] converged in %d sweeps" % (iters + 1))
    print("  photo hull |V*| = %d cells (%.1f%%);  true scene = %d;  V* superset of true: %s"
          % (hull.sum(), 100 * hull.mean(), occ.sum(),
             bool(np.all(occ.ravel() <= hull))))
    print("  cells carved away: %d" % (~hull).sum())

    # ---- [Q2] does carving distinguish unseen from seen-and-carved? ----
    print("-" * 74)
    print("[Q2] Does the carving output distinguish never-seen from seen-and-carved?")
    ks_never_seen = kscount == 0
    print("  dead cells (our ray mass == 0):                    %d" % dead.sum())
    print("  of those, INSIDE the photo hull V*:                %d (%.1f%%)"
          % ((dead & hull).sum(), 100 * (dead & hull).sum() / max(dead.sum(), 1)))
    print("  of those, CARVED away:                             %d (%.1f%%)"
          % ((dead & ~hull).sum(), 100 * (dead & ~hull).sum() / max(dead.sum(), 1)))
    print("  cells in V* with Vis_V*(v) == 0 (interior+occluded): %d" % (hull & ks_never_seen).sum())
    print("  --> the algorithm emits ONE bit (in/out of V*).  Compare label sets:")
    for name, m in [("in-hull & never-seen", hull & ks_never_seen),
                    ("in-hull & seen", hull & ~ks_never_seen),
                    ("carved", ~hull)]:
        if m.sum():
            print("      %-22s n=%5d  mean|err|=%.4f  mean raymass=%.3f"
                  % (name, m.sum(), err[m].mean(), ray_mass[m].mean()))

    # ---- [Q3] predictor comparison ----
    print("-" * 74)
    print("[Q3] Spearman rho vs per-cell |error|  (more negative = better predictor)")
    preds = {
        "ray mass (A^T 1)_j        [BASELINE]": ray_mass,
        "seen-count (#rays, A_rj>0)":            seen_count.astype(float),
        "photo-hull membership (binary)":        hull.astype(float),
        "K&S visibility |Vis_V*(v)|":            kscount,
        "K&S vis, hull-restricted":              np.where(hull, kscount, 0.0),
    }
    res = {}
    for k, v in preds.items():
        rho, p = spearmanr(v, err)
        res[k] = rho
        print("  %-38s rho = %+.4f   (p=%.1e)" % (k, rho, p))

    base = res["ray mass (A^T 1)_j        [BASELINE]"]
    best_ks = min(res["photo-hull membership (binary)"], res["K&S visibility |Vis_V*(v)|"],
                  res["K&S vis, hull-restricted"])
    print("-" * 74)
    print("VERDICT: baseline rho=%.4f, best K&S-derived rho=%.4f  -> K&S %s"
          % (base, best_ks, "BEATS baseline" if best_ks < base else "does NOT beat baseline"))

    # ---- does hull membership add anything ON TOP of ray mass? ----
    # partial: within live cells only, does hull membership still correlate?
    live = ~dead
    rho_h, p_h = spearmanr(hull[live].astype(float), err[live])
    print("  hull membership vs err, LIVE cells only: rho=%+.4f (p=%.1e)  <- incremental value"
          % (rho_h, p_h))
    rho_k, p_k = spearmanr(kscount[live], err[live])
    print("  K&S vis-count  vs err, LIVE cells only: rho=%+.4f (p=%.1e)" % (rho_k, p_k))
    rho_r, p_r = spearmanr(ray_mass[live], err[live])
    print("  ray mass       vs err, LIVE cells only: rho=%+.4f (p=%.1e)" % (rho_r, p_r))
    print("=" * 74)


if __name__ == "__main__":
    main()
