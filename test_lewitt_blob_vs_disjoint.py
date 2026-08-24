"""
test_lewitt_blob_vs_disjoint.py

MATCHED comparison of a DISJOINT (pixel/Voronoi indicator) basis against an
OVERLAPPING (Kaiser-Bessel blob, Lewitt JOSA-A 7(10):1834, 1990) basis on a
small 2D parallel-beam tomography problem.

Matched: same J centers, same rays, same phantom, same continuous-domain error
metric (L2 error of the RECONSTRUCTED FUNCTION vs the continuous phantom on a
fine grid), same solver family.

Reports for both bases:
  - cond(G), G_jk = <phi_j, phi_k>_L2   (Gram / function-vs-coefficient gap)
  - cond(A^T A) = (smax/smin)^2, smin, numerical rank, # exactly-dead columns
  - reconstruction L2 error, min-norm LSQ and best-Tikhonov, clean and noisy

Regimes:
  A) well-determined, full angular coverage
  B) under-determined, poor coverage (limited angles + truncated detector, so a
     rim of cells is NEVER touched by any ray) -- mirrors our 9.44% dead cells.

CPU-only, tiny, no writes outside this file's stdout.
"""
import numpy as np
from scipy.special import iv

np.random.seed(0)

# ----------------------------------------------------------------- geometry --
NG = 22                 # grid side; centers inside unit disk
DELTA = 2.0 / NG        # grid spacing on [-1,1]^2
FINE = 176              # fine evaluation grid side


def make_centers():
    c1 = (np.arange(NG) + 0.5) * DELTA - 1.0
    X, Y = np.meshgrid(c1, c1, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    keep = np.hypot(P[:, 0], P[:, 1]) <= 1.0 - 0.5 * DELTA
    return P[keep]


CENTERS = make_centers()
J = len(CENTERS)

# ------------------------------------------------------- Kaiser-Bessel blob --
# Lewitt Eq. (A1): w_n^m(r) = [sqrt(1-(r/a)^2)]^m I_m(alpha sqrt(1-(r/a)^2)) / I_m(alpha)
M_ORDER = 2
ALPHA = 10.4            # standard blob taper (Lewitt/Matej-Lewitt convention)


def blob(r, a, alpha=ALPHA, m=M_ORDER):
    r = np.asarray(r, float)
    z = 1.0 - (r / a) ** 2
    out = np.zeros_like(r)
    ok = z > 0
    s = np.sqrt(z[ok])
    out[ok] = (s ** m) * iv(m, alpha * s) / iv(m, alpha)
    return out


def blob_xray(s, a, alpha=ALPHA, m=M_ORDER):
    """Lewitt Eq. (A7): Abel transform of the generalized KB window.
    p^m(s) = a*sqrt(2*pi/alpha)/I_m(alpha) * u^(m+1/2) * I_{m+1/2}(alpha*u),
    u = sqrt(1-(s/a)^2).   Verified numerically against direct quadrature below.
    """
    s = np.asarray(s, float)
    z = 1.0 - (s / a) ** 2
    out = np.zeros_like(s)
    ok = z > 0
    u = np.sqrt(z[ok])
    out[ok] = (a * np.sqrt(2.0 * np.pi / alpha) / iv(m, alpha)) * \
              (u ** (m + 0.5)) * iv(m + 0.5, alpha * u)
    return out


def verify_xray(a):
    """Check Eq. (A7) against direct numerical Abel integration."""
    t = np.linspace(-a, a, 20001)
    errs = []
    for s in [0.0, 0.25 * a, 0.5 * a, 0.8 * a]:
        num = np.trapezoid(blob(np.hypot(s, t), a), t)
        errs.append(abs(num - blob_xray(np.array([s]), a)[0]) / max(num, 1e-30))
    return max(errs)


# ------------------------------------------------------------------ phantom --
def phantom(x, y):
    """Continuous phantom: two flat disks (edges) + a smooth Gaussian bump."""
    v = np.zeros_like(x)
    v += 1.0 * (((x + 0.30) ** 2 + (y - 0.15) ** 2) < 0.32 ** 2)
    v += 0.6 * (((x - 0.35) ** 2 + (y + 0.28) ** 2) < 0.22 ** 2)
    v += 0.8 * np.exp(-((x - 0.10) ** 2 + (y - 0.45) ** 2) / (2 * 0.16 ** 2))
    v *= (np.hypot(x, y) < 0.97)
    return v


# --------------------------------------------------------------------- rays --
def make_rays(n_ang, n_det, ang_lo, ang_hi, det_max):
    ang = np.linspace(ang_lo, ang_hi, n_ang, endpoint=False)
    det = np.linspace(-det_max, det_max, n_det)
    A, D = np.meshgrid(ang, det, indexing="ij")
    return A.ravel(), D.ravel()


NQ = 900  # quadrature samples along each ray


def ray_points(theta, s):
    """Sample points along ray (theta, s) in [-1,1]^2. Returns (I,NQ,2), dt."""
    t = np.linspace(-1.45, 1.45, NQ)
    dt = t[1] - t[0]
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    # normal (ct,st), direction (-st,ct)
    px = s[:, None] * ct - t[None, :] * st
    py = s[:, None] * st + t[None, :] * ct
    return px, py, dt


def exact_data(theta, s):
    px, py, dt = ray_points(theta, s)
    return phantom(px, py).sum(1) * dt


# ------------------------------------------------------- system matrices A ---
def A_disjoint(theta, s):
    """Ray-length of ray i inside square pixel j (quadrature; matched to the
    same quadrature used for the exact data, so no quadrature bias)."""
    px, py, dt = ray_points(theta, s)
    ix = np.floor((px + 1.0) / DELTA).astype(np.int64)
    iy = np.floor((py + 1.0) / DELTA).astype(np.int64)
    valid = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)
    # map (ix,iy) -> column index of retained centers
    lut = -np.ones(NG * NG, np.int64)
    ci = np.round((CENTERS[:, 0] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    cj = np.round((CENTERS[:, 1] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    lut[ci * NG + cj] = np.arange(J)
    col = np.where(valid, lut[np.clip(ix, 0, NG - 1) * NG + np.clip(iy, 0, NG - 1)], -1)
    A = np.zeros((len(theta), J))
    rows = np.repeat(np.arange(len(theta)), NQ)
    m = col.ravel() >= 0
    np.add.at(A, (rows[m], col.ravel()[m]), dt)
    return A


def A_blob(theta, s, a):
    """Analytic x-ray transform: A_ij = p(dist from ray i to center j)."""
    ct, st = np.cos(theta), np.sin(theta)
    # signed distance of center j from ray i = <c_j,(ct,st)> - s_i
    d = CENTERS[None, :, 0] * ct[:, None] + CENTERS[None, :, 1] * st[:, None] - s[:, None]
    return blob_xray(np.abs(d), a)


# --------------------------------------------------------------- Gram / eval --
def fine_grid():
    g = (np.arange(FINE) + 0.5) * (2.0 / FINE) - 1.0
    X, Y = np.meshgrid(g, g, indexing="ij")
    return X, Y, (2.0 / FINE) ** 2


FX, FY, dA = fine_grid()


def gram_disjoint():
    return np.eye(J) * (DELTA ** 2)


def gram_blob(a):
    """G_jk = g(|c_j-c_k|); radial overlap g computed by 2D quadrature."""
    dmax = 2 * a
    dq = np.linspace(0, dmax, 241)
    h = min(DELTA / 12.0, a / 24.0)
    n = int(np.ceil(2 * a / h))
    u = (np.arange(-n, n + 1)) * h
    UX, UY = np.meshgrid(u, u, indexing="ij")
    b0 = blob(np.hypot(UX, UY), a)
    g = np.array([(b0 * blob(np.hypot(UX - d, UY), a)).sum() * h * h for d in dq])
    D = np.linalg.norm(CENTERS[:, None, :] - CENTERS[None, :, :], axis=-1)
    return np.interp(np.clip(D, 0, dmax), dq, g) * (D < dmax)


def eval_disjoint(c):
    ix = np.clip(np.floor((FX + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    iy = np.clip(np.floor((FY + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    lut = np.zeros(NG * NG)
    ci = np.round((CENTERS[:, 0] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    cj = np.round((CENTERS[:, 1] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    lut[ci * NG + cj] = c
    return lut[ix * NG + iy]


def eval_blob(c, a):
    out = np.zeros_like(FX)
    for j in range(J):
        r = np.hypot(FX - CENTERS[j, 0], FY - CENTERS[j, 1])
        msk = r < a
        if msk.any():
            out[msk] += c[j] * blob(r[msk], a)
    return out


PH = phantom(FX, FY)
PHNORM = np.sqrt((PH ** 2).sum() * dA)


def l2err(img):
    return np.sqrt(((img - PH) ** 2).sum() * dA) / PHNORM


# -------------------------------------------------------------- diagnostics --
def cond_report(A):
    sv = np.linalg.svd(A, compute_uv=False)
    smax, smin = sv[0], sv[-1]
    tol = max(A.shape) * np.finfo(float).eps * smax
    rank = int((sv > tol).sum())
    dead = int((np.abs(A).max(0) == 0).sum())
    return dict(smax=smax, smin=smin, condA=smax / max(smin, 1e-300),
                condAtA=(smax / max(smin, 1e-300)) ** 2, rank=rank,
                dead=dead, sv=sv)


def solve_and_score(A, b, evalfn, lam_grid):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    ub = U.T @ b
    # min-norm least squares (pseudo-inverse, tol-truncated)
    tol = max(A.shape) * np.finfo(float).eps * S[0]
    Si = np.where(S > tol, 1.0 / np.maximum(S, 1e-300), 0.0)
    c_mn = Vt.T @ (Si * ub)
    e_mn = l2err(evalfn(c_mn))
    # Tikhonov sweep -> best achievable
    best = (np.inf, None)
    for lam in lam_grid:
        c = Vt.T @ ((S / (S ** 2 + lam ** 2)) * ub)
        e = l2err(evalfn(c))
        if e < best[0]:
            best = (e, lam)
    return e_mn, best[0], best[1]


# ------------------------------------------------------------------- driver --
def run_regime(name, theta, s, a, noise_frac=0.0):
    b = exact_data(theta, s)
    if noise_frac > 0:
        b = b + noise_frac * b.std() * np.random.randn(len(b))
    Ad, Ab = A_disjoint(theta, s), A_blob(theta, s, a)
    Gd, Gb = gram_disjoint(), gram_blob(a)
    lam = np.concatenate([[0.0], np.logspace(-8, 1, 60)])
    rows = []
    for tag, A, G, ev in [("disjoint", Ad, Gd, eval_disjoint),
                          ("blob a=%.3f(%.1fD)" % (a, a / DELTA), Ab, Gb,
                           lambda c: eval_blob(c, a))]:
        r = cond_report(A)
        cg = np.linalg.cond(G)
        e_mn, e_tk, l_best = solve_and_score(A, b, ev, lam)
        rows.append((tag, cg, r, e_mn, e_tk, l_best))
    print("\n=== %s ===   rays I=%d, unknowns J=%d, I/J=%.2f, noise=%.0f%%"
          % (name, len(theta), J, len(theta) / J, 100 * noise_frac))
    print("%-22s %10s %8s %11s %11s %6s %6s %9s %9s %9s" %
          ("basis", "cond(G)", "sqrtcG", "cond(A)", "cond(AtA)", "rank", "dead",
           "err_minn", "err_tikh", "lam*"))
    for tag, cg, r, e_mn, e_tk, lb in rows:
        print("%-22s %10.3e %8.1f %11.3e %11.3e %6d %6d %9.4f %9.4f %9.2e" %
              (tag, cg, np.sqrt(cg), r["condA"], r["condAtA"], r["rank"],
               r["dead"], e_mn, e_tk, lb))
    return rows


def best_representation_error(a):
    """Floor: best possible L2 error of each basis given PERFECT data
    (least-squares fit of the continuous phantom on the fine grid)."""
    # disjoint: cell averages
    ix = np.clip(np.floor((FX + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    iy = np.clip(np.floor((FY + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    ci = np.round((CENTERS[:, 0] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    cj = np.round((CENTERS[:, 1] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    lut = -np.ones(NG * NG, np.int64)
    lut[ci * NG + cj] = np.arange(J)
    col = lut[ix.ravel() * NG + iy.ravel()]
    cd = np.zeros(J); cnt = np.zeros(J)
    m = col >= 0
    np.add.at(cd, col[m], PH.ravel()[m]); np.add.at(cnt, col[m], 1.0)
    cd /= np.maximum(cnt, 1)
    ed = l2err(eval_disjoint(cd))
    # blob: normal equations on the fine grid
    Phi = np.zeros((FINE * FINE, J))
    for j in range(J):
        r = np.hypot(FX - CENTERS[j, 0], FY - CENTERS[j, 1]).ravel()
        msk = r < a
        Phi[msk, j] = blob(r[msk], a)
    cb = np.linalg.lstsq(Phi, PH.ravel(), rcond=None)[0]
    eb = l2err(eval_blob(cb, a))
    return ed, eb


def whitened_cond(A, G):
    """Basis-INDEPENDENT conditioning of the FUNCTION-recovery problem.
    f = sum c_j phi_j, ||f||^2 = c^T G c. Put c = G^{-1/2} y so ||f||=||y||.
    Then data = (A G^{-1/2}) y, and cond(A G^{-1/2}) is the conditioning of
    recovering the FUNCTION, free of any coefficient-parameterisation artefact.
    Comparing cond(A) to cond(A G^{-1/2}) isolates the coefficient-vs-function gap.
    """
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, 1e-300)
    Gmh = V @ np.diag(w ** -0.5) @ V.T
    sv = np.linalg.svd(A @ Gmh, compute_uv=False)
    return sv[0] / max(sv[-1], 1e-300), sv


def uncovered_mask(theta, s):
    """Fine-grid mask of points no ray passes near (the truly unobserved set)."""
    ct, st = np.cos(theta), np.sin(theta)
    P = np.stack([FX.ravel(), FY.ravel()], 1)
    d = np.abs(P[:, None, 0] * ct[None, :] + P[:, None, 1] * st[None, :] - s[None, :])
    return (d.min(1) > 0.5 * DELTA).reshape(FX.shape) & (np.hypot(FX, FY) < 0.97)


def phantom_smooth(x, y):
    v = (0.8 * np.exp(-((x - 0.10) ** 2 + (y - 0.45) ** 2) / (2 * 0.20 ** 2))
         + 1.0 * np.exp(-((x + 0.30) ** 2 + (y - 0.15) ** 2) / (2 * 0.26 ** 2))
         + 0.6 * np.exp(-((x - 0.35) ** 2 + (y + 0.28) ** 2) / (2 * 0.18 ** 2)))
    return v * (np.hypot(x, y) < 0.97)


def phantom_pw(x, y):
    v = 1.0 * (((x + 0.30) ** 2 + (y - 0.15) ** 2) < 0.32 ** 2)
    v += 0.6 * (((x - 0.35) ** 2 + (y + 0.28) ** 2) < 0.22 ** 2)
    v += 0.9 * ((np.abs(x - 0.10) < 0.20) & (np.abs(y - 0.45) < 0.16))
    return v * (np.hypot(x, y) < 0.97)


def run_phantom(pf, name, theta, s, a):
    """Re-run the matched comparison against a chosen phantom (breaks the
    smooth-vs-discontinuous confound). Also reports error inside the
    NEVER-OBSERVED region only, and the representation floor for each basis."""
    global PH, PHNORM
    PH_old, PHN_old = PH, PHNORM
    PH = pf(FX, FY); PHNORM = np.sqrt((PH ** 2).sum() * dA)
    px, py, dt = ray_points(theta, s)
    b = PH_ray(pf, theta, s)
    unc = uncovered_mask(theta, s)
    # normalise by the GLOBAL phantom norm: err_UNCOV is then the share of the
    # total error contributed by the never-observed region (uncovered support
    # is mostly background, so a local normalisation is degenerate).
    nrm_u = PHNORM
    lam = np.concatenate([[0.0], np.logspace(-8, 1, 60)])
    out = []
    for tag, A, G, ev in [("disjoint", A_disjoint(theta, s), gram_disjoint(), eval_disjoint),
                          ("blob(2D)", A_blob(theta, s, a), gram_blob(a),
                           lambda c: eval_blob(c, a))]:
        # representation floor
        if tag == "disjoint":
            fl = floor_disjoint()
        else:
            fl = floor_blob(a)
        cw, _ = whitened_cond(A, G)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        ub = U.T @ b
        best = (np.inf, None, None)
        for l in lam:
            c = Vt.T @ ((S / (S ** 2 + l ** 2)) * ub)
            img = ev(c); e = l2err(img)
            if e < best[0]:
                eu = np.sqrt(((img - PH)[unc] ** 2).sum() * dA) / max(nrm_u, 1e-12)
                best = (e, eu, l)
        out.append((tag, np.linalg.cond(A), cw, fl, best[0], best[1]))
    print("\n--- phantom: %s   (uncovered area = %.1f%% of disk) ---"
          % (name, 100 * unc.sum() / max((np.hypot(FX, FY) < 0.97).sum(), 1)))
    print("%-10s %11s %11s %9s %9s %11s %11s" %
          ("basis", "cond(A)", "cond(AG^-1/2)", "floor", "err_all", "excess",
           "err_UNCOV"))
    for tag, ca, cw, fl, e, eu in out:
        print("%-10s %11.3e %11.3e %9.4f %9.4f %11.4f %11.4f"
              % (tag, ca, cw, fl, e, np.sqrt(max(e * e - fl * fl, 0)), eu))
    PH, PHNORM = PH_old, PHN_old
    return out


def PH_ray(pf, theta, s):
    px, py, dt = ray_points(theta, s)
    return pf(px, py).sum(1) * dt


def floor_disjoint():
    ix = np.clip(np.floor((FX + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    iy = np.clip(np.floor((FY + 1.0) / DELTA).astype(np.int64), 0, NG - 1)
    ci = np.round((CENTERS[:, 0] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    cj = np.round((CENTERS[:, 1] + 1.0 - 0.5 * DELTA) / DELTA).astype(np.int64)
    lut = -np.ones(NG * NG, np.int64); lut[ci * NG + cj] = np.arange(J)
    col = lut[ix.ravel() * NG + iy.ravel()]
    cd = np.zeros(J); cnt = np.zeros(J); m = col >= 0
    np.add.at(cd, col[m], PH.ravel()[m]); np.add.at(cnt, col[m], 1.0)
    return l2err(eval_disjoint(cd / np.maximum(cnt, 1)))


def floor_blob(a):
    Phi = np.zeros((FINE * FINE, J))
    for j in range(J):
        r = np.hypot(FX - CENTERS[j, 0], FY - CENTERS[j, 1]).ravel()
        msk = r < a
        Phi[msk, j] = blob(r[msk], a)
    cb = np.linalg.lstsq(Phi, PH.ravel(), rcond=None)[0]
    return l2err(eval_blob(cb, a))


if __name__ == "__main__":
    a2 = 2.0 * DELTA
    print("grid %dx%d, DELTA=%.4f, J=%d unknowns, blob alpha=%.1f m=%d"
          % (NG, NG, DELTA, J, ALPHA, M_ORDER))
    print("Eq.(A7) x-ray transform vs direct quadrature, max rel err: %.2e"
          % verify_xray(a2))
    ed, eb = best_representation_error(a2)
    print("BEST-CASE representation error (perfect data): disjoint %.4f  blob %.4f"
          % (ed, eb))

    # A) well-determined, full angular coverage
    th, sd = make_rays(60, 41, 0.0, np.pi, 1.0)
    run_regime("A well-determined, full coverage", th, sd, a2)

    # B) under-determined, POOR coverage: limited angle + truncated detector
    th, sd = make_rays(9, 25, 0.0, np.pi / 3, 0.62)
    Ad = A_disjoint(th, sd)
    print("\n[poor-coverage geometry: %d/%d disjoint cells NEVER touched = %.1f%%]"
          % ((np.abs(Ad).max(0) == 0).sum(), J,
             100 * (np.abs(Ad).max(0) == 0).sum() / J))
    run_regime("B under-determined, POOR coverage (clean)", th, sd, a2)
    run_regime("B under-determined, POOR coverage (2% noise)", th, sd, a2, 0.02)
    run_regime("B under-determined, POOR coverage (10% noise)", th, sd, a2, 0.10)

    # C) overlap sweep: does cond(G) blow up with overlap, as our note claims?
    print("\n=== C overlap sweep (cond(G) vs blob radius), poor-coverage A ===")
    print("%-10s %12s %12s %12s %8s %10s" %
          ("a/DELTA", "cond(G)", "cond(AtA)", "smin(A)", "dead", "err_tikh"))
    lam = np.concatenate([[0.0], np.logspace(-8, 1, 60)])
    b = exact_data(th, sd)
    for f in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        a = f * DELTA
        Ab = A_blob(th, sd, a)
        r = cond_report(Ab)
        cg = np.linalg.cond(gram_blob(a))
        _, e_tk, _ = solve_and_score(Ab, b, lambda c, a=a: eval_blob(c, a), lam)
        print("%-10.1f %12.3e %12.3e %12.3e %8d %10.4f"
              % (f, cg, r["condAtA"], r["smin"], r["dead"], e_tk))

    # D) coefficient-vs-function gap: is cond(A) an honest number for blobs?
    print("\n=== D coefficient-vs-function conditioning gap (poor coverage) ===")
    print("%-10s %13s %15s %10s %12s" %
          ("basis", "cond(A) coef", "cond(AG^-1/2) fn", "ratio", "sqrt(cond G)"))
    for tag, A, G in [("disjoint", A_disjoint(th, sd), gram_disjoint()),
                      ("blob 2D", A_blob(th, sd, a2), gram_blob(a2)),
                      ("blob 4D", A_blob(th, sd, 4 * DELTA), gram_blob(4 * DELTA))]:
        ca = np.linalg.cond(A)
        cw, _ = whitened_cond(A, G)
        print("%-10s %13.3e %15.3e %10.3f %12.1f"
              % (tag, ca, cw, ca / cw, np.sqrt(np.linalg.cond(G))))
    # effective rank of the FUNCTION-recovery operator: how many independent
    # function-space directions the geometry actually constrains. This is
    # basis-independent and far more meaningful than cond() when both are
    # numerically singular.
    print("\n  effective rank of A*G^-1/2 (basis-independent), J=%d" % J)
    print("  %-10s %10s %10s %10s %10s" %
          ("basis", "s>1e-2", "s>1e-3", "s>1e-4", "s>1e-6"))
    for tag, A, G in [("disjoint", A_disjoint(th, sd), gram_disjoint()),
                      ("blob 2D", A_blob(th, sd, a2), gram_blob(a2)),
                      ("blob 4D", A_blob(th, sd, 4 * DELTA), gram_blob(4 * DELTA))]:
        _, sv = whitened_cond(A, G)
        rel = sv / sv[0]
        print("  %-10s %10d %10d %10d %10d" %
              (tag, (rel > 1e-2).sum(), (rel > 1e-3).sum(),
               (rel > 1e-4).sum(), (rel > 1e-6).sum()))

    # E) break the smooth-vs-discontinuous confound + error in UNOBSERVED region
    print("\n=== E confound control: same geometry, three phantoms ===")
    for pf, nm in [(phantom_smooth, "SMOOTH (blob-favourable)"),
                   (phantom_pw, "PIECEWISE-CONSTANT (disjoint-favourable)"),
                   (phantom, "MIXED")]:
        u = uncovered_mask(th, sd); P = pf(FX, FY)
        print("  [signal mass in uncovered region: %.2f%% of ||f||]"
              % (100 * np.sqrt((P[u] ** 2).sum()) / np.sqrt((P ** 2).sum())))
        run_phantom(pf, nm, th, sd, a2)

    # F) THE decisive coverage test: uncovered region provably CONTAINS signal.
    # Full 180deg angular coverage but a truncated detector (|s|<0.45), so the
    # outer annulus r>0.45 is crossed by NO ray, and we place mass there.
    # Build the phantom FROM the provably dead cells of the regime-B geometry,
    # so the unobserved region certainly carries signal.
    thF, sdF = make_rays(9, 25, 0.0, np.pi / 3, 0.62)
    dead_idx = np.where(np.abs(A_disjoint(thF, sdF)).max(0) == 0)[0]
    DEADC = CENTERS[dead_idx]

    def phantom_annulus(x, y):
        r = np.hypot(x, y)
        v = 1.0 * (r < 0.35)                       # observed core
        for cx, cy in DEADC:                       # mass ON the dead cells
            v = v + 1.2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2)
                                 / (2 * (0.8 * DELTA) ** 2))
        return v * (r < 0.95)

    uF = uncovered_mask(thF, sdF); PF = phantom_annulus(FX, FY)
    AdF = A_disjoint(thF, sdF)
    print("\n=== F decisive coverage test: signal INSIDE the unobserved region ===")
    print("  dead disjoint cells %d/%d (%.1f%%); uncovered area %.1f%% of disk; "
          "signal mass there %.1f%% of ||f||"
          % ((np.abs(AdF).max(0) == 0).sum(), J,
             100 * (np.abs(AdF).max(0) == 0).sum() / J,
             100 * uF.sum() / (np.hypot(FX, FY) < 0.95).sum(),
             100 * np.sqrt((PF[uF] ** 2).sum()) / np.sqrt((PF ** 2).sum())))
    run_phantom(phantom_annulus, "ANNULUS (mass in unobserved rim)", thF, sdF, a2)
