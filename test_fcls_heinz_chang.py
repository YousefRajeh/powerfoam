"""
test_fcls_heinz_chang.py

Reimplementation + verification of:
  D. C. Heinz and C.-I Chang, "Fully Constrained Least Squares Linear Spectral
  Mixture Analysis Method for Material Quantification in Hyperspectral Imagery",
  IEEE TGRS 39(3):529-545, 2001.

Methods implemented exactly as described in the paper:
  LS     -- unconstrained least squares, eq. (5)                (Sec. III-B, p.531)
  SCLS   -- sum-to-one constrained LS, closed form eq. (8)-(9)  (Sec. IV-A, p.531-532)
  NSCLS  -- SCLS, negatives zeroed, remainder renormalised      (Sec. V,    p.533)
  NCLS   -- nonnegativity constrained LS (NNLS), eq. (10)-(15)  (Sec. IV-B, p.532)
  NNCLS  -- NCLS renormalised to sum one                        (Sec. V,    p.533)
  FCLS   -- NCLS applied to the ASC-augmented signature matrix
            of eq. (17)-(18) with delta = 1e-5                  (Sec. V-A,  p.533)

Experiments:
  A. Paper Example 1 (Sec. VI-A, p.537): 3 endmembers, all present.
  B. Paper Example 2 (Sec. VI-A, p.538): 2 extra, spectrally similar endmembers
     that are present in M but have 0% abundance in EVERY pixel.
  C. NEW -- true rank-deficient regime: an endmember whose signature is
     unobservable (zero column in the data rows), i.e. a parameter with no
     supporting data at all.  This is the analogue of an unobserved cell.
  D. NEW -- transposed / per-parameter regime matching a lifting problem:
     one global unknown vector, each unknown observed by a variable number of
     rows, some unknowns observed by no row at all.

Run:  D:\\conda\\envs\\powerfoam\\python.exe D:\\Downloads\\powerfoam\\test_fcls_heinz_chang.py
CPU only, ~seconds.
"""
import numpy as np
from scipy.optimize import nnls

RNG = np.random.default_rng(0)
DELTA = 1e-5          # paper: "the value of delta used in (17) and (18) was
                      #  fixed at 1.0 x 10^-5" (Sec. VI, p.537)


# ----------------------------------------------------------------- estimators
def ls(M, x):
    """eq. (5): alpha_hat = (M^T M)^-1 M^T x"""
    return np.linalg.pinv(M) @ x


def scls(M, x):
    """eq. (8)-(9): unconstrained LS plus a sum-to-one correction term."""
    a = ls(M, x)
    G = np.linalg.pinv(M.T @ M)
    one = np.ones(M.shape[1])
    corr = G @ one * ((1.0 - one @ a) / (one @ G @ one))
    return a + corr


def nscls(M, x):
    a = scls(M, x).copy()
    a[a < 0] = 0.0
    s = a.sum()
    return a / s if s > 0 else a


def ncls(M, x):
    """eq. (10): min ||x - M a||^2 s.t. a >= 0.  Solved with NNLS ([30],[33])."""
    return nnls(M, x)[0]


def nncls(M, x):
    a = ncls(M, x)
    s = a.sum()
    return a / s if s > 0 else a


def fcls(M, x, delta=DELTA):
    """eq. (17)-(18): augment with the ASC row, then run the NCLS algorithm."""
    L, p = M.shape
    Mt = np.vstack([delta * M, np.ones((1, p))])
    xt = np.concatenate([delta * x, [1.0]])
    return nnls(Mt, xt)[0]


METHODS = [("LS", ls), ("SCLS", scls), ("NSCLS", nscls),
           ("NCLS", ncls), ("NNCLS", nncls), ("FCLS", fcls)]


# ------------------------------------------------------------------ spectra
def make_spectra(L=100):
    """Five smooth non-negative 'reflectance' spectra in [0,1].
    #2 (creosote-like), #3 (blackbrush-like) and #4 (sagebrush-like) are made
    deliberately similar, matching the paper's 'less spectrally distinct' pair.
    """
    w = np.linspace(0, 1, L)

    def band(c, s, a):
        return a * np.exp(-0.5 * ((w - c) / s) ** 2)

    dry_grass = 0.15 + band(0.30, 0.12, 0.35) + band(0.75, 0.20, 0.25)
    red_soil = 0.10 + band(0.85, 0.35, 0.55) + band(0.35, 0.25, 0.10)
    creosote = 0.08 + band(0.55, 0.10, 0.40) + band(0.20, 0.08, 0.10)
    blackbrush = creosote + 0.02 * np.sin(9 * np.pi * w) + 0.015
    sagebrush = creosote + 0.02 * np.cos(7 * np.pi * w) + 0.020
    S = np.stack([dry_grass, red_soil, creosote, blackbrush, sagebrush], 1)
    return np.clip(S, 1e-3, 1.0)


def spectral_angle(a, b):
    return np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def make_pixels(S, n=400, snr=30.0, seed=0):
    """Paper's simulation (Sec. VI-A, p.537): pixel 1 = 100% red soil,
    dry grass +0.25%/red soil -0.25% per pixel up to 100% dry grass at pixel
    400; creosote inserted at 10% in pixels 198-202 with the other two scaled
    by 90%.  White Gaussian noise for 30:1 SNR = 50% reflectance / sigma."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, 5))
    for i in range(n):
        g = i * 0.0025
        A[i, 0] = g
        A[i, 1] = 1.0 - g
    for i in range(197, 202):          # pixel numbers 198..202, 1-based
        A[i, :2] *= 0.9
        A[i, 2] = 0.10
    X = S @ A.T
    sigma = 0.5 / snr
    X = X + rng.normal(0, sigma, X.shape)
    return X, A


def run(M, X, A_true, target, cols, title):
    """Mean squared abundance error of `target` over all pixels."""
    n = X.shape[1]
    print("\n" + title)
    print("  signature matrix M: %d bands x %d signatures" % M.shape)
    out = {}
    for name, fn in METHODS:
        est = np.array([fn(M, X[:, i]) for i in range(n)])
        err = np.mean((est[:, cols.index(target)] - A_true[:, target]) ** 2)
        out[name] = err
        print("    %-6s mean squared abundance error = %.4e" % (name, err))
    best = min(out, key=out.get)
    print("    -> best: %s" % best)
    return out


def main():
    np.set_printoptions(precision=4, suppress=True)
    S = make_spectra()
    names = ["dry grass", "red soil", "creosote", "blackbrush", "sagebrush"]
    print("=" * 78)
    print("Spectral angles to creosote (deg): " + ", ".join(
        "%s %.2f" % (names[j], spectral_angle(S[:, 2], S[:, j]))
        for j in (0, 1, 3, 4)))
    X, A_true = make_pixels(S)

    # ---------------------------------------------------- A: paper Example 1
    e1 = run(S[:, :3], X, A_true, target=2, cols=[0, 1, 2],
             title="[A] Example 1 (p.537): M = 3 signatures, all present in pixels")

    # ---------------------------------------------------- B: paper Example 2
    e2 = run(S, X, A_true, target=2, cols=[0, 1, 2, 3, 4],
             title="[B] Example 2 (p.538): M = 5 signatures; blackbrush+sagebrush "
                   "have\n    0% abundance in ALL 400 pixels (over-complete endmember set)")

    print("\n  Degradation factor Example 1 -> Example 2 "
          "(how much the two never-present endmembers cost):")
    for name, _ in METHODS:
        print("    %-6s x%.1f" % (name, e2[name] / e1[name]))

    # ------------------------------------- C: truly unobservable endmember
    print("\n" + "=" * 78)
    print("[C] RANK-DEFICIENT REGIME (not in the paper): endmember #6 has a ZERO")
    print("    signature, i.e. its column of the data-fit design is identically 0.")
    print("    Nothing in the radiance data can ever say anything about it.")
    print("    (This is the analogue of a cell that no ray ever touches.)")
    S6 = np.hstack([S, np.zeros((S.shape[0], 1))])
    n = X.shape[1]
    for name, fn in METHODS:
        est = np.array([fn(S6, X[:, i]) for i in range(n)])
        ghost = est[:, 5]
        cre = np.mean((est[:, 2] - A_true[:, 2]) ** 2)
        print("    %-6s  ghost abundance: mean %+.4f  median %+.4f  min %+.4f  "
              "max %+.4f | creosote MSE %.3e"
              % (name, ghost.mean(), np.median(ghost), ghost.min(), ghost.max(), cre))
    print("    (true ghost abundance is undefined / unidentifiable; a graceful")
    print("     estimator would refuse or flag it, not return a confident number)")

    # rank check
    Mt = np.vstack([DELTA * S6, np.ones((1, 6))])
    print("    rank(M6 data rows) = %d of 6 ; rank(FCLS augmented M6) = %d of 6"
          % (np.linalg.matrix_rank(S6), np.linalg.matrix_rank(Mt)))
    print("    -> the ASC row is what restores rank: the augmented column for the")
    print("       unobservable endmember is [0...0, 1]^T, so the ASC makes it a")
    print("       SLACK VARIABLE that soaks up whatever mass the others leave.")

    # what does that do to the ASC's protective effect on the rest?
    est_f = np.array([fcls(S6, X[:, i]) for i in range(n)])
    est_f5 = np.array([fcls(S, X[:, i]) for i in range(n)])
    print("    FCLS creosote MSE with 5 real endmembers      : %.4e"
          % np.mean((est_f5[:, 2] - A_true[:, 2]) ** 2))
    print("    FCLS creosote MSE with 1 unobservable added   : %.4e"
          % np.mean((est_f[:, 2] - A_true[:, 2]) ** 2))
    print("    mean sum of the 5 REAL abundances under FCLS+ghost: %.4f "
          "(ASC on the real ones is destroyed)" % est_f[:, :5].sum(1).mean())

    # ---- C2: same, but the real abundances do NOT sum to one (shade / leak),
    #      which is exactly our row-SUBstochastic situation.
    print("\n[C2] Same unobservable endmember, but the true abundances of the real")
    print("     materials sum to 0.70 (30% 'shade'/leak) -- the substochastic case.")
    Xs = 0.70 * (S @ A_true.T) + RNG.normal(0, 0.5 / 30.0, (S.shape[0], n))
    est_c2 = np.array([fcls(S6, Xs[:, i]) for i in range(n)])
    print("     FCLS ghost abundance: mean %.4f  median %.4f  min %.4f  max %.4f"
          % (est_c2[:, 5].mean(), np.median(est_c2[:, 5]),
             est_c2[:, 5].min(), est_c2[:, 5].max()))
    print("     -> FCLS confidently assigns ~%.0f%% of every pixel to a material it"
          % (100 * est_c2[:, 5].mean()))
    print("        has, and can have, NO evidence about.  No warning, no flag.")
    est_c2r = np.array([fcls(S, Xs[:, i]) for i in range(n)])
    print("     creosote MSE, no ghost in M : %.4e" % np.mean((est_c2r[:, 2] - A_true[:, 2]) ** 2))
    print("     creosote MSE, ghost in M    : %.4e" % np.mean((est_c2[:, 2] - A_true[:, 2]) ** 2))

    # ---- C3: the ghost is a pure bookkeeping slack, independent of delta
    print("\n[C3] Is the ghost an estimate?  Sweep delta (eq. 17-18) on one pixel:")
    i0 = 100
    print("     %-10s %-12s %-14s" % ("delta", "ghost alpha", "data residual"))
    for d in (1e-7, 1e-5, 1e-3, 1e-1, 1e0, 1e1):
        a = fcls(S6, Xs[:, i0], delta=d)
        print("     %-10.0e %-12.4f %-14.4f"
              % (d, a[5], np.linalg.norm(Xs[:, i0] - S6 @ a)))
    print("     -> invariant: the ghost is exactly 1 - sum(real alphas).  It is pure")
    print("        mass bookkeeping, carrying zero information about that material.")

    # ---- C4: TWO unobservable endmembers -> the split between them is arbitrary
    print("\n[C4] TWO unobservable endmembers (our case: 9.4%% of cells are dead,")
    print("     not one).  Data cannot distinguish them at all.")
    S7 = np.hstack([S, np.zeros((S.shape[0], 2))])
    e7 = np.array([fcls(S7, Xs[:, i]) for i in range(n)])
    print("     ghost A: mean %.4f   ghost B: mean %.4f   (sum %.4f)"
          % (e7[:, 5].mean(), e7[:, 6].mean(), (e7[:, 5] + e7[:, 6]).mean()))
    print("     per-pixel |ghostA - ghostB|: mean %.4f  max %.4f"
          % (np.abs(e7[:, 5] - e7[:, 6]).mean(), np.abs(e7[:, 5] - e7[:, 6]).max()))
    print("     -> the TOTAL unobserved mass is pinned by the ASC; its DIVISION among")
    print("        the unobserved parameters is decided by the NNLS pivoting rule,")
    print("        not by data.  Any split summing to the same total fits identically:")
    a0 = e7[100].copy()
    r0 = np.linalg.norm(Xs[:, 100] - S7 @ a0)
    alt = a0.copy()
    t = alt[5] + alt[6]
    alt[5], alt[6] = 0.0, t
    print("        residual of the FCLS answer %.6f vs residual of an arbitrary "
          "re-split %.6f" % (r0, np.linalg.norm(Xs[:, 100] - S7 @ alt)))

    # ------------------------------------- D: transposed / lifting structure
    print("\n" + "=" * 78)
    print("[D] LIFTING STRUCTURE (each parameter is its own unknown, one global")
    print("    coupled system).  b_r = sum_j A_rj f_j, f_j in R^D, A row-substochastic.")
    D, Ncell, Nray = 8, 60, 300
    rngd = np.random.default_rng(3)
    # ground-truth cell features on the unit sphere (CLIP-like)
    F = rngd.normal(size=(Ncell, D))
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    A = np.zeros((Nray, Ncell))
    for r in range(Nray):
        k = rngd.integers(3, 8)
        cells = rngd.choice(Ncell - 6, size=k, replace=False)  # last 6 unobserved
        wts = rngd.random(k)
        wts = wts / wts.sum() * rngd.uniform(0.7, 1.0)         # substochastic
        A[r, cells] = wts
    B = A @ F
    colsum = A.sum(0)
    dead = np.where(colsum == 0)[0]
    print("    cells never touched by any ray: %d of %d" % (len(dead), Ncell))
    print("    rank(A) = %d, cond of Gram on live cells vs full:" % np.linalg.matrix_rank(A))
    G = A.T @ A
    live = colsum > 0
    print("      cond(A^T A) full  = %.3e" % np.linalg.cond(G))
    print("      cond(A^T A) live  = %.3e" % np.linalg.cond(G[np.ix_(live, live)]))

    with np.errstate(invalid="ignore", divide="ignore"):
        F_bp = (A.T @ B) / colsum[:, None]
    err_live = np.linalg.norm(F_bp[live] - F[live], axis=1)
    print("    column-normalised back-projection f_j = (A^T b)_j / (A^T 1)_j :")
    print("      live cells: mean ||f_hat - f|| = %.4f ; unobserved cells: %s"
          % (err_live.mean(), "NaN (0/0) -- the estimator itself refuses"))
    print("    NOTE the index: the weights that sum to one here are w_rj = A_rj /")
    print("    sum_r A_rj, i.e. a sum over RAYS for a fixed CELL.  FCLS's ASC is")
    print("    sum over ENDMEMBERS for a fixed PIXEL.  These are transposes of one")
    print("    another; they are not the same constraint.")

    # what if we impose FCLS's ASC on the lifting problem, per ray?
    print("\n    Row-sums of A (the 'ASC' of the forward model): min %.3f max %.3f"
          % (A.sum(1).min(), A.sum(1).max()))
    print("    -> already satisfied by construction of A; it is a property of the")
    print("       KNOWN operator, not a constraint available to impose on the")
    print("       UNKNOWN f.  Imposing it buys nothing.")
    print("=" * 78)


if __name__ == "__main__":
    main()
