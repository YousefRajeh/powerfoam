"""
test_nullspace_shuttle_dewit.py

Tests the null-space shuttle of de Wit, Trampert & van der Hilst (2012),
JGR Solid Earth 117, B03301, doi:10.1029/2011JB008754, against our setting:
a row-substochastic ray/cell operator WITH OCCLUSION and DEAD COLUMNS,
and a CLASSIFICATION output (argmax cosine of a per-cell feature vector
against class text embeddings) rather than a scalar wave-speed perturbation.

Their method (paper eqs. 1-7, section 3, p.3):
    m_t = m_t^range + m_t^null                      (1)
    G m_t = d_t,  G m_t^null = 0  =>  G m_t^range = d_t     (2),(3)
    m~_t^range = L d_t = R m_t                      (4)   [L = LSQR operator]
    m~_t^null  = m_t - m~_t^range = (I - R) m_t     (5)
    m~_new = m_d + alpha * m~_t^null                (6)
    dm     = m~_new - m_d = alpha (I-R) m_d         (7)   [test model m_t = m_d]
Robustness criterion (para [24],[25],[38]): a parameter is robust iff its SIGN
is the same in every model of the acceptable range Delta-m.

We check four things:
  A. Does the dead-cell set span null(A) exactly, or is null(A) larger?
  B. Does the LSQR-based (I-R) shuttle recover the dead cells by itself?
  C. For a classification output, what fraction of cells keep their label
     across the null-space family (label-stability = sign-stability analogue)?
  D. Does label-stability correlate with the ray-mass statistic (A^T 1)_j
     better or worse than rho = -0.740 (ray mass vs per-cell error)?

CPU only, seconds to run.
"""

import numpy as np
from scipy.linalg import null_space, svdvals
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 1. Build a small ray/cell system with occlusion and deliberately dead cells
# ----------------------------------------------------------------------------
NX, NY = 20, 20
NCELL = NX * NY
NRAY = 260
DEAD_BOX = (12, 20, 12, 20)   # x0,x1,y0,y1: region no ray is allowed to enter


def build_A():
    """Rays from the left/bottom borders; each ray accumulates transmittance
    and is TERMINATED once cumulative opacity passes 1 (occlusion), which makes
    A row-substochastic with disjoint per-cell contributions (telescoping
    transmittance), exactly as in our forward operator.
    A whole box of cells is made unreachable by construction (dead columns)."""
    rows, cols, vals = [], [], []
    r = 0
    while r < NRAY:
        # random ray: start on left edge or bottom edge, random direction
        if RNG.random() < 0.5:
            p = np.array([0.0, RNG.uniform(0, NY)])
            th = RNG.uniform(-0.9, 0.9)
        else:
            p = np.array([RNG.uniform(0, NX), 0.0])
            th = RNG.uniform(0.7, 2.4)
        d = np.array([np.cos(th), np.sin(th)])
        T = 1.0
        seen = {}
        for step in range(400):
            p = p + 0.25 * d
            ix, iy = int(p[0]), int(p[1])
            if not (0 <= ix < NX and 0 <= iy < NY):
                break
            if DEAD_BOX[0] <= ix < DEAD_BOX[1] and DEAD_BOX[2] <= iy < DEAD_BOX[3]:
                break                      # occluder / never sampled
            j = iy * NX + ix
            if j in seen:
                continue
            alpha = RNG.uniform(0.05, 0.25)
            w = T * alpha                  # telescoping transmittance weight
            seen[j] = w
            T *= (1 - alpha)
            if T < 0.02:                   # fully occluded -> ray ends
                break
        if len(seen) < 3:
            continue
        for j, w in seen.items():
            rows.append(r); cols.append(j); vals.append(w)
        r += 1
    A = np.zeros((NRAY, NCELL))
    A[np.array(rows), np.array(cols)] = np.array(vals)
    return A


A = build_A()
raymass = A.sum(axis=0)                    # (A^T 1)_j
dead = raymass == 0.0
live = ~dead
rowsums = A.sum(axis=1)

print("=" * 74)
print("SYSTEM")
print("=" * 74)
print(f"A shape                      : {A.shape}  ({NRAY} rays x {NCELL} cells)")
print(f"nnz / row (mean)             : {(A != 0).sum() / NRAY:.1f} cells per ray")
print(f"row sums in [{rowsums.min():.3f}, {rowsums.max():.3f}]  (substochastic: "
      f"{bool((rowsums <= 1 + 1e-12).all())})")
print(f"dead cells                   : {dead.sum()} / {NCELL} = "
      f"{100*dead.sum()/NCELL:.2f}%")
sv = svdvals(A)
print(f"singular values              : max {sv[0]:.4e}  min {sv[-1]:.4e}")
nz = sv[sv > 1e-12]
print(f"cond(A) incl. zeros          : inf (rank deficient)")
print(f"cond(A) on nonzero sv        : {nz[0]/nz[-1]:.4e}")

# ----------------------------------------------------------------------------
# 2. (A) Does the dead set span the null space exactly?
# ----------------------------------------------------------------------------
N = null_space(A)                          # NCELL x k, orthonormal columns
k = N.shape[1]
# energy of each null vector on dead coordinates
dead_energy = (N[dead, :] ** 2).sum()
tot_energy = (N ** 2).sum()
# dimension of the null space restricted to live coordinates:
N_live_block = null_space(A[:, live])
print()
print("=" * 74)
print("A. IS null(A) EXACTLY THE DEAD SET?")
print("=" * 74)
print(f"dim null(A)                          : {k}")
print(f"# dead cells                         : {dead.sum()}")
print(f"dim null(A restricted to live cols)  : {N_live_block.shape[1]}")
print(f"  -> null(A) = span(dead unit vecs) (+) null(A_live);  "
      f"{dead.sum()} + {N_live_block.shape[1]} = {dead.sum()+N_live_block.shape[1]}")
print(f"fraction of null-space energy on DEAD coords : "
      f"{dead_energy/tot_energy:.4f}")
print(f"fraction on LIVE coords                      : "
      f"{1-dead_energy/tot_energy:.4f}")
print("VERDICT: the null space is STRICTLY LARGER than the dead set whenever"
      " dim null(A_live) > 0.")

# ----------------------------------------------------------------------------
# 3. Ground truth features + classification setup
# ----------------------------------------------------------------------------
D = 8            # feature dim (CLIP analogue)
K = 4            # number of text classes


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


C = unit(RNG.normal(size=(K, D)))                   # class "text embeddings"
gt_label = RNG.integers(0, K, size=NCELL)
# spatially smooth the labels a bit so it looks like a scene
lab2d = gt_label.reshape(NY, NX)
for _ in range(3):
    lab2d = np.round(
        0.5 * lab2d + 0.5 * np.roll(lab2d, 1, axis=0)).astype(int) % K
gt_label = lab2d.reshape(-1)
F_true = unit(C[gt_label] + 0.25 * RNG.normal(size=(NCELL, D)))   # per-cell feat
B = A @ F_true                                                    # ray features


def classify(F):
    return np.argmax(unit(F) @ C.T, axis=1)


# ----------------------------------------------------------------------------
# 4. Reconstruct m_d the way a regularised solver would (LSQR, per channel)
# ----------------------------------------------------------------------------
Asp = csr_matrix(A)
DAMP = 1e-3
M_d = np.column_stack([lsqr(Asp, B[:, c], damp=DAMP, iter_lim=200)[0]
                       for c in range(D)])
misfit0 = np.linalg.norm(A @ M_d - B) / np.sqrt(B.size)
print()
print("=" * 74)
print("B. LSQR (I-R) SHUTTLE, AS ACTUALLY IMPLEMENTED IN THE PAPER (eq. 5)")
print("=" * 74)
print(f"RMS data misfit of m_d               : {misfit0:.3e}")
print(f"||m_d|| on dead cells                : "
      f"{np.linalg.norm(M_d[dead]):.3e}   (LSQR from 0 stays in row space)")

# Their shuttle: m~_t^null = (I - R) m_t  with m_t = m_d
Dt = A @ M_d                                        # synthetic data d_t
M_range = np.column_stack([lsqr(Asp, Dt[:, c], damp=DAMP, iter_lim=200)[0]
                           for c in range(D)])
M_null_approx = M_d - M_range                       # (I - R) m_d
print(f"||(I-R)m_d||                          : "
      f"{np.linalg.norm(M_null_approx):.3e}")
print(f"||A (I-R) m_d|| / ||A m_d||           : "
      f"{np.linalg.norm(A@M_null_approx)/np.linalg.norm(A@M_d):.3e}"
      "   (should be ~0 if it is really null)")

# The crucial dead-cell question, for a test model that IS nonzero on dead cells
M_t = M_d + 0.0
M_t[dead] = unit(RNG.normal(size=(dead.sum(), D)))  # plant structure on dead cells
Dt2 = A @ M_t
M_rng2 = np.column_stack([lsqr(Asp, Dt2[:, c], damp=DAMP, iter_lim=200)[0]
                          for c in range(D)])
M_null2 = M_t - M_rng2
recov = (np.linalg.norm(M_null2[dead] - M_t[dead]) /
         np.linalg.norm(M_t[dead]))
print(f"planted dead-cell structure recovered by (I-R): relative error "
      f"{recov:.3e}")
print("  -> the shuttle returns dead-cell content UNCHANGED: it re-identifies"
      " the dead set, it does not constrain it.")

# ----------------------------------------------------------------------------
# 5. (C) Label stability across the null-space family
# ----------------------------------------------------------------------------
# Two families:
#   (i)  EXACT null space, bounded by a MODEL-NORM budget beta (data misfit is
#        identically unchanged, so the data supplies NO bound at all).
#   (ii) the paper's (I-R) family, bounded by a DATA-MISFIT tolerance.
lab_d = classify(M_d)
norm_d = np.linalg.norm(M_d)


def family_exact(beta, ndraw=200):
    labs = []
    for _ in range(ndraw):
        Z = RNG.normal(size=(k, D))
        dM = N @ Z
        dM *= beta * norm_d / np.linalg.norm(dM)
        labs.append(classify(M_d + dM))
    return np.array(labs)


def family_paper(alphas):
    """de Wit eq. (6)/(7): m_new = m_d + alpha (I-R) m_d, alpha scanned."""
    labs, misfits = [], []
    for a in alphas:
        Mn = M_d + a * M_null_approx
        labs.append(classify(Mn))
        misfits.append(np.linalg.norm(A @ Mn - B) / np.sqrt(B.size))
    return np.array(labs), np.array(misfits)


print()
print("=" * 74)
print("C. LABEL STABILITY  (sign-stability analogue for a classifier)")
print("=" * 74)

alphas = np.linspace(-10, 10, 201)
labs_p, mis_p = family_paper(alphas)
# de Wit use +/-0.1 s on an RMS misfit of 1.46 s against a pre-inversion misfit
# of 1.92 s, i.e. a tolerance ~7% of the RMS DATA amplitude. We mirror that.
rms_b = np.linalg.norm(B) / np.sqrt(B.size)
for frac in (0.02, 0.07, 0.20):
    tol = frac * rms_b
    ok = np.abs(mis_p - misfit0) <= tol
    stable_p = (labs_p[ok] == lab_d).all(axis=0)
    print(f"paper (I-R) family, tol={frac:.2f}*RMS(b): "
          f"{ok.sum():3d}/{len(alphas)} models, alpha in "
          f"[{alphas[ok].min():+.1f},{alphas[ok].max():+.1f}]  "
          f"P_label all {100*stable_p.mean():5.2f}%  "
          f"live {100*stable_p[live].mean():5.2f}%  "
          f"dead {100*stable_p[dead].mean():5.2f}%")

print(f"  misfit over the whole alpha scan: [{mis_p.min():.3e},"
      f" {mis_p.max():.3e}] vs RMS(b) = {rms_b:.3e}"
      "  -> the data barely bound alpha at all")
print(f"  ||(I-R)m_d|| restricted to DEAD cells : "
      f"{np.linalg.norm(M_null_approx[dead]):.3e}   <-- the paper's family does"
      " NOT move dead cells at all when m_t = m_d, so dead cells are reported"
      " 100% 'sign-robust'. That is a FALSE PASS.")
print()
for beta in (0.05, 0.10, 0.25, 0.50):
    labs = family_exact(beta)
    stable = (labs == lab_d).all(axis=0)
    frac_live = stable[live].mean()
    frac_dead = stable[dead].mean()
    print(f"exact-null family, ||dm||={beta:4.2f}||m_d|| : "
          f"P_label all {100*stable.mean():5.2f}%  live {100*frac_live:5.2f}%  "
          f"dead {100*frac_dead:5.2f}%")

# ----------------------------------------------------------------------------
# 6. (D) Does label stability beat the ray-mass baseline?
# ----------------------------------------------------------------------------
BETA = 0.25
labs = family_exact(BETA, ndraw=400)
stab_cont = (labs == lab_d).mean(axis=0)     # continuous stability in [0,1]
err = np.linalg.norm(unit(M_d) - unit(F_true), axis=1)   # per-cell feature error
mis_gt = (classify(M_d) != gt_label).astype(float)

print()
print("=" * 74)
print("D. RAY-MASS BASELINE COMPARISON  (baseline to beat: rho = -0.740)")
print("=" * 74)


def rep(name, x, y, mask):
    r, p = spearmanr(x[mask], y[mask])
    print(f"  {name:<52s} rho = {r:+.3f}  (p={p:.1e}, n={mask.sum()})")


rep("raymass vs per-cell feature error, ALL cells", raymass, err,
    np.ones(NCELL, bool))
rep("raymass vs per-cell feature error, LIVE only", raymass, err, live)
rep("raymass vs shuttle label-stability, ALL cells", raymass, stab_cont,
    np.ones(NCELL, bool))
rep("raymass vs shuttle label-stability, LIVE only", raymass, stab_cont, live)
rep("shuttle label-stability vs per-cell error, ALL", stab_cont, err,
    np.ones(NCELL, bool))
rep("shuttle label-stability vs per-cell error, LIVE", stab_cont, err, live)
rep("shuttle label-stability vs misclassification, LIVE", stab_cont, mis_gt,
    live)

# does flagging unstable cells actually catch the errors?
for thr in (0.99, 0.9, 0.75):
    flag = stab_cont < thr
    if flag[live].sum() == 0:
        continue
    acc_flag = 1 - mis_gt[live & flag].mean()
    acc_ok = 1 - mis_gt[live & ~flag].mean() if (live & ~flag).sum() else np.nan
    print(f"  live cells flagged unstable (<{thr:.2f}): "
          f"{100*flag[live].mean():5.1f}%  accuracy {100*acc_flag:5.1f}%  vs "
          f"stable-cell accuracy {100*acc_ok:5.1f}%")

# Does stability add anything BEYOND ray mass? Partial correlation via ranks.
def partial_spearman(x, y, z, mask):
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v[mask]) for v in (x, y, z))
    rx, ry, rz = (v - v.mean() for v in (rx, ry, rz))
    bx = rx - rz * (rx @ rz) / (rz @ rz)
    by = ry - rz * (ry @ rz) / (rz @ rz)
    return (bx @ by) / np.sqrt((bx @ bx) * (by @ by))


print()
print("  INCREMENTAL VALUE OVER RAY MASS (live cells only):")
print(f"    partial rho(stability, error | raymass) = "
      f"{partial_spearman(stab_cont, err, raymass, live):+.3f}")
print(f"    partial rho(raymass, error | stability) = "
      f"{partial_spearman(raymass, err, stab_cont, live):+.3f}")

print()
print("=" * 74)
print("COST NOTE")
print("=" * 74)
print("Per shuttle model the paper needs ONE extra LSQR solve of the same size")
print("as the original inversion (eq. 4). Nothing else. No SVD anywhere.")
print(f"Here: {D} channels x 200 LSQR iters per shuttle model.")
