"""
test_sqs_majorizer_erdogan_fessler.py

Numerical verification of the separable-surrogate (SQS) majorizer as it appears in
  H. Erdogan & J. A. Fessler, "Ordered subsets algorithms for transmission tomography",
  Phys. Med. Biol. 44 (1999) 2835-2851.

The paper does NOT state a matrix (Loewner) inequality. It states a *function*
majorization (eq. 7, p.2839) obtained by Jensen/De Pierro convexity with weights
    alpha_ij = a_ij / gamma_i,   gamma_i = sum_k a_ik      ("additive form", p.2839)
and the resulting separable curvature (eq. 12, p.2841)
    d_j = sum_i a_ij * gamma_i * c_i.

Specializing to a quadratic h_i (c_i == 1) this is exactly
    d_j = sum_i a_ij * sum_k a_ik = (A^T A 1)_j,
i.e. D = diag(A^T A 1), and the function inequality is equivalent to the Loewner
inequality  A^T A  <=  diag(A^T A 1).  This script tests that Loewner statement
directly, and the weighted version A^T C A <= diag(A^T C A 1) with C = diag(c), c>=0.

Focus case: COLUMNS OF A THAT ARE ENTIRELY ZERO (unobserved cells) and
ROWS OF A THAT ARE ENTIRELY ZERO (rays that touch nothing) -- the two ways
gamma_i = 0 or d_j = 0 can arise and break the preconditioner D^{-1}.

Run:  D:\conda\envs\powerfoam\python.exe D:\Downloads\powerfoam\test_sqs_majorizer_erdogan_fessler.py
CPU only, seconds, no GPU.
"""

import numpy as np

rng = np.random.default_rng(20260824)


def min_eig_gap(A, c=None):
    """min eigenvalue of D - S where S = A^T diag(c) A, D = diag(S 1)."""
    A = np.asarray(A, dtype=np.float64)
    if c is None:
        S = A.T @ A
    else:
        S = A.T @ (np.asarray(c, dtype=np.float64)[:, None] * A)
    d = S.sum(axis=1)
    G = np.diag(d) - S
    w = np.linalg.eigvalsh(0.5 * (G + G.T))
    return w.min(), S, d, G


def report(name, A, c=None, extra=""):
    lam, S, d, G = min_eig_gap(A, c)
    scale = max(np.abs(S).max(), 1e-300)
    print(f"{name:52s} min_eig(D-S) = {lam: .6e}   (rel {lam/scale: .3e})  "
          f"n_zero_d = {int((d <= 0).sum())}  {extra}")
    return lam, d


def substochastic_rows(n, p, density=1.0, rowmax=1.0):
    """Non-negative, row sums <= rowmax (row-substochastic scaled by rowmax)."""
    A = rng.random((n, p))
    if density < 1.0:
        A *= (rng.random((n, p)) < density)
    s = A.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    # random total mass in (0, rowmax]
    A = A / s * (rowmax * rng.random((n, 1)))
    return A


print("=" * 100)
print("PART 1 -- random row-substochastic A, unweighted (c = 1): is diag(A^T A 1) - A^T A PSD?")
print("=" * 100)
worst = np.inf
for trial in range(300):
    n = int(rng.integers(3, 60))
    p = int(rng.integers(2, 40))
    dens = float(rng.choice([1.0, 0.6, 0.25, 0.08]))
    A = substochastic_rows(n, p, density=dens)
    lam, _, S, _ = (None, None, None, None)
    lam, S, d, G = min_eig_gap(A)
    worst = min(worst, lam / max(np.abs(S).max(), 1e-300))
print(f"300 random draws: worst RELATIVE min eigenvalue of (D - S) = {worst: .3e}")
print(f"  -> PSD holds (>= -1e-12 numerically): {worst > -1e-12}")

print()
print("=" * 100)
print("PART 2 -- adversarial structures")
print("=" * 100)

# (a) each row all mass on a single column  -> S diagonal, D = S, gap exactly 0
A = np.zeros((12, 6))
for i in range(12):
    A[i, rng.integers(0, 6)] = rng.random()
report("(a) every row concentrated on ONE column", A,
       extra="expect gap == 0 exactly (S already diagonal)")

# (b) wildly different row sparsity: one dense row + many 1-sparse rows
A = np.zeros((20, 10))
A[0] = rng.random(10) / 10.0
for i in range(1, 20):
    A[i, rng.integers(0, 10)] = rng.random()
report("(b) one dense row + 19 one-sparse rows", A)

# (c) wildly different magnitudes: row sums spanning 1e-8 .. 1
A = substochastic_rows(30, 12)
A *= np.geomspace(1e-8, 1.0, 30)[:, None]
report("(c) row masses spanning 1e-8..1", A)

# (d) two identical rows (max collinearity)
A = substochastic_rows(2, 5)
A = np.vstack([A[0], A[0], A[1]])
report("(d) duplicated rows (collinear)", A)

# (e) rank-1 A: all rows proportional
v = rng.random(8); v /= v.sum()
A = rng.random((25, 1)) * v[None, :]
report("(e) rank-1 A (all rows proportional)", A)

# (f) A with a hard "telescoping transmittance" flavour: row = alpha_k * T_k, sum < 1
def telescoping_row(p, ncells=8):
    row = np.zeros(p)
    idx = rng.choice(p, size=ncells, replace=False)
    alpha = rng.random(ncells) * 0.5
    T = 1.0
    for k, j in enumerate(idx):
        row[j] = alpha[k] * T
        T *= (1.0 - alpha[k])
    return row  # sum = 1 - T_final <= 1

A = np.array([telescoping_row(30, 8) for _ in range(60)])
report("(f) telescoping alpha*T rows (sum = 1 - T_final <= 1)", A,
       extra=f"max row sum = {A.sum(1).max():.4f}")

print()
print("=" * 100)
print("PART 3 -- THE CASE WE CARE ABOUT: ZERO COLUMNS (unobserved cells)")
print("=" * 100)

p = 20
n = 50
A = substochastic_rows(n, p, density=0.4)
dead = np.array([3, 7, 11, 12, 19])
A[:, dead] = 0.0          # 5 of 20 cells never touched by any ray
lam, S, d, G = min_eig_gap(A)
print(f"A has {len(dead)} zero columns out of {p}")
print(f"  min_eig(D - S)                 = {lam: .6e}   -> majorizer STILL holds (PSD)")
print(f"  d_j on dead columns            = {d[dead]}")
print(f"  d_j min over live columns      = {d[np.setdiff1d(np.arange(p), dead)].min(): .6e}")
print(f"  rank(A)                        = {np.linalg.matrix_rank(A)}  (of p={p})")
print(f"  eigenvalues of S=A^T A, smallest 7: {np.sort(np.linalg.eigvalsh(S))[:7]}")
print()
print("  What the PRECONDITIONER does there:")
with np.errstate(divide="ignore", invalid="ignore"):
    Dinv = 1.0 / d
print(f"    1/d_j on dead columns          = {Dinv[dead]}   <-- DIVISION BY ZERO")
print("    The SQS update mu_j := mu_j - grad_j / d_j is 0/0 on a dead cell:")
grad = A.T @ rng.random(n)          # gradient contribution from data
print(f"    numerator grad_j on dead cols  = {grad[dead]}   (exactly 0: A^T anything is 0 there)")
print("    => 0/0. The algorithm does not move dead cells at all; they are FROZEN at")
print("       their initial value. SQS is silent/undefined there, not stabilising.")

# Verify the D^{-1/2} scaling claim on the LIVE submatrix only
live = np.setdiff1d(np.arange(p), dead)
Sl = S[np.ix_(live, live)]
dl = Sl.sum(axis=1)
M = (Sl / np.sqrt(dl)[:, None]) / np.sqrt(dl)[None, :]
ev = np.linalg.eigvalsh(M)
print()
print(f"  After diag(d)^{{-1/2}} scaling restricted to LIVE cells:")
print(f"    lambda_max(D^-1/2 S D^-1/2)  = {ev.max(): .6f}   (claim: <= 1)")
print(f"    lambda_min                    = {ev.min(): .6e}")
print(f"    cond                          = {ev.max()/max(ev.min(),1e-300): .6e}")
print("  Note: lambda_max <= 1 is scale-INVARIANT and says nothing about lambda_min.")
print("  The dead cells contribute a 0/0 entry, not a large condition number.")

print()
print("=" * 100)
print("PART 4 -- ZERO ROWS (rays that hit nothing): gamma_i = 0 breaks alpha_ij = a_ij/gamma_i")
print("=" * 100)
A = substochastic_rows(20, 8, density=0.5)
A[4] = 0.0
A[11] = 0.0
gamma = A.sum(axis=1)
print(f"  gamma_i for the two empty rays = {gamma[[4, 11]]}")
with np.errstate(divide="ignore", invalid="ignore"):
    alpha = A / gamma[:, None]
print(f"  alpha_ij on empty rows         = {alpha[4][:4]} ... (0/0 = nan)")
lam, S, d, G = min_eig_gap(A)
print(f"  but min_eig(D - S)             = {lam: .6e}  -> the MATRIX inequality is unharmed")
print("  (an all-zero row contributes nothing to S or D; the 0/0 is only in the")
print("   intermediate De Pierro weights, eq. (6) p.2839 requires sum_j alpha_ij = 1)")

print()
print("=" * 100)
print("PART 5 -- weighted version A^T diag(c) A <= diag(A^T diag(c) A 1), c >= 0")
print("  (this is the paper's actual d_j = sum_i a_ij gamma_i c_i, eq. 12 p.2841)")
print("=" * 100)
worst = np.inf
neg_c_broke = 0
for trial in range(200):
    n = int(rng.integers(4, 50)); p = int(rng.integers(2, 30))
    A = substochastic_rows(n, p, density=float(rng.choice([1.0, 0.4, 0.15])))
    c = rng.random(n) * float(rng.choice([1.0, 1e3, 1e-3]))
    if rng.random() < 0.3:
        c[rng.random(n) < 0.4] = 0.0     # zero curvature rays (paper's y_i <= r_i footnote, p.2843)
    lam, S, d, G = min_eig_gap(A, c)
    worst = min(worst, lam / max(np.abs(S).max(), 1e-300))
print(f"200 draws, c >= 0 (incl. c_i = 0 rays): worst relative min_eig(D-S) = {worst: .3e}")
print(f"  -> holds: {worst > -1e-12}")

# counterexample: NEGATIVE curvature breaks it (why the paper takes [.]_+ in eq. 4)
bad = 0
for trial in range(200):
    n, p = 10, 5
    A = substochastic_rows(n, p)
    c = rng.normal(size=n)   # allow negative
    lam, S, d, G = min_eig_gap(A, c)
    if lam < -1e-10 * max(np.abs(S).max(), 1e-300):
        bad += 1
print(f"200 draws with SIGNED c: majorizer FAILS in {bad}/200 cases")
print("  -> this is exactly why eq. (4) p.2839 wraps the curvature in [ . ]_+")

print()
print("=" * 100)
print("PART 6 -- is diag(A^T A 1) actually TIGHT? compare to the trivial alternative")
print("  bound lambda_max(A^T A) * I, and to diag(S) (Jacobi).")
print("=" * 100)
for name, Afun in [("dense substochastic", lambda: substochastic_rows(40, 15, 1.0)),
                   ("sparse (8/58-ish) substochastic", lambda: substochastic_rows(200, 60, 8 / 60)),
                   ("telescoping rows", lambda: np.array([telescoping_row(60, 8) for _ in range(200)]))]:
    A = Afun()
    S = A.T @ A
    d = S.sum(1)
    dj = np.diag(S)
    lam_sqs, _, _, _ = min_eig_gap(A)
    lam_jac = np.linalg.eigvalsh(np.diag(dj) - S).min()
    print(f"  {name:34s}  trace(D_SQS)={d.sum():.4f}  trace(D_Jacobi)={dj.sum():.4f}  "
          f"ratio={d.sum()/dj.sum():.3f}  min_eig(Jacobi gap)={lam_jac: .3e}")
print("  (Jacobi diag(S) is NOT a valid majorizer in general -- negative gaps above")
print("   confirm it; SQS trace tells you how much slack SQS pays for validity.)")
print()
print("=" * 100)
print("PART 7 -- THE PAPER'S ACTUAL ANSWER TO d_j = 0: the PENALTY denominator.")
print("  Erdogan & Fessler p.2841, just after eq. (11):")
print("     D_jj = d_j^n + 2*beta*sum_k w_jk")
print("  So in the PENALIZED case the SQS denominator on an unobserved cell is")
print("  2*beta*sum_k w_jk > 0, NOT zero.  ML (beta=0, eq. 24 p.2844) is 0/0.")
print("=" * 100)

p = 20
A = substochastic_rows(50, p, density=0.4)
dead = np.array([3, 7, 11, 12, 19])
A[:, dead] = 0.0
S = A.T @ A
d = S.sum(axis=1)

# 1-D chain neighbourhood, w_jk = 1 -> graph Laplacian L (connected)
W = np.zeros((p, p))
for j in range(p - 1):
    W[j, j + 1] = W[j + 1, j] = 1.0
L = np.diag(W.sum(1)) - W          # Hessian of R = 1/2 sum_j sum_k w_jk (mu_j-mu_k)^2 / ... quadratic case

for beta in [0.0, 1e-4, 1e-2, 1.0]:
    Djj = d + 2.0 * beta * W.sum(1)
    H = S + beta * L               # exact PL Hessian (quadratic penalty)
    ev = np.linalg.eigvalsh(H)
    print(f"  beta={beta:<8g} D_jj on dead cells = {Djj[dead]}")
    print(f"  {'':14s} lambda_min(A^T A + beta*L) = {ev[0]: .6e}   "
          f"cond = {ev[-1]/max(ev[0], 1e-300):.4e}")
print()
print("  Loewner check that D_PL = diag(d + 2 beta sum_k w_jk) still majorizes the PL Hessian:")
for beta in [1e-4, 1e-2, 1.0]:
    Djj = d + 2.0 * beta * W.sum(1)
    H = S + beta * L
    lam = np.linalg.eigvalsh(np.diag(Djj) - H).min()
    print(f"    beta={beta:<8g} min_eig(D_PL - H) = {lam: .6e}   -> majorizer holds: {lam > -1e-10}")

print()
print("  Rank argument: null(A^T A) contains the 5 dead unit vectors; null(L) = span(1)")
print("  for a CONNECTED neighbourhood graph. The intersection is {0} because 1 is not")
print("  supported only on dead cells. Hence A^T A + beta L is positive definite for any")
print("  beta > 0 -- rank deficiency is removed by the prior, NOT by the preconditioner.")
Ldis = L.copy()                     # now DISCONNECT the dead cells from the graph
for j in dead:
    Ldis[j, :] = 0.0; Ldis[:, j] = 0.0
Hdis = S + 1.0 * Ldis
print(f"  Control: if dead cells are DISCONNECTED from the penalty graph,")
print(f"    lambda_min(A^T A + L_disconnected) = {np.linalg.eigvalsh(Hdis)[0]: .6e}  (singular again)")
print("  => connectivity of the neighbourhood graph TO the dead cells is the load-bearing")
print("     condition, not the penalty strength.")

print()
print("DONE.")
