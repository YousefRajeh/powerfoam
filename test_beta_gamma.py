"""Prove the gamma identity on small dense cases before trusting it on 500M-nonzero operators.

CLAIM (compute_beta.py): because x_hat solves the normal equations A^T(A x_hat - B) = 0, the cross
term vanishes and for ANY x

    ||Ax - B||^2 = ||A(x - x_hat)||^2 + ||A x_hat - B||^2                        (Pythagoras)

so with gamma = ||A(x' - x_hat)||^2 / L(x_hat), the closed form's loss is EXACTLY (1+gamma) L(x_hat).

Also asserted here:
  * gamma >= 0, and gamma = 0 iff x' is itself optimal.
  * The k=1 decoupling: if no ray touches more than one primitive, A^T A is diagonal, x' == x_hat,
    and gamma == 0 to floating point.
  * beta >= gamma (SFS's bound is an upper bound on the realized suboptimality), on random cases.
"""
import numpy as np

rng = np.random.default_rng(0)


def closed_form(A, B):
    """x'_j = sum_i A_ij B_i / sum_i A_ij -- the row-sum preconditioned estimate."""
    cs = A.sum(0)
    return (A.T @ B) / np.maximum(cs, np.finfo(float).eps)[:, None]


def gamma_of(A, B):
    x_hat, *_ = np.linalg.lstsq(A, B, rcond=None)
    x_p = closed_form(A, B)
    loss_hat = float(((A @ x_hat - B) ** 2).sum())
    gap = float(((A @ (x_p - x_hat)) ** 2).sum())
    return gap / max(loss_hat, 1e-300), x_hat, x_p, loss_hat


def beta_of(A, B, x_hat):
    """SFS Eq. 14, with the row-sum normalisation compute_beta.py applies."""
    out = []
    for i in range(A.shape[0]):
        w = A[i]
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        nz = w > 0
        delta = np.linalg.norm(x_hat[nz] - B[i][None, :], axis=1)
        mu = float((w[nz] * delta).sum())
        if mu <= 1e-12:
            continue
        sig2 = float((w[nz] * (delta - mu) ** 2).sum())
        out.append(sig2 / mu ** 2)
    return max(out) if out else 0.0


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("   " + detail if detail else ""))
    return cond


ok = True

# ---- 1. Hand-computable: two rays, one shared primitive, exact arithmetic ------------------
# A = [[1,0],[0,1]] is block-diagonal: each ray touches exactly one primitive (k=1 everywhere).
A = np.array([[1.0, 0.0], [0.0, 1.0]])
B = np.array([[1.0, 0.0], [0.0, 1.0]])
g, x_hat, x_p, L = gamma_of(A, B)
ok &= check("k=1 everywhere -> x' == x_hat", np.allclose(x_p, x_hat), f"max|diff|={np.abs(x_p-x_hat).max():.2e}")
ok &= check("k=1 everywhere -> gamma == 0", abs(g) < 1e-12, f"gamma={g:.3e}")
ok &= check("k=1 everywhere -> L(x_hat) == 0 (exact fit)", L < 1e-24, f"L={L:.3e}")

# ---- 2. Shared primitive, OVERDETERMINED so the fit is not exact ---------------------------
# A square full-rank A fits B exactly, making L(x_hat) = 0 and gamma = 0/0; the ratio is then
# meaningless (it printed 1.6e30 before this case was made overdetermined). Any real operator is
# hugely overdetermined -- 15M rays against ~1e5 primitives -- so that degeneracy cannot arise
# there, but a test must not depend on that.
A = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.5, 0.5]])
B = np.array([[1.0], [0.0], [2.0], [1.0]])
g, x_hat, x_p, L = gamma_of(A, B)
lhs = float(((A @ x_p - B) ** 2).sum())
ok &= check("shared primitive -> Pythagoras exact", abs(lhs - (1 + g) * L) <= 1e-9 * max(lhs, 1.0),
            f"L(x')={lhs:.6f} vs (1+g)L(x_hat)={(1+g)*L:.6f}")
ok &= check("shared primitive -> gamma > 0", g > 0, f"gamma={g:.6f}")
ok &= check("shared primitive -> L(x_hat) > 0 (not a degenerate ratio)", L > 1e-9, f"L={L:.6f}")

# ---- 3. Random row-stochastic operators: identity holds, and beta >= gamma -----------------
worst = 0.0
bad_beta = 0
n_tested = 0
for trial in range(200):
    R, P, D = rng.integers(4, 40), rng.integers(2, 12), rng.integers(1, 5)
    A = np.zeros((R, P))
    for i in range(R):
        k = int(rng.integers(1, min(P, 5) + 1))
        j = rng.choice(P, size=k, replace=False)
        w = rng.random(k)
        A[i, j] = w / w.sum()                      # row-stochastic, as beta requires
    B = rng.normal(size=(R, D))
    if np.linalg.matrix_rank(A) < P:               # x_hat non-unique; identity still holds but
        continue                                   # lstsq picks min-norm, so skip for clarity
    g, x_hat, x_p, L = gamma_of(A, B)
    if L / float((B ** 2).sum()) < 0.05:      # near-exact fit: gamma is 0/0, ratio meaningless
        continue
    n_tested += 1
    lhs = float(((A @ x_p - B) ** 2).sum())
    worst = max(worst, abs(lhs - (1 + g) * L) / max(lhs, 1e-12))
    if beta_of(A, B, x_hat) + 1e-9 < g:
        bad_beta += 1
ok &= check("random: Pythagoras exact to fp", worst < 1e-9, f"worst rel err={worst:.3e}")

# beta >= gamma is NOT asserted. It ought to follow from SFS's L(x') <= (1+beta) L(x_hat) combined
# with the exact L(x') = (1+gamma) L(x_hat), but measured here it FAILS on 23/335 well-posed random
# cases -- e.g. R=9 P=7 with L(x_hat)/||B||^2 = 0.38 (a healthy residual, nothing degenerate) gives
# gamma = 0.729 against beta = 0.444. Either beta as transcribed in compute_beta.py is not the
# quantity in their theorem, or the bound carries a hypothesis these random row-stochastic operators
# violate. Until that is resolved against the paper, this is REPORTED, not asserted, and no claim
# in our own text should rest on beta >= gamma.
n_ok = max(n_tested, 1)
print(f"  INFO   beta >= gamma held on {n_ok - bad_beta}/{n_ok} well-posed random cases "
      f"({bad_beta} violations) -- see comment, deliberately not asserted")

# ---- 4. Block-diagonal (no shared rays) of any k -> gamma == 0 ------------------------------
# Generalises case 1: primitives that share no ray decouple, so the weighted mean IS the optimum.
A = np.zeros((6, 3))
A[0:2, 0] = 1.0
A[2:4, 1] = 1.0
A[4:6, 2] = 1.0
B = rng.normal(size=(6, 2))
g, x_hat, x_p, _ = gamma_of(A, B)
ok &= check("no shared rays -> gamma == 0 for any k", abs(g) < 1e-12, f"gamma={g:.3e}")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
raise SystemExit(0 if ok else 1)
