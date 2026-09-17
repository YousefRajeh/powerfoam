"""Theory for the solve term: WHY k=1, proved rather than measured.

Two results, each verified numerically here before being claimed in the paper.

-------------------------------------------------------------------------------
THEOREM 1 (stability, unconditional -- no noise model, no assumptions on B)

    lambda_max(D^-1 G) <= max_i r_i = max_i (1 - T_i^final) <= 1

  where G = A^T A, D = diag(A^T 1), and r_i = sum_j A_ij is ray i's total
  compositing weight.

  Proof. A >= 0 entrywise, so G >= 0 and D > 0, hence D^-1 G >= 0. For a
  nonnegative matrix the spectral radius is bounded by the max row sum
  (rho <= ||.||_inf). Row j of D^-1 G sums to

      (1/D_jj) sum_k G_jk = (sum_i A_ij r_i) / (sum_i A_ij),

  a convex combination of {r_i} with weights A_ij >= 0, hence <= max_i r_i.
  Finally A_ij = alpha_j T_j along ray i, so r_i = sum_j alpha_j T_j = 1 - T_i^final
  <= 1: the alpha-compositing partition of unity. Since D^-1 G is similar to the
  symmetric PSD C = D^-1/2 G D^-1/2, its spectrum is real and nonnegative, so
  lambda_max = rho <= 1.  []

  CONSEQUENCE: with omega = 1 the iteration matrix I - D^-1 G has spectrum in
  [0, 1), so the iteration converges monotonically with NO tuning and no line
  search. The step size is not a hyperparameter -- it is fixed at 1 by the fact
  that rays partition unity. This is why the measured lambda_max was 0.981-0.986
  on every scene: it is <= 1 by construction, and close to 1 because most rays
  terminate on an opaque surface (r_i -> 1).

-------------------------------------------------------------------------------
THEOREM 2 (exact risk, and the stopping rule)

  Model B = A X_true + N with E[N] = 0 and Cov(vec N) = sigma^2 I. Write
  C = D^-1/2 G D^-1/2 = U Lambda U^T, y_true = U^T D^1/2 X_true, s_m = ||y_true,m||^2.
  The Richardson iterates X_k (omega = 1, X_0 = 0) satisfy, in the D^1/2 metric,

      E ||X_k - X_true||_D^2 = sum_m (1-lambda_m)^{2k} s_m           <- bias, DECREASING in k
                             + sigma^2 d sum_m phi_k(lambda_m)^2 / lambda_m   <- variance, INCREASING

  with phi_k(lambda) = 1 - (1-lambda)^k. Both monotonicities are immediate since
  0 <= 1-lambda_m < 1.

  STOPPING RULE. k=1 beats k=2 exactly when

      sigma^2 d * sum_m lambda_m mu_m (mu_m + 2)  >=  sum_m lambda_m mu_m^2 (1 + mu_m) s_m

  where mu_m = 1 - lambda_m. Equivalently, defining the critical noise level

      sigma_crit^2 = [ sum_m lambda_m mu_m^2 (1+mu_m) s_m ] / [ d * sum_m lambda_m mu_m (mu_m+2) ]

  k=1 is preferred to k=2 iff sigma^2 >= sigma_crit^2.

  THIS IS THE ANSWER TO "could a better solve exist". At sigma = 0 the right side
  is 0 and the condition FAILS: with clean data you should iterate, and k=1 is
  strictly suboptimal. The closed form is optimal BECAUSE of the noise, not in
  spite of it. Any solver that beats X' on this data must therefore be exploiting
  something outside this model -- it cannot be found by running the same iteration
  longer or solving the same least-squares problem better.

  Note phi_1(lambda)/lambda = 1 identically: k=1 applies NO inverse amplification
  to any mode. Every k > 1 amplifies near-null directions by
  phi_k(lambda)/lambda -> 1/lambda. So k=1 is the extreme point of the family --
  the maximally regularised nontrivial member -- which is why the optimum sits on
  the boundary rather than in the interior.
"""
from __future__ import annotations

import numpy as np


def risk_exact(lam, s, sigma2, d, k):
    """The Theorem 2 identity."""
    mu = 1.0 - lam
    phi = 1.0 - mu ** k
    return float((mu ** (2 * k) * s).sum() + sigma2 * d * (phi ** 2 / np.maximum(lam, 1e-300)).sum())


def sigma_crit2(lam, s, d):
    """Critical noise level at which k=1 overtakes k=2."""
    mu = 1.0 - lam
    num = (lam * mu ** 2 * (1.0 + mu) * s).sum()
    den = d * (lam * mu * (mu + 2.0)).sum()
    return float(num / max(den, 1e-300))


def _richardson(A, B, K):
    D = A.sum(0)
    X = np.zeros((A.shape[1], B.shape[1]))
    out = []
    for _ in range(K):
        X = X + (A.T @ (B - A @ X)) / D[:, None]
        out.append(X.copy())
    return out


def test_theorem1():
    """lambda_max(D^-1 G) <= max_i r_i <= 1 whenever rows of A sum to <= 1."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        R, P = rng.integers(5, 40), rng.integers(3, 25)
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.5)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        rs = A.sum(1)
        A = A / np.maximum(rs[:, None], 1e-30) * rng.random((R, 1))   # partition of unity: r_i <= 1
        r = A.sum(1)
        D = A.sum(0); G = A.T @ A
        Dh = np.diag(1.0 / np.sqrt(D))
        lam = np.linalg.eigvalsh(Dh @ G @ Dh)
        assert lam.min() > -1e-10, lam.min()
        assert lam.max() <= r.max() + 1e-9, (lam.max(), r.max())
        assert lam.max() <= 1.0 + 1e-9, lam.max()
        # and the row-sum bound itself
        rowsum = (G / D[:, None]).sum(1)
        assert lam.max() <= rowsum.max() + 1e-9
        worst = max(worst, lam.max())
    print(f"  T1 OK: 300 random operators, lambda_max <= max_i r_i <= 1 always "
          f"(largest lambda_max seen {worst:.6f})")


def test_theorem2():
    """The risk identity matches Monte-Carlo, and the stopping rule predicts the argmin."""
    rng = np.random.default_rng(1)
    for trial in range(6):
        R, P, d = 60, 14, 3
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.5, 1.0, (R, 1))
        D = A.sum(0); G = A.T @ A
        Dh = np.diag(1.0 / np.sqrt(D)); Dhi = np.diag(np.sqrt(D))
        C = Dh @ G @ Dh
        lam, U = np.linalg.eigh(C)
        lam = np.clip(lam, 1e-12, None)
        Xt = rng.normal(size=(P, d))
        ytrue = U.T @ (Dhi @ Xt)
        s = (ytrue ** 2).sum(1)
        sigma = 0.35
        K = 12
        # Monte-Carlo risk in the D^1/2 metric
        NT = 4000
        acc = np.zeros(K)
        for _ in range(NT):
            B = A @ Xt + sigma * rng.normal(size=(R, d))
            for k, Xk in enumerate(_richardson(A, B, K)):
                acc[k] += ((Dhi @ (Xk - Xt)) ** 2).sum()
        mc = acc / NT
        ex = np.array([risk_exact(lam, s, sigma ** 2, d, k + 1) for k in range(K)])
        rel = np.abs(mc - ex) / np.maximum(ex, 1e-30)
        assert rel.max() < 0.06, (trial, rel.max(), mc[:4], ex[:4])
        # stopping rule must agree with the identity's own k=1 vs k=2 comparison
        sc2 = sigma_crit2(lam, s, d)
        pred = sigma ** 2 >= sc2
        actual = ex[0] <= ex[1]
        assert pred == actual, (trial, sigma ** 2, sc2, ex[0], ex[1])
    print("  T2 OK: exact risk matches Monte-Carlo to <6% over 6 operators x 12 iterates; "
          "stopping rule agrees with the identity in every case")


def test_noise_drives_it():
    """The point of the theorem: at sigma=0, k=1 is NOT optimal. Noise is what makes it optimal."""
    rng = np.random.default_rng(2)
    R, P, d = 60, 14, 3
    A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
    A[:, A.sum(0) == 0] = 1.0
    A = A / A.sum(1)[:, None] * rng.uniform(0.5, 1.0, (R, 1))
    D = A.sum(0); G = A.T @ A
    Dh = np.diag(1.0 / np.sqrt(D)); Dhi = np.diag(np.sqrt(D))
    lam, U = np.linalg.eigh(Dh @ G @ Dh); lam = np.clip(lam, 1e-12, None)
    Xt = rng.normal(size=(P, d)); s = ((U.T @ (Dhi @ Xt)) ** 2).sum(1)
    sc2 = sigma_crit2(lam, s, d)
    rows = []
    for sig in (0.0, 0.5 * np.sqrt(sc2), np.sqrt(sc2) * 0.999, np.sqrt(sc2) * 1.001, 3 * np.sqrt(sc2)):
        r = [risk_exact(lam, s, sig ** 2, d, k) for k in range(1, 40)]
        rows.append((sig, int(np.argmin(r)) + 1))
    assert rows[0][1] > 1, "at sigma=0 the optimum must NOT be k=1"
    assert rows[-1][1] == 1, "at high noise the optimum must be k=1"
    below = [k for sig, k in rows if sig < np.sqrt(sc2)]
    above = [k for sig, k in rows if sig > np.sqrt(sc2)]
    assert all(k > 1 for k in below) and all(k == 1 for k in above), rows
    print(f"  T3 OK: sigma_crit = {np.sqrt(sc2):.4f}. argmin_k risk = "
          + ", ".join(f"sigma={sig:.3f}->k={k}" for sig, k in rows))
    print("        i.e. with CLEAN data you should iterate; k=1 is optimal BECAUSE of the noise.")


if __name__ == "__main__":
    print("verifying the solve-stopping theory:")
    test_theorem1()
    test_theorem2()
    test_noise_drives_it()
    print("all proofs numerically verified")
