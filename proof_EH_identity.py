"""The corrected E_H identity for SUB-STOCHASTIC rays, and what it does and does not change.

We have long written  J - L = E_H = 1/2 sum_jk G_jk ||X_j - X_k||^2.  That graph-energy form requires
r_i = sum_j A_ij = 1 for every ray. Our rays are sub-stochastic (r_i = 1 - T_i^final, mean measured
0.9965-0.9971), so it is false as stated. This derives the correct statement, checks which downstream
claims survive, and measures the size of the correction.

DEFINITIONS.  J(X) = sum_i sum_j A_ij ||X_j - B_i||^2,  L(X) = ||AX - B||_F^2,
D = diag(A^T 1) (column masses), G = A^T A, H = D - G, r = A 1 (row masses).

DERIVATION.
  J(X) = sum_j D_j ||X_j||^2 - 2<X, A^T B> + sum_i r_i ||B_i||^2
  L(X) = tr(X^T G X)         - 2<X, A^T B> + sum_i    ||B_i||^2
so
  J - L = tr(X^T (D - G) X) - sum_i (1 - r_i) ||B_i||^2 = E_H(X) - c,     c = sum_i (1-r_i)||B_i||^2

and the graph decomposition of E_H follows from (G1)_j = sum_i A_ij r_i = [A^T r]_j:

  D - G = [diag(A^T 1) - diag(A^T r)] + [diag(G1) - G]
  ==> E_H(X) = 1/2 sum_jk G_jk ||X_j - X_k||^2  +  sum_j [A^T(1-r)]_j ||X_j||^2      (*)

The second term is the correction. It vanishes iff r = 1 on every ray that touches a primitive.

WHAT SURVIVES (all verified below):
  1. X' = D^-1 A^T B still minimises J exactly. grad J = 2(DX - A^T B), independent of r.
  2. H >= 0 still, so E_H >= 0. This follows from THEOREM 1: G <= r_max D gives H >= (1-r_max) D >= 0.
     The partition of unity is what makes E_H a valid non-negative bound -- the two results are
     linked, which was not previously stated.
  3. The suboptimality bound is UNCHANGED: J(X') <= J(Xhat) with J = L + E_H - c and c independent
     of X gives L(X') - L(Xhat) <= E_H(Xhat) - E_H(X') <= E_H(Xhat).
  4. The measured flip-rate result is unaffected: `compute_beta.py` computes the excess directly as
     ||A(X' - Xhat)||_F^2, never via the graph formula. Only the prose identity was wrong.

WHAT CHANGES:
  5. "J = L iff G is diagonal" becomes: J = L as functions of X iff H = 0 AND c = 0, i.e. every ray
     deposits on exactly one primitive WITH FULL WEIGHT (r_i = 1). One-hot is no longer sufficient --
     it must also be opaque. Sub-stochastic one-hot rays give J - L = -c != 0, a constant.
"""
from __future__ import annotations

import numpy as np


def terms(A, B, X):
    D = A.sum(0); r = A.sum(1); G = A.T @ A
    J = float(sum(A[i, j] * ((X[j] - B[i]) ** 2).sum()
                  for i in range(A.shape[0]) for j in range(A.shape[1])))
    L = float(((A @ X - B) ** 2).sum())
    EH = float(np.trace(X.T @ (np.diag(D) - G) @ X))
    graph = 0.5 * float(sum(G[j, k] * ((X[j] - X[k]) ** 2).sum()
                            for j in range(A.shape[1]) for k in range(A.shape[1])))
    extra = float(((A.T @ (1.0 - r)) * (X ** 2).sum(1)).sum())
    c = float(((1.0 - r) * (B ** 2).sum(1)).sum())
    return J, L, EH, graph, extra, c


def matvec_forms(A, X):
    """EXACTLY what the GPU path computes -- no P x P matrix is ever formed.

    tr(X^T H X) = tr(X^T D X) - tr(X^T G X) = sum_j D_j ||X_j||^2 - ||A X||_F^2
    graph       = tr(X^T (diag(G1) - G) X), and (G1)_j = sum_i A_ij r_i = [A^T r]_j
    extra       = sum_j (D_j - [A^T r]_j) ||X_j||^2
    One ||A X||^2 is shared by the first two.
    """
    D = A.sum(0); r = A.sum(1); Atr = A.T @ r
    xn2 = (X ** 2).sum(1)
    AX2 = float(((A @ X) ** 2).sum())
    return (float((D * xn2).sum()) - AX2,
            float((Atr * xn2).sum()) - AX2,
            float(((D - Atr) * xn2).sum()))


def test_matvec_equivalence():
    """The optimisation must be EXACT, not merely plausible. Dense reference vs matvec form."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(300):
        R, P, d = 25, 7, 4
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        X = rng.normal(size=(P, d)); B = rng.normal(size=(R, d))
        _, _, EH_d, graph_d, extra_d, _ = terms(A, B, X)
        EH_m, graph_m, extra_m = matvec_forms(A, X)
        for nm, a_, b_ in (("E_H", EH_d, EH_m), ("graph", graph_d, graph_m), ("extra", extra_d, extra_m)):
            rel = abs(a_ - b_) / max(abs(a_), 1e-30)
            worst = max(worst, rel)
            assert rel < 1e-9, (nm, a_, b_, rel)
    print(f"  matvec form == dense form (E_H, graph, extra)  : 300/300, worst rel {worst:.2e}")


def main():
    rng = np.random.default_rng(0)
    print("verifying the corrected E_H identity for sub-stochastic rays:\n")
    n_old_fails = 0
    for t in range(300):
        R, P, d = 30, 6, 3
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.6)
        A[:, A.sum(0) == 0] = np.abs(rng.normal(size=(R, int((A.sum(0) == 0).sum()))))
        A = A / np.maximum(A.sum(1)[:, None], 1e-30) * rng.uniform(0.3, 1.0, (R, 1))
        B = rng.normal(size=(R, d)); X = rng.normal(size=(P, d))
        J, L, EH, graph, extra, c = terms(A, B, X)
        # the corrected identity
        assert abs((J - L) - (EH - c)) < 1e-8 * max(1, abs(J - L)), (t, J - L, EH - c)
        # the graph decomposition (*)
        assert abs(EH - (graph + extra)) < 1e-8 * max(1, abs(EH)), (t, EH, graph + extra)
        # the OLD claim fails
        if abs((J - L) - graph) > 1e-6 * max(1, abs(J - L)):
            n_old_fails += 1
        # 1. X' minimises J
        D = A.sum(0); Xp = (A.T @ B) / D[:, None]
        for _ in range(4):
            Y = Xp + 1e-3 * rng.normal(size=Xp.shape)
            assert terms(A, B, Xp)[0] <= terms(A, B, Y)[0] + 1e-9
        # 2. H >= 0, and the Theorem 1 refinement H >= (1-r_max) D
        G = A.T @ A; Hm = np.diag(D) - G
        ev = np.linalg.eigvalsh(Hm)
        assert ev.min() > -1e-9, (t, ev.min())
        rmax = float(A.sum(1).max())
        ev2 = np.linalg.eigvalsh(Hm - (1.0 - rmax) * np.diag(D))
        assert ev2.min() > -1e-8, (t, ev2.min())
        # 3. the suboptimality bound still holds
        Xhat = np.linalg.pinv(A) @ B
        lhs = float(((A @ Xp - B) ** 2).sum() - ((A @ Xhat - B) ** 2).sum())
        EHhat = float(np.trace(Xhat.T @ Hm @ Xhat))
        assert -1e-9 <= lhs <= EHhat + 1e-8, (t, lhs, EHhat)
    print(f"  corrected identity  J - L = E_H - c            : 300/300")
    print(f"  graph decomposition E_H = graph + extra        : 300/300")
    print(f"  OLD claim J - L = graph                        : FAILS on {n_old_fails}/300")
    print(f"  X' still minimises J                           : 300/300")
    print(f"  H >= 0, and H >= (1-r_max) D  (via Theorem 1)   : 300/300")
    print(f"  bound L(X')-L(Xhat) <= E_H(Xhat) still holds    : 300/300")
    test_matvec_equivalence()

    # 5. the iff condition
    print("\n  the 'J = L' condition, tested on three operators:")
    for name, A in (("one-hot, r=1 (opaque)", np.eye(3)),
                    ("one-hot, r=0.5 (translucent)", 0.5 * np.eye(3)),
                    ("mixed rays, r=1", np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]))):
        B = rng.normal(size=(A.shape[0], 2)); X = rng.normal(size=(A.shape[1], 2))
        J, L, EH, graph, extra, c = terms(A, B, X)
        print(f"    {name:30s} J-L = {J-L:+9.4f}   E_H = {EH:7.4f}   c = {c:7.4f}")
    print("    -> one-hot alone is NOT enough; the ray must also be opaque (r=1) for J = L.")


if __name__ == "__main__":
    main()
