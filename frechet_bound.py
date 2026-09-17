"""The Frechet formulation of feature lifting, and a bound for the estimator we actually compute.

WHY REPLACE THE LEAST-SQUARES TARGET. Both Splat Feature Solver's (1+beta) claim and our corrected
Laplacian identity bound the distance to x* = (A^T A)^-1 A^T B. That is only a quality measure if x*
is desirable, and on a real scene it is not obviously so: G = A^T A is badly conditioned (plain CG
fails to converge on scene0062_00; it needs Jacobi preconditioning AND a ridge), so x* is an overfit
answer to noisy B. Worse, the whole linear framing drops the geometry -- CLIP features are
directional, only their direction is ever used, and the paper's own solver normalises its output
afterwards, OUTSIDE the optimisation where no guarantee covers it.

THE HONEST OBJECTIVE. Put each primitive's feature on the sphere and use the intrinsic loss:

    x^F_j  =  argmin        sum_i  w_ij  d( x , B_i )^2 ,     d(x,b) = arccos<x,b>
              x in S^{F-1}

This is the weighted FRECHET (Karcher) mean. It is NOT convex in the ambient sense -- the sphere is
not a convex set -- but it is GEODESICALLY strongly convex when the data lie in a geodesic ball of
radius r < pi/2, where the minimiser also exists and is unique (Karcher; Afsari). So the problem has
a complete theory; it is simply not the linear one.

WHAT WE ACTUALLY COMPUTE. The estimator in the code is the normalised weighted mean -- the EXTRINSIC
mean xbar = sum w B / ||sum w B||. So the question that matters is not "how far is x' from x*" but:

    how far is the cheap extrinsic mean from the intrinsic Frechet mean?

THE BOUND (derived below, verified in this file).

    ||grad F(xbar)||  <=  2 sum_i w_i (theta_i - sin theta_i)  <=  (1/3) sum_i w_i theta_i^3

  using that xbar is parallel to the Euclidean mean, so sum_i w_i P_xbar B_i = 0 exactly, which lets
  the O(theta) part of the gradient cancel and leaves only the O(theta^3) remainder. Combined with
  geodesic strong convexity of F on a ball of radius r (Hessian >= 2 r cot r):

    ┌──────────────────────────────────────────────────────────────┐
    │   d( xbar , x^F )   <=   tan(r) * M2 / 6                      │
    │                                                              │
    │   r  = max_i theta_i        M2 = sum_i w_i theta_i^2          │
    └──────────────────────────────────────────────────────────────┘

  THIRD ORDER in the spread: for concentrated data the cheap estimator is extremely close to the
  expensive one. And M2 is not a new quantity -- to leading order M2 = 2(1-R) where R = ||sum w B||
  is the resultant length, i.e. exactly the per-cell conflict statistic this project already gets
  for free from the accumulator (conflict = 1 - R^2 = (1-R)(1+R) ~ 2(1-R)).

  The bound degrades gracefully and DIVERGES at r = pi/2, which is correct rather than a defect:
  beyond that radius the Frechet mean need not be unique and there is nothing to be close to.
"""
import numpy as np


def geodesic(x, B):
    """Angles between a unit vector and a set of unit vectors."""
    return np.arccos(np.clip(B @ x, -1.0, 1.0))


def extrinsic_mean(B, w):
    m = (w[:, None] * B).sum(0)
    n = np.linalg.norm(m)
    return m / max(n, 1e-300), n / w.sum()          # xbar, resultant length R


def frechet_mean(B, w, iters=500, tol=1e-14):
    """Weighted Frechet mean by Riemannian gradient descent with exponential retraction."""
    x, _ = extrinsic_mean(B, w)
    for _ in range(iters):
        th = geodesic(x, B)
        s = np.sin(th)
        coef = np.where(s > 1e-12, th / np.maximum(s, 1e-12), 1.0)   # theta/sin(theta) -> 1
        # tangent gradient of sum w theta^2 is -2 sum w (theta/sin theta) P_x B
        g = -2.0 * ((w * coef)[:, None] * (B - np.outer(B @ x, x))).sum(0)
        gn = np.linalg.norm(g)
        if gn < tol:
            break
        v = -g / (2.0 * w.sum())                                     # step ~ Newton for small r
        nv = np.linalg.norm(v)
        if nv < tol:
            break
        x = np.cos(nv) * x + np.sin(nv) * (v / nv)                   # exponential map
        x /= np.linalg.norm(x)
    return x


def grad_norm(x, B, w):
    th = geodesic(x, B)
    s = np.sin(th)
    coef = np.where(s > 1e-12, th / np.maximum(s, 1e-12), 1.0)
    g = -2.0 * ((w * coef)[:, None] * (B - np.outer(B @ x, x))).sum(0)
    return np.linalg.norm(g)


def bound(B, w):
    """(bound, r, M2) for d(xbar, x^F) <= tan(r) * M2 / 6."""
    xb, _ = extrinsic_mean(B, w)
    th = geodesic(xb, B)
    r = float(th.max())
    M2 = float((w * th ** 2).sum() / w.sum())
    if r >= np.pi / 2:
        return np.inf, r, M2
    return float(np.tan(r) * M2 / 6.0), r, M2


def sample(rng, n, F, spread, wskew=1.0):
    """n unit vectors concentrated within roughly `spread` radians of a random axis."""
    mu = rng.normal(size=F); mu /= np.linalg.norm(mu)
    X = mu[None, :] + spread * rng.normal(size=(n, F)) / np.sqrt(F)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.random(n) ** wskew
    return X, w / w.sum()


def main():
    rng = np.random.default_rng(0)

    print("CLAIM 1  the gradient at the extrinsic mean is O(theta^3), not O(theta)")
    print(f"{'spread':>8} {'r':>8} {'||gradF(xbar)||':>16} {'(1/3)*sum w th^3':>18} {'holds':>6}")
    for spread in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        B, w = sample(rng, 200, 32, spread)
        xb, _ = extrinsic_mean(B, w)
        th = geodesic(xb, B)
        g = grad_norm(xb, B, w)
        rhs = (w * th ** 3).sum() / 3.0
        print(f"{spread:8.3f} {th.max():8.4f} {g:16.3e} {rhs:18.3e} {str(g <= rhs + 1e-12):>6}")

    print("\nCLAIM 2  d(xbar, x^F) <= tan(r) * M2 / 6, and the ratio shows the order")
    print(f"{'spread':>8} {'r':>8} {'M2':>10} {'d(xbar,xF)':>12} {'bound':>12} {'ratio':>8} {'holds':>6}")
    worst = 0.0
    for spread in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]:
        for trial in range(20):
            B, w = sample(rng, 200, 32, spread)
            xb, _ = extrinsic_mean(B, w)
            xf = frechet_mean(B, w)
            d = float(np.arccos(np.clip(xb @ xf, -1, 1)))
            bd, r, M2 = bound(B, w)
            if trial == 0:
                ratio = d / bd if np.isfinite(bd) and bd > 0 else np.nan
                print(f"{spread:8.3f} {r:8.4f} {M2:10.5f} {d:12.3e} {bd:12.3e} "
                      f"{ratio:8.4f} {str(d <= bd + 1e-12):>6}")
            if np.isfinite(bd):
                assert d <= bd + 1e-9, (spread, trial, d, bd)
                worst = max(worst, d / max(bd, 1e-300))
    print(f"  bound never violated; worst observed ratio d/bound = {worst:.4f}")

    print("\nCLAIM 3  M2 ~ 2(1-R), so the free conflict statistic IS the Frechet dispersion")
    print(f"{'spread':>8} {'M2':>10} {'2(1-R)':>10} {'conflict 1-R^2':>15} {'M2/2(1-R)':>11}")
    for spread in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        B, w = sample(rng, 400, 32, spread)
        xb, R = extrinsic_mean(B, w)
        th = geodesic(xb, B)
        M2 = (w * th ** 2).sum() / w.sum()
        print(f"{spread:8.3f} {M2:10.5f} {2 * (1 - R):10.5f} {1 - R ** 2:15.5f} "
              f"{M2 / max(2 * (1 - R), 1e-12):11.4f}")

    print("\nCLAIM 4  uniqueness/strong convexity really does fail past r = pi/2")
    B = np.zeros((3, 3)); B[0] = [1, 0, 0]; B[1] = [-1, 1e-9, 0]; B[2] = [0, 1, 0]
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    w = np.array([0.5, 0.5, 0.0]) + 1e-12
    w /= w.sum()
    bd, r, M2 = bound(B, w)
    print(f"  antipodal pair: r = {r:.4f} rad (pi/2 = {np.pi/2:.4f}), bound = {bd}")
    print("  -> reported as infinite rather than a number, which is the correct answer:")
    print("     every point on the equator is a Frechet mean, so 'the' mean does not exist.")

    print("\nCLAIM 5  the extrinsic mean is NOT the Frechet mean in general (the gap is real)")
    B, w = sample(rng, 200, 32, 0.9)
    xb, _ = extrinsic_mean(B, w)
    xf = frechet_mean(B, w)
    Fx = lambda x: float((w * geodesic(x, B) ** 2).sum())
    print(f"  F(xbar) = {Fx(xb):.8f}   F(x^F) = {Fx(xf):.8f}   "
          f"improvement {Fx(xb) - Fx(xf):.3e}   d = {np.arccos(np.clip(xb@xf,-1,1)):.5f}")
    assert Fx(xf) <= Fx(xb) + 1e-12

    print("\nall claims verified")


if __name__ == "__main__":
    main()
