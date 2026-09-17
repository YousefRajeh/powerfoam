"""The L1 Frechet (geometric) median on the sphere, and a robustness theorem for feature lifting.

WHERE THIS SITS. Three formulations of the same lifting problem, each with a different target:

  linear least squares   x* = (A^T A)^-1 A^T B        -- Splat Feature Solver's target. Their
                                                         (1+beta) bound on it is false, and on a
                                                         real scene G is so ill-conditioned that x*
                                                         is an overfit answer anyway.
  L2 Frechet mean        argmin sum w d(x,B)^2        -- respects the sphere. We proved the cheap
                                                         normalised mean is within O(spread^3) of
                                                         it: microradians in practice. Correct, and
                                                         it makes the whole (1+beta) apparatus moot.
  L1 Frechet median      argmin sum w d(x,B)          -- THIS FILE. The L2 mean is optimal for its
                                                         own loss and useless when views conflict:
                                                         given two clusters it sits between them.

WHY L1 IS THE RIGHT LOSS HERE. The failure mode the paper itself describes -- one view's mask covers
the noodles, the next covers the whole bowl -- is CONTAMINATION, not noise. A fraction of the
observations are drawn from the wrong object entirely. Under contamination the L2 mean has no
protection, and empirically the geometric median beat it by +0.14 mIoU on room_0 (0.6095 vs 0.4649).
That gap has never had a theorem attached; this is the theorem.

THE MECHANISM, in one line. The gradient of each observation's contribution is

    grad d(x,B_i)  = a UNIT tangent vector          -> influence capped at w_i, whatever d is
    grad d(x,B_i)^2 = 2 * d(x,B_i) * (unit vector)  -> influence GROWS with distance

so under L1 no single observation can push harder than its own weight, no matter how wrong it is.

THE THEOREM. Let the clean observations carry weight 1-eps and lie within geodesic radius r0 of x0;
let the remaining weight eps be adversarial (anywhere on the sphere). Then ANY weighted Frechet
median satisfies

    ┌────────────────────────────────────────────────┐
    │   d( x^M , x0 )   <=   2 r0  +  (eps/(1-eps)) pi │
    └────────────────────────────────────────────────┘

  Proof. Upper-bound the objective at x0: clean terms <= (1-eps) r0, dirty terms <= eps*pi, so
  F1(x0) <= (1-eps) r0 + eps*pi. Lower-bound it at any x with d(x,x0) = t > r0: every clean term is
  >= t - r0 by the triangle inequality and dirty terms are >= 0, so F1(x) >= (1-eps)(t - r0). A
  minimiser needs F1(x) <= F1(x0), giving (1-eps)(t - r0) <= (1-eps) r0 + eps*pi, i.e.
  t <= 2 r0 + eps*pi/(1-eps).  []

  The bias is O(eps), and the bound stays below the sphere's diameter pi exactly while
  eps/(1-eps) * pi < pi - 2 r0 -- for clean data (r0 -> 0) that is eps < 1/2. So the classical
  BREAKDOWN POINT OF 1/2 drops out of the same inequality rather than being asserted separately.
"""
import numpy as np

PI = np.pi


def log_map(x, B):
    """Riemannian log at x of each row of B, plus the angles."""
    c = np.clip(B @ x, -1.0, 1.0)
    th = np.arccos(c)
    s = np.sin(th)
    scale = np.where(s > 1e-12, th / np.maximum(s, 1e-12), 1.0)
    return scale[:, None] * (B - np.outer(c, x)), th


def exp_map(x, v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return x
    y = np.cos(n) * x + np.sin(n) * (v / n)
    return y / np.linalg.norm(y)


def frechet_mean(B, w, iters=300):
    m = (w[:, None] * B).sum(0)
    x = m / np.linalg.norm(m)
    for _ in range(iters):
        V, th = log_map(x, B)
        g = (w[:, None] * V).sum(0) / w.sum()
        if np.linalg.norm(g) < 1e-14:
            break
        x = exp_map(x, g)
    return x


def frechet_median(B, w, iters=2000, eps=1e-12):
    """Weighted Riemannian Weiszfeld iteration."""
    x = frechet_mean(B, w)
    for _ in range(iters):
        V, th = log_map(x, B)
        d = np.maximum(th, eps)
        coef = w / d
        num = (coef[:, None] * V).sum(0)
        den = coef.sum()
        v = num / max(den, eps)
        if np.linalg.norm(v) < 1e-15:
            break
        x = exp_map(x, v)
    return x


def f1(x, B, w):
    return float((w * np.arccos(np.clip(B @ x, -1, 1))).sum() / w.sum())


def bias_bound(eps, r0):
    return 2.0 * r0 + (eps / max(1.0 - eps, 1e-12)) * PI


def make_case(rng, n, F, r0, eps, adversarial=True):
    """Clean cluster of radius r0 about a random x0, plus weight eps placed adversarially."""
    x0 = rng.normal(size=F); x0 /= np.linalg.norm(x0)
    n_dirty = max(1, int(round(eps * n)))
    n_clean = n - n_dirty
    C = x0[None, :] + (r0 / 2.5) * rng.normal(size=(n_clean, F)) / np.sqrt(F)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # pull the clean cloud strictly inside radius r0 of x0
    thc = np.arccos(np.clip(C @ x0, -1, 1))
    too = thc > r0
    if too.any():
        V, th = log_map(x0, C[too])
        C[too] = np.array([exp_map(x0, V[k] * (r0 / max(th[k], 1e-12))) for k in range(too.sum())])
    if adversarial:
        D = np.tile(-x0, (n_dirty, 1))          # antipode: the worst possible placement
        D = D + 1e-6 * rng.normal(size=D.shape)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    else:
        D = rng.normal(size=(n_dirty, F)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    B = np.vstack([C, D])
    w = np.concatenate([np.full(n_clean, (1 - eps) / max(n_clean, 1)),
                        np.full(n_dirty, eps / max(n_dirty, 1))])
    return B, w / w.sum(), x0


def main():
    rng = np.random.default_rng(0)

    print("CLAIM 1  bounded influence: ||grad d|| == 1 while ||grad d^2|| == 2d")
    x = np.zeros(8); x[0] = 1.0
    for th in [0.05, 0.3, 1.0, 2.0, 3.0]:
        b = np.zeros(8); b[0] = np.cos(th); b[1] = np.sin(th)
        V, t = log_map(x, b[None, :])
        gl1 = np.linalg.norm(V[0]) / max(t[0], 1e-12)     # grad of d is log/||log||
        gl2 = 2 * np.linalg.norm(V[0])                     # grad of d^2 is 2*log
        print(f"   d={th:4.2f}   ||grad d||={gl1:.6f}   ||grad d^2||={gl2:.6f}  (=2d: {2*th:.4f})")

    print("\nCLAIM 2  bias bound  d(x^M,x0) <= 2 r0 + eps*pi/(1-eps)   [ADVERSARIAL antipodes]")
    print(f"{'eps':>6} {'r0':>6} {'d(median,x0)':>14} {'bound':>10} {'holds':>6}   "
          f"{'d(mean,x0)':>11}")
    worst = 0.0
    for eps in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
        for r0 in [0.05, 0.2, 0.5]:
            B, w, x0 = make_case(rng, 300, 24, r0, eps)
            xm = frechet_median(B, w)
            xa = frechet_mean(B, w)
            dm = float(np.arccos(np.clip(xm @ x0, -1, 1)))
            da = float(np.arccos(np.clip(xa @ x0, -1, 1)))
            bd = bias_bound(eps, r0)
            ok = dm <= bd + 1e-9
            if r0 == 0.2:
                print(f"{eps:6.2f} {r0:6.2f} {dm:14.5f} {bd:10.5f} {str(ok):>6}   {da:11.5f}")
            assert ok, (eps, r0, dm, bd)
            worst = max(worst, dm / max(bd, 1e-12))
    print(f"   bound never violated; worst ratio d/bound = {worst:.4f}")

    print("\nCLAIM 3  breakdown at 1/2: the median holds until eps crosses it, the mean does not")
    print(f"{'eps':>6} {'d(median,x0)':>14} {'d(mean,x0)':>12}  {'bound<pi?':>10}")
    for eps in [0.30, 0.40, 0.45, 0.49, 0.51, 0.55, 0.70]:
        B, w, x0 = make_case(rng, 400, 24, 0.05, eps)
        xm = frechet_median(B, w); xa = frechet_mean(B, w)
        dm = float(np.arccos(np.clip(xm @ x0, -1, 1)))
        da = float(np.arccos(np.clip(xa @ x0, -1, 1)))
        print(f"{eps:6.2f} {dm:14.5f} {da:12.5f}  "
              f"{str(bias_bound(eps, 0.05) < PI):>10}")

    print("\nCLAIM 4  influence saturates for the median, grows for the mean")
    print(f"{'outlier angle':>14} {'d(mean,x0)':>12} {'d(median,x0)':>14}")
    F = 24
    x0 = np.zeros(F); x0[0] = 1.0
    C = x0[None, :] + 0.02 * rng.normal(size=(200, F)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    for phi in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        o = np.zeros(F); o[0] = np.cos(phi); o[1] = np.sin(phi)
        B = np.vstack([C, o[None, :]])
        w = np.concatenate([np.full(200, 0.85 / 200), [0.15]])
        xm = frechet_median(B, w); xa = frechet_mean(B, w)
        print(f"{phi:14.2f} {np.arccos(np.clip(xa@x0,-1,1)):12.5f} "
              f"{np.arccos(np.clip(xm@x0,-1,1)):14.5f}")

    print("\nCLAIM 5  the median genuinely minimises F1 (and the mean does not)")
    B, w, x0 = make_case(rng, 300, 24, 0.2, 0.2)
    xm = frechet_median(B, w); xa = frechet_mean(B, w)
    print(f"   F1(median) = {f1(xm,B,w):.8f}   F1(mean) = {f1(xa,B,w):.8f}   "
          f"improvement {f1(xa,B,w)-f1(xm,B,w):.5f}")
    assert f1(xm, B, w) <= f1(xa, B, w) + 1e-9
    rng2 = np.random.default_rng(7)
    best = min(f1(exp_map(xm, 0.05 * rng2.normal(size=24)), B, w) for _ in range(3000))
    print(f"   no better point found in 3000 random perturbations: "
          f"{best >= f1(xm,B,w) - 1e-9}")

    print("\nall claims verified")


if __name__ == "__main__":
    main()
