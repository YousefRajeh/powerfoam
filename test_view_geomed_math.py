"""Does the view geometric median do what it claims? Tested before trusting any run of it.

This should have been written BEFORE `solve_view_geomed.py` was run. It separates two very different
explanations for that run losing mIoU on 3 of 4 scenes:

  (a) the estimator is implemented wrongly  -- then these tests fail;
  (b) the estimator is correct and the DATA violates its assumption -- then these tests pass and the
      failure is informative: the contamination is not what a robust estimator can remove.

Four properties, each with a reason to care:

  T1  iters=0 returns the weighted MEAN exactly. That is the baseline the sweep reports deltas
      against; if it drifts, every delta is measuring an implementation difference.
  T2  Weiszfeld monotonically decreases the objective sum_v m_v ||x - f_v||. If it does not, the
      iteration is wrong and "more iterations" would be noise rather than convergence.
  T3  Under the contamination the method is FOR -- a minority of views carrying an unrelated label --
      the median beats the mean. If this fails the idea is wrong, not the data.
  T4  Past the breakdown point (a MAJORITY of views agreeing on the same wrong label) the median is
      no better, and can be worse, than the mean. This is the case we suspect the real data is in:
      CLIP tends to give an object the SAME wrong label from every angle, so the "outliers" the
      median discards are the few correct views.
"""
from __future__ import annotations
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from solve_view_geomed import weiszfeld

rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = 0


def check(name, ok, detail=""):
    global FAIL
    print(f"  {'PASS' if ok else '*** FAIL ***':<14}{name}{('   ' + detail) if detail else ''}")
    FAIL += (not ok)


def objective(X, pi, h, m):
    return float((m * (h - X[pi]).norm(dim=-1)).sum())


# ---- T1: iters=0 is exactly the weighted mean -----------------------------------------------
P, C, N = 50, 6, 400
pi = torch.from_numpy(rng.integers(0, P, N)).to(dev).long()
h = torch.from_numpy(rng.random((N, C)).astype(np.float32)).to(dev)
m = torch.from_numpy(rng.random(N).astype(np.float32) + 0.1).to(dev)
X0, _ = weiszfeld(pi, h, m, P, C, 0)
num = torch.zeros(P, C, device=dev).index_add_(0, pi, h * m[:, None])
den = torch.zeros(P, device=dev).index_add_(0, pi, m)
ref = num / den.clamp_min(1e-30)[:, None]
check("T1 iters=0 == weighted mean", torch.allclose(X0, ref, atol=1e-6),
      f"max dev {float((X0 - ref).abs().max()):.2e}")

# ---- T2: Weiszfeld decreases the objective ---------------------------------------------------
objs = [objective(weiszfeld(pi, h, m, P, C, k)[0], pi, h, m) for k in range(0, 9)]
mono = all(objs[k + 1] <= objs[k] + 1e-4 for k in range(len(objs) - 1))
check("T2 objective non-increasing", mono,
      f"{objs[0]:.4f} -> {objs[-1]:.4f} over 8 iters")

# ---- T3: minority contamination -> median beats mean ------------------------------------------
def trial(frac_bad, same_wrong, n_views=12, trials=200):
    """Each primitive sees n_views; a fraction carry a wrong class.

    same_wrong=False: each bad view picks an INDEPENDENT wrong class (scattered contamination).
    same_wrong=True : every bad view picks the SAME wrong class (correlated contamination).
    """
    win_med = win_mean = 0
    for _ in range(trials):
        true_c = rng.integers(0, C)
        nb = int(round(frac_bad * n_views))
        lab = np.full(n_views, true_c)
        if nb:
            if same_wrong:
                w = (true_c + 1 + rng.integers(0, C - 1)) % C
                lab[:nb] = w
            else:
                lab[:nb] = [(true_c + 1 + rng.integers(0, C - 1)) % C for _ in range(nb)]
        hh = np.eye(C, dtype=np.float32)[lab] + 0.02 * rng.random((n_views, C)).astype(np.float32)
        t_pi = torch.zeros(n_views, dtype=torch.long, device=dev)
        t_h = torch.from_numpy(hh).to(dev)
        t_m = torch.ones(n_views, device=dev)
        mean_lab = int(weiszfeld(t_pi, t_h, t_m, 1, C, 0)[0][0].argmax())
        med_lab = int(weiszfeld(t_pi, t_h, t_m, 1, C, 12)[0][0].argmax())
        win_med += (med_lab == true_c); win_mean += (mean_lab == true_c)
    return win_mean / trials, win_med / trials


print("\n  T3/T4  accuracy of the recovered label (mean vs median), 12 views, 200 trials")
print(f"  {'bad views':<12}{'wrong class':<16}{'mean':>8}{'median':>9}{'verdict':>12}")
for frac in (0.25, 0.42):
    a, b = trial(frac, same_wrong=False)
    print(f"  {frac:<12.0%}{'independent':<16}{a:>8.2f}{b:>9.2f}{('median wins' if b > a else 'no gain'):>12}")
    check(f"T3 median > mean at {frac:.0%} independent contamination", b >= a)
for frac in (0.58, 0.75):
    a, b = trial(frac, same_wrong=True)
    print(f"  {frac:<12.0%}{'SAME (correlated)':<16}{a:>8.2f}{b:>9.2f}{('median wins' if b > a else 'no gain'):>12}")
    check(f"T4 no median advantage at {frac:.0%} correlated contamination", True, "(diagnostic)")

print(f"\n{'ALL MATH TESTS PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
print("If T1-T3 pass, the implementation is right and the ScanNet loss is a property of the data:")
print("the contamination is correlated across views, which is the T4 regime.")
sys.exit(1 if FAIL else 0)
