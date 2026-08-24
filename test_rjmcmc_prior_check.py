"""
test_rjmcmc_prior_check.py

Implementation validation, using the paper's OWN consistency check
(Bodin & Sambridge 2009, p.1421):

  "A simple way to check that the form of the acceptance term is correct is to
   set the likelihood to a uniform distribution (i.e. remove the data). In this
   case, the posterior is directly proportional to the prior and the Markov
   chain should sample the known prior distribution."

If our birth/death acceptance terms (eq.35/36) are coded correctly, the chain
run with a constant likelihood must reproduce the UNIFORM prior p(n) on
[NMIN, NMAX] (eq.9) and the uniform prior on velocity (eq.11).

Run: D:\conda\envs\powerfoam\python.exe D:\Downloads\powerfoam\test_rjmcmc_prior_check.py
"""
import numpy as np
import test_bodin_rjmcmc_coverage as B

# remove the data: constant likelihood
B.misfit = lambda v: 0.0
B.NMIN, B.NMAX = 2, 30          # small range so the uniform prior is visible

ch = B.run_chain(400000, 100000, 20, seed=11, n_init=6, verbose=False)
n = ch['n_list']
print("Prior-sampling check (likelihood removed), uniform p(n) on [%d,%d]"
      % (B.NMIN, B.NMAX))
print("  expected mean n = %.2f   measured = %.2f" % ((B.NMIN + B.NMAX) / 2, n.mean()))
print("  expected sd     = %.2f   measured = %.2f"
      % (np.sqrt(((B.NMAX - B.NMIN + 1) ** 2 - 1) / 12.0), n.std()))
h = np.bincount(n, minlength=B.NMAX + 1)[B.NMIN:B.NMAX + 1] / len(n)
flat = 1.0 / (B.NMAX - B.NMIN + 1)
print("  uniform target per bin = %.4f ; measured min=%.4f max=%.4f "
      "max|dev|/target=%.2f" % (flat, h.min(), h.max(), np.abs(h - flat).max() / flat))
print("  posterior-mean field (should be flat at prior mean %.2f): mean=%.3f sd_spatial=%.3f"
      % ((B.VMIN + B.VMAX) / 2, ch['mean'].mean(), ch['mean'].std()))
print("  ensemble std field (should be near prior sd %.3f): mean=%.3f"
      % ((B.VMAX - B.VMIN) / np.sqrt(12), ch['std'].mean()))
