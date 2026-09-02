"""Show why frac(beta=0) undercounts: the one-pass variance formula cancels catastrophically.

beta_i = sigma_i^2 / mu_i^2 with sigma_i^2 accumulated as E[D^2] - mu^2. For a k=1 ray the two terms
are equal and large, so their difference is pure rounding: in float32 each carries ~1e-7 relative
error, leaving sigma^2 ~ 1e-7 * D^2 and hence beta ~ 1e-7 -- far above any sensible "is it zero"
threshold, which is why 31.2% of rays with k=1 produced only 27.4% with beta=0.

The two-pass form sigma^2 = sum_j w_j (D_j - mu)^2 has no cancellation: for k=1 the normalised
weight is v/v = 1 EXACTLY in IEEE754, so the residual is (D - mu)^2 with mu accurate to 1 ulp,
giving beta ~1e-14. This is a numerical-stability fix, not a loosened threshold.
"""
import torch

torch.manual_seed(0)


def betas(d, v, two_pass):
    """One ray, k=1: weight v, residual d. Returns beta_i under each formulation."""
    rs = v                                    # row sum, single term
    mu = (v * d) / rs
    if two_pass:
        w = v / rs                            # exactly 1.0 for k=1
        sig2 = w * (d - mu) ** 2
    else:
        m2 = (v * d * d) / rs
        sig2 = m2 - mu * mu
    return (sig2 / (mu * mu)).abs()


print(f"{'weight v':>10s} {'residual d':>11s} {'one-pass beta':>15s} {'two-pass beta':>15s}")
worst1 = worst2 = 0.0
for v in (0.3, 0.7, 0.9998, 1.0):
    for d in (0.05, 0.5, 1.7):
        t = torch.tensor([v], dtype=torch.float32)
        u = torch.tensor([d], dtype=torch.float32)
        b1 = float(betas(u, t, False))
        b2 = float(betas(u, t, True))
        worst1, worst2 = max(worst1, b1), max(worst2, b2)
        print(f"{v:10.4f} {d:11.3f} {b1:15.3e} {b2:15.3e}")

print(f"\nworst one-pass beta for a k=1 ray: {worst1:.3e}")
print(f"worst two-pass beta for a k=1 ray: {worst2:.3e}")
print(f"\nthreshold 1e-12 counts k=1 as zero?  one-pass: {worst1 <= 1e-12}   "
      f"two-pass: {worst2 <= 1e-12}")
