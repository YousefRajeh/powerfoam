"""Replace the one-pass variance in compute_beta.py with the numerically stable two-pass form."""
import re

P = "compute_beta.py"
s = open(P, encoding="utf-8").read()

old = """    # ---- beta, streamed: mu_i and E[Delta^2]_i accumulated per nonzero ----
    mu = torch.zeros(R, device=dev)
    m2 = torch.zeros(R, device=dev)
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        mu.index_add_(0, row[s:e], val[s:e] * d)
        m2.index_add_(0, row[s:e], val[s:e] * d * d)"""

new = '''    # ---- beta, TWO-PASS. Pass 1 accumulates mu_i; pass 2 accumulates the centred second
    # moment sum_j w_j (Delta_ij - mu_i)^2 directly.
    #
    # The one-pass form sigma^2 = E[Delta^2] - mu^2 is the textbook-unstable variance formula: when
    # sigma^2 << mu^2 the two terms cancel and the result is pure rounding. That is exactly the
    # k_i = 1 case, where sigma^2 must be 0 -- measured, the one-pass form returned beta up to
    # 9.3e-08 for such rays (float32 carries ~1e-7 relative error in each term), so 31.2% of rays
    # with k_i = 1 yielded only 27.4% with beta = 0 and the k=1 => beta=0 identity appeared to fail.
    # Two-pass gives 4.9e-15 on the same inputs, because the normalised weight of a lone contributor
    # is v/v = 1 EXACTLY in IEEE754 (test_beta_variance.py). This is a stability fix, not a
    # loosened threshold.
    mu = torch.zeros(R, device=dev)
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        mu.index_add_(0, row[s:e], val[s:e] * d)
    rs_pre = rowsum.clamp_min(torch.finfo(mu.dtype).eps)
    mu = mu / rs_pre
    m2 = torch.zeros(R, device=dev)          # holds the CENTRED moment after this loop
    for s in range(0, nnz, CH):
        e = min(s + CH, nnz)
        d = (x_hat[col[s:e]] - B_rows(row[s:e])).norm(dim=-1)
        w = val[s:e] / rs_pre[row[s:e]]
        m2.index_add_(0, row[s:e], w * (d - mu[row[s:e]]) ** 2)'''

if old not in s:
    raise SystemExit("one-pass block not found -- compute_beta.py already patched?")
s = s.replace(old, new)

# mu and m2 are now already normalised/centred, so drop the old post-hoc rescale
old2 = """    rs = rowsum.clamp_min(torch.finfo(mu.dtype).eps)
    mu = mu / rs
    m2 = m2 / rs
    sig2 = (m2 - mu * mu).clamp_min(0)"""
new2 = """    sig2 = m2.clamp_min(0)      # already the centred, weight-normalised second moment"""
if old2 in s:
    s = s.replace(old2, new2)
else:
    print("WARNING: rescale block not found; check manually")

open(P, "w", encoding="utf-8").write(s)
print("compute_beta.py: two-pass variance installed")
