"""Feature Foam Test B: synthetic ground-truth recovery.

Uses the actual exported garden train_operator.pt (real geometry/sparsity, not a
toy example). Draws a random X_true, forward-renders B = A @ X_true (optionally
+ noise), solves X_hat with every solver, and reports cosine similarity /
angular error between X_hat and X_true restricted to valid_mask (support > 0).

See docs/feature-foam-phase1-pipeline-and-tests.md section 4.2 for the full
rationale and expected qualitative ordering across solvers.
"""

import argparse

import torch

from feature_foam_lifting.operator import (
    SparseFeatureOperator,
    feature_metrics,
    ridge_closed_form,
    ridge_pcg,
    weighted_average,
)


def run(operator_path, feature_dim, noise_std, seed, device):
    torch.manual_seed(seed)
    a = SparseFeatureOperator.load(operator_path, device)
    valid_mask = a.support() > 0

    x_true = torch.randn(a.num_primitives, feature_dim, device=device)
    b_clean = a.matmul(x_true)
    b_noisy = b_clean + noise_std * torch.randn_like(b_clean)

    solvers = {
        "weighted": lambda b: weighted_average(a, b)[0],
        "squared-weighted": lambda b: weighted_average(a, b, squared=True)[0],
        "ridge (default)": lambda b: ridge_pcg(a, b, "default")[0],
        "ridge-closed-form (default)": lambda b: ridge_closed_form(a, b, "default")[0],
    }

    rows = []
    for name, solve in solvers.items():
        x_clean = solve(b_clean)
        x_noisy = solve(b_noisy)
        m_clean = feature_metrics(x_clean[valid_mask], x_true[valid_mask])
        m_noisy = feature_metrics(x_noisy[valid_mask], x_true[valid_mask])
        rows.append((name, m_clean["cos_sim"], m_clean["angular_error_deg_mean"],
                     m_noisy["cos_sim"], m_noisy["angular_error_deg_mean"]))

    header = f"{'solver':<30}{'clean cos_sim':>15}{'clean angle(deg)':>20}{'noisy cos_sim':>16}{'noisy angle(deg)':>18}"
    print(header)
    print("-" * len(header))
    for name, cc, ca, nc, na in rows:
        print(f"{name:<30}{cc:>15.4f}{ca:>20.2f}{nc:>16.4f}{na:>18.2f}")

    print(f"\nnum_primitives={a.num_primitives} valid_fraction={float(valid_mask.float().mean()):.4f} "
          f"feature_dim={feature_dim} noise_std={noise_std}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--operator", default="artifacts/garden/train_operator.pt")
    p.add_argument("--feature-dim", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args.operator, args.feature_dim, args.noise_std, args.seed, args.device)
