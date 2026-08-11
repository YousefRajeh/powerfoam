import gsplat_env_gsview  # noqa: F401

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_garden_capped import cap_to_budget  # noqa: E402

DEVICE = "cuda"


def make_params(n, tied_value_count):
    """n gaussians, with `tied_value_count` of them sharing the exact same
    (low) opacity value -- simulating gsplat's reset_opa behavior."""
    opac = torch.rand(n, device=DEVICE) * 5 - 2.5  # pre-sigmoid logits, spread out
    opac[:tied_value_count] = torch.logit(torch.tensor(0.01, device=DEVICE))  # huge tied plateau
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3, device=DEVICE)),
        "quats": torch.nn.Parameter(torch.zeros(n, 4, device=DEVICE)),
        "scales": torch.nn.Parameter(torch.zeros(n, 3, device=DEVICE)),
        "opacities": torch.nn.Parameter(opac),
        "colors": torch.nn.Parameter(torch.rand(n, 3, device=DEVICE)),
    }).to(DEVICE)
    optimizers = {name: torch.optim.Adam([params[name]], lr=1e-3) for name in params}
    # Touch each optimizer once so its state dict is non-empty (matches real training).
    for name, opt in optimizers.items():
        params[name].sum().backward()
        opt.step()
        opt.zero_grad()
    state = {"grad2d": torch.zeros(n, device=DEVICE), "count": torch.zeros(n, device=DEVICE), "scene_scale": 1.0}
    return params, optimizers, state


def test_case(n, tied_value_count, target_max, label):
    params, optimizers, state = make_params(n, tied_value_count)
    n_removed = cap_to_budget(params, optimizers, state, target_max)
    n_after = params["means"].shape[0]
    expected_after = min(n, target_max)  # under-budget case is a no-op, not a top-up
    ok = n_after == expected_after and n_removed == n - expected_after
    print(f"[{label}] n={n} tied={tied_value_count} target={target_max} -> "
          f"removed={n_removed} n_after={n_after} {'PASS' if ok else 'FAIL'}")
    assert ok, f"{label}: expected n_after={expected_after}, got {n_after}"


if __name__ == "__main__":
    # The exact failure mode observed: n_remove smaller than the tied
    # plateau's size -- old threshold-based code would remove the WHOLE
    # plateau (way more than n_remove); the fix must remove exactly n_remove.
    test_case(n=1_286_832, tied_value_count=900_000, target_max=1_200_000, label="reproduces original crash shape")
    test_case(n=2_000_000, tied_value_count=2_000_000, target_max=1_200_000, label="all values tied")
    test_case(n=1_500_000, tied_value_count=0, target_max=1_200_000, label="no ties")
    test_case(n=1_000_000, tied_value_count=500_000, target_max=1_200_000, label="under budget, no-op")
    print("ALL PASSED")
