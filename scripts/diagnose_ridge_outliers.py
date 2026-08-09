import torch

for name in ["weighted", "ridge"]:
    field = torch.load(f"artifacts/garden/roundtrip/x_{name}.pt", map_location="cuda", weights_only=True)
    x = field["primitive_features"].float()
    valid = field["valid_mask"]
    norms = x[valid].norm(dim=-1)
    q = torch.tensor([0.0, 0.5, 0.9, 0.99, 0.999, 1.0], device=norms.device)
    print(f"{name}: valid_count={int(valid.sum())} norm quantiles {q.tolist()} -> {torch.quantile(norms, q).tolist()}")
    print(f"{name}: mean={norms.mean().item():.4f} std={norms.std().item():.4f} max={norms.max().item():.4f}")
