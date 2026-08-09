import torch
import numpy as np
from PIL import Image

from feature_foam_lifting.operator import SparseFeatureOperator

device = "cuda"
a = SparseFeatureOperator.load("artifacts/garden/test_operator.pt", device)

H, W = 420, 648
view_id = 3

for name, path in [("before", "artifacts/garden/streaming161/x_weighted_v3.pt"),
                    ("after_refine", "artifacts/garden/streaming161/x_weighted_refined_v2.pt")]:
    field = torch.load(path, map_location=device, weights_only=True)
    valid_mask = field["valid_mask"]
    coverage = a.matmul(valid_mask.float())
    mask = a.row_view_ids == view_id
    coords = a.row_pixels[mask]
    coverage_img = torch.zeros(H, W, device=device)
    coverage_img[coords[:, 0], coords[:, 1]] = coverage[mask]
    black_frac = (coverage_img < 0.05).float().mean().item()
    print(f"{name}: valid_fraction={float(valid_mask.float().mean()):.4f} view {view_id} low-coverage(<0.05) fraction={black_frac:.5f}")
    img = (coverage_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(f"artifacts/garden/coverage_view3_{name}.png")
