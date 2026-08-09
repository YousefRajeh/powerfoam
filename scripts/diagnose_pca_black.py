import numpy as np
import torch
from PIL import Image

from feature_foam_lifting.operator import SparseFeatureOperator

device = "cuda"
a = SparseFeatureOperator.load("artifacts/garden/test_operator.pt", device)
field = torch.load("artifacts/garden/roundtrip/x_ridge.pt", map_location=device, weights_only=True)
valid_mask = field["valid_mask"]
x = field["primitive_features"].float()

H, W = 420, 648
view_id = 3  # the panel that looked worst

# "coverage" = how much of this pixel's opacity budget comes from VALID primitives
coverage = a.matmul(valid_mask.float())
mask = a.row_view_ids == view_id
coords = a.row_pixels[mask]
row_sum_img = torch.zeros(H, W, device=device)
row_sum_img[coords[:, 0], coords[:, 1]] = a.row_sums()[mask]
coverage_img = torch.zeros(H, W, device=device)
coverage_img[coords[:, 0], coords[:, 1]] = coverage[mask]

# fraction of this view's foreground opacity that is "explained" by valid primitives
frac_valid_of_foreground = (coverage_img.sum() / row_sum_img.clamp_min(1e-8).sum()).item()
black_frac = (coverage_img < 0.05).float().mean().item()
print(f"view {view_id}: row_sum(mean)={row_sum_img.mean().item():.4f} "
      f"coverage(mean)={coverage_img.mean().item():.4f} "
      f"frac_pixels_coverage<0.05={black_frac:.4f} "
      f"frac_valid_of_total_opacity={frac_valid_of_foreground:.4f}")

# visualize: coverage as grayscale, and pca-black-mask overlay
img = (coverage_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(img).save("artifacts/garden/coverage_view3.png")
print("wrote artifacts/garden/coverage_view3.png (bright = well covered by train views, dark = low/no support)")
