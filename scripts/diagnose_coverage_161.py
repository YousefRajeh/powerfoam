import torch
from PIL import Image
import numpy as np

from feature_foam_lifting.operator import SparseFeatureOperator

device = "cuda"
a = SparseFeatureOperator.load("artifacts/garden/test_operator.pt", device)
field = torch.load("artifacts/garden/streaming161/x_weighted.pt", map_location=device, weights_only=True)
valid_mask = field["valid_mask"]

H, W = 420, 648
view_id = 3
coverage = a.matmul(valid_mask.float())
mask = a.row_view_ids == view_id
coords = a.row_pixels[mask]
coverage_img = torch.zeros(H, W, device=device)
coverage_img[coords[:, 0], coords[:, 1]] = coverage[mask]

black_frac = (coverage_img < 0.05).float().mean().item()
print(f"161-view weighted: valid_fraction={float(valid_mask.float().mean()):.4f} "
      f"view {view_id} low-coverage(<0.05) pixel fraction={black_frac:.4f}")

img = (coverage_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(img).save("artifacts/garden/coverage_view3_161views.png")
