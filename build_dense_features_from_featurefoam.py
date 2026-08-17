"""Convert Feature Foam's already-extracted OpenCLIP features (native patch grid,
ViT-B-16-quickgelu/openai, 787 train views) into per-image .pt tensors that
splat-distiller's plain dense-feature loader (gsplat_ext.datasets.normalize.load_image_features,
the branch with no `masks/` folder) can read directly -- so the Splat Feature Solver's solve
step (distill.py) uses the EXACT SAME 2D CLIP features as Feature Foam, for an apples-to-apples
comparison of the lifting method + 3D representation, not the feature extractor.

Saves the RAW NATIVE small grid (e.g. 14x14x512), not upsampled to full image resolution --
distill.py's own dataset pipeline already bilinear-upsamples + L2-normalizes at load time
(features.permute+interpolate+F.normalize in distill.py's distill() loop), exactly mirroring
Feature Foam's own "store small, upsample lazily" convention. Saving upsampled instead would
write ~787 x 480x640x512 float tensors to disk -- ~18GB and climbing, the exact waste Feature
Foam's own extract_openclip_features.py comment warns about.
"""
import json
import re
from pathlib import Path

import torch

MANIFEST = Path(r"D:\Downloads\powerfoam\artifacts\replica_room0\openclip_train\feature_manifest.json")
ARCHIVE_DIR = MANIFEST.parent
OUT_DIR = Path(r"D:\Downloads\powerfoam\data\replica\room_0_colmap\openclip_dense_featurefoam")

manifest = json.loads(MANIFEST.read_text())
assert manifest["encoder"]["model"] == "ViT-B-16-quickgelu"
assert manifest["encoder"]["pretrained"] == "openai"
feature_maps = torch.load(ARCHIVE_DIR / manifest["feature_archive"], map_location="cpu", weights_only=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

written = 0
for view in manifest["views"]:
    view_id = view["id"]
    m = re.search(r"rgb_(\d+)\.png$", view["image"])
    assert m, f"unexpected image name: {view['image']}"
    global_frame = int(m.group(1))
    colmap_name = f"rgb_{global_frame:03d}"

    grid = feature_maps[view_id].to(torch.float32)  # (gh, gw, C), raw -- e.g. (14, 14, 512)
    torch.save(grid, OUT_DIR / f"{colmap_name}.pt")
    written += 1

print(f"wrote {written} native-resolution feature tensors to {OUT_DIR}")
