"""Generic version of build_dense_features_from_featurefoam.py: convert a scene's
Feature Foam dense OpenCLIP archive into per-image .pt files splat-distiller's plain
dense-tensor loader can read directly (no masks/ folder -> torch.load(...) verbatim).
Saves RAW native small grids (not upsampled) -- distill.py upsamples+normalizes lazily.
"""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--out-dir", required=True)
args = p.parse_args()

manifest = json.loads(Path(args.manifest).read_text())
assert manifest["encoder"]["model"] == "ViT-B-16-quickgelu"
assert manifest["encoder"]["pretrained"] == "openai"
archive_dir = Path(args.manifest).parent
feature_maps = torch.load(archive_dir / manifest["feature_archive"], map_location="cpu", weights_only=True)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

written = 0
for view in manifest["views"]:
    view_id = view["id"]
    stem = Path(view["image"]).stem
    grid = feature_maps[view_id].to(torch.float32)
    torch.save(grid, out_dir / f"{stem}.pt")
    written += 1

print(f"wrote {written} native-resolution feature tensors to {out_dir}")
