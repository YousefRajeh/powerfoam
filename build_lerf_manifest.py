"""Build an OpenCLIP extraction manifest for a LERF-OVS scene: every image in
<scene>/images, no train/test split (both reconstructions trained on all frames --
test_every defaults far exceed these scene sizes)."""
import argparse
import json
import os

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--scene-dir", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()

images_dir = os.path.join(args.scene_dir, "images")
names = sorted(os.listdir(images_dir))

views = []
for i, name in enumerate(names):
    path = os.path.join(images_dir, name)
    with Image.open(path) as im:
        w, h = im.size
    views.append({"id": i, "image": path, "height": h, "width": w})

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"views": views}, f, indent=2)
print(f"wrote {len(views)} views to {args.output}")
