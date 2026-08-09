"""Build an OpenCLIP extraction manifest for a subset of garden train views,
matching the exact filenames/order/resolution export_feature_operator.py used
(row_view_ids in the exported operator are indices into DataHandler.reload's
camera list for that split, which itself mirrors COLMAPDataset's sorted,
every-8th-filtered image name list)."""

import argparse
import json
import os

from PIL import Image


def colmap_sorted_names(sparse_dir):
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(sparse_dir)
    return sorted(im.name for im in reconstruction.images.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--scene", default="garden")
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--views", required=True, help="comma-separated indices, or 'all'")
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    scene_dir = os.path.join(args.data_path, args.scene)
    names = colmap_sorted_names(os.path.join(scene_dir, "sparse/0"))
    indices = list(range(len(names)))
    if args.split == "train":
        names = [n for i, n in zip(indices, names) if i % 8 != 0]
    else:
        names = [n for i, n in zip(indices, names) if i % 8 == 0]

    if args.views.strip().lower() == "all":
        view_ids = list(range(len(names)))
    else:
        view_ids = [int(v) for v in args.views.split(",") if v.strip() != ""]

    images_dir = os.path.join(scene_dir, "images" if args.downsample == 1 else f"images_{args.downsample}")

    views = []
    for view_id in view_ids:
        image_path = os.path.join(images_dir, names[view_id])
        with Image.open(image_path) as im:
            w, h = im.size
        views.append({"id": view_id, "image": image_path, "height": h, "width": w})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"views": views}, f, indent=2)
    print(f"wrote {len(views)} views to {args.output}")


if __name__ == "__main__":
    main()
