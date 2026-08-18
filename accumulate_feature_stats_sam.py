"""Feature Foam accumulation using the REAL SAM+CLIP features (task #5), reusing
accumulate_feature_stats_for_views (unchanged, the same function used for every other
Feature Foam result in this project) with a SAM-compatible feature-map loader.

The reconstruction formula below is copied VERBATIM from splat-distiller's own
gsplat_ext/datasets/normalize.py::load_image_feature_from_SAMOpenCLIP (the exact function
Splat Feature Solver's own pipeline uses to turn SAM masks + per-mask CLIP embeddings into a
dense per-pixel map) -- not reimplemented from scratch, copied, specifically so both methods
consume byte-identical reconstructed features and the only thing that differs is the 3D
representation + solver, matching this project's controlling fairness principle throughout.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.feature_operator import accumulate_feature_stats_for_views
from feature_foam_lifting.operator import AccumulatedFeatureStats  # noqa: F401 (import path sanity)


def load_image_feature_from_SAMOpenCLIP(feature_folder: Path, image_stem: str, height: int = 480, width: int = 640, sam_level=None) -> torch.Tensor:
    """Copied verbatim from gsplat_ext/datasets/normalize.py (with the float() fix already
    applied there this session for the fp16-on-disk issue), EXCEPT the missing-feature
    placeholder shape, which is now parameterized by the actual camera resolution rather than
    hardcoded to Replica's fixed 480x640. Real bug found on ScanNet retraining (commit after
    324775e): scene0062_00's frame 0 has no extracted SAM+CLIP feature (a handful of frames are
    intentionally skipped by the extractor when every SAM mask is degenerate), so this fallback
    fires -- but ScanNet cameras are NOT 480x640, so the placeholder's shape silently mismatched
    the real per-view pixel count, and downstream `lazy_b`'s `row_local // W, row_local % W`
    indexing (built from the REAL camera H*W) read out of bounds of the WRONG-shaped placeholder,
    triggering a CUDA device-side assert deep in a later kernel (confirmed via
    CUDA_LAUNCH_BLOCKING=1 -- the assert fires exactly at this indexing line, not spuriously
    elsewhere). Passing height/width through so the placeholder always matches the view it's
    standing in for fixes this for any dataset, not just Replica's fixed resolution."""
    feature_path = feature_folder / f"{image_stem}_f.npy"
    segment_path = feature_folder / f"{image_stem}_s.npy"
    if not feature_path.exists():
        # Same tolerance as normalize.py::load_image_features -- a handful of images have
        # every SAM mask degenerate and were intentionally skipped by the extractor. Shape
        # must match a real feat_map (H, W, C), not normalize.py's (1,1,512) placeholder --
        # this loader's caller (accumulate_feature_stats_for_views) needs a real per-pixel map.
        print(f"[accumulate_feature_stats_sam] WARNING: no SAMOpenCLIP feature for {image_stem} -- using zero feature map ({height}x{width}x512)")
        return torch.zeros(height, width, 512, device="cuda")
    features = torch.from_numpy(np.load(feature_path)).to("cuda").float()
    segment = torch.from_numpy(np.load(segment_path)).to("cuda").to(torch.long) + 1
    if sam_level is not None:
        if isinstance(sam_level, str):
            sam_level = [int(x) for x in sam_level.split(",")]
        # Use ONLY this SAM granularity level (LangSplat hierarchy in _s.npy: 0=default,
        # 1=subpart(s), 2=part(m), 3=whole(l)). OpenGaussian's README: "we only use the
        # large-level mask"; NormLift (per its author) likewise uses the l-level. Summing
        # all 4 levels per pixel (the splat-distiller loader default) blends up to 4 mask
        # embeddings per pixel -- the measured feature-contamination source.
        segment = segment[sam_level] if isinstance(sam_level, list) else segment[sam_level:sam_level + 1]
    zero_row = torch.zeros(1, 512, device=features.device, dtype=features.dtype)
    features_pad = torch.cat([zero_row, features], dim=0)
    feat_map = F.embedding(segment, features_pad).sum(dim=0)
    feat_map = feat_map / (feat_map.norm(dim=-1, keepdim=True) + 1e-6)
    return feat_map


def main(scene: str, config_path: str, feature_folder: str, output_path: str, batch_size: int = 1,
         images_subdir: str = "images", feature_name_format: str = None, sam_level=None):
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", config_path])

    checkpoint = config_path.replace("/config.yaml", "").replace("\\config.yaml", "")
    data_handler = DataHandler(args)
    data_handler.reload("all", downsample=args.downsample[-1])

    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    # No sort_points()/resample() -- same index-stability requirement as every other export in this project.

    cameras = data_handler.cameras
    # split="all" on ReplicaDataset actually returns the TRAIN split only (split != "test" ->
    # ~test_mask, see replica.py) -- cameras is already train-only and 0-indexed by POSITION in
    # that filtered list, not by original global frame number. So no further i % 8 filtering
    # here, and camera position i does NOT equal its global frame index -- recover the real
    # global indices the same way replica.py itself does, to compute the right rgb_XXX names.
    indices = list(range(len(cameras)))

    if feature_name_format is not None:
        # Explicit override for loaders (e.g. data_loader/replica.py) whose camera index i
        # corresponds directly to a numerically-named frame (rgb_{i}.png, NOT zero-padded --
        # lexicographic directory sort would be wrong order), while the SAM feature files on
        # disk use a different (typically zero-padded, COLMAP-convention) naming for the same
        # global frame index.
        rgb_dir = Path(args.data_path) / args.scene / "rgb"
        num_frames = len([f for f in rgb_dir.iterdir() if f.name.startswith("rgb_") and f.name.endswith(".png")])
        all_idx = np.arange(num_frames)
        test_mask = all_idx % 8 == 0
        frame_indices = all_idx[~test_mask]  # matches ReplicaDataset's split="all" (non-"test") branch
        assert len(frame_indices) == len(cameras), f"{len(frame_indices)} frame indices vs {len(cameras)} cameras"
        image_names = [feature_name_format.format(fi) for fi in frame_indices]
    else:
        # Relies on the SAME sorted-name convention used throughout this project for LERF-OVS
        # (colmap.py's "all" split preserves sorted-name order, matching the plain
        # sorted(os.listdir(images_dir)) used to build every other manifest in this project).
        images_dir = Path(args.data_path) / args.scene / images_subdir
        image_names = sorted(p.stem for p in images_dir.iterdir())
    assert len(image_names) == len(cameras), f"{len(image_names)} images vs {len(cameras)} cameras"

    feature_dir = Path(feature_folder)

    def load_feature_map(view_id):
        camera = cameras[view_id]
        return load_image_feature_from_SAMOpenCLIP(feature_dir, image_names[view_id], height=camera.height, width=camera.width, sam_level=sam_level)

    print(f"[accumulate_feature_stats_sam] scene={scene} views={len(indices)} num_primitives={model.points.shape[0]} batch_size={batch_size}")
    torch.cuda.synchronize()
    t0 = time.time()
    stats = accumulate_feature_stats_for_views(model, cameras, indices, load_feature_map, batch_size=batch_size)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[accumulate_feature_stats_sam] TIMING batch_size={batch_size} num_views={len(indices)} elapsed_sec={elapsed:.3f} sec_per_view={elapsed / max(len(indices), 1):.4f}")

    stats.positions = model.points.detach()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stats.save(output_path)
    print(f"[accumulate_feature_stats_sam] wrote {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--feature-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--feature-name-format", default=None)
    p.add_argument("--sam-level", type=str, default=None,
                   help="use only this SAM granularity level (3 = whole/l-level, the "
                        "OpenGaussian/NormLift convention); default sums all levels")
    cli_args = p.parse_args()
    main(cli_args.scene, cli_args.config, cli_args.feature_folder, cli_args.output, cli_args.batch_size,
         images_subdir=cli_args.images_subdir, feature_name_format=cli_args.feature_name_format, sam_level=cli_args.sam_level)
