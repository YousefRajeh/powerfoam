"""Stream-accumulate Feature Foam per-primitive statistics across ALL (or many)
train views, at memory bounded by one view at a time -- see
powerfoam/feature_operator.py::accumulate_feature_stats_for_views and
docs/feature-foam-phase1-pipeline-and-tests.md for why this replaces the
10-view-capped batch export for solving (the batch export/test_operator.pt path
is still used for held-out evaluation, which is forward-only and much cheaper).

Usage:
    python accumulate_feature_stats.py -c outputs/garden/config.yaml \
        --data_path D:\\Downloads\\powerfoam\\data\\mipnerf360 \
        --split train --views all --downsample 8 \
        --feature-manifest artifacts/garden/openclip_train_all/feature_manifest.json \
        --output artifacts/garden/train_stats_all161.pt
"""

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import configargparse
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from configs import *
from data_loader import DataHandler
from feature_foam_lifting.operator import AccumulatedFeatureStats, normalize_features
from powerfoam.feature_operator import accumulate_feature_stats_for_views
from powerfoam.scene import PowerfoamScene

seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)


def parse_views(views_arg, num_available):
    if views_arg is None or views_arg.strip().lower() == "all":
        return list(range(num_available))
    return [int(v) for v in views_arg.split(",") if v.strip() != ""]


def main(args, config_path, split, views_arg, feature_manifest_path, output_path, batch_size=8, skip_save=False):
    wp.init()

    checkpoint = config_path.replace("/config.yaml", "")

    data_handler = DataHandler(args)
    data_handler.reload(split, downsample=args.downsample[-1])

    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    # No sort_points()/resample() -- see export_feature_operator.py's identical note.

    indices = parse_views(views_arg, len(data_handler.cameras))
    cameras = [data_handler.cameras[i] for i in indices]

    manifest = json.loads(Path(feature_manifest_path).read_text())
    manifest_dir = Path(feature_manifest_path).parent
    views_by_id = {v["id"]: v for v in manifest["views"]}
    for idx in indices:
        if idx not in views_by_id:
            raise SystemExit(f"feature manifest has no view {idx} -- extract OpenCLIP features for it first")
    if "feature_archive" not in manifest:
        raise SystemExit("feature manifest has no feature_archive -- re-extract with the current extract_openclip_features")

    # One combined CPU load instead of one torch.load() per view. The archive
    # stores the NATIVE (small, e.g. 14x14) CLIP patch grid per view, not a
    # render-resolution map -- so this whole archive is now tens of MB, not
    # tens of GB (see extract_openclip_features.py's comment for why storing
    # the upsampled version was pure waste: bilinear upsampling manufactures
    # values at every render pixel from a much smaller set of real patch
    # values, adding zero information). Individual grids still move to CUDA
    # and get upsampled+normalized lazily, one view at a time inside
    # `load_feature_map`, to keep GPU memory bounded exactly as before.
    print(f"[accumulate_feature_stats] loading combined feature archive ({manifest['feature_archive']})...")
    feature_maps_cpu = torch.load(manifest_dir / manifest["feature_archive"], map_location="cpu", weights_only=True)

    def load_feature_map(view_id):
        if view_id not in feature_maps_cpu:
            raise SystemExit(f"feature archive has no view {view_id}")
        grid = feature_maps_cpu[view_id].to("cuda").float()  # (grid_h, grid_w, C), raw
        record = views_by_id[view_id]
        upsampled = F.interpolate(
            grid.permute(2, 0, 1)[None], size=(int(record["height"]), int(record["width"])),
            mode="bilinear", align_corners=False,
        )[0]
        return normalize_features(upsampled.permute(1, 2, 0))

    print(f"[accumulate_feature_stats] split={split} views={len(indices)} num_primitives={model.points.shape[0]} batch_size={batch_size}")

    torch.cuda.synchronize()
    t0 = time.time()
    stats = accumulate_feature_stats_for_views(model, cameras, indices, load_feature_map, batch_size=batch_size)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[accumulate_feature_stats] TIMING batch_size={batch_size} num_views={len(indices)} "
          f"elapsed_sec={elapsed:.3f} sec_per_view={elapsed / max(len(indices), 1):.4f}")

    # Carried along purely so a later, optional KNN refinement stage
    # (feature_foam_lifting.refine) has primitive positions without needing to
    # reload the checkpoint separately -- not used by any accumulation math.
    stats.positions = model.points.detach()

    if not skip_save:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stats.save(output_path)
        print(f"[accumulate_feature_stats] wrote {output_path}")


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--split", choices=("train", "test", "all"), required=True)
    parser.add_argument("--views", default="all", help='Comma-separated view indices, or "all"')
    parser.add_argument("--feature-manifest", required=True, help="OpenCLIP feature_manifest.json from feature-foam-extract-openclip")
    parser.add_argument("--output", required=False, default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-save", action="store_true", help="Benchmark-only: skip writing the (large) stats file to disk")

    cli_args = parser.parse_args()
    if not cli_args.skip_save and not cli_args.output:
        raise SystemExit("--output is required unless --skip-save is set")
    main(get_params(cli_args), cli_args.config, cli_args.split, cli_args.views, cli_args.feature_manifest,
         cli_args.output, cli_args.batch_size, cli_args.skip_save)
