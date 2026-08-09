"""Export a Feature Foam sparse rendering-weight operator from a trained,
frozen PowerFoam checkpoint. See docs/feature-foam-phase1-pipeline-and-tests.md
for the full design.

Usage:
    python export_feature_operator.py -c outputs/garden/config.yaml \
        --data_path D:\\Downloads\\powerfoam\\data\\mipnerf360 \
        --split train --views 0,5,10,15,20,25,30,40,60,90 \
        --downsample 8 --max_hits_per_pixel 64 \
        --output artifacts/garden/train_operator.pt
"""

import os
import warnings

warnings.filterwarnings("ignore")

import configargparse
import numpy as np
import torch
import warp as wp

from configs import *
from data_loader import DataHandler
from powerfoam.feature_operator import export_operator_for_views
from powerfoam.scene import PowerfoamScene

seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)


def parse_views(views_arg, num_available):
    if views_arg is None or views_arg.strip().lower() == "all":
        return list(range(num_available))
    indices = [int(v) for v in views_arg.split(",") if v.strip() != ""]
    for idx in indices:
        if idx < 0 or idx >= num_available:
            raise ValueError(f"view index {idx} out of range [0, {num_available})")
    return indices


def main(args, config_path, split, views_arg, max_hits_per_pixel, transmittance_threshold, max_intersections, output_path):
    wp.init()

    checkpoint = config_path.replace("/config.yaml", "")

    data_handler = DataHandler(args)
    data_handler.reload(split, downsample=args.downsample[-1])

    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    # Deliberately no sort_points()/resample() here -- primitive index p must
    # stay identical to whatever a separate train/test export run (or a later
    # viewer load) produces for the same model.pt.

    indices = parse_views(views_arg, len(data_handler.cameras))
    cameras = [data_handler.cameras[i] for i in indices]

    print(
        f"[export_feature_operator] split={split} views={indices} "
        f"num_primitives={model.points.shape[0]} downsample={args.downsample[-1]}"
    )

    operator = export_operator_for_views(
        model,
        cameras,
        indices,
        transmittance_threshold=transmittance_threshold,
        max_intersections=max_intersections,
        max_hits_per_pixel=max_hits_per_pixel,
    )

    diagnostics = operator.diagnostics()
    print(
        f"[export_feature_operator] num_rows={operator.num_rows} nnz={operator.values.numel()} "
        f"row_sum_mean={diagnostics['row_sum_mean']:.4f} "
        f"valid_primitive_fraction={float(diagnostics['valid_p'].float().mean()):.4f}"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    operator.save(output_path)
    print(f"[export_feature_operator] wrote {output_path}")


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--views",
        default="all",
        help='Comma-separated view indices into the split, or "all"',
    )
    parser.add_argument("--max_hits_per_pixel", type=int, default=64)
    parser.add_argument("--transmittance_threshold", type=float, default=1e-3)
    parser.add_argument("--export_max_intersections", type=int, default=1024)
    parser.add_argument("--output", required=True)

    cli_args = parser.parse_args()
    main(
        get_params(cli_args),
        cli_args.config,
        cli_args.split,
        cli_args.views,
        cli_args.max_hits_per_pixel,
        cli_args.transmittance_threshold,
        cli_args.export_max_intersections,
        cli_args.output,
    )
