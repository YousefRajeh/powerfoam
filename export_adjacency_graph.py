"""Tier 1 idea #5: export PowerFoam's real power-diagram adjacency graph (the CSR
(adjacent, offsets) structure powerfoam/bvh.py::AABBTree.build_cech_complex already
computes for ray-marching/interpenetration-loss purposes) from a trained checkpoint, for
reuse in a post-clustering graph-cut smoothing pass -- see
ResearchVault/Ideas/powerfoam-structural-ideas.md Idea 1.

Usage: python export_adjacency_graph.py -c <config.yaml> --output <adjacency.pt>
"""
import argparse
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import torch
import warp as wp
import configargparse

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def main(config_path: str, output_path: str):
    wp.init()
    checkpoint = config_path.replace("/config.yaml", "").replace("\\config.yaml", "")
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", config_path])

    data_handler = DataHandler(args)
    data_handler.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")

    centers = model.points.detach()
    radii = model.get_radii().detach()
    model.aabb_tree.update(centers, radii)
    adjacent, offsets = model.aabb_tree.build_cech_complex()

    print(f"[export_adjacency_graph] {centers.shape[0]} primitives, {adjacent.numel()} adjacency edges "
          f"({adjacent.numel() / centers.shape[0]:.2f} avg neighbors/primitive)")
    torch.save({"adjacent": adjacent.cpu(), "offsets": offsets.cpu(), "num_primitives": centers.shape[0]}, output_path)
    print(f"[export_adjacency_graph] wrote {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--output", required=True)
    cli_args = p.parse_args()
    main(cli_args.config, cli_args.output)
