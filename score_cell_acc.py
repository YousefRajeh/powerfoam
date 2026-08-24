"""Per-cell accuracy (the screening metric) for arbitrary solved-feature files.

Lifted verbatim from the scoring block of diagnose_lifting_optimality.py so the numbers are
directly comparable to the 55.08% mean baseline and the 78.59% observed-view oracle reported
there: a cell's GT label is the MAJORITY GT label of the points assigned to it, a cell is scored
only if it owns at least one GT point, and the prediction is argmax of cosine to the class-name
text embeddings.

The cell->GT assignment is computed ONCE from the reference entry's valid mask and reused for
every entry, so the entries are scored against an identical cell_gt and identical has_gt set.
mIoU (single seed, no clustering) is reported alongside as a secondary number.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
import configargparse

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt,
                                       calculate_metrics)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES


def main():
    enable_determinism()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=os.environ.get("SCANNET_GT_ROOT",
                                                       r"D:\Downloads\scannet_pointcept"))
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--entry", action="append", required=True, help="name=path.pt ; first is ref")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    device = "cuda"

    entries = [e.split("=", 1) for e in a.entry]
    entries = [(n, q) for n, q in entries if os.path.exists(q) or print(f"[skip] missing {q}")]

    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    P = model.points.shape[0]
    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()

    split = SCENES[a.scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        os.path.join(a.gt_root, split, a.scene), "segment20")

    ref = torch.load(entries[0][1], map_location="cpu", weights_only=True)
    vmask = ref["valid_mask"].numpy()
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vmask, k=64)
    owned = assigned >= 0

    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    tids, tnames = [i for i, _ in kept], [n for _, n in kept]
    gt_t = remap_gt_labels(raw_labels, tids)
    n_classes = len(tids) + 1
    text = embed_class_names(tnames, device)

    cnt = np.zeros((P, n_classes), dtype=np.int64)
    np.add.at(cnt, (assigned[owned], gt_t[owned]), 1)
    cnt[:, 0] = 0
    cell_gt = cnt.argmax(1)
    has_gt = cnt.sum(1) > 0
    print(f"[{a.scene}] {P:,} cells, {int(has_gt.sum()):,} own a GT point, "
          f"{n_classes-1} classes present ({a.class_set})", flush=True)

    res = {}
    for name, path in entries:
        t = torch.load(path, map_location=device, weights_only=True)
        f = t["primitive_features"].to(device).float()
        vm = t["valid_mask"].to(device)
        nrm = f.norm(dim=-1)
        f = F.normalize(f, dim=-1)
        cls = (f @ text.T).argmax(-1).cpu().numpy() + 1
        acc = float((cls == cell_gt)[has_gt].mean())
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = cls[assigned[owned]]
        _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_t).long(),
                                         torch.from_numpy(pred).long(), n_classes)
        nz = int((vm & (nrm > 0)).sum())
        res[name] = {"cell_acc": acc, "mIoU_nocluster": float(mi), "mAcc_nocluster": float(ma),
                     "valid": int(vm.sum()), "valid_nonzero": nz,
                     "coverage_of_has_gt": float(((nrm > 0).cpu().numpy())[has_gt].mean())}
        print(f"  {name:<16} cell_acc {acc*100:6.2f}%   mIoU(nocluster) {mi*100:6.2f}   "
              f"valid {int(vm.sum()):,} (nonzero {nz:,})   "
              f"cov(has_gt) {res[name]['coverage_of_has_gt']*100:6.2f}%", flush=True)
        del f, t
        torch.cuda.empty_cache()

    base = res[entries[0][0]]["cell_acc"]
    print(f"\n=== {a.scene} per-cell accuracy vs {entries[0][0]} ===")
    for n in res:
        print(f"  {n:<16} {res[n]['cell_acc']*100:6.2f}%   "
              f"({(res[n]['cell_acc']-base)*100:+5.2f})")
    if a.output:
        json.dump(res, open(a.output, "w"), indent=2)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
