"""Compute del.% -- the fraction of labelled GT points OpenGaussian's opacity rule deletes.

The rule (from their scripts/eval_scannet.py, and the one column of Table 3 that is not a quality
metric): a GT point is dropped from scoring when the Gaussian it is assigned to has opacity < 0.1.
It is a property of (reconstruction, assignment), not of the class set, so it is computed once per
method+scene and is identical across the 19/15/10 splits.

This matters for reading the table: a method that deletes a quarter of the labelled points is
scored on an easier subset than one that deletes 4%. It is reported per row rather than averaged
into the metrics, for the same reason coverage is.

Mirrors evaluate_point_cloud_miou.apply_gt_opacity_mask exactly: assignment over the FULL Gaussian
set (not the opacity-filtered one), then drop labelled points whose owner is below threshold.
Filtering candidates first would make the mask a no-op -- the exact trap documented in that file.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_point_cloud_miou import (load_gaussian_means_opacities,  # noqa: E402
                                       load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_nearest_center  # noqa: E402

GT_ROOT = r"D:\Downloads\scannet_pointcept"
MANIFEST = os.path.join(HERE, "artifacts", "baseline_eval", "manifest.json")
OUT = os.path.join(HERE, "artifacts", "baseline_eval", "del_pct.json")


def gt_dir(scene):
    for split in ("train", "val", "test"):
        d = os.path.join(GT_ROOT, split, scene)
        if os.path.isdir(d):
            return d
    return None


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for e in json.load(open(MANIFEST)):
    key = f"{e['tag']}|{e['scene']}"
    if key in res:
        continue
    d = gt_dir(e["scene"])
    if d is None:
        continue
    means, opac = load_gaussian_means_opacities(e["ckpt"], "cuda")
    gt_points, raw, _ = load_scannet_pointcept_gt(d, "segment20")
    # assignment over the FULL set, exactly as the masked protocol requires
    assigned = assign_points_to_nearest_center(gt_points, means, valid=None)
    labelled = raw >= 0
    owned = assigned >= 0
    drop = labelled & owned & (opac[np.clip(assigned, 0, None)] < 0.1)
    n_lab = int(labelled.sum())
    res[key] = dict(tag=e["tag"], scene=e["scene"], n_labelled=n_lab,
                    n_dropped=int(drop.sum()),
                    del_pct=100.0 * float(drop.sum()) / max(n_lab, 1))
    print(f"  {e['tag']:22s} {e['scene']} del={res[key]['del_pct']:.2f}%", flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

from collections import defaultdict  # noqa: E402
agg = defaultdict(list)
for v in res.values():
    agg[v["tag"]].append(v["del_pct"])
print()
for t, v in sorted(agg.items()):
    print(f"  {t:24s} del.% = {sum(v)/len(v):5.2f}  (n={len(v)})")
