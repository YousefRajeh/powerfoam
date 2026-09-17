"""Rebuild artifacts/baseline_eval/manifest.json from the materialised inputs on disk.

make_baseline_eval_inputs.py writes the manifest from scratch each run, so invoking it with --only
or --scenes truncates the manifest to just that subset -- which is how a 70-entry manifest became a
3-entry one. This reconstructs it by scanning the directories that actually exist, so it is correct
regardless of which subsets were built when, and it can be re-run safely at any time.

n_original comes from the source arm checkpoint (the pre-pruning Gaussian count), which is what
makes kept_frac meaningful for the pruning question.
"""
import glob
import json
import os

import torch

ROOT = r"D:\Downloads\powerfoam\artifacts\baseline_eval"
RECON = r"D:\Downloads\powerfoam\recon_remote"
ARM_CKPT = {"frozen": RECON + r"\gs_froz\{scene}\ckpt.pt",
            "unfrozen": RECON + r"\gs_unfroz\{scene}\ckpt.pt"}

_norig = {}


def n_original(arm, scene):
    key = (arm, scene)
    if key not in _norig:
        p = ARM_CKPT[arm].format(scene=scene)
        if not os.path.exists(p):
            _norig[key] = None
        else:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
            _norig[key] = int(sp["means"].shape[0])
    return _norig[key]


rows = []
for tag_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(tag_dir):
        continue
    tag = os.path.basename(tag_dir)
    parts = tag.split("_")
    baseline = parts[0]
    arm = "frozen" if "frozen" in tag and "unfrozen" not in tag else "unfrozen"
    level = int(parts[-1][1:]) if parts[-1].startswith("l") and parts[-1][1:].isdigit() else None
    for scene_dir in sorted(glob.glob(os.path.join(tag_dir, "scene*"))):
        scene = os.path.basename(scene_dir)
        ck = os.path.join(scene_dir, "ckpt.pt")
        ft = os.path.join(scene_dir, "features.pt")
        if not (os.path.exists(ck) and os.path.exists(ft)):
            continue
        n_kept = int(torch.load(ck, map_location="cpu", weights_only=False)["splats"]["means"].shape[0])
        n_all = n_original(arm, scene)
        rows.append(dict(baseline=baseline, arm=arm, scene=scene, level=level, tag=tag,
                         n_original=n_all, n_kept=n_kept,
                         kept_frac=(n_kept / n_all) if n_all else None,
                         ckpt=ck, features=ft))

out = os.path.join(ROOT, "manifest.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=1)

from collections import Counter
print(f"{len(rows)} entries -> {out}")
for t, c in sorted(Counter(r["tag"] for r in rows).items()):
    print(f"  {t:24s} {c}")
