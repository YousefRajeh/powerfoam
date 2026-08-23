"""Evaluate arbitrary solved-feature files over MULTIPLE clustering seeds.

Why multi-seed is mandatory here, measured today: the dominant noise source in this protocol is
not float nondeterminism (fixed and verified bitwise earlier) but the k-means seed itself.
Across three seeds on identical features and geometry, the per-arm standard deviation is 0.71
mIoU on average and 1.64 at worst -- scene0070_00's power arm ranged 39.53 to 43.55. So a
single-seed delta below roughly 1.5 mIoU is not evidence, and every comparison from here on
reports mean and spread over seeds.

Usage:
    python eval_solved_multiseed.py --scene scene0347_00 --seeds 0,1,2 \
        --entry diagonal=artifacts/.../solved_coupled_ridge_diagbaseline.pt \
        --entry cone=artifacts/.../solved_cone_constrained.pt

The first --entry is treated as the reference and deltas are reported against it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch

CLASS_SETS = ("opengaussian19", "opengaussian15", "opengaussian10")


def main():
    enable_determinism()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--entry", action="append", required=True,
                   help="name=path/to/solved.pt ; first entry is the reference")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("voro", "run_voronoi_feature_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    seeds = [int(x) for x in a.seeds.split(",")]
    entries = []
    for e in a.entry:
        name, path = e.split("=", 1)
        entries.append((name, path))

    results = {}
    for name, path in entries:
        if not os.path.exists(path):
            print(f"[{name}] MISSING {path}, skipping", flush=True)
            continue
        per_seed = []
        for sd in seeds:
            r = m.evaluate(a.scene, ckpt, path, seed=sd)
            per_seed.append(r)
            print(f"  [{name}] seed {sd}: " + "  ".join(
                f"{cs[12:]}cls={r[cs]['mIoU']*100:6.2f}" for cs in CLASS_SETS), flush=True)
        results[name] = {
            cs: {"mean": float(np.mean([r[cs]["mIoU"] for r in per_seed])) ,
                 "std": float(np.std([r[cs]["mIoU"] for r in per_seed])),
                 "per_seed": [float(r[cs]["mIoU"]) for r in per_seed]}
            for cs in CLASS_SETS}
        t = torch.load(path, map_location="cpu", weights_only=True)
        n = t["primitive_features"][t["valid_mask"]].norm(dim=-1)
        results[name]["frac_norm_gt1"] = float((n > 1).float().mean())
        results[name]["max_norm"] = float(n.max())

    ref = entries[0][0]
    print(f"\n=== {a.scene}, {len(seeds)} seeds, mean +/- std (delta vs {ref}) ===", flush=True)
    hdr = f"{'entry':<14}" + "".join(f"{cs[12:]+'cls':>20}" for cs in CLASS_SETS) + f"{'off-cone':>10}"
    print(hdr); print("-" * len(hdr))
    for name in results:
        row = f"{name:<14}"
        for cs in CLASS_SETS:
            r = results[name][cs]
            d = (r["mean"] - results[ref][cs]["mean"]) * 100 if name != ref else 0.0
            cell = f"{r['mean']*100:6.2f}+/-{r['std']*100:4.2f}"
            cell += "      " if name == ref else f" ({d:+5.2f})"
            row += f"{cell:>20}"
        row += f"{results[name]['frac_norm_gt1']*100:9.2f}%"
        print(row, flush=True)
    print("\n  off-cone = fraction of cells with ||f|| > 1, impossible for a convex "
          "combination of unit vectors")
    print(f"  NOTE: seed std is typically ~0.7 mIoU here; deltas below ~1.5 are not evidence.")
    if a.output:
        json.dump(results, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
