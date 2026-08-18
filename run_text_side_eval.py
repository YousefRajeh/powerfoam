"""Text-side ablation: prompt ensembles x score calibration, holding the best clustering
protocol fixed (position-aware 64x5 k-means, the 34.16/34.44/38.73 baseline), so any delta
is attributable to the classification rule alone.

Prompt modes:
  raw        -- bare class name (current baseline; verified byte-identical to what
                OpenGaussian ships in assets/text_features.json)
  templates  -- mean of N generic CLIP zero-shot templates per class (standard trick)
  indoor     -- templates + indoor/structural-specific phrasings (extra synonyms for the
                diagnosed failure classes: wall/floor get surface-specific prompts)

Calibration modes (all unsupervised -- no GT is used to fit anything):
  none    -- plain cosine argmax (OpenGaussian's rule)
  zscore  -- per-class z-score standardization (current hubness correction)
  center  -- per-class mean subtraction only (no variance scaling)
  rank    -- per-class percentile rank of the similarity, argmax over ranks
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, CLIP_MODEL, CLIP_PRETRAINED,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt,
)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES, CLASS_SETS

GENERIC_TEMPLATES = [
    "{}",
    "a photo of a {}.",
    "a photo of the {} in a room.",
    "a {} in an indoor scene.",
    "a photo of one {}.",
    "a cropped photo of a {}.",
    "a rendering of a {}.",
]
# extra phrasings for the diagnosed structural failure classes only
INDOOR_EXTRA = {
    "wall": ["a blank interior wall.", "the wall of a room.", "a painted wall surface.", "an empty wall."],
    "floor": ["the floor of a room.", "a wooden floor.", "a carpeted floor.", "the ground surface of an indoor room."],
    "ceiling": ["the ceiling of a room."],
    "table": ["a table in a room.", "a dining table.", "a wooden table."],
    "door": ["a closed interior door.", "the door of a room."],
    "window": ["a window of a room.", "a glass window."],
}

_clip_cache = {}


def embed_prompts(class_names, prompt_mode, device):
    key = (tuple(class_names), prompt_mode)
    if key in _clip_cache:
        return _clip_cache[key]
    if "model" not in _clip_cache:
        model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
        _clip_cache["model"] = model.eval().to(device)
        _clip_cache["tok"] = open_clip.get_tokenizer(CLIP_MODEL)
    model, tok = _clip_cache["model"], _clip_cache["tok"]
    out = []
    with torch.no_grad():
        for name in class_names:
            if prompt_mode == "raw":
                prompts = [name]
            elif prompt_mode == "templates":
                prompts = [t.format(name) for t in GENERIC_TEMPLATES]
            elif prompt_mode == "indoor":
                prompts = [t.format(name) for t in GENERIC_TEMPLATES] + INDOOR_EXTRA.get(name, [])
            else:
                raise ValueError(prompt_mode)
            feats = model.encode_text(tok(prompts).to(device)).float()
            feats = F.normalize(feats, dim=-1).mean(0)
            out.append(F.normalize(feats, dim=0))
    res = torch.stack(out)
    _clip_cache[key] = res
    return res


def classify(pooled_unit, text_feats, calibration):
    sim = pooled_unit @ text_feats.T  # (R, K)
    if calibration == "none":
        pass
    elif calibration == "zscore":
        sim = (sim - sim.mean(0, keepdim=True)) / sim.std(0, keepdim=True).clamp_min(1e-6)
    elif calibration == "center":
        sim = sim - sim.mean(0, keepdim=True)
    elif calibration == "rank":
        sim = sim.argsort(dim=0).argsort(dim=0).float() / max(sim.shape[0] - 1, 1)
    else:
        raise ValueError(calibration)
    return sim.argmax(dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="all")
    p.add_argument("--class-sets", default="all")
    p.add_argument("--prompts", default="raw,templates,indoor")
    p.add_argument("--calibrations", default="none,zscore,center,rank")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scenes = SCENES if args.scenes == "all" else {s: SCENES[s] for s in args.scenes.split(",")}
    class_sets = CLASS_SETS if args.class_sets == "all" else args.class_sets.split(",")
    prompt_modes = args.prompts.split(",")
    calibrations = args.calibrations.split(",")

    results = {}
    for scene, split in scenes.items():
        gt_dir = rf"D:\Downloads\scannet_pointcept\{split}\{scene}"
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
        centers, radii = load_foam(f"output/scannet_{scene}_nonfrozen", device)
        solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen.pt",
                            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        valid_mask = solved["valid_mask"].cpu().numpy()
        vm_t = torch.from_numpy(valid_mask).to(device)
        vi = torch.where(vm_t)[0]
        unit = F.normalize(feats[vi], dim=-1)
        positions = torch.from_numpy(centers[vi.cpu().numpy()]).to(device).float()

        leaf = two_level_position_aware(positions, unit, seed=0)
        pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
        pooled.index_add_(0, leaf, unit)
        nonempty = pooled.norm(dim=-1) > 1e-8
        pooled = F.normalize(pooled, dim=-1)

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
        owned = assigned >= 0
        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())

        for cs in class_sets:
            kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if name_to_id[n] in present]
            target_ids = [i for i, _ in kept]
            target_names = [n for _, n in kept]
            gt_t = torch.from_numpy(remap_gt_labels(raw_labels, target_ids)).long()
            for pm in prompt_modes:
                text_feats = embed_prompts(target_names, pm, device)
                for cal in calibrations:
                    cls = torch.full((K_FLAT,), 0, dtype=torch.long, device=device)
                    cls[nonempty] = classify(pooled[nonempty], text_feats, cal)
                    prim_class = np.zeros(centers.shape[0], dtype=np.int64)
                    prim_class[vi.cpu().numpy()] = cls[leaf].cpu().numpy()
                    pred = np.zeros(gt_points.shape[0], dtype=np.int64)
                    pred[owned] = prim_class[assigned[owned]] + 1
                    _, miou, acc, macc = calculate_metrics(
                        gt_t, torch.from_numpy(pred).long(), len(target_ids) + 1)
                    results.setdefault((pm, cal, cs), {})[scene] = (miou, macc)
                    print(f"  {scene} {cs} prompt={pm} cal={cal}: mIoU={miou:.4f} mAcc={macc:.4f}", flush=True)

    print("\n=== averages over evaluated scenes ===")
    summary = {}
    for (pm, cal, cs), per_scene in sorted(results.items()):
        mi = float(np.mean([v[0] for v in per_scene.values()]))
        ma = float(np.mean([v[1] for v in per_scene.values()]))
        summary[f"{pm}|{cal}|{cs}"] = {"mean_mIoU": mi, "mean_mAcc": ma, "n": len(per_scene)}
        print(f"{pm:>10} {cal:>7} {cs}: {mi*100:.2f}/{ma*100:.2f} (n={len(per_scene)})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
