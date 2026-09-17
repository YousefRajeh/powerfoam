"""Camera-free per-class surface metric for every manifest baseline, on the same footing as our arms.

WHY THIS NEEDS A SHAPE-RECOVERY STEP. `make_baseline_eval_inputs.py` normalises each baseline to a
(ckpt, features) pair carrying only `means` and `opacities` -- everything the point-mIoU protocol
needs, since that protocol assigns GT points to the nearest centre and never asks how big a Gaussian
is. The camera-free surface metric evaluates the DENSITY FIELD, which needs `scales` and `quats`
too, so those have to come back from the reconstruction each baseline lifted onto:
`ARM_CKPT` in that script shows every baseline uses OUR gs_froz / gs_unfroz reconstruction, so the
source is on disk and no baseline is being re-fit or re-trained here.

Each baseline prunes that reconstruction differently (VALA keeps ~12% via its Weiszfeld gate, Occam
keeps ~55%), and no index back to the source is stored -- only the surviving `means`. We therefore
recover the mapping by EXACT COORDINATE MATCH: a KD-tree lookup of each kept centre in the source
centres. The match is asserted, not assumed -- if any centre is further than `--match-tol` from its
nearest source centre the tag is skipped and reported, so a silent mis-association can never quietly
produce a plausible-looking row.

Everything downstream is identical to run_camera_free_gs.py: negatives-rejected per-primitive
classification, per-class voxelised isosurface, scored against the class-restricted GT mesh.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")
MANIFEST = os.environ.get("PF_MANIFEST", "artifacts/baseline_eval/manifest.json")
RECON_ROOT = os.environ.get("PF_RECON_ROOT", "recon_remote")
ARM_CKPT = {"frozen": RECON_ROOT + "/gs_froz/{scene}/ckpt.pt",
            "unfrozen": RECON_ROOT + "/gs_unfroz/{scene}/ckpt.pt"}


def load_source_splats(arm, scene):
    d = torch.load(ARM_CKPT[arm].format(scene=scene), map_location="cpu", weights_only=False)
    sp = d["splats"] if "splats" in d else d
    return (sp["means"].detach().float().numpy(),
            torch.exp(sp["scales"].detach().float()).numpy(),
            sp["quats"].detach().float().numpy())


def recover_shapes(means, arm, scene, tol):
    """Map each kept centre onto its source Gaussian and return (scales, quats, max_dist)."""
    from scipy.spatial import cKDTree
    src_means, src_scales, src_quats = load_source_splats(arm, scene)
    dist, idx = cKDTree(src_means).query(means)
    return src_scales[idx], src_quats[idx], float(dist.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--tags", nargs="*", default=None, help="default: every tag in the manifest")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--match-tol", type=float, default=1e-4, help="metres; exact match expected")
    ap.add_argument("--out", default="artifacts/cf_baselines_k1.json")
    a = ap.parse_args()

    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from mesh_surface import MeshSurfaceIndex, load_mesh, semantic_surface_metrics_mesh
    from semantic_reject import CANONICAL_NEGATIVES, classify_with_rejection
    from surface_extract import gaussian_volume, grid_from_bbox, isosurface_samples

    enable_determinism()
    man = json.load(open(MANIFEST))
    by_key = {(e["tag"], e["scene"]): e for e in man}
    tags = a.tags or sorted({e["tag"] for e in man})
    rows = []
    if os.path.exists(a.out):                       # resume: never recompute a finished row
        rows = json.load(open(a.out))
    done = {(r["tag"], r["scene"]) for r in rows}
    print(f"{len(tags)} tags x {len(a.scenes)} scenes, {len(done)} already done", flush=True)

    for scene in a.scenes:
        mesh = load_mesh(scene)
        V = np.asarray(mesh.vertices)
        _, raw, all_names = load_scannet_pointcept_gt(
            os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
        if len(V) != len(raw):
            print(f"[MISALIGNED] {scene}", flush=True)
            continue
        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw).tolist())
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
        text = embed_class_names(names, "cuda")
        neg = embed_class_names(CANONICAL_NEGATIVES, "cuda")
        idx = MeshSurfaceIndex(scene, gt_lab, len(names) + 1)
        lo, dims = grid_from_bbox(V.min(0), V.max(0), a.voxel)
        print(f"\n=== {scene} ({len(names)} classes, GT {mesh.get_surface_area():.1f} m2) ===",
              flush=True)

        for tag in tags:
            if (tag, scene) in done:
                continue
            e = by_key.get((tag, scene))
            if e is None or not (os.path.exists(e["ckpt"]) and os.path.exists(e["features"])):
                print(f"  [miss] {tag}", flush=True)
                continue
            sp = torch.load(e["ckpt"], map_location="cpu", weights_only=False)["splats"]
            # unfrozen checkpoints are saved mid-optimisation, so their tensors still
            # carry requires_grad -- detach before leaving torch
            means = sp["means"].detach().float().numpy()
            opac = torch.sigmoid(sp["opacities"].detach().float()).numpy().reshape(-1)
            feats = torch.load(e["features"], map_location="cpu",
                               weights_only=True).detach().float()
            try:
                scales, quats, mx = recover_shapes(means, e["arm"], scene, a.match_tol)
            except FileNotFoundError:
                print(f"  [no source recon] {tag}", flush=True)
                continue
            if mx > a.match_tol:
                print(f"  [SKIP] {tag}: centres do not match the source reconstruction "
                      f"(max {mx:.2e} m > {a.match_tol:.0e}); shapes cannot be recovered",
                      flush=True)
                continue

            cls = classify_with_rejection(feats.cuda(), text, neg, a.margin).cpu().numpy()
            dens, cvol = gaussian_volume(means, scales, quats, opac, cls, lo, dims, a.voxel)
            pts, pc, areas = isosurface_samples(dens, cvol, lo, a.voxel, a.alpha)
            if len(pts) == 0:
                print(f"  {tag:24} EMPTY at alpha {a.alpha}", flush=True)
                continue
            m = semantic_surface_metrics_mesh(idx, pts, pc)
            rec = {"tag": tag, "scene": scene, "arm": e["arm"], "alpha": a.alpha,
                   "n_prims": int(len(means)), "kept_frac": e.get("kept_frac"),
                   "claimed_frac": float((cls > 0).mean()),
                   "area_m2": float(sum(areas.values())),
                   "gt_area_m2": float(mesh.get_surface_area())}
            rec.update({k: float(m[k]) for k in
                        ("scd", "hd95", "boundary_f1", "n_missed", "n_classes_present")
                        if isinstance(m.get(k), (int, float))})
            rows.append(rec)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"  {tag:24} {len(means):>9,} prims  claims {rec['claimed_frac']:5.1%}  "
                  f"area {rec['area_m2']:7.1f}  SCD {100*rec['scd']:6.2f}cm  "
                  f"HD95 {100*rec['hd95']:7.2f}cm  BF1 {rec['boundary_f1']:.3f}  "
                  f"missed {rec.get('n_missed', 0):.0f}", flush=True)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
