"""Re-lift onto two checkpoints and score them under PLAIN PER-CELL ARGMAX only.

WHY NOT eval_checkpoint_full.py. It calls `eval_semantic_surface.py`, which offers two protocols and
neither is what a representation comparison needs:
  * `champion`      -- mode-vote refinement + position-aware two-level clustering + R-weighted
                       voting on partially-centred similarities. A stack, not a representation test.
  * `opengaussian`  -- OpenGaussian's 64x5=320 codebook, pooled cluster features, then argmax. Still
                       a GROUPING; the clustering can absorb or manufacture a difference between two
                       geometries all by itself.
Grouping strategies are a separate question and can be revisited later. For "does the exp geometry
give better features", the classifier must be the simplest thing that exists: each cell's own
feature, cosine against the raw class names, argmax. Nothing pooled, nothing refined, nothing
weighted -- the same `percell-argmax` that Table 4 uses, so the numbers are directly readable
against it.

SURFACE METRICS ARE SCORED THE SAME WAY, per the same argument: the predicted region for a class is
the set of points whose OWN cell argmaxes to that class. A clustering step upstream of the surface
metric would make the surface number a statement about the clustering.

THE REFERENCE IS THE LABELLED MESH, not its vertices (`mesh_surface.py`): exact point-to-triangle
distance for pred->GT, and area-uniform sampling for GT->pred, which removes both the vertex-spacing
floor and the density confound.

TWO FIXES THE EARLIER ATTEMPT NEEDED, recorded because both fail loudly but non-obviously:
  * `--sam-level 0`, not 3. These artifacts are single-level (SAM_ONLY_LEVEL extraction), which
    stores the chosen granularity at index 0; asking for 3 selects nothing.
  * the checkpoint's own `config.yaml` cannot be fed back to configargparse: train.py writes every
    Params field including optional ones as `null`, plus trainer-only keys it never registers. Both
    are stripped into a temp config here.

BOTH checkpoints are re-lifted through this same path, including the baseline that already has
features elsewhere. That doubles the cost and removes the confound of comparing two feature sets
that were produced differently.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from mesh_surface import MeshSurfaceIndex, semantic_surface_metrics_mesh
from point_cloud_query import assign_points_to_power_cells
from run_percell_masked import OPACITY_THRESH, SPLIT

PY = r"D:\conda\envs\powerfoam\python.exe"
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def shadow_checkpoint(ckpt_dir, work):
    """A directory holding a SANITISED `config.yaml` next to the checkpoint's `model.pt`.

    Two constraints force this shape. The config must be sanitised, because train.py writes every
    optional Params field as `null` plus trainer-only keys it never registers, and configargparse
    dies on both. But the config must ALSO be named `config.yaml` and sit beside `model.pt`,
    because accumulate_feature_stats_sam.py derives the checkpoint directory from the config path
    (`config_path.replace("/config.yaml", "")`). Passing a cleaned config from anywhere else makes
    it look for model.pt in that other place -- which is exactly how the first attempt failed.

    So: copy model.pt in and write the cleaned config beside it. The copy is ~400 MB and takes a
    few seconds; mutating the real checkpoint directory to avoid it would be worse.
    """
    import shutil
    src = os.path.join(ckpt_dir, "config.yaml")
    drop = ("ckpt_every", "resume")
    kept = []
    for ln in open(src, encoding="utf-8").read().splitlines():
        k = ln.split(":", 1)[0].strip()
        v = ln.split(":", 1)[-1].strip()
        if v in ("null", "None") or k in drop:
            continue
        kept.append(ln)
    shadow = os.path.join(work, "ckpt")
    os.makedirs(shadow, exist_ok=True)
    with open(os.path.join(shadow, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    dst_pt = os.path.join(shadow, "model.pt")
    src_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(dst_pt) or os.path.getsize(dst_pt) != os.path.getsize(src_pt):
        shutil.copyfile(src_pt, dst_pt)
    return os.path.join(shadow, "config.yaml")


def sh(cmd, tag):
    print(f"  [{tag}] {' '.join(str(c) for c in cmd[-6:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  [{tag}] FAILED rc={r.returncode}", flush=True)
        print("\n".join((r.stdout + r.stderr).splitlines()[-15:]), flush=True)
        return False
    return True


def lift(ckpt_dir, scene, feat_dir, work, sam_level):
    stats = os.path.join(work, "stats.pt")
    solved = os.path.join(work, "solved.pt")
    if os.path.exists(solved):
        print("  [lift] solved.pt exists, reusing", flush=True)
        return solved
    cfg = shadow_checkpoint(ckpt_dir, work)
    if not os.path.exists(stats):
        if not sh([PY, "accumulate_feature_stats_sam.py", "--scene", scene, "--config", cfg,
                   "--feature-folder", feat_dir, "--output", stats,
                   "--sam-level", str(sam_level)], "accumulate"):
            return None
    if not sh([PY, "solve_geometric_median.py", "--stats", stats, "--output", solved], "solve"):
        return None
    return solved


def score(ckpt_dir, solved, scene, dev):
    """Plain per-cell argmax, OpenGaussian's masked protocol, mesh surface metrics."""
    m = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=False)
    pts_c = m["points"].float().numpy().astype(np.float64)
    radii = F.softplus(m["radii"].float(), beta=100).numpy().astype(np.float64)
    cfg_txt = open(os.path.join(ckpt_dir, "config.yaml"), encoding="utf-8").read()
    act = "exp" if "density_activation: exp" in cfg_txt else "softplus"
    dens = torch.exp(m["density"].float()) if act == "exp" \
        else F.softplus(m["density"].float(), beta=100)
    alpha = (1.0 - torch.exp(-dens * 2.0 * torch.from_numpy(radii).float())).numpy()

    d = torch.load(solved, map_location=dev, weights_only=True)
    feats = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()
    vt = torch.from_numpy(valid).to(dev)
    unit = torch.zeros_like(feats)
    unit[vt] = F.normalize(feats[vt], dim=-1)

    gt_pts, raw, names_all = load_scannet_pointcept_gt(
        rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
    pts = np.asarray(gt_pts, dtype=np.float64)
    assigned = assign_points_to_power_cells(pts, pts_c, radii)
    owned = assigned >= 0
    low = np.zeros(len(pts), dtype=bool)
    low[owned] = alpha[assigned[owned]] < OPACITY_THRESH

    n2i = {n: q for q, n in enumerate(names_all)}
    present = set(np.unique(raw).tolist())
    out = {"act": act, "n_prim": int(pts_c.shape[0]),
           "valid_frac": float(valid.mean()), "owned_frac": float(owned.mean())}
    for cs in CLASS_SETS:
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        gt = remap_gt_labels(raw, [n2i[n] for n in names])
        gt[low] = 0
        nc = len(names) + 1
        text = embed_class_names(names, dev)
        cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1        # PLAIN per-cell argmax
        sc = owned.copy()
        sc[owned] = valid[assigned[owned]]
        pred = np.zeros(len(gt), dtype=np.int64)
        pred[sc] = cls[assigned[sc]]
        _, miou, _, macc = calculate_metrics(torch.from_numpy(gt).long(),
                                             torch.from_numpy(pred).long(), nc)
        rec = {"mIoU": float(miou) * 100, "mAcc": float(macc) * 100}
        if cs == "opengaussian19":
            sm = semantic_surface_metrics_mesh(MeshSurfaceIndex(scene, gt, nc), pts, pred)
            rec.update({k: sm[k] for k in ("mae_pred2gt", "mae_gt2pred", "scd", "hd95",
                                           "boundary_f1", "n_missed") if k in sm})
        out[cs] = rec
        print(f"    {cs[-2:]}cls  mIoU={rec['mIoU']:6.2f}  mAcc={rec['mAcc']:6.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--feature-dir",
                    default=r"data\scannet\scene0347_00_colmap\openclip_features_sam_l3")
    ap.add_argument("--sam-level", type=int, default=0)
    ap.add_argument("--lift-only", action="store_true",
                    help="accumulate + solve, then stop. `score()` reads ScanNet's Pointcept GT and "
                         "is meaningless for LERF, which is scored by eval_lerf_iou.py against "
                         "polygon masks instead. Without this the lift succeeds and then dies on "
                         "KeyError: 'figurines' -- work done, nothing reported.")
    ap.add_argument("--out", default="artifacts/scannet/relift_percell.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for ckpt in a.ckpt_dirs:
        name = os.path.basename(ckpt.rstrip("/\\"))
        print(f"\n=== {name} ===", flush=True)
        work = os.path.join("artifacts/scannet/relift", name)
        os.makedirs(work, exist_ok=True)
        solved = lift(ckpt, a.scene, a.feature_dir, work, a.sam_level)
        if solved is None:
            continue
        if a.lift_only:
            results[name] = {"solved": solved, "lift_only": True}
            print(f"  [lift-only] {solved}", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(results, open(a.out, "w"), indent=2)
            continue
        results[name] = score(ckpt, solved, a.scene, dev)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print(f"\n{'run':34s} " + "  ".join(f"{c[-2:]}cls mIoU" for c in CLASS_SETS))
    for k, v in results.items():
        print(f"{k:34s} " + "  ".join(f"{v[c]['mIoU']:10.2f}" for c in CLASS_SETS if c in v))


if __name__ == "__main__":
    main()
