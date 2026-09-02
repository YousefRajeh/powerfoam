"""Plain per-primitive argmax, scored WITH OpenGaussian's low-opacity GT mask.

WHY THIS EXISTS. Table 4 of Paper A compares four reconstructions under one identical pipeline:
solved features -> per-primitive cosine argmax -> nothing else. The rows already in
`results_unified` for `method='percell-argmax'` all carry `masked=0`, i.e. they do NOT apply
OpenGaussian's `sigmoid(opacity) < 0.1 -> GT label 0` deletion (`eval_scannet.py:127-129`). Every
published row we print above the rule -- OpenGaussian, LangSplat, NormLift, SFS, ... -- was scored
WITH it, because NormLift copies their evaluator verbatim. So our rows and their rows are not
scored on the same GT vector, and the gap is not a method difference.

This re-scores the same four arms with the mask on, changing nothing else. Predictions, features
and assignment are byte-identical to the unmasked run; only the GT vector differs, so the
comparison is exactly paired.

THE FOAM HAS NO sigmoid(opacity), AND THE CONVERSION IS NOT JUST AN ACTIVATION. A Gaussian's
opacity is already an alpha in (0,1), so their threshold is well defined per primitive. The foam
has no such scalar. Softplus is only the activation of the raw density parameter into sigma
(`scene.py:get_density`, beta=100); the quantity comparable to a Gaussian's opacity is the
renderer's own alpha, and that is a volume-rendering integral, not an activation:

    dt    = t_far - t_near                  # rasterize.py:870-873
    alpha = 1 - exp(-sigma * dt)

Two things make `dt` awkward to reduce to one number per primitive:

  1. It is the chord of the RAY through the power cell, so it depends on the view direction and on
     where the ray enters -- a cell has no single alpha, it has a distribution over rays.
  2. That chord is further CLIPPED BY THE PRIMITIVE'S OWN DIPOLE SURFACE (`t_surf`, `dp` from
     plane_intersection_fwd_local, rasterize.py:855-868): a ray approaching the front is cut at
     the surface rather than crossing the whole cell. So even the mean chord is strictly shorter
     than the geometric chord of the cell.

There is therefore NO choice of L that makes the foam's number mean exactly what the Gaussian's
means. We take L = mult * radius as an explicit, stated proxy and SWEEP mult, which is the only
honest handling: if the ordering of the four arms is stable across the sweep the conclusion does
not hinge on the choice, and if it is not, the masked foam number is not reportable at all. This
mirrors run_gt_opacity_mask_10scene.py, which sweeps L for the same reason.

Note the asymmetry this leaves: for the FROZEN Gaussian arm the mask is OpenGaussian's rule
literally (assignment is the identity -- verified, point i maps to Gaussian i), while for the foam
arms it is a proxy. That asymmetry is a property of the protocol, not of this script.

A MASK THAT DELETES POINTS CANNOT BE JUDGED BY mIoU ALONE -- deleting hard points raises mIoU
with no method improving -- so the dropped fraction is recorded next to every score.

Reads only cached artifacts plus the raw checkpoints (`torch.load`, no warp/scene rebuild), so it
is CPU-light and does not contend for the GPU beyond the CLIP text embedding.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from mesh_surface import MeshSurfaceIndex, semantic_surface_metrics_mesh
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

DB = "artifacts/ablation.sqlite"
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
SCENES = list(SPLIT)
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
OPACITY_THRESH = 0.1

# verbatim from backfill_surface_cross_recon.py so the arms line up with the unmasked rows
FEATURES = {
    "pf_nonfroz": "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt",
    "pf_tfroz":   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt",
    "gs_froz":    "artifacts/scannet/{s}/solved_weighted_gs_froz_ogl3.pt",
    "gs_unfroz":  "artifacts/scannet/{s}/solved_weighted_gs_unfroz_ogl3.pt",
}
# verbatim from ablation_assign.py
CKPT = {
    "pf_tfroz":   ("foam", "output/scannet_{s}_truefrozen/model.pt"),
    "pf_nonfroz": ("foam", "output/scannet_{s}_nonfrozen/model.pt"),
    "gs_froz":    ("gs",   "recon_remote/gs_froz/{s}/ckpt.pt"),
    "gs_unfroz":  ("gs",   "recon_remote/gs_unfroz/{s}/ckpt.pt"),
}


def primitive_alpha(recon, scene, mult):
    """Per-primitive alpha in (0,1), comparable to a Gaussian's opacity.

    Raw checkpoint parameters are PRE-activation (`scene.py:load_pt` assigns straight into
    `.data`), so the same activations the model uses at render time are applied here: softplus
    with beta=100 for both density and radius.
    """
    kind, tmpl = CKPT[recon]
    path = tmpl.format(s=scene)
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "gs":
        s = d["splats"] if isinstance(d, dict) and "splats" in d else d
        return torch.sigmoid(s["opacities"].float()).numpy()
    sigma = F.softplus(d["density"].float(), beta=100)
    radius = F.softplus(d["radii"].float(), beta=100)
    return (1.0 - torch.exp(-sigma * mult * radius)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(FEATURES))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--mults", default="2.0",
                    help="comma-separated path-length multipliers L = mult*radius (foam only)")
    ap.add_argument("--surface", action="store_true", help="also compute surface metrics")
    ap.add_argument("--surface-ref", choices=("vertex", "mesh"), default="vertex",
                    help="what the predicted region is compared AGAINST. `vertex` is the original "
                         "ablation_surface behaviour: both sides are subsets of the GT point cloud, "
                         "so distances have a floor at the vertex spacing (1.26 cm) and the "
                         "GT->pred direction falls for free as the prediction gets denser. `mesh` "
                         "uses exact point-to-triangle distance to the labelled surface and samples "
                         "that surface area-uniformly for the reverse direction, removing both. "
                         "The numbers currently in the paper are `vertex`.")
    ap.add_argument("--out", default="artifacts/scannet/percell_masked.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    con = sqlite3.connect(DB, timeout=120)
    mults = [float(x) for x in a.mults.split(",")]
    rows, skipped = [], []

    for mult in mults:
        for recon in a.recons.split(","):
            for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
                fp = FEATURES[recon].format(s=scene)
                apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
                if not all(os.path.exists(p) for p in (fp, apth)):
                    skipped.append(f"{recon}/{scene}: missing feature or assignment")
                    continue
                alpha = primitive_alpha(recon, scene, mult)
                if alpha is None:
                    skipped.append(f"{recon}/{scene}: missing checkpoint")
                    continue
                t0 = time.time()
                d = torch.load(fp, map_location=dev, weights_only=True)
                feats = d["primitive_features"].to(dev).float()
                valid = d["valid_mask"].cpu().numpy()
                vt = torch.from_numpy(valid).to(dev)
                unit = torch.zeros_like(feats)
                unit[vt] = F.normalize(feats[vt], dim=-1)
                if alpha.shape[0] != feats.shape[0]:
                    skipped.append(f"{recon}/{scene}: alpha {alpha.shape[0]} vs feats {feats.shape[0]}")
                    continue
                assign = np.load(apth)
                owned = assign >= 0

                gt_pts, raw, names_all = load_scannet_pointcept_gt(
                    rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
                pts = np.asarray(gt_pts, dtype=np.float64)
                n2i = {n: q for q, n in enumerate(names_all)}
                present = set(np.unique(raw).tolist())

                # OpenGaussian's rule, generalised off the frozen-point index identity: mask the
                # point whose ASSIGNED primitive is transparent. Under a frozen checkpoint the
                # assignment of point i IS primitive i, so this reduces to their rule exactly.
                # A point owned by NO primitive keeps its label and scores as a miss -- masking it
                # would let a method delete the points it fails to cover.
                low = np.zeros(len(pts), dtype=bool)
                low[owned] = alpha[assign[owned]] < OPACITY_THRESH

                for cs in CLASS_SETS:
                    names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                    gt = remap_gt_labels(raw, [n2i[n] for n in names])
                    nc = len(names) + 1
                    text = embed_class_names(names, dev)
                    cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1

                    sc = owned.copy()
                    sc[owned] = valid[assign[owned]]
                    pred = np.zeros(len(gt), dtype=np.int64)
                    pred[sc] = cls[assign[sc]]

                    scored_before = int((gt != 0).sum())
                    gt_m = gt.copy()
                    gt_m[low] = 0
                    dropped = scored_before - int((gt_m != 0).sum())

                    _, miou, _, macc = calculate_metrics(
                        torch.from_numpy(gt_m).long(), torch.from_numpy(pred).long(), nc)
                    rec = {"mult": mult, "recon": recon, "scene": scene, "class_set": cs,
                           "miou": float(miou) * 100, "macc": float(macc) * 100,
                           "dropped": dropped, "scored_before": scored_before,
                           "dropped_pct": dropped / max(scored_before, 1) * 100}

                    if a.surface:
                        # surface metrics use the MASKED GT so they describe the same scored set
                        if a.surface_ref == "mesh":
                            sm = semantic_surface_metrics_mesh(
                                MeshSurfaceIndex(scene, gt_m, nc), pts, pred)
                        else:
                            sm = semantic_surface_metrics(GTSurfaceIndex(pts, gt_m, nc), pred)
                        # `sm` also carries a nested `per_class` dict; keep only the scalars
                        rec.update({k: float(sm[k]) for k in
                                    ("mae_pred2gt", "mae_gt2pred", "scd", "hd95",
                                     "boundary_f1", "n_missed", "n_classes_present", "n_scored")
                                    if isinstance(sm.get(k), (int, float))})
                        con.execute(
                            "INSERT OR IGNORE INTO results_unified (scene,recon,features,solver,"
                            "method,family,class_set,n_classes,miou,macc,scd,mae_pred2gt,"
                            "mae_gt2pred,hd95,boundary_f1,n_missed,assignment,masked,source,"
                            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
                            (scene, recon, "ogl3",
                             "geometric_median" if recon.startswith("pf") else "weighted",
                             # the surface reference goes IN THE METHOD NAME. Without it the
                             # mesh-scored row collides with the vertex-scored row on the UNIQUE
                             # key and INSERT OR IGNORE silently keeps the OLD numbers -- the
                             # failure mode would be "the rerun did nothing and looked fine".
                             f"percell-argmax+opacitymask@{OPACITY_THRESH}"
                             + ("+meshsurf" if a.surface_ref == "mesh" else ""), "percell",
                             cs, nc - 1, rec["miou"], rec["macc"], sm.get("scd"),
                             sm.get("mae_pred2gt"), sm.get("mae_gt2pred"), sm.get("hd95"),
                             sm.get("boundary_f1"), sm.get("n_missed"), "geometric", 1,
                             "run_percell_masked.py"))
                        con.commit()
                    rows.append(rec)
                    print(f"[L={mult}] {recon:11s} {scene} {cs:15s} mIoU={rec['miou']:5.2f} "
                          f"mAcc={rec['macc']:5.2f} dropped={rec['dropped_pct']:5.2f}% "
                          f"({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    for s in skipped:
        print("  SKIPPED", s)


if __name__ == "__main__":
    main()
