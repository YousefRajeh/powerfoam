"""The "Ours + dipole" row of Paper A's Table 4, on BOTH the frozen and unfrozen foam.

WHAT THIS ADDS TO run_percell_masked.py, AND NOTHING ELSE. The table's four reconstruction rows
are plain per-primitive cosine argmax. This row adds exactly one operator on top of that argmax:

    dipole_fill  a cell KEEPS ITS OWN prediction wherever it has one, and borrows its geometric
                 segment's pooled prediction only where it has none.

That is a pure COVERAGE operator. It cannot change a cell the lifting already reached, so it
cannot be confused with a smoothing or refinement method -- which is the point, since the rest of
the table is deliberately free of those. `dipole_pool` (every cell takes its segment's pooled
feature, overwriting good cells too) is also reported, because the honest comparison for a
coverage claim is against the version that is allowed to overwrite.

WHY THE SEGMENTS ARE A FOAM PROPERTY. Two adjacent cells are joined when their dipole surfaces are
coplanar: normals nearly parallel (|cos| > TAU_N) and the mutual plane offsets small relative to
the radii (< TAU_D). Connected components of that relation are the geometric segments. Both tests
read the primitive's OWN surface parameters -- a Gaussian has no surface to be coplanar with, so
this operator has no 3DGS analogue and is not something the Gaussian rows are being denied.

CONSTANTS ARE INHERITED, NOT RETUNED. TAU_N=0.98, TAU_D=1.0 are taken verbatim from
run_dipole_stack_eval.py, where they were the deliberately mid-range setting of the pilot sweep
rather than its best-scoring one. They are not re-swept here: this row reports what the operator
does at a fixed, previously-chosen setting, on scenes that setting was not chosen on.

SCORED UNDER THE SAME PROTOCOL AS THE REST OF THE TABLE -- OpenGaussian's low-opacity GT deletion
included (see run_percell_masked.py for why the foam's alpha needs a path-length proxy and why the
choice does not matter). A coverage operator changes which points are classifiable, so `cov` is
reported for every arm: under the coverage law the classifiable fraction is what predicts mIoU,
and two arms with the same coverage are doing the same thing however differently they are named.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from run_percell_masked import OPACITY_THRESH, SPLIT, primitive_alpha

DB = "artifacts/ablation.sqlite"
SCENES = list(SPLIT)
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
TAU_N, TAU_D = 0.98, 1.0        # inherited from run_dipole_stack_eval.py; NOT retuned here

ARMS = {
    "pf_tfroz":   ("output/scannet_{s}_truefrozen/model.pt",
                   "artifacts/scannet/{s}/solved_geometric_median_truefrozen_ogl3.pt"),
    "pf_nonfroz": ("output/scannet_{s}_nonfrozen/model.pt",
                   "artifacts/scannet/{s}/solved_geometric_median_nonfrozen_ogl3.pt"),
}


def quat_normal(q):
    """Dipole surface normal from the primitive's quaternion (verbatim from
    run_dipole_stack_eval.py so the segments are identical to the ones measured there)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def geometric_segments(P, radii, Nrm, adjacent, offsets, n_prim, dev):
    src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                  offsets[1:] - offsets[:-1])
    k = src < adjacent
    i, j = src[k], adjacent[k]
    cos_n = (Nrm[i] * Nrm[j]).sum(-1).abs().clamp(0, 1)
    dp = P[j] - P[i]
    rr = (radii[i] + radii[j]).clamp_min(1e-20)
    offs = ((dp * Nrm[i]).sum(-1).abs() + (dp * Nrm[j]).sum(-1).abs()) / rr
    geo = (cos_n > TAU_N) & (offs < TAU_D)
    ii, jj = i[geo].cpu().numpy(), j[geo].cpu().numpy()
    G = sp.coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n_prim, n_prim))
    ns, lab = connected_components(G, directed=False)
    return ns, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recons", default=",".join(ARMS))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--no-surface", action="store_true")
    ap.add_argument("--no-mask", action="store_true",
                    help="score WITHOUT OpenGaussian's opacity deletion. Off-protocol, and only "
                         "for isolating how much of the dipole's coverage gain lands on points "
                         "the mask deletes anyway. Never write these to the paper table.")
    ap.add_argument("--out", default="artifacts/scannet/dipole_row.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    con = sqlite3.connect(DB, timeout=120)
    rows, skipped = [], []

    for recon in a.recons.split(","):
        mtmpl, ftmpl = ARMS[recon]
        for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
            mp, fp = mtmpl.format(s=scene), ftmpl.format(s=scene)
            apth = f"artifacts/ablation_cache/{scene}_{recon}_assign.npy"
            if not all(os.path.exists(p) for p in (mp, fp, apth)):
                skipped.append(f"{recon}/{scene}: missing artifact")
                continue
            t0 = time.time()
            m = torch.load(mp, map_location="cpu", weights_only=False)
            P = m["points"].float().to(dev)
            radii = F.softplus(m["radii"].float().to(dev), beta=100)
            Nrm = quat_normal(m["quaternions"].float().to(dev))
            adjacent = m["adjacency"].long().to(dev)
            offsets = m["adjacency_offsets"].long().to(dev)
            n_prim = P.shape[0]
            del m

            d = torch.load(fp, map_location=dev, weights_only=True)
            feats = d["primitive_features"].to(dev).float()
            valid = d["valid_mask"].cpu().numpy()
            vt = torch.from_numpy(valid).to(dev)
            unit = torch.zeros_like(feats)
            unit[vt] = F.normalize(feats[vt], dim=-1)
            if feats.shape[0] != n_prim:
                skipped.append(f"{recon}/{scene}: feats {feats.shape[0]} vs model {n_prim}")
                continue

            alpha = primitive_alpha(recon, scene, a.mult)
            assign = np.load(apth)
            owned = assign >= 0
            ns, lab = geometric_segments(P, radii, Nrm, adjacent, offsets, n_prim, dev)
            st = torch.from_numpy(lab).long().to(dev)
            print(f"[{recon}/{scene}] {n_prim:,} cells, {ns:,} segments, "
                  f"{100*valid.mean():.1f}% have a feature", flush=True)

            gt_pts, raw, names_all = load_scannet_pointcept_gt(
                rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
            pts = np.asarray(gt_pts, dtype=np.float64)
            n2i = {n: q for q, n in enumerate(names_all)}
            present = set(np.unique(raw).tolist())
            low = np.zeros(len(pts), dtype=bool)
            if not a.no_mask:
                low[owned] = alpha[assign[owned]] < OPACITY_THRESH

            for cs in CLASS_SETS:
                names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
                gt = remap_gt_labels(raw, [n2i[n] for n in names])
                nc = len(names) + 1
                text = embed_class_names(names, dev)
                gt_m = gt.copy()
                gt_m[low] = 0

                base_cls = (unit @ text.T).argmax(-1).cpu().numpy() + 1
                # segment-pooled class, and whether a segment carries any feature at all
                pooled = torch.zeros(ns, unit.shape[1], device=dev).index_add_(0, st[vt], unit[vt])
                cnt = torch.zeros(ns, device=dev).index_add_(
                    0, st[vt], torch.ones(int(vt.sum()), device=dev))
                seg_c = (F.normalize(pooled, dim=-1) @ text.T).argmax(-1).cpu().numpy() + 1
                sc_cls, sc_live = seg_c[lab], (cnt > 0).cpu().numpy()[lab]

                fill_cls = base_cls.copy()
                fill_cls[~valid] = sc_cls[~valid]      # borrow ONLY where the cell is blind
                arms = {"percell-argmax": (base_cls, valid),
                        "percell-argmax+dipfill": (fill_cls, valid | sc_live),
                        "dipole-pool": (sc_cls, sc_live)}

                for tag, (cls, live) in arms.items():
                    sc = owned.copy()
                    sc[owned] = live[assign[owned]]
                    pred = np.zeros(len(gt_m), dtype=np.int64)
                    pred[sc] = cls[assign[sc]]
                    _, miou, _, macc = calculate_metrics(
                        torch.from_numpy(gt_m).long(), torch.from_numpy(pred).long(), nc)
                    rec = {"recon": recon, "scene": scene, "class_set": cs, "arm": tag,
                           "miou": float(miou) * 100, "macc": float(macc) * 100,
                           "cov": float(sc.mean()) * 100, "n_segments": int(ns)}
                    if not a.no_surface:
                        sm = semantic_surface_metrics(GTSurfaceIndex(pts, gt_m, nc), pred)
                        rec.update({k: float(sm[k]) for k in
                                    ("mae_pred2gt", "mae_gt2pred", "scd", "hd95",
                                     "boundary_f1", "n_missed")
                                    if isinstance(sm.get(k), (int, float))})
                        con.execute(
                            "INSERT OR IGNORE INTO results_unified (scene,recon,features,solver,"
                            "method,family,class_set,n_classes,miou,macc,coverage,scd,mae_pred2gt,"
                            "mae_gt2pred,hd95,boundary_f1,n_missed,assignment,masked,source,"
                            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                            "datetime('now'))",
                            (scene, recon, "ogl3", "geometric_median",
                             f"{tag}(tn{TAU_N},td{TAU_D})+opacitymask@{OPACITY_THRESH}"
                             if tag != "percell-argmax"
                             else f"percell-argmax+opacitymask@{OPACITY_THRESH}",
                             "dipole" if tag != "percell-argmax" else "percell",
                             cs, nc - 1, rec["miou"], rec["macc"], rec["cov"], sm.get("scd"),
                             sm.get("mae_pred2gt"), sm.get("mae_gt2pred"), sm.get("hd95"),
                             sm.get("boundary_f1"), sm.get("n_missed"), "geometric", 1,
                             "run_dipole_seg.py"))
                        con.commit()
                    rows.append(rec)
                    print(f"   {cs[11:]:>3} {tag:<24} mIoU={rec['miou']:6.2f} "
                          f"mAcc={rec['macc']:6.2f} cov={rec['cov']:6.2f}%", flush=True)
            print(f"[done] {recon}/{scene} {(time.time()-t0)/60:.1f} min", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print(f"\nwrote {len(rows)} rows -> {a.out}")
    for s in skipped:
        print("  SKIPPED", s)


if __name__ == "__main__":
    main()
