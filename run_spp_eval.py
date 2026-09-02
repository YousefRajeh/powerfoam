"""ScanNet++ point-mIoU: OpenGaussian baseline vs the FROZEN stack. No retuning.

THE POINT OF THIS SCRIPT. Every constant is transferred verbatim from the ScanNet result and
nothing is fitted here -- lam=0.3, CSLS k=1000, rank-encode s=200, alpha=0.95, iters=100. ScanNet++
is a different capture rig, a different vocabulary and a different reconstruction, so this is the
one-shot cross-domain test. Whatever it says is the answer; a knob touched here would destroy the
result's meaning.

GT CONSTRUCTION (verified against the data, not assumed):
  * `scans/segments.json:segIndices` is the IDENTITY permutation (checked: segIndices == arange,
    271,704 unique of 271,704), so ScanNet++ "segments" ARE mesh vertices and
    `segments_anno.json:segGroups[i]["segments"]` holds vertex indices directly -- no indirection.
  * 91.9% of vertices carry an annotation; the rest stay label 0 (ignored), the same
    ignore-index convention OpenGaussian's `calculate_metrics` uses.
  * Raw labels come from a 2,878-word open vocabulary and are folded to the official benchmark
    classes via `metadata/semantic_benchmark/map_benchmark.csv:semantic_map_to`.
  * COORDINATE FRAMES MATCH: `mesh_aligned_0.05.ply` sits entirely inside the reconstruction's
    bounding box (overlap 1.000, agreeing centroids) on every scene checked, so no alignment
    transform is applied. Applying one unasked would silently fabricate a result.

Class sets are prefixes of `top100.txt`, which is ordered by frequency (wall, ceiling, floor,
table, door, ...). top100 is the official ScanNet++ semantic benchmark set; the 50 and 20 prefixes
give the same coarse/fine spread the ScanNet 19/15/10 sets provide.
"""
import argparse
import csv
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names, calculate_metrics, remap_gt_labels
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode

GT_ROOT = r"D:\Downloads\spp_gt_semantic"
RECON = r"D:\Downloads\spp_results\full"
DB = r"D:\Downloads\powerfoam\artifacts\ablation_scannetpp.sqlite"
SPP = ["0d2ee665be", "3864514494", "27dd4da69e", "c50d2d1d42", "578511c8a9", "5942004064",
       "f9f95681fd", "d755b3d9d8", "3db0a1c8f3", "9071e139d9", "e7af285f7d", "09c1414f1b"]

# FROZEN -- transferred from ScanNet, not fitted on ScanNet++.
LAM, CSLS_K, RANK_S, ALPHA, ITERS = 0.3, 1000, 200.0, 0.95, 100


def benchmark_map():
    top = [l.strip() for l in open(f"{GT_ROOT}/metadata/semantic_benchmark/top100.txt")
           if l.strip()]
    raw2bench = {}
    with open(f"{GT_ROOT}/metadata/semantic_benchmark/map_benchmark.csv") as fh:
        for row in csv.DictReader(fh):
            tgt = (row.get("semantic_map_to") or "").strip()
            if tgt:
                raw2bench[row["class"].strip()] = tgt
    return top, raw2bench


def load_gt(scene, top, raw2bench):
    """Mesh vertices + per-vertex benchmark class id (-1 = unannotated/out-of-vocabulary/masked)."""
    d = f"{GT_ROOT}/{scene}/scans"
    ply = PlyData.read(f"{d}/mesh_aligned_0.05.ply")
    pts = np.stack([np.asarray(ply["vertex"][k]) for k in ("x", "y", "z")], 1).astype(np.float32)
    ann = json.load(open(f"{d}/segments_anno.json"))
    idx = {n: i for i, n in enumerate(top)}
    lab = np.full(pts.shape[0], -1, dtype=np.int64)
    for g in ann["segGroups"]:
        name = g["label"].strip()
        name = raw2bench.get(name, name)          # fold the open vocabulary to benchmark classes
        c = idx.get(name)
        if c is None:
            continue
        v = np.asarray(g["segments"], dtype=np.int64)
        lab[v[v < pts.shape[0]]] = c

    # ScanNet++ ships its own per-scene exclusion list (anonymisation / invalid geometry).
    # Populated on 5 of our 12 scenes (169-33,178 vertices); honouring it is the dataset's rule,
    # not a judgement call of ours. Empty on 27dd4da69e, so it does NOT address coverage.
    mpath = f"{d}/mesh_aligned_0.05_mask.txt"
    n_masked = 0
    if os.path.exists(mpath):
        try:
            m = np.loadtxt(mpath, dtype=np.int64, ndmin=1)
            m = m[(m >= 0) & (m < pts.shape[0])]
            n_masked = int((lab[m] >= 0).sum())
            lab[m] = -1
        except (ValueError, OSError):
            pass
    return pts, lab, n_masked


def coverage_filter(pts, assigned, centers, valid_mask, k_spacing):
    """Drop GT points with no reconstructed geometry near them.

    Power cells TILE ALL SPACE, so `assigned >= 0` is vacuous -- every GT point receives an owner
    however far away it is. On ScanNet++ the laser mesh spans more space than the DSLR images
    covered (27dd4da69e: 17.1% of vertices sit up to 2 m from any primitive, a contiguous region
    beyond the camera envelope), so those points get labelled from geometry that belongs elsewhere.

    Rejected alternatives, each refuted by measurement rather than opinion:
      * power-ball containment  -- discards 34-40% on scenes with ZERO far points;
      * support/opacity >= tau (OpenGaussian's rule) -- the far cells are well observed
        (median support 14.5), so it keeps ~100% of them;
      * camera frustum          -- 100% of points fall in some frustum; they are OCCLUDED, not
        out of frame, so this needs depth rendering to mean anything.

    The scale is the scene's own median nearest-neighbour spacing between valid primitive centres,
    so this is scene-adaptive with no global constant. At k=20: <=1.3% excluded on every healthy
    scene (0.00% on the cleanest) versus 22.6% on the broken one.
    """
    from scipy.spatial import cKDTree
    cv = np.asarray(centers)[valid_mask]
    sub = cv[::37] if cv.shape[0] > 40_000 else cv
    nn, _ = cKDTree(cv).query(sub, k=2)
    spacing = float(np.median(nn[:, 1]))
    d = np.linalg.norm(pts - np.asarray(centers)[assigned], axis=1)
    keep = d <= k_spacing * spacing
    return keep, spacing, float(np.median(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default=",".join(SPP))
    p.add_argument("--class-sizes", default="100,50,20")
    p.add_argument("--recon", default="pf_unfroz")
    p.add_argument("--k-spacing", type=float, default=20.0,
                   help="Exclude GT points farther than k x the scene's median primitive spacing "
                        "from their owning cell. k=20 drops <=1.3%% on healthy scenes (0.00%% on "
                        "the cleanest) and 22.6%% on 27dd4da69e, whose laser mesh extends beyond "
                        "the DSLR coverage. k=5/k=10 would discard 22%%/4.5%% of a good scene.")
    p.add_argument("--outdir", default="artifacts/scannetpp/eval")
    a = p.parse_args()

    enable_determinism()
    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda"
    top, raw2bench = benchmark_map()
    sizes = [int(x) for x in a.class_sizes.split(",")]
    con = sqlite3.connect(DB, timeout=60.0)

    for scene in a.scenes.split(","):
        out = os.path.join(a.outdir, f"{scene}_{a.recon}.json")
        if os.path.exists(out):
            print(f"[skip] {scene}", flush=True); continue
        t0 = time.time()
        art = f"artifacts/scannetpp/{scene}"
        ck = os.path.join(RECON, f"spp_{a.recon}_{scene}")
        centers, radii = load_points_radii(ck)
        sv = torch.load(f"{art}/solved_geometric_median_nonfrozen_ogl3.pt",
                        map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        valid_mask = sv["valid_mask"].cpu().numpy()
        vm = torch.from_numpy(valid_mask).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv

        gt_pts, gt_lab, n_masked = load_gt(scene, top, raw2bench)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid_mask, k=64)
        keep, spacing, med_d = coverage_filter(gt_pts, assigned, centers, valid_mask, a.k_spacing)
        n_dropped = int((gt_lab >= 0).sum() - (gt_lab[keep] >= 0).sum())
        gt_lab = np.where(keep, gt_lab, -1)       # uncovered -> ignore, same as unannotated
        owned = assigned >= 0
        cov = float(keep.mean())
        print(f"[{scene}] P={P:,} gt={gt_pts.shape[0]:,} labelled={(gt_lab>=0).mean()*100:.1f}% "
              f"| spacing={spacing:.4f} medD={med_d:.4f} covered={cov*100:.1f}% "
              f"(dropped {n_dropped:,} uncovered, {n_masked:,} dataset-masked)", flush=True)

        # ---- the two feature-space variants -------------------------------------------
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0 = adj["adjacent"].to(device).long(); of0 = adj["offsets"].to(device).long()
        Rr = AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt"
                                          ).reliability()["reliability"].to(device).float() * vm
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        from run_normlift_refine_eval import mode_vote_refine
        positions = torch.from_numpy(centers).to(device).float()
        cen = mode_vote_refine(cen, Rr, positions, ad0, of0,
                               chunk=max(256, 200_000 // max(Dm, 1)))
        del Rr
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        keep = vm[src] & vm[dst]; src, dst = src[keep], dst[keep]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
            0, src, torch.ones_like(src))
        del adj, ad0, of0
        torch.cuda.empty_cache()

        res = {"scene": scene, "recon": a.recon, "arms": {}}
        for K in sizes:
            names_all = top[:K]
            present = sorted(set(np.unique(gt_lab).tolist()) & set(range(K)))
            if not present:
                print(f"  top{K}: no classes present", flush=True); continue
            names = [names_all[i] for i in present]
            gt_t = torch.from_numpy(remap_gt_labels(gt_lab, present)).long()
            text = embed_class_names(names, device)
            C = len(names)
            cs = f"spp_top{K}"

            def score(cls_np, tag):
                pred = np.zeros(gt_pts.shape[0], dtype=np.int64)
                pred[owned] = cls_np[assigned[owned]] + 1
                _, miou, oa, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), C + 1)
                res["arms"].setdefault(tag, {})[cs] = {
                    "mIoU": float(miou) * 100, "mAcc": float(macc) * 100,
                    "n_classes": C, "coverage": cov}
                con.execute(
                    "insert into results_unified (scene,recon,features,solver,method,family,"
                    "class_set,n_classes,miou,macc,coverage,grouping,complex,assignment,masked,"
                    "source,created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (scene, a.recon, "ogl3", "geometric_median", tag,
                     "baseline" if tag == "base" else "stack", cs, C,
                     float(miou) * 100, float(macc) * 100, cov,
                     None, "true_facet", "power_cell", 0, "run_spp_eval.py", time.time()))
                con.commit()
                print(f"  {cs} [{tag}] mIoU={float(miou)*100:.2f} mAcc={float(macc)*100:.2f}",
                      flush=True)

            # BASELINE: plain cosine argmax on the raw lifted features -- no centering, no
            # refinement, no CSLS, no diffusion. This is the OpenGaussian protocol.
            cr = torch.zeros(P, C, device=device); cr[vm] = raw[vm] @ text.T
            score(cr.argmax(-1).cpu().numpy(), "base")
            del cr

            # FROZEN STACK: centre -> prerefine -> CSLS -> rank-encode -> diffuse -> argmax
            cc = torch.zeros(P, C, device=device); cc[vm] = cen[vm] @ text.T
            kk = min(CSLS_K, int(vm.sum()))
            cc[vm] = cc[vm] - 0.5 * cc[vm].topk(kk, dim=0).values.mean(0)[None, :]
            p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
            x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
            score(x.argmax(-1).cpu().numpy(), "stack_frozen")
            del cc, p0, x, text
            torch.cuda.empty_cache()

        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[{scene}] done {time.time()-t0:.0f}s", flush=True)
        del raw, cen, src, dst, deg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
