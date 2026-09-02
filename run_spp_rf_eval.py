"""ScanNet++ evaluation for the RadFoam arm, matching the PowerFoam protocol exactly.

Only three things differ from `run_overnight.stage_spp_full`, and all are properties of the
representation rather than choices:

  * CENTRES come from the RadFoam checkpoint (`radfoam_adapter.load_radfoam_foam`), whose radii are
    exactly ZERO -- RadFoam cells are Voronoi, i.e. power cells with null weights. So
    `assign_points_to_power_cells(..., radii=0)` is the correct nearest-centre membership, the same
    exact-partition query used for PowerFoam, not an approximation.
  * ADJACENCY is read straight out of the checkpoint: RadFoam persists its own Delaunay CSR
    (`adjacency` / `adjacency_offsets`), which is the same object `build_true_facet_graph` derives
    for PowerFoam. No graph is re-derived.
  * The solved artefact is `solved_gm_rf_unfroz_ogl3.pt`, produced by radfoam's own
    `scripts/lift_clip_features.py` (the lift needs RadFoam's renderer, which is why it runs in WSL).

Everything downstream -- constants, ablation ladder, coverage variants, surface metrics, DB schema
-- is shared with the PowerFoam path so a delta between the arms is attributable to the
reconstruction, not to two different evaluators.
"""
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

from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from radfoam_adapter import load_radfoam_checkpoint, load_radfoam_foam, native_csr
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine
from eval_semantic_surface import semantic_surface_metrics
from run_overnight import (SPP, DB, RECON, LAM, CSLS_K, RANK_S, ALPHA, ITERS,
                           log, csls, score_pred)
from run_spp_eval import benchmark_map, load_gt, coverage_filter

ART = "artifacts/scannetpp_rf"


def rf_checkpoint(scene):
    for v in (f"spp_rf_unfroz4x_{scene}", f"spp_rf_unfroz_{scene}"):
        p = os.path.join(RECON, v, "model.pt")
        if os.path.exists(p):
            return p, v
    return None, None


def run(outdir="artifacts/scannetpp/eval_rf", k_list=(20.0, 0.0), text_lowdin=False):
    device = "cuda"
    os.makedirs(outdir, exist_ok=True)
    top, raw2bench = benchmark_map()
    con = sqlite3.connect(DB, timeout=60.0)
    for scene in SPP:
        solved = f"{ART}/{scene}/solved_gm_rf_unfroz_ogl3.pt"
        stats = f"{ART}/{scene}/stats_rf_unfroz_ogl3.pt"
        ck, variant = rf_checkpoint(scene)
        if not (os.path.exists(solved) and ck):
            log(f"  [miss] rf {scene}"); continue
        outp = os.path.join(outdir, f"{scene}_rf_unfroz.json")
        if os.path.exists(outp):
            log(f"  [skip] rf {scene}"); continue
        t0 = time.time()
        centers, radii = load_radfoam_foam(ck)          # radii are exactly zero (Voronoi)
        sv = torch.load(solved, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
        P = feats.shape[0]
        if centers.shape[0] != P:
            log(f"  [skip] rf {scene}: P mismatch {P} vs {centers.shape[0]}"); continue
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)

        sd = load_radfoam_checkpoint(ck)                 # native Delaunay CSR, not re-derived
        g = native_csr(sd, centers=centers, with_dist=False)
        ad0 = g["adjacent"].to(device).long(); of0 = g["offsets"].to(device).long()
        del sd, g
        R = (AccumulatedFeatureStats.load(stats).reliability()["reliability"]
             .to(device).float() * vm) if os.path.exists(stats) else (raw.norm(dim=-1) * vm)
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        chunk = max(256, 200_000 // max(Dm, 1))
        cen_r = mode_vote_refine(cen, R, pos, ad0, of0, chunk=chunk)
        raw_r = mode_vote_refine(raw, R, pos, ad0, of0, chunk=chunk)
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del ad0, of0, R
        torch.cuda.empty_cache()

        gt_pts, gt_lab0, n_masked = load_gt(scene, top, raw2bench)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        res = {"scene": scene, "recon": "rf_unfroz", "variant": variant,
               "n_masked": n_masked, "arms": {}}
        for ks in k_list:
            if ks > 0:
                keepc, spacing, med_d = coverage_filter(gt_pts, assigned, centers, vmn, ks)
                gt_lab = np.where(keepc, gt_lab0, -1); cov = float(keepc.mean()); tagc = f"cov{ks:g}"
            else:
                gt_lab = gt_lab0; cov = 1.0; tagc = "covNONE"
            for K in (100, 50, 20):
                present = sorted(set(np.unique(gt_lab).tolist()) & set(range(K)))
                if not present: continue
                nm = [top[:K][i] for i in present]
                gt_t = torch.from_numpy(remap_gt_labels(gt_lab, present)).long()
                txt = embed_class_names(nm, device); C = len(nm)
                # Loewdin prototype orthogonalisation is REPRESENTATION-AGNOSTIC: it touches
                # only the class embeddings, never the 3D field. Applied identically to every
                # arm so the DELTA answers whether the representation changes how much a
                # decorrelated classifier can actually exploit.
                if text_lowdin:
                    from run_text_and_pseudo_eval import text_lowdin as _lowdin
                    txt = _lowdin(txt)
                cs = f"spp_top{K}"
                SURF = {"A_base", "F_full_stack", "B_normlift"}

                def emit(cls_np, tag):
                    mi, ma = score_pred(cls_np, assigned, owned, gt_t, C, gt_pts.shape[0])
                    sm = {}
                    if tag in SURF and K == 100 and ks > 0:
                        pr = np.zeros(gt_pts.shape[0], dtype=np.int64)
                        pr[owned] = cls_np[assigned[owned]] + 1
                        try:
                            sm = semantic_surface_metrics(gt_pts, gt_t.numpy(), pr, C + 1)
                        except Exception as e:
                            log(f"    [surf err] {tag}: {e}")
                    res["arms"].setdefault(f"{tag}|{tagc}", {})[cs] = {
                        "mIoU": mi, "mAcc": ma, "n_classes": C, "coverage": cov,
                        "scd": sm.get("scd"), "hd95": sm.get("hd95"),
                        "boundary_f1": sm.get("boundary_f1")}
                    con.execute(
                        "insert or replace into results_unified (scene,recon,features,solver,"
                        "method,family,class_set,n_classes,miou,macc,coverage,grouping,complex,"
                        "assignment,masked,scd,hd95,boundary_f1,mae_pred2gt,mae_gt2pred,n_missed,"
                        "source,created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (scene, "rf_unfroz", "ogl3", "geometric_median", f"{tag}|{tagc}", tagc,
                         cs, C, mi, ma, cov, None, "delaunay_native", "voronoi_cell", 0,
                         sm.get("scd"), sm.get("hd95"), sm.get("boundary_f1"),
                         sm.get("mae_pred2gt"), sm.get("mae_gt2pred"), sm.get("n_missed"),
                         "run_spp_rf_eval.py", time.time()))
                    con.commit()

                def cosof(u):
                    c = torch.zeros(P, C, device=device); c[vm] = u[vm] @ txt.T; return c

                emit(cosof(raw).argmax(-1).cpu().numpy(), "A_base")
                emit(cosof(raw_r).argmax(-1).cpu().numpy(), "B_normlift")
                emit(cosof(cen).argmax(-1).cpu().numpy(), "C_centre")
                emit(cosof(cen_r).argmax(-1).cpu().numpy(), "D_centre_refine")
                cc = csls(cosof(cen_r), vm)
                emit(cc.argmax(-1).cpu().numpy(), "E_plus_csls")
                p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
                emit(diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy(),
                     "F_full_stack")
                p0n = rank_encode(cosof(cen_r), RANK_S, device); p0n[~vm] = 0.0
                emit(diffuse(p0n, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy(),
                     "G_stack_noCSLS")
                del txt, cc, p0, p0n
                torch.cuda.empty_cache()
        with open(outp, "w") as fh:
            json.dump(res, fh, indent=1)
        log(f"  [ok] rf {scene} {time.time()-t0:.0f}s")
        del raw, cen, cen_r, raw_r, src, dst, deg, pos
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from determinism import enable_determinism
    enable_determinism()
    import sys
    kw = {}
    if "--outdir" in sys.argv: kw["outdir"] = sys.argv[sys.argv.index("--outdir") + 1]
    if "--text-lowdin" in sys.argv: kw["text_lowdin"] = True
    run(**kw)
    print("RF_EVAL DONE")
