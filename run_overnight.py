"""Unattended overnight driver: NormLift replication -> ScanNet++ full evaluation -> DB.

DESIGN RULE. Stages are ordered by (value x certainty) and each is wrapped independently: a stage
that throws logs the traceback and the next one still runs. Nothing here may leave the run in a
state where a single failure loses the whole night.

WHERE THE USER SAID "DO BOTH", BOTH ARE RUN:
  * coverage handling -- every ScanNet++ number is produced BOTH unfiltered and with the k=20
    coverage filter, as separate rows, so the effect of the protocol choice is visible rather than
    baked in;
  * neighbour graph for NormLift -- knn30 (their Euclidean KNN, faithful) AND true_facet (our exact
    power-diagram adjacency), so the graph substitution is measured, not assumed;
  * solver -- weighted (NormLift's Eq. 5) AND geometric_median (our default), because that
    difference was invisible until the paper was read properly.

NORMLIFT, NOW READ FROM THE PDF RATHER THAN INFERRED:
    Eq (5)  u*_j = f_j/||f_j||, f_j weight-normalised by sum_i A_ij  -> the WEIGHTED solver,
            NOT the geometric median this project defaults to. Every earlier "NormLift refinement"
            run in this repo sat on geometric-median features, i.e. the wrong base.
    Eq (8)  R(j) = ||f_j|| * N_eff(j)/(N_eff(j)+beta),  beta = 1
    Eq (9-10) reliability-guided mode-voting over Euclidean KNN, K~30 ("flat above K=30"),
            (sigma_d, tau, gamma, Delta) fixed across all their experiments.
    Published ScanNet (OpenGaussian protocol): 35.77/54.02 (19), 39.62/59.26 (15), 48.93/68.83 (10).
    They evaluate on 3DGS Gaussians queried independently, so `gs_froz` is the closest arm.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import traceback

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names, calculate_metrics,
    remap_gt_labels, load_scannet_pointcept_gt,
)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from point_cloud_query import assign_points_to_power_cells
from build_true_facet_graph import load_points_radii
from run_cluster_classify_eval import SCENES as SN_SCENES
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_derived_stack_eval import rank_encode
from run_normlift_refine_eval import mode_vote_refine, build_knn_csr
from eval_semantic_surface import semantic_surface_metrics

DB = r"D:\Downloads\powerfoam\artifacts\ablation_scannetpp.sqlite"
SN_DB = r"D:\Downloads\powerfoam\artifacts\ablation.sqlite"
GT_ROOT = r"D:\Downloads\spp_gt_semantic"
SN_GT = r"D:\Downloads\scannet_pointcept"
RECON = r"D:\Downloads\spp_results\full"
SPP = ["0d2ee665be", "3864514494", "27dd4da69e", "c50d2d1d42", "578511c8a9", "5942004064",
       "f9f95681fd", "d755b3d9d8", "3db0a1c8f3", "9071e139d9", "e7af285f7d", "09c1414f1b"]
LAM, CSLS_K, RANK_S, ALPHA, ITERS = 0.3, 1000, 200.0, 0.95, 100
NL_PUBLISHED = {"opengaussian19": (35.77, 54.02), "opengaussian15": (39.62, 59.26),
                "opengaussian10": (48.93, 68.83)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def csls(cos, vm, k=CSLS_K):
    out = cos.clone()
    kk = min(k, int(vm.sum()))
    out[vm] = cos[vm] - 0.5 * cos[vm].topk(kk, dim=0).values.mean(0)[None, :]
    return out


def score_pred(cls_np, assigned, owned, gt_t, n_cls, n_gt):
    pred = np.zeros(n_gt, dtype=np.int64)
    pred[owned] = cls_np[assigned[owned]] + 1
    _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), n_cls + 1)
    return float(miou) * 100, float(macc) * 100


# ---------------------------------------------------------------- stage 1: NormLift on ScanNet
def stage_normlift_scannet(out_json):
    """Delegated to normlift_replication.run -- NormLift on every arm, both solvers, both graphs."""
    import normlift_replication as NR
    return NR.run(out_json)


# ------------------------------------------------- stage 2: ScanNet++ full evaluation + ablation
def spp_gt(scene, top, raw2bench):
    from run_spp_eval import load_gt
    return load_gt(scene, top, raw2bench)


def stage_spp_full(recon, outdir, k_spacing_list=(20.0, 0.0), text_lowdin=False):
    """Every method x every coverage setting. k_spacing=0 means NO filter (the 'do both' arm)."""
    from run_spp_eval import benchmark_map, load_gt, coverage_filter
    device = "cuda"
    os.makedirs(outdir, exist_ok=True)
    top, raw2bench = benchmark_map()
    con = sqlite3.connect(DB, timeout=60.0)
    for scene in SPP:
        art = f"artifacts/scannetpp/{scene}"
        solved = f"{art}/solved_geometric_median_nonfrozen_ogl3.pt"
        ck = os.path.join(RECON, f"spp_{recon}_{scene}")
        if not (os.path.exists(solved) and os.path.isdir(ck)):
            log(f"  [miss] {recon} {scene}"); continue
        outp = os.path.join(outdir, f"{scene}_{recon}.json")
        if os.path.exists(outp):
            log(f"  [skip] {scene}"); continue
        t0 = time.time()
        centers, radii = load_points_radii(ck)
        sv = torch.load(solved, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
        P = feats.shape[0]
        raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
        del feats, sv
        pos = torch.from_numpy(centers).to(device).float()
        mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
        cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
        adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
        ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
        R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
             .reliability()["reliability"].to(device).float() * vm)
        Dm = int((of0[1:] - of0[:-1]).max()) + 1
        chunk = max(256, 200_000 // max(Dm, 1))
        cen_r = mode_vote_refine(cen, R, pos, ad0, of0, chunk=chunk)   # centred + prerefine
        raw_r = mode_vote_refine(raw, R, pos, ad0, of0, chunk=chunk)   # NormLift-style, no centring
        src, dst, _ = csr_to_edges(ad0, of0, P, device)
        keep_e = vm[src] & vm[dst]; src, dst = src[keep_e], dst[keep_e]
        deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))
        del adj, ad0, of0, R
        torch.cuda.empty_cache()

        gt_pts, gt_lab0, n_masked = load_gt(scene, top, raw2bench)
        assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
        owned = assigned >= 0
        res = {"scene": scene, "recon": recon, "n_masked": n_masked, "arms": {}}
        for ks in k_spacing_list:
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
                # Loewdin prototype orthogonalisation -- REPRESENTATION-AGNOSTIC, applied
                # identically to pf/rf/gs so the DELTA isolates whether the 3D
                # representation changes how much a decorrelated classifier can exploit.
                if text_lowdin:
                    from run_text_and_pseudo_eval import text_lowdin as _lowdin
                    txt = _lowdin(txt)
                cs = f"spp_top{K}"

                # Surface metrics are a per-class KD-tree pair, so 42 combos/scene is
                # prohibitive on a 4.7M-vertex mesh. Restrict to the two rows the comparison
                # actually turns on, at the primary coverage setting and finest class set.
                SURF = {"A_base", "F_full_stack", "B_normlift"}

                def emit(cls_np, tag):
                    mi, ma = score_pred(cls_np, assigned, owned, gt_t, C, gt_pts.shape[0])
                    sm = {}
                    if tag in SURF and K == 100 and ks > 0:
                        pred = np.zeros(gt_pts.shape[0], dtype=np.int64)
                        pred[owned] = cls_np[assigned[owned]] + 1
                        try:
                            sm = semantic_surface_metrics(gt_pts, gt_t.numpy(), pred, C + 1)
                        except Exception as e:
                            log(f"    [surf err] {tag}: {e}")
                    res["arms"].setdefault(f"{tag}|{tagc}", {})[cs] = {
                        "mIoU": mi, "mAcc": ma, "n_classes": C, "coverage": cov,
                        "scd": sm.get("scd"), "hd95": sm.get("hd95"),
                        "boundary_f1": sm.get("boundary_f1"),
                        "mae_pred2gt": sm.get("mae_pred2gt"),
                        "mae_gt2pred": sm.get("mae_gt2pred"), "n_missed": sm.get("n_missed")}
                    con.execute(
                        # coverage variant belongs in METHOD: results_unified is unique on
                        # (scene,recon,features,solver,method,class_set,source,assignment), so
                        # keeping it only in `family` made the two variants collide.
                        "insert or replace into results_unified (scene,recon,features,solver,method,family,"
                        "class_set,n_classes,miou,macc,coverage,grouping,complex,assignment,masked,"
                        "scd,hd95,boundary_f1,mae_pred2gt,mae_gt2pred,n_missed,"
                        "source,created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (scene, recon, "ogl3", "geometric_median", f"{tag}|{tagc}",
                         tagc, cs, C, mi, ma, cov, None, "true_facet", "power_cell", 0,
                         sm.get("scd"), sm.get("hd95"), sm.get("boundary_f1"),
                         sm.get("mae_pred2gt"), sm.get("mae_gt2pred"), sm.get("n_missed"),
                         "run_overnight.py", time.time()))
                    con.commit()

                def cosof(u):
                    c = torch.zeros(P, C, device=device); c[vm] = u[vm] @ txt.T; return c

                # --- ablation ladder: each row adds exactly one component -----------------
                emit(cosof(raw).argmax(-1).cpu().numpy(), "A_base")            # OpenGaussian baseline
                emit(cosof(raw_r).argmax(-1).cpu().numpy(), "B_normlift")      # + mode-vote only
                emit(cosof(cen).argmax(-1).cpu().numpy(), "C_centre")          # + centring only
                emit(cosof(cen_r).argmax(-1).cpu().numpy(), "D_centre_refine")  # + both
                cc = csls(cosof(cen_r), vm)
                emit(cc.argmax(-1).cpu().numpy(), "E_plus_csls")               # + CSLS
                p0 = rank_encode(cc, RANK_S, device); p0[~vm] = 0.0
                emit(diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy(),
                     "F_full_stack")                                            # + diffusion
                p0n = rank_encode(cosof(cen_r), RANK_S, device); p0n[~vm] = 0.0
                emit(diffuse(p0n, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy(),
                     "G_stack_noCSLS")                                          # isolates CSLS
                del txt, cc, p0, p0n
                torch.cuda.empty_cache()
        with open(outp, "w") as fh:
            json.dump(res, fh, indent=1)
        log(f"  [ok] {scene} {time.time()-t0:.0f}s")
        del raw, cen, cen_r, raw_r, src, dst, deg, pos
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="normlift,spp")
    a = p.parse_args()
    enable_determinism()
    stages = a.stages.split(",")
    if "normlift" in stages:
        log("=== STAGE 1: NormLift replication on ScanNet ===")
        try:
            stage_normlift_scannet("artifacts/scannet/normlift_replication.json")
        except Exception:
            log("STAGE 1 FAILED\n" + traceback.format_exc())
    if "spp" in stages:
        log("=== STAGE 2: ScanNet++ full evaluation + ablation (both coverage settings) ===")
        try:
            stage_spp_full("pf_unfroz", "artifacts/scannetpp/eval_full")
        except Exception:
            log("STAGE 2 FAILED\n" + traceback.format_exc())
    log("=== OVERNIGHT DONE ===")


if __name__ == "__main__":
    main()
