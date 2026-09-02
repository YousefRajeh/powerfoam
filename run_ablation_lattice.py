"""Task #55: the FULL on/off ablation lattice over all six contributions, GS only, 12 scenes.

WHY THIS EXISTS. The joint grid (`run_grid_search.py`) swept CONSTANTS with every component switched
permanently ON (`use_consensus=1, use_diffusion=1, text_transform='none'` for all 9,184 rows). It
therefore measured tuning sensitivity, not whether any contribution helps -- and it never ran the two
prototype-side components at all. Its "+0.23, not significant" result is about TUNING and says
nothing about the methods.

THE SIX CONTRIBUTIONS, as defined in Paper-B-v2:
    C1  centering            feature-side, subtract lam * scene mean, renormalise
    C2a feature consensus    mode-vote refinement over the primitive graph
    C2b rank-encode+diffuse  simplex diffusion over the same graph
    C3a CSLS offset          per-class offset from local density
    C3b Loewdin              T' = (T T^T)^{-1/2} T, parameter-free
    C3c prototype whitening  covariance whitening with alpha=0.25 (tunable variant of C3b)

C3b and C3c are ALTERNATIVES on the same axis (both are linear maps of the prototype matrix), so the
lattice is 2^4 x 3 = 48 configurations rather than 2^6.

WHY A LATTICE AND NOT A LADDER. The sequential ladder that produced the published per-stage deltas
is order-dependent and has already been measured to lie: removing feature consensus IMPROVED the full
pipeline by +0.62 while the ladder credited it +1.05 (12/12). Only a full lattice gives each
component's effect IN THE PRESENCE OF the others, and the interaction terms.

Constants are held at the BEST configuration found by the 12-scene confirmation
(lam=0.2, csls_k=1000, rank_s=50, alpha=0.99, iters=100). Re-tuning was worth only
+0.23 [95% CI -0.48,+0.92], i.e. indistinguishable from zero, so this choice is not a confound --
it just means each component is measured in the best stack we have rather than an arbitrary one.

OPACITY CULLING IS ON. OpenGaussian/NormLift delete GT points whose Gaussian has
sigmoid(opacity) < 0.1, and our pipeline had never applied it -- so every previous ScanNet++ GS
number was scored on points their protocol removes. Those primitives carry the least reliable
features, so including them added noise that dilutes every component's measured effect. It is now
applied as standard, not as an option, and the kept fraction is logged per scene.

The metric is OUR point-level unweighted IoU (each GT point -> nearest Gaussian by Mahalanobis).
Dr-Splat's Gaussian-level significance-weighted IoU is deliberately NOT used here; see OPEN_ISSUES N
for why the two are not comparable.

Cost: (lam x consensus) = 4 expensive consensus passes per scene; everything downstream reuses them.
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

import reliability as rel
import sweep_db
from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
from graph_variants import BUILDERS
from run_derived_stack_eval import rank_encode
from run_grid_search import SCENES, csr_from_edges, diffuse_snapshots
from run_normlift_refine_eval import mode_vote_refine
from run_overnight import log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign
# reuse the SHIPPED transforms verbatim -- a second implementation could drift from the one that
# produced the published foam numbers, which is exactly what this run is trying to check
from run_text_and_pseudo_eval import text_lowdin, text_whiten

ART = "artifacts/scannetpp_gs"
TEXT_MODES = ("none", "lowdin", "whiten")
WHITEN_ALPHA = 0.25
# best configuration from the 12-scene confirmation (weighted solver)
LAM, CSLS_K, RANK_S, ALPHA, ITERS = 0.2, 1000, 50.0, 0.99, 100
OPACITY_THRESH = 0.1          # OpenGaussian/NormLift rule -- applied as standard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--solvers", default="weighted")
    ap.add_argument("--grouping", default="knn_pos")
    ap.add_argument("--graph-k", type=int, default=30)
    ap.add_argument("--class-size", type=int, default=100)
    ap.add_argument("--stack-only", action="store_true",
                    help="run only the headline configurations (baseline / published full stack / "
                         "scene-1 best) across all scenes, before spending hours on the full 48-cell "
                         "lattice -- this is the number the paper actually reports")
    a = ap.parse_args()
    enable_determinism()
    device = "cuda"
    con = sweep_db.connect()
    top_names, r2b = benchmark_map()

    for scene in SCENES[:a.scenes]:
        means, scales, quats = load_gaussians(scene)
        # opacity is not returned by load_gaussians; read the raw logit and sigmoid it, matching how
        # 3DGS stores it and how OpenGaussian applies the 0.1 rule.
        from plyfile import PlyData
        from run_spp_gs_eval import GS as _GS
        _ply = os.path.join(_GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                            "scene_point_cloud.ply")
        _op = np.asarray(PlyData.read(_ply)["vertex"]["opacity"]).astype(np.float64)
        alpha_op = 1.0 / (1.0 + np.exp(-_op))
        gt_pts, lab0, _ = load_gt(scene, top_names, r2b)
        pos = torch.from_numpy(means).to(device).float()
        sc = torch.from_numpy(scales).to(device).float()
        qt = torch.from_numpy(quats).to(device).float()

        for solver in a.solvers.split(","):
            path = f"{ART}/{scene}/solved_{solver}_gs_unfroz_ogl3.pt"
            if not os.path.exists(path):
                log(f"  [miss] {scene} {solver}")
                continue
            sv = torch.load(path, map_location="cpu", weights_only=True)
            feats = sv["primitive_features"].float().to(device)
            vmn = sv["valid_mask"].numpy()
            vm = sv["valid_mask"].to(device)
            P, nvalid = feats.shape[0], int(vm.sum())
            raw = torch.zeros_like(feats)
            raw[vm] = F.normalize(feats[vm], dim=-1)
            R, r_source = rel.get(feats, vm, f"{ART}/{scene}/stats_gs_unfroz_ogl3.pt", device)
            del feats, sv

            assigned = mahalanobis_assign(gt_pts.astype(np.float64), means, scales, quats)
            assigned = np.where(vmn[assigned], assigned, -1)
            owned = assigned >= 0
            keepc, _, _ = coverage_filter(gt_pts, assigned, means, vmn, 20.0)
            # OPACITY CULLING: DELETE the GT label of any point whose assigned Gaussian has
            # sigmoid(opacity) < 0.1, exactly as OpenGaussian/NormLift do. Deleting (label -1)
            # rather than reweighting also changes which classes are present, so the present-class
            # average is computed on the surviving population -- matching their convention.
            op_keep = alpha_op[np.clip(assigned, 0, None)] >= OPACITY_THRESH
            kept_frac = float((op_keep & owned).sum()) / max(int(owned.sum()), 1)
            lab = np.where(keepc & owned & op_keep, lab0, -1)
            pres = sorted(set(np.unique(lab).tolist()) & set(range(a.class_size)))
            if not pres:
                continue
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt0 = embed_class_names([top_names[:a.class_size][i] for i in pres], device)
            C = len(pres)
            texts = {"none": txt0, "lowdin": text_lowdin(txt0),
                     "whiten": text_whiten(txt0, WHITEN_ALPHA)[0]}
            mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)

            src, dst, _ = BUILDERS[a.grouping](pos=pos, vm=vm, feat=raw, scales=sc, quats=qt,
                                               K=a.graph_k, device=device)
            keep = vm[src] & vm[dst]
            src, dst = src[keep], dst[keep]
            deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
                0, src, torch.ones_like(src))
            adj, off = csr_from_edges(src, dst, P, device)
            Dm = int((off[1:] - off[:-1]).max()) + 1
            log(f"  {scene}/{solver}: P={P:,} C={C} R[{r_source}] "
                f"opacity>=0.1 keeps {kept_frac*100:.1f}% of owned GT points")

            # (C1, consensus, text, CSLS, diffusion)
            STACKS = [(0, 0, "none",   0, 0),      # baseline: no contribution at all
                      (1, 1, "lowdin", 1, 1),      # published full stack (C1+C2a+C2b+C3a+C3b)
                      (0, 0, "lowdin", 1, 1)]      # best cell from the scene-1 lattice
            want = set(STACKS) if a.stack_only else None
            for use_c1 in (0, 1):
                for use_cons in (0, 1):
                    u = raw.clone()
                    if use_c1:
                        u[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
                    if use_cons:
                        u = mode_vote_refine(u, R, pos, adj, off,
                                             chunk=max(256, 200_000 // max(Dm, 1)))
                    for tmode in TEXT_MODES:
                        cvb = torch.zeros(P, C, device=device)
                        cvb[vm] = u[vm] @ texts[tmode].T
                        for use_csls in (0, 1):
                            cv = cvb.clone()
                            if use_csls:
                                cv[vm] = cv[vm] - 0.5 * cv[vm].topk(
                                    min(int(CSLS_K), nvalid), dim=0).values.mean(0)
                            for use_diff in (0, 1):
                                if want is not None and                                         (use_c1, use_cons, tmode, use_csls, use_diff) not in want:
                                    continue
                                cfg = {"representation": "gs_unfroz", "solver": solver,
                                       "dataset": "scannetpp", "scene": scene,
                                       "class_set": f"spp_top{a.class_size}",
                                       "lam": LAM if use_c1 else 0.0,
                                       "csls_k": float(CSLS_K) if use_csls else 0.0,
                                       "csls_frac": 0.0, "graph_k": float(a.graph_k),
                                       "alpha": ALPHA, "iters": float(ITERS) if use_diff else 0.0,
                                       "rank_s": RANK_S,
                                       "use_consensus": float(use_cons),
                                       "use_diffusion": float(use_diff),
                                       "text_transform": tmode,
                                       "text_alpha": WHITEN_ALPHA if tmode == "whiten" else 0.0,
                                       "coverage_k": 20.0, "grouping": a.grouping,
                                       "reliability_source": r_source,
                                       "opacity_mask": OPACITY_THRESH}
                                if sweep_db.already_done(con, cfg):
                                    continue
                                if use_diff:
                                    p0 = rank_encode(cv, RANK_S, device)
                                    p0[~vm] = 0.0
                                    pred = diffuse_snapshots(p0, src, dst, deg, ALPHA,
                                                             {int(ITERS)})[int(ITERS)].argmax(-1)
                                    del p0
                                else:
                                    pred = cv.argmax(-1)
                                miou, macc = score_pred(pred.cpu().numpy(), assigned, owned,
                                                        gt_t, C, gt_pts.shape[0])[:2]
                                sweep_db.record(con, cfg, miou, macc, C, nvalid, "lattice",
                                                "run_ablation_lattice.py")
                                del pred
                            del cv
                        del cvb
                    del u
            log(f"    {scene}/{solver} lattice done")
            del raw, R, texts, txt0
            torch.cuda.empty_cache()
        del pos, sc, qt
        torch.cuda.empty_cache()

    print("\n=== MAIN EFFECT of each contribution, averaged over the whole lattice ===")
    print("(mean mIoU with the component ON minus with it OFF, over all other combinations)")
    for col, name in (("use_consensus", "C2a feature consensus"),
                      ("use_diffusion", "C2b rank+diffusion"),
                      ("lam", "C1  centering"),
                      ("csls_k", "C3a CSLS offset")):
        on = con.execute(f"SELECT AVG(miou) FROM runs WHERE phase='lattice' AND {col}>0").fetchone()[0]
        off = con.execute(f"SELECT AVG(miou) FROM runs WHERE phase='lattice' AND {col}=0").fetchone()[0]
        if on is not None and off is not None:
            print(f"  {name:24s} ON {on:6.2f}   OFF {off:6.2f}   effect {on-off:+.2f}")
    for tm in TEXT_MODES:
        m = con.execute("SELECT AVG(miou) FROM runs WHERE phase='lattice' AND text_transform=?",
                        (tm,)).fetchone()[0]
        if m is not None:
            print(f"  C3b/c text={tm:8s}       {m:6.2f}")


if __name__ == "__main__":
    main()
