"""Task #45/#54/#55: JOINT grid over the interacting constants -- not a coordinate sweep.

WHY THE COORDINATE SWEEP WAS WRONG. `run_constants_rederive.py` varies one knob at a time around
the defaults, which finds a COORDINATE optimum. The smoke run showed exactly why that is not enough:
`lam` peaked at 0.7 while csls_k was held at 1000, and `csls_k=0` won while lam was held at 0.3.
Those two answers were measured at different points, so the pair (0.7, 0) was never evaluated and
the true joint optimum could be at neither. `lam` and `csls_k` are both corrections applied to the
same similarity scores, so an interaction is expected rather than hypothetical.

HOW A FULL GRID IS MADE AFFORDABLE. Evaluated naively, lam(8) x csls_k(6) x rank_s(2) x alpha(4) x
iters(3) = 1,152 pipeline runs per (scene, solver) at ~30s each -- about 10 hours per scene. Three
structural facts collapse that:

  1. `mode_vote_refine` (feature consensus) depends only on (graph, lam). Computed once per lam and
     reused across every csls_k / rank_s / alpha / iters -- 8 consensus passes instead of 1,152.
  2. `cv = u @ txt.T` likewise depends only on lam; the CSLS correction is a rank-K subtraction on
     top of it, so csls_k costs a topk, not a re-run.
  3. `iters` IS FREE. `diffuse` is a fixed-point iteration whose state after 10 steps is exactly the
     state a 10-iteration call would return, so a single 100-step run snapshotted at {10, 30, 100}
     yields all three -- BITWISE, verified in `test_diffuse_snapshots.py`, not assumed.

That leaves lam(8) x csls_k(6) x rank_s(2) x alpha(4) = 384 diffusion runs per (scene, solver),
each yielding 3 recorded configs.

`graph_k` is held at 30 here and swept separately: it changes the graph, so it cannot share the
cached consensus, and folding it in would multiply the whole grid by 4. That is a stated scope
limit, not an oversight -- the grid is complete over the knobs that interact through the scores.
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
from run_normlift_refine_eval import mode_vote_refine
from run_overnight import log, score_pred
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign

ART = "artifacts/scannetpp_gs"
SCENES = ["e7af285f7d", "27dd4da69e", "0d2ee665be", "c50d2d1d42", "f9f95681fd",
          "3db0a1c8f3", "09c1414f1b", "5942004064", "3864514494", "578511c8a9",
          "9071e139d9", "d755b3d9d8"]

LAMS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)
CSLS = (0, 300, 1000, 3000, 10000, 30000)
RANK_S = (200.0, 50.0)
ALPHAS = (0.5, 0.8, 0.95, 0.99)
ITER_SNAPS = (10, 30, 100)


def diffuse_snapshots(p0, src, dst, deg, alpha, snaps, chunk=8_000_000):
    """`diffuse` reimplemented to yield intermediate states -- arithmetic identical, verified bitwise.

    Kept line-for-line with run_simplex_diffusion_eval.diffuse (same row-normalisation, same dead-row
    handling, same update order) so the snapshot at `n` equals a fresh call with iters=n exactly.
    """
    P, K = p0.shape
    w = torch.ones(src.numel(), device=p0.device)
    rowsum = torch.zeros(P, device=p0.device).index_add_(0, src, w)
    w = w / rowsum.clamp_min(1e-30)[src]
    a = torch.full((P, 1), alpha, device=p0.device)
    a = torch.where((deg > 0).unsqueeze(1), a, torch.zeros_like(a))
    p = p0.clone()
    out = {}
    for it in range(1, max(snaps) + 1):
        acc = torch.zeros_like(p)
        for s in range(0, src.numel(), chunk):
            e = min(s + chunk, src.numel())
            acc.index_add_(0, src[s:e], p[dst[s:e]] * w[s:e, None])
        p = (1 - a) * p0 + a * acc
        if it in snaps:
            out[it] = p.clone()
    return out


def csr_from_edges(src, dst, P, device):
    order = torch.argsort(src)
    s, d = src[order], dst[order]
    counts = torch.bincount(s, minlength=P)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
    return d.contiguous(), offsets.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--solvers", default="geometric_median,weighted")
    ap.add_argument("--grouping", default="knn_pos")
    ap.add_argument("--graph-k", type=int, default=30)
    ap.add_argument("--class-size", type=int, default=100)
    a = ap.parse_args()
    enable_determinism()
    device = "cuda"
    con = sweep_db.connect()
    top, r2b = benchmark_map()

    for scene in SCENES[:a.scenes]:
        means, scales, quats = load_gaussians(scene)
        gt_pts, lab0, _ = load_gt(scene, top, r2b)
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
            lab = np.where(keepc, lab0, -1)
            pres = sorted(set(np.unique(lab).tolist()) & set(range(a.class_size)))
            if not pres:
                continue
            gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
            txt = embed_class_names([top[:a.class_size][i] for i in pres], device)
            C = len(pres)
            mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)

            src, dst, _ = BUILDERS[a.grouping](pos=pos, vm=vm, feat=raw, scales=sc, quats=qt,
                                               K=a.graph_k, device=device)
            keep = vm[src] & vm[dst]
            src, dst = src[keep], dst[keep]
            deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
                0, src, torch.ones_like(src))
            adj, off = csr_from_edges(src, dst, P, device)
            Dm = int((off[1:] - off[:-1]).max()) + 1
            log(f"  {scene}/{solver}: P={P:,} valid={nvalid:,} C={C} edges={src.numel():,} "
                f"R[{r_source}] {rel.describe(R, vm)}")

            n_done = 0
            for lam in LAMS:
                u = raw.clone()
                if lam > 0:
                    u[vm] = F.normalize(raw[vm] - lam * mu, dim=-1)
                u = mode_vote_refine(u, R, pos, adj, off, chunk=max(256, 200_000 // max(Dm, 1)))
                cv_base = torch.zeros(P, C, device=device)
                cv_base[vm] = u[vm] @ txt.T
                del u
                for ck in CSLS:
                    cv = cv_base.clone()
                    if ck > 0:
                        cv[vm] = cv[vm] - 0.5 * cv[vm].topk(min(int(ck), nvalid), dim=0).values.mean(0)
                    for rs in RANK_S:
                        p0 = rank_encode(cv, rs, device)
                        p0[~vm] = 0.0
                        for al in ALPHAS:
                            cfgs = []
                            for it in ITER_SNAPS:
                                cfg = {"representation": "gs_unfroz", "solver": solver,
                                       "dataset": "scannetpp", "scene": scene,
                                       "class_set": f"spp_top{a.class_size}",
                                       "lam": lam, "csls_k": float(ck), "csls_frac": 0.0,
                                       "graph_k": float(a.graph_k), "alpha": al,
                                       "iters": float(it), "rank_s": rs,
                                       "use_consensus": 1.0, "use_diffusion": 1.0,
                                       "text_transform": "none", "text_alpha": 0.0,
                                       "coverage_k": 20.0, "grouping": a.grouping,
                                       "reliability_source": r_source}
                                if not sweep_db.already_done(con, cfg):
                                    cfgs.append((it, cfg))
                            if not cfgs:
                                continue
                            snaps = diffuse_snapshots(p0, src, dst, deg, al,
                                                      {it for it, _ in cfgs})
                            for it, cfg in cfgs:
                                miou, macc = score_pred(snaps[it].argmax(-1).cpu().numpy(),
                                                        assigned, owned, gt_t, C,
                                                        gt_pts.shape[0])[:2]
                                sweep_db.record(con, cfg, miou, macc, C, nvalid, "grid",
                                                "run_grid_search.py")
                                n_done += 1
                            del snaps
                        del p0
                    del cv
                del cv_base
                log(f"    lam={lam}: {n_done} configs recorded")
            del raw, R, txt
            torch.cuda.empty_cache()
        del pos, sc, qt
        torch.cuda.empty_cache()

    print("\n=== JOINT grid: top 20 by mean mIoU across scenes ===")
    rows = con.execute("""
        SELECT solver, lam, csls_k, rank_s, alpha, iters, COUNT(*) n, AVG(miou) m
        FROM runs WHERE phase='grid' GROUP BY solver, lam, csls_k, rank_s, alpha, iters
        ORDER BY m DESC LIMIT 20""").fetchall()
    for r in rows:
        print(f"  {r[0]:17s} lam={r[1]:<5} csls={int(r[2]):<6} rank_s={r[3]:<6} "
              f"alpha={r[4]:<5} iters={int(r[5]):<4} n={r[6]:<2} mIoU={r[7]:.2f}")


if __name__ == "__main__":
    main()
