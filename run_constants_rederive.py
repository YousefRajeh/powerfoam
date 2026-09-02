"""Task #54/#45: re-derive the pipeline constants on the GEOMETRIC-MEDIAN 3DGS features.

WHY THIS RUNS BEFORE ANY GROUPING OR ABLATION WORK. Every constant we have (LAM=0.3, CSLS_K=1000,
RANK_S=200, ALPHA=0.95, ITERS=100, graph K=30) was chosen on foam, and the only 3DGS sweep we ever
ran (`run_hyperparam_transfer.py`) read `solved_weighted_gs_unfroz_ogl3.pt` -- the WEIGHTED solve,
which is confound #1 that task #53 existed to remove. Sweeping groupings at constants tuned for a
different solver would confound grouping with tuning, so this comes first.

Two things this measures that the earlier sweep could not:
  - geometric-median vs weighted at EACH constant, on identical features and identical graphs, so
    the solver effect is isolated rather than entangled with the representation.
  - reliability from the REAL accumulator stats, not the ||f|| proxy (OPEN_ISSUES K). The proxy is
    identically 1 on the geometric-median arm because that solver renormalises every update, so
    consensus weighting was INERT wherever the proxy was used. `stats.reliability()` is live there
    (measured median 0.7318, std 0.3119). The source is recorded per row, because a number produced
    with inert weighting is not comparable to one produced with live weighting.

Coordinate-wise, not factorial: 6*5*4*3 = 360 configs per (solver, scene) is ~27 days across the
grid; one-knob-at-a-time is ~26 configs and answers the same question about where the foam defaults
sit. Results go to artifacts/sweep.db (cfg_hash UNIQUE => resumable, no double-counting).

Run:  python run_constants_rederive.py --scenes 4 --solvers geometric_median,weighted
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
from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS, log, score_pred
from run_simplex_diffusion_eval import csr_to_edges, diffuse
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_spp_gs_eval import load_gaussians, mahalanobis_assign

ART = "artifacts/scannetpp_gs"
# hardest (largest) first, per standing practice: if a constant behaves differently at scale we
# want to know before spending hours on the easy scenes.
SCENES = ["d755b3d9d8", "9071e139d9", "578511c8a9", "3864514494",
          "5942004064", "09c1414f1b", "3db0a1c8f3", "f9f95681fd",
          "c50d2d1d42", "0d2ee665be", "27dd4da69e", "e7af285f7d"]


def edges_from_builder(name, pos, vm, feat, scales, quats, K, device):
    src, dst, _ = BUILDERS[name](pos=pos, vm=vm, feat=feat, scales=scales, quats=quats,
                                 K=K, device=device)
    keep = vm[src] & vm[dst]
    src, dst = src[keep], dst[keep]
    deg = torch.zeros(pos.shape[0], dtype=torch.long, device=device).index_add_(
        0, src, torch.ones_like(src))
    return src, dst, deg


def csr_from_edges(src, dst, P, device):
    """CSR for mode_vote_refine, which wants (adjacent, offsets) rather than an edge list."""
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
    ap.add_argument("--class-size", type=int, default=100)
    ap.add_argument("--smallest", action="store_true",
                    help="take scenes from the SMALL end of the list -- for smoke tests, where the "
                         "point is to fail fast on a script error rather than stress memory")
    a = ap.parse_args()
    enable_determinism()
    device = "cuda"
    con = sweep_db.connect()
    top, r2b = benchmark_map()
    scenes = list(reversed(SCENES))[:a.scenes] if a.smallest else SCENES[:a.scenes]

    for scene in scenes:
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
            # OPEN_ISSUES K: use NormLift's real reliability from the accumulator stats, not the
            # ||f|| proxy. On the geometric-median arm the proxy is IDENTICALLY 1 (the solver
            # renormalises every update), so consensus weighting was inert; the stats reliability is
            # live there (median 0.73, std 0.31). The source tag is logged and must be reported.
            R, r_source = rel.get(feats, vm, f"{ART}/{scene}/stats_gs_unfroz_ogl3.pt", device)
            r_med = float(R[vm].median())
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
            log(f"  {scene}/{solver}: P={P:,} valid={nvalid:,} R[{r_source}] {rel.describe(R, vm)} "
                f"cos(f,mu)={float((raw[vm] @ mu.T).mean()):.3f} C={C}")

            graphs = {}

            def graph(K):
                if K not in graphs:
                    s, d, deg = edges_from_builder(a.grouping, pos, vm, raw, sc, qt, K, device)
                    adj, off = csr_from_edges(s, d, P, device)
                    graphs[K] = (s, d, deg, adj, off)
                return graphs[K]

            def run(lam=LAM, csls_k=CSLS_K, gK=30, alpha=ALPHA, iters=ITERS, rs=RANK_S,
                    consensus=True, diffusion=True):
                u = raw.clone()
                if lam > 0:
                    u[vm] = F.normalize(raw[vm] - lam * mu, dim=-1)
                s, d, deg, adj, off = graph(gK)
                if consensus:
                    Dm = int((off[1:] - off[:-1]).max()) + 1
                    u = mode_vote_refine(u, R, pos, adj, off,
                                         chunk=max(256, 200_000 // max(Dm, 1)))
                cv = torch.zeros(P, C, device=device)
                cv[vm] = u[vm] @ txt.T
                if csls_k > 0:
                    cv[vm] = cv[vm] - 0.5 * cv[vm].topk(min(int(csls_k), nvalid), dim=0).values.mean(0)
                if not diffusion:
                    pred = cv.argmax(-1)
                else:
                    p0 = rank_encode(cv, rs, device)
                    p0[~vm] = 0.0
                    pred = diffuse(p0, s, d, deg, alpha, iters).argmax(-1)
                return score_pred(pred.cpu().numpy(), assigned, owned, gt_t, C, gt_pts.shape[0])

            def record(tag, **kw):
                cfg = {"representation": "gs_unfroz", "solver": solver, "dataset": "scannetpp",
                       "scene": scene, "class_set": f"spp_top{a.class_size}",
                       "lam": kw.get("lam", LAM), "csls_k": kw.get("csls_k", CSLS_K),
                       "csls_frac": 0.0, "graph_k": kw.get("gK", 30),
                       "alpha": kw.get("alpha", ALPHA), "iters": kw.get("iters", ITERS),
                       "rank_s": kw.get("rs", RANK_S),
                       "use_consensus": float(kw.get("consensus", True)),
                       "use_diffusion": float(kw.get("diffusion", True)),
                       "text_transform": "none", "text_alpha": 0.0, "coverage_k": 20.0,
                       "grouping": a.grouping, "reliability_source": r_source}
                if sweep_db.already_done(con, cfg):
                    return None
                miou, macc = run(**kw)[:2]
                sweep_db.record(con, cfg, miou, macc, C, nvalid, tag,
                                "run_constants_rederive.py")
                return miou

            # lam extended past 0.7: the single-scene smoke had mIoU still RISING at 0.7, so the
            # optimum sat on the boundary and was unmeasured. 1.0 removes the scene mean entirely.
            sweeps = [("lam", "lam", (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)),
                      ("csls_k", "csls_k", (0, 300, 1000, 3000, 10000, 30000)),
                      ("graph_k", "gK", (8, 15, 30, 60)),
                      ("alpha", "alpha", (0.5, 0.8, 0.95, 0.99)),
                      ("iters", "iters", (10, 30, 100)),
                      ("rank_s", "rs", (50.0, 200.0, 500.0))]
            for tag, key, vals in sweeps:
                out = []
                for v in vals:
                    m = record(f"rederive:{tag}", **{key: v})
                    out.append(f"{v}={m:.2f}" if m is not None else f"{v}=cached")
                log(f"    {tag:8s} " + "  ".join(out))
            for tag, kw in (("no_consensus", {"consensus": False}),
                            ("no_diffusion", {"diffusion": False})):
                m = record(f"rederive:{tag}", **kw)
                log(f"    {tag:14s} {m:.2f}" if m is not None else f"    {tag:14s} cached")

            del raw, R, txt, graphs
            torch.cuda.empty_cache()
        del pos, sc, qt
        torch.cuda.empty_cache()

    print("\n=== best per (solver, knob), averaged over scenes ===")
    for row in sweep_db.summary(con, "phase LIKE 'rederive:%'")[:40]:
        print(row)


if __name__ == "__main__":
    main()
