"""Confirm the grid's top configurations on ALL 12 ScanNet++ scenes, GS only.

WHY. The joint grid runs on 4 scenes to keep the search affordable. A 4-scene winner is a candidate,
not a result -- this project has already caught four single-scene or small-sample reversals, most
recently "csls_k=0 wins" which vanished under cross-scene averaging. The top configs are therefore
re-run on the full 12 before anything is reported.

Selection is on the PAIRED mean over scenes where a config is complete, and configs are deduplicated
on the axes that actually matter: `rank_s` changed results by 0.02 mIoU in the grid (consistent with
the documented rank-encoding invariance), so three "top" configs differing only in rank_s would waste
two thirds of the run. Distinct (lam, csls_k, alpha, iters) tuples are required.

The inherited-defaults configuration is ALWAYS included as a control, so the reported gain is a
paired difference measured on the same 12 scenes rather than against a number from another run.

Run:  python run_top_configs_12scene.py --top 3 --solvers geometric_median,weighted
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

ART = "artifacts/scannetpp_gs"
DEFAULTS = (0.3, 1000.0, 200.0, 0.95, 100.0)   # inherited (foam-chosen) constants -- the control


def pick_top(con, solver, k):
    """Top-k DISTINCT (lam, csls_k, alpha, iters), best rank_s per tuple, by paired mean."""
    n_sc = con.execute("SELECT COUNT(DISTINCT scene) FROM runs WHERE phase='grid' AND solver=?",
                       (solver,)).fetchone()[0]
    rows = con.execute("""
        SELECT lam, csls_k, rank_s, alpha, iters, COUNT(*) n, AVG(miou) m
        FROM runs WHERE phase='grid' AND solver=?
        GROUP BY lam, csls_k, rank_s, alpha, iters HAVING n=? ORDER BY m DESC""",
        (solver, n_sc)).fetchall()
    out, seen = [], set()
    for lam, ck, rs, al, it, n, m in rows:
        tup = (lam, ck, al, it)
        if tup in seen:
            continue
        seen.add(tup)
        out.append((lam, ck, rs, al, it, m))
        if len(out) == k:
            break
    return out, n_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--solvers", default="geometric_median,weighted")
    ap.add_argument("--grouping", default="knn_pos")
    ap.add_argument("--graph-k", type=int, default=30)
    ap.add_argument("--class-size", type=int, default=100)
    a = ap.parse_args()
    enable_determinism()
    device = "cuda"
    con = sweep_db.connect()
    top_names, r2b = benchmark_map()

    plan = {}
    for solver in a.solvers.split(","):
        cfgs, n_sc = pick_top(con, solver, a.top)
        if not cfgs:
            log(f"[skip] {solver}: no complete grid rows yet")
            continue
        cfgs.append((*DEFAULTS, None))          # control
        plan[solver] = cfgs
        log(f"[plan] {solver}: {len(cfgs)-1} top configs (from {n_sc}-scene grid) + inherited control")
        for lam, ck, rs, al, it, m in cfgs:
            tag = "CONTROL (inherited)" if m is None else f"grid mean {m:.2f}"
            log(f"        lam={lam} csls={int(ck)} rank_s={rs} alpha={al} iters={int(it)}  {tag}")
    if not plan:
        return

    for scene in SCENES:
        means, scales, quats = load_gaussians(scene)
        gt_pts, lab0, _ = load_gt(scene, top_names, r2b)
        pos = torch.from_numpy(means).to(device).float()
        sc = torch.from_numpy(scales).to(device).float()
        qt = torch.from_numpy(quats).to(device).float()

        for solver, cfgs in plan.items():
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
            txt = embed_class_names([top_names[:a.class_size][i] for i in pres], device)
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

            # group by lam so the consensus pass is shared, as in the grid
            by_lam = {}
            for lam, ck, rs, al, it, _ in cfgs:
                by_lam.setdefault(lam, []).append((ck, rs, al, it))
            for lam, items in by_lam.items():
                u = raw.clone()
                if lam > 0:
                    u[vm] = F.normalize(raw[vm] - lam * mu, dim=-1)
                u = mode_vote_refine(u, R, pos, adj, off, chunk=max(256, 200_000 // max(Dm, 1)))
                cvb = torch.zeros(P, C, device=device)
                cvb[vm] = u[vm] @ txt.T
                del u
                for ck, rs, al, it in items:
                    cfg = {"representation": "gs_unfroz", "solver": solver,
                           "dataset": "scannetpp", "scene": scene,
                           "class_set": f"spp_top{a.class_size}", "lam": lam,
                           "csls_k": float(ck), "csls_frac": 0.0, "graph_k": float(a.graph_k),
                           "alpha": al, "iters": float(it), "rank_s": rs,
                           "use_consensus": 1.0, "use_diffusion": 1.0, "text_transform": "none",
                           "text_alpha": 0.0, "coverage_k": 20.0, "grouping": a.grouping,
                           "reliability_source": r_source}
                    if sweep_db.already_done(con, dict(cfg)):
                        # the 4 grid scenes already hold these rows; phase differs, so record a
                        # confirm row only where it is genuinely new
                        pass
                    cv = cvb.clone()
                    if ck > 0:
                        cv[vm] = cv[vm] - 0.5 * cv[vm].topk(min(int(ck), nvalid),
                                                            dim=0).values.mean(0)
                    p0 = rank_encode(cv, rs, device)
                    p0[~vm] = 0.0
                    snaps = diffuse_snapshots(p0, src, dst, deg, al, {int(it)})
                    miou, macc = score_pred(snaps[int(it)].argmax(-1).cpu().numpy(),
                                            assigned, owned, gt_t, C, gt_pts.shape[0])[:2]
                    sweep_db.record(con, cfg, miou, macc, C, nvalid, "confirm12",
                                    "run_top_configs_12scene.py")
                    log(f"    {scene}/{solver} lam={lam} csls={int(ck)} a={al} it={int(it)}"
                        f" -> {miou:.2f}")
                    del cv, p0, snaps
                del cvb
            del raw, R, txt
            torch.cuda.empty_cache()
        del pos, sc, qt
        torch.cuda.empty_cache()

    print("\n=== 12-scene confirmation, paired against inherited defaults ===")
    for solver in plan:
        d = con.execute("""SELECT AVG(miou), COUNT(*) FROM runs WHERE phase='confirm12'
            AND solver=? AND lam=? AND csls_k=? AND rank_s=? AND alpha=? AND iters=?""",
            (solver,) + DEFAULTS).fetchone()
        print(f"-- {solver} --")
        print(f"   inherited defaults: {d[0]:.2f}  (n={d[1]})" if d[0] else "   control missing")
        rows = con.execute("""SELECT lam,csls_k,rank_s,alpha,iters,COUNT(*) n,AVG(miou) m
            FROM runs WHERE phase='confirm12' AND solver=?
            GROUP BY lam,csls_k,rank_s,alpha,iters ORDER BY m DESC""", (solver,)).fetchall()
        for r in rows:
            gain = f"{r[6]-d[0]:+.2f}" if d[0] else "n/a"
            ctl = "  <- control" if (r[0], r[1], r[2], r[3], r[4]) == DEFAULTS else ""
            print(f"   lam={r[0]:<5} csls={int(r[1]):<6} rank_s={r[2]:<6} alpha={r[3]:<5} "
                  f"iters={int(r[4]):<4} n={r[5]:<3} mean={r[6]:.2f}  {gain}{ctl}")


if __name__ == "__main__":
    main()
