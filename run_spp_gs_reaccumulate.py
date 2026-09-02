"""Re-accumulate the 3DGS arm WITH full stats, so 3DGS can use the same solvers as foam.

WHY (task #53, OPEN_ISSUES section F). Foam features come from the streaming geometric-median solve;
3DGS features came from the weighted mean, because `distill.py` accumulated only `sum_r A[r,j] f_r`
and `sum_r A[r,j]` and then divided, discarding everything else. Every foam-vs-3DGS number we have
therefore compares two PIPELINES (solver + representation), not two representations. This removes
that confound by persisting the full accumulator state on a single re-accumulation pass.

HOW. `distill.py` is patched (backup: `distill.py.bak_pre_stats`) so that when `FFL_STATS_OUT` is
set it replays each view's per-primitive aggregates through the project's real
`AccumulatedFeatureStats.accumulate_view` -- one "nonzero" per touched primitive, value = that view's
weight, b = that view's mean direction. Replaying rather than reimplementing the VALA update is
deliberate: a second implementation could drift from the foam path, which is the exact thing this is
meant to make comparable.

WHAT IS EXACT AND WHAT IS NOT -- measured, not assumed:
  EXACT (max relative difference vs a true ray-level accumulation, float32 accumulator):
      support 5.0e-07, numerator 3.4e-07, sum_view_weight_sq 1.9e-07, intra_sum 1.1e-07,
      gm_z 2.4e-07, gm_weight 1.3e-07
      -> geometric-median and weighted features reproduce at cosine 1.00000000
  NOT RECONSTRUCTIBLE (ray-level second moments; view-level aggregates cannot recover them):
      support2 and sq_numerator differ by ~13x
      -> the inverse-variance and ridge solvers are INVALID on these stats and are blocked by the
         `<stats>.valid.json` sidecar written next to every file.

Run:  python run_spp_gs_reaccumulate.py            # all 12 ScanNet++ scenes
      python run_spp_gs_reaccumulate.py --scenes f9f95681fd
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")

import torch

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\spp_data_1600"
GS = r"D:\Downloads\refbench_3dgs_12scenes\output"
FEAT_DIR = "openclip_features_sam_l3"
ART = "artifacts/scannetpp_gs"
SPP = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494", "578511c8a9", "5942004064",
       "27dd4da69e", "3db0a1c8f3", "9071e139d9", "d755b3d9d8", "e7af285f7d", "09c1414f1b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--solvers", default="geometric_median,weighted")
    ap.add_argument("--tikhonov", type=float, default=None,
                    help="Splat Feature Solver's Tikhonov Guidance (paper optimum 1.2): squeezes "
                         "opacity as sigmoid(theta*lambda), Eq.23. The SQUARED-weight half of "
                         "Eq.24 is already unconditional in their CUDA kernel. Artifacts are "
                         "written under a separate _tikh tag so the two lifts never collide.")
    a = ap.parse_args()
    scenes = [s for s in a.scenes.split(",") if s] or SPP
    from feature_foam_lifting.operator import (AccumulatedFeatureStats,
                                               solve_geometric_median_from_stats,
                                               solve_weighted_from_stats)
    SOLVERS = {"geometric_median": solve_geometric_median_from_stats,
               "weighted": solve_weighted_from_stats}

    for scene in scenes:
        os.makedirs(f"{ART}/{scene}", exist_ok=True)
        tag = "gs_tikh_ogl3" if a.tikhonov is not None else "gs_unfroz_ogl3"
        stats_path = os.path.abspath(f"{ART}/{scene}/stats_{tag}.pt")
        src = os.path.join(DATA, scene)
        ply = os.path.join(GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                           "scene_point_cloud.ply")
        if not os.path.exists(ply):
            print(f"[miss] {scene}: no ply", flush=True); continue

        if not os.path.exists(stats_path):
            t0 = time.time()
            print(f"[accumulate] {scene}", flush=True)
            env = dict(os.environ, FFL_STATS_OUT=stats_path)
            r = subprocess.run(
                [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", ply,
                 "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"]
                + ([] if a.tikhonov is None else ["--tikhonov", str(a.tikhonov)]),
                env=env, capture_output=True, text=True, cwd=r"D:\Downloads\splat-distiller")
            if not os.path.exists(stats_path):
                print(f"[FAIL] {scene} rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}",
                      flush=True)
                continue
            print(f"[ok] {scene} accumulated in {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"[skip] {scene}: stats exist", flush=True)

        # guard: refuse any solver the sidecar marks invalid for these stats
        side = stats_path + ".valid.json"
        allowed = json.load(open(side))["valid_solvers"] if os.path.exists(side) else list(SOLVERS)
        st = AccumulatedFeatureStats.load(stats_path)
        for name in a.solvers.split(","):
            if name not in allowed:
                print(f"  [blocked] {name} is invalid for replay-built stats -- skipping",
                      flush=True)
                continue
            out = f"{ART}/{scene}/solved_{name}_{tag}.pt"
            if os.path.exists(out):
                print(f"  [skip] {name}", flush=True); continue
            # solve_weighted_from_stats returns (x, valid); solve_geometric_median_from_stats
            # returns (x, valid, info). Unpacking a fixed 3 crashed the first tikhonov run after a
            # 9-minute accumulation -- and had never fired before because every previous run found
            # the weighted solve already on disk and skipped the branch entirely.
            res = SOLVERS[name](st)
            x, v = res[0], res[1]
            torch.save({"primitive_features": x.cpu(), "valid_mask": v.cpu()}, out)
            n = x[v].norm(dim=-1)
            print(f"  [solved] {name}: valid={int(v.sum()):,} "
                  f"||f|| median={float(n.median()):.4f} -> {out}", flush=True)
        del st
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
