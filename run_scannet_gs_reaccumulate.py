"""Re-accumulate the ScanNet 3DGS arms WITH full stats, so 3DGS can be solved with the SAME solver
as the foam. The ScanNet counterpart of run_spp_gs_reaccumulate.py (task #53, OPEN_ISSUES F).

WHY THIS IS NEEDED AT ALL, stated precisely. `artifacts/scannet/*/` contains only
`solved_weighted_gs_{froz,unfroz}_ogl3.pt`: every ScanNet 3DGS feature field we have was produced by
the weighted mean, because splat-distiller's `distill.py` accumulated only `sum_r A[r,j] f_r` and
`sum_r A[r,j]` and divided. The foam arms use the streaming geometric median. So every
foam-vs-3DGS ScanNet number compares two PIPELINES, not two representations.

BEWARE A MISLEADING COLUMN. `results_unified.solver` says `geometric_median` for the cross-recon
3DGS rows, but `backfill_surface_cross_recon.py:139` writes that string as a LITERAL for every row
it inserts while its own FEATURES map (lines 54-55) points the 3DGS arms at `solved_weighted_gs_*`.
Those rows are weighted solves mislabelled. Do not read that column as evidence of a matched solver;
this script is what would make it true.

HOW. Identical mechanism to the ScanNet++ version: `distill.py` is already patched so that with
`FFL_STATS_OUT` set it replays each view's per-primitive aggregates through the project's real
`AccumulatedFeatureStats.accumulate_view`. Replaying rather than reimplementing keeps the 3DGS path
bit-comparable with the foam path.

WHAT THE REPLAY CAN AND CANNOT GIVE (measured on the ScanNet++ run, not assumed here):
  EXACT   support, numerator, sum_view_weight_sq, intra_sum, gm_z, gm_weight  (<= 5e-7 rel)
          -> geometric-median and weighted solves are valid
  INVALID support2 and sq_numerator (~13x off; ray-level second moments are not recoverable from
          view-level aggregates)
          -> ridge / inverse-variance solvers are blocked by the `.valid.json` sidecar, AND
          -> rho_j = 1 - support2/support MUST NOT be read off these stats. The only valid route to
             a 3DGS rho is the true per-ray operator, which run_rgb_roundtrip.py builds directly.

INDEX ALIGNMENT IS ASSERTED, NOT ASSUMED. Every downstream evaluator indexes the solved features by
the Gaussian order of `recon_remote/<arm>/<scene>/ckpt.pt` (the cached assignment is built from that
file). distill.py has a filtering path that writes `*_filtered.pt`; if it ever pruned, the stats
would silently describe a different primitive set. The primitive count is therefore checked against
the checkpoint and the scene is skipped, loudly, on any mismatch.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import torch

PY_DISTILL = r"D:\conda\envs\splat-distiller\python.exe"
DISTILL = r"D:\Downloads\splat-distiller\distill.py"
DATA = r"D:\Downloads\powerfoam\data\scannet"
FEAT_DIR = "openclip_features_sam_l3"
ART = "artifacts/scannet"
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--arms", default="gs_froz,gs_unfroz")
    ap.add_argument("--solvers", default="geometric_median")
    ap.add_argument("--keep-stats", action="store_true",
                    help="keep the intermediate stats file after solving. Off by default: the "
                         "unfrozen arm writes ~7 GB per scene and keeping all 10 filled the disk "
                         "mid-write, which produced a truncated file that then failed to load.")
    ap.add_argument("--min-free-gb", type=float, default=25.0,
                    help="refuse to start a scene with less than this free on the artifact drive")
    a = ap.parse_args()
    scenes = [s for s in a.scenes.split(",") if s] or SCENES

    from feature_foam_lifting.operator import (AccumulatedFeatureStats,
                                               solve_geometric_median_from_stats,
                                               solve_weighted_from_stats)
    SOLVERS = {"geometric_median": solve_geometric_median_from_stats,
               "weighted": solve_weighted_from_stats}

    for arm in [x for x in a.arms.split(",") if x]:
        for scene in scenes:
            ck = os.path.abspath(f"recon_remote/{arm}/{scene}/ckpt.pt")
            src = os.path.join(DATA, f"{scene}_colmap")
            if not (os.path.exists(ck) and os.path.isdir(src)):
                print(f"[miss] {arm}/{scene}", flush=True)
                continue
            os.makedirs(f"{ART}/{scene}", exist_ok=True)
            tag = f"{arm}_ogl3"
            stats_path = os.path.abspath(f"{ART}/{scene}/stats_{tag}.pt")

            # A stats file that exists but cannot be opened is worse than one that is missing:
            # the old code skipped accumulation on mere existence and then died on load. Treat
            # unreadable as absent and rebuild it.
            # zipfile.is_zipfile reads only the central directory, which is precisely what was
            # missing from the truncated file, so this costs milliseconds instead of re-reading
            # 7 GB through torch.load just to find out.
            if os.path.exists(stats_path) and not zipfile.is_zipfile(stats_path):
                print(f"[corrupt] {stats_path}: no central directory -- rebuilding", flush=True)
                os.remove(stats_path)

            free_gb = shutil.disk_usage(os.path.dirname(stats_path)).free / 2**30
            if not os.path.exists(stats_path) and free_gb < a.min_free_gb:
                print(f"[STOP] only {free_gb:.1f} GB free, need {a.min_free_gb:.0f} GB for "
                      f"{arm}/{scene} -- a short write here is how the last run corrupted a file",
                      flush=True)
                return

            if not os.path.exists(stats_path):
                t0 = time.time()
                print(f"[accumulate] {arm}/{scene}", flush=True)
                env = dict(os.environ, FFL_STATS_OUT=stats_path)
                r = subprocess.run(
                    [PY_DISTILL, "-u", DISTILL, "--dir", src, "--ckpt", ck,
                     "--feature_folder", FEAT_DIR, "--method", "3DGS", "--factor", "1"],
                    env=env, capture_output=True, text=True,
                    cwd=r"D:\Downloads\splat-distiller")
                if not os.path.exists(stats_path):
                    print(f"[FAIL] {arm}/{scene} rc={r.returncode}\n{r.stdout[-2000:]}\n"
                          f"{r.stderr[-2000:]}", flush=True)
                    continue
                print(f"[ok] {arm}/{scene} accumulated in {time.time()-t0:.0f}s", flush=True)
            else:
                print(f"[skip] {arm}/{scene}: stats exist", flush=True)

            st = AccumulatedFeatureStats.load(stats_path)
            n_ck = torch.load(ck, map_location="cpu", weights_only=False)
            n_ck = (n_ck["splats"] if "splats" in n_ck else n_ck)["means"].shape[0]
            if int(st.support.shape[0]) != n_ck:
                print(f"[MISALIGNED] {arm}/{scene}: stats P={int(st.support.shape[0]):,} vs "
                      f"checkpoint {n_ck:,} -- refusing to solve, the assignment cache indexes the "
                      f"checkpoint order", flush=True)
                del st
                continue

            side = stats_path + ".valid.json"
            allowed = (json.load(open(side))["valid_solvers"]
                       if os.path.exists(side) else list(SOLVERS))
            for name in [x for x in a.solvers.split(",") if x]:
                if name not in allowed:
                    print(f"  [blocked] {name} invalid for replay-built stats", flush=True)
                    continue
                out = f"{ART}/{scene}/solved_{name}_{tag}.pt"
                if os.path.exists(out):
                    print(f"  [skip] {name} exists", flush=True)
                    continue
                res = SOLVERS[name](st)
                x, v = res[0], res[1]
                torch.save({"primitive_features": x.cpu(), "valid_mask": v.cpu()}, out)
                print(f"  [solved] {name}: valid={int(v.sum()):,}/{x.shape[0]:,} "
                      f"||f|| median={float(x[v].norm(dim=-1).median()):.4f} -> {out}", flush=True)
            del st
            torch.cuda.empty_cache()
            if not a.keep_stats:
                # only after every requested solver has produced a file on disk
                done = all(os.path.exists(f"{ART}/{scene}/solved_{n}_{tag}.pt")
                           for n in [x for x in a.solvers.split(",") if x] if n in allowed)
                if done:
                    sz = os.path.getsize(stats_path) / 2**30
                    os.remove(stats_path)
                    print(f"  [stats removed] {sz:.1f} GB freed (solved features kept)", flush=True)


if __name__ == "__main__":
    main()
