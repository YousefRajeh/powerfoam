"""Measure what each grouping actually COSTS to build, before committing to a sweep over all of them.

WHY THIS RUNS FIRST. The sweep estimate (~2 days coordinate-wise) assumes graph construction is a
rounding error next to the 34s-per-config pipeline cost. That assumption is untested at 3DGS
primitive counts, and two builders look dangerous on paper:

  - knn_maha is O(n^2) chunked: at P=2.25M a single chunk materialises (chunk, P, 3), which is
    ~13 GB at chunk=512, repeated P/chunk = 4,400 times. This may simply not be tractable.
  - delaunay/regular SUBSAMPLE above 150k points, so on 1-2M-primitive scenes they are approximate
    while knn_pos is exact -- a fairness problem for any comparison between them, independent of cost.

So each builder is timed at the hyperparameters we would actually sweep around ("assumption
optimum": K=30, kmeans 2048 clusters, codebook 64x5), on scenes of DIFFERENT size, so cost can be
extrapolated to the largest scene rather than discovered by hanging on it.

EACH BUILDER RUNS IN ITS OWN SUBPROCESS WITH A WALL-CLOCK CAP. An in-process loop would let one
pathological builder hang the whole measurement, and an OOM in one would poison the others' timings
through allocator fragmentation. A builder that exceeds the cap is recorded as TIMEOUT -- a result,
not a crash.

Reported per builder: wall time, edge count, mean degree, peak GPU memory. Degree matters as much
as time: measured degrees differ 5x across builders (knn_pos 9.9 vs delaunay 47.4), so diffusion
cost per config -- and any accuracy difference -- is confounded with connectivity unless reported.

Run:  python measure_graph_cost.py                      # driver, all builders x 2 scenes
      python measure_graph_cost.py --one knn_pos --scene e7af285f7d    # single (used by driver)
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

ART = "artifacts/scannetpp_gs"
# Two sizes so cost can be fitted and extrapolated to the 2.25M worst case instead of risking a
# multi-hour hang there. e7af=881k is the smallest, 3db0=1.28M is mid.
SCENES = ["e7af285f7d", "3db0a1c8f3"]
BUILDER_NAMES = ["knn_pos", "knn_feat", "knn_maha", "radius",
                 "delaunay", "kmeans", "codebook"]   # `regular` dropped: see graph_variants.BUILDERS
CAP_SECONDS = 900          # per builder; exceeded => TIMEOUT, recorded and moved past


def run_one(scene, name):
    import numpy as np
    import torch
    from plyfile import PlyData

    from graph_variants import BUILDERS
    from run_spp_gs_eval import load_gaussians, GS

    dev = "cuda"
    means, scales, quats = load_gaussians(scene)
    sv = torch.load(f"{ART}/{scene}/solved_geometric_median_gs_unfroz_ogl3.pt",
                    map_location="cpu", weights_only=True)
    feat = sv["primitive_features"].float().to(dev)
    vm = sv["valid_mask"].to(dev)
    pos = torch.from_numpy(means).float().to(dev)
    sc = torch.from_numpy(scales).float().to(dev)
    qt = torch.from_numpy(quats).float().to(dev)
    # opacity is not returned by load_gaussians but regular_graph needs it (r_i^2 = 2 sigma^2 log w).
    # Stored as a logit in the ply, hence the sigmoid.
    p = os.path.join(GS, f"refbench-{scene}", "point_cloud", "iteration_30000",
                     "scene_point_cloud.ply")
    op = torch.from_numpy(1.0 / (1.0 + np.exp(-np.asarray(PlyData.read(p)["vertex"]["opacity"]),
                                              dtype=np.float64))).float().to(dev)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    src, dst, _ = BUILDERS[name](pos=pos, vm=vm, feat=feat, scales=sc, quats=qt,
                                 opacity=op, device=dev,
                                 K=30, n_clusters=2048, root=64, leaf=5)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_valid = int(vm.sum())
    out = {"scene": scene, "builder": name, "P": int(pos.shape[0]), "valid": n_valid,
           "seconds": round(dt, 2), "edges": int(src.numel()),
           "mean_degree": round(float(src.numel()) / max(n_valid, 1), 2),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print("RESULT " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--scene")
    ap.add_argument("--out", default="artifacts/graph_build_cost.json")
    a = ap.parse_args()
    if a.one:
        run_one(a.scene, a.one)
        return

    rows = []
    for scene in SCENES:
        if not os.path.exists(f"{ART}/{scene}/solved_geometric_median_gs_unfroz_ogl3.pt"):
            print(f"[skip] {scene}: no geometric-median solve yet", flush=True)
            continue
        for name in BUILDER_NAMES:
            print(f"[run] {scene} {name} (cap {CAP_SECONDS}s)", flush=True)
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, "-u", __file__, "--one", name,
                                    "--scene", scene],
                                   capture_output=True, text=True, timeout=CAP_SECONDS)
                line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
                if line:
                    row = json.loads(line[len("RESULT "):])
                    print(f"    {row['seconds']:8.2f}s  edges={row['edges']:>12,}  "
                          f"deg={row['mean_degree']:6.2f}  peak={row['peak_gb']:5.2f} GB", flush=True)
                else:
                    tail = (r.stderr or r.stdout).strip().splitlines()
                    msg = tail[-1][:120] if tail else f"rc={r.returncode}"
                    row = {"scene": scene, "builder": name, "status": "FAILED", "error": msg}
                    print(f"    FAILED: {msg}", flush=True)
            except subprocess.TimeoutExpired:
                row = {"scene": scene, "builder": name, "status": "TIMEOUT",
                       "seconds": CAP_SECONDS}
                print(f"    TIMEOUT after {CAP_SECONDS}s (> cap; recorded, not retried)", flush=True)
            row["wall"] = round(time.time() - t0, 1)
            rows.append(row)
            json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== graph build cost @ assumption-optimum hyperparameters ===")
    print(f"{'builder':10s} " + " ".join(f"{s[:8]:>22s}" for s in SCENES))
    for name in BUILDER_NAMES:
        cells = []
        for s in SCENES:
            r = next((x for x in rows if x["builder"] == name and x["scene"] == s), None)
            if r is None:
                cells.append(f"{'-':>22s}")
            elif r.get("status"):
                cells.append(f"{r['status']:>22s}")
            else:
                cells.append(f"{r['seconds']:>8.1f}s deg={r['mean_degree']:<6.1f}")
        print(f"{name:10s} " + " ".join(cells))

    # extrapolate to the 2.25M worst case using the two measured sizes
    print("\nextrapolation to d755b3d9d8 (P=2,252,236), fitted exponent from the two scenes:")
    import math
    for name in BUILDER_NAMES:
        a_row = next((x for x in rows if x["builder"] == name and x["scene"] == SCENES[0]
                      and not x.get("status")), None)
        b_row = next((x for x in rows if x["builder"] == name and x["scene"] == SCENES[1]
                      and not x.get("status")), None)
        if not (a_row and b_row) or a_row["seconds"] <= 0:
            print(f"  {name:10s} insufficient data")
            continue
        k = math.log(b_row["seconds"] / a_row["seconds"]) / math.log(
            b_row["valid"] / max(a_row["valid"], 1))
        est = b_row["seconds"] * (2_108_694 / b_row["valid"]) ** k
        print(f"  {name:10s} exponent={k:5.2f}  est {est:8.1f}s  "
              f"(x12 scenes ~ {est*12/60:6.1f} min of pure graph building)")


if __name__ == "__main__":
    main()
