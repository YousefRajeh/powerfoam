"""Prove the evaluation path is reproducible, rather than assuming it.

`enable_determinism()` uses `warn_only=True` so that an op without a deterministic kernel
degrades to a warning instead of killing a multi-hour run. That is the right trade-off for long
jobs, but it means enabling it is NOT itself evidence of reproducibility -- some op may still be
silently nondeterministic.

So this measures it directly, the same way the problem was found: evaluate the SAME solved
feature file N times in SEPARATE PROCESSES and compare the resulting mIoU to full precision.
Separate processes matter -- running twice inside one process shares RNG state and would hide
exactly the failure being tested.

Baseline to beat: before this fix, the identical diagonal file scored 36.12 and 36.60 at 19
classes on scene0347_00, a spread of ~0.5 mIoU, which is larger than most effects this project
measures.

Usage:
    python verify_determinism.py --solved artifacts/scannet/scene0347_00/solved_....pt
    python verify_determinism.py --child ...        (internal; one measurement)
"""
import argparse
import json
import subprocess
import sys

PY = sys.executable


def measure(scene, solved, ckpt_dir):
    from determinism import enable_determinism
    enable_determinism(verbose=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("voro", "run_voronoi_feature_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.evaluate(scene, ckpt_dir, solved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--solved",
                   default="artifacts/scannet/scene0347_00/solved_coupled_ridge_diagbaseline.pt")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--child", action="store_true")
    a = p.parse_args()
    ckpt = a.ckpt_dir or f"output/scannet_{a.scene}_nonfrozen"

    if a.child:
        r = measure(a.scene, a.solved, ckpt)
        print("RESULT " + json.dumps({k: r[k] for k in r if k.startswith("opengaussian")}))
        return

    vals = []
    for i in range(a.runs):
        out = subprocess.run(
            [PY, "verify_determinism.py", "--child", "--scene", a.scene,
             "--solved", a.solved, "--ckpt-dir", ckpt],
            capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"run {i}: FAILED\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
            return
        r = json.loads(line[0][len("RESULT "):])
        vals.append(r)
        print(f"  run {i}: " + "  ".join(
            f"{cs[12:]}cls={r[cs]['mIoU']*100:.10f}" for cs in sorted(r)), flush=True)

    print("\n=== reproducibility ===")
    ok = True
    for cs in sorted(vals[0]):
        s = {v[cs]["mIoU"] for v in vals}
        spread = (max(s) - min(s)) * 100
        ok &= len(s) == 1
        print(f"  {cs:<16} distinct values across {a.runs} runs: {len(s)}   "
              f"spread: {spread:.10f} mIoU")
    print("\n  " + ("BITWISE REPRODUCIBLE" if ok else
                    "STILL NONDETERMINISTIC -- some op lacks a deterministic kernel"))
    print("  (pre-fix baseline on this same file: 36.12 vs 36.60, spread ~0.5 mIoU)")


if __name__ == "__main__":
    main()
