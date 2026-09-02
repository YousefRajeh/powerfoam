"""Matched held-out PowerFoam runs, to pair with the radfoam held-out runs.

Both codebases split identically (indices % 8 == 0 -> test), so setting `eval: true` here
gives the SAME partition radfoam gets from `no_holdout: false`. Everything else is copied
from the config that produced the existing checkpoint, so the only difference between this
run and that one is whether the test eighth was withheld from training.

The point is to test whether radfoam's +6.17 dB train-set advantage survives on unseen views.
Both existing sets of numbers are train-set PSNRs measured under the same protocol -- fair,
but inflated, and inflated most for the highest-capacity arm (rf_unfroz fits 3x the points to
the very views it is scored on).
"""
import os, subprocess, sys, time, yaml

PY = r"D:\conda\envs\powerfoam\python.exe"
SCENES = ["scene0062_00", "scene0347_00", "scene0070_00"]
ARMS = {"truefrozen": "output/scannet_{scene}_truefrozen",
        "nonfrozen":  "output/scannet_{scene}_nonfrozen"}

for scene in SCENES:
    for arm, tmpl in ARMS.items():
        src = os.path.join(tmpl.format(scene=scene), "config.yaml")
        if not os.path.exists(src):
            print(f"[miss] {scene}/{arm}: no config", flush=True); continue
        tag = f"ho_pf_{arm}_{scene}"
        out = f"output/{tag}"
        if os.path.exists(os.path.join(out, "metrics.txt")):
            print(f"[skip] {tag}", flush=True); continue
        # TEXT-LEVEL edit, not a yaml round-trip. safe_dump rewrites lists in block style
        # ("- 500" on its own line) while this config loader requires inline flow sequences
        # ("[0, 500]"), and the block form reached argparse as the literal option "--=500".
        # Editing the two lines in place leaves every other line byte-identical to the config
        # that produced the existing checkpoint, which is the point: the ONLY difference
        # between this run and that one must be the holdout.
        lines = open(src).read().splitlines()
        out_lines, saw_eval = [], False
        for ln in lines:
            if ln.startswith("eval:"):
                out_lines.append("eval: true"); saw_eval = True
            elif ln.startswith("experiment_name:"):
                out_lines.append(f"experiment_name: {tag}")
            else:
                out_lines.append(ln)
        if not saw_eval:
            out_lines.append("eval: true")
        cpath = f"configs/{tag}.yaml"
        open(cpath, "w").write(chr(10).join(out_lines) + chr(10))
        t0 = time.time()
        print(f"[train] {tag} {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run([PY, "train.py", "-c", cpath],
                           stdout=open(f"logs_{tag}.log", "w"), stderr=subprocess.STDOUT)
        m = os.path.join(out, "metrics.txt")
        psnr = open(m).read().strip().split("\n")[0] if os.path.exists(m) else "no metrics"
        print(f"  rc={r.returncode} {(time.time()-t0)/60:.1f} min  {psnr}", flush=True)
print("[POWERFOAM HELDOUT DONE]", flush=True)
