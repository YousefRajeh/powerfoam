"""Matched-budget PowerFoam reruns on the LOCAL GPU, strictly one scene at a time.

WHY LOCAL AT ALL. There is a standing rule in the plan file that scene retraining goes on remote
GPUs and the local card is reserved for idea-iteration -- it exists because concurrent local GPU jobs
once nearly crashed the machine. This run is a deliberate, narrowed exception agreed with the user:
Ibex has 12 jobs pending and 0 running (fairshare 0.18, no idle V100s), all three 995 A6000s are at
98-100% under another user, and the local RTX 6000 Ada is completely idle at 2/49 GB. It is also the
fastest of the three for this workload (~2x a V100).

The safety conditions that make it acceptable: ONE scene at a time (never concurrent, which is what
caused the earlier incident), only the two SMALLEST scenes, and a GPU-free check before each start.

WHAT IT CHANGES. final_points = that scene's GAUSSIAN COUNT instead of the flat 700,000 every
ScanNet++ PowerFoam run used, so the arms become like-for-like. init_points is untouched -- only the
densification target moves, keeping the experiment single-variable.

DISJOINTNESS. Both scenes were scancel'd from Ibex before this runs. Two runners writing one output
directory is a bug this project has already hit; relying on the `metrics.txt` skip-guard would be a
race, so the split is structural.
"""
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
PF = r"D:\Downloads\powerfoam"
SRC = r"D:\Downloads\spp_results\full"
DATA = r"D:\Downloads\spp_data_1600"
TARGET = {"27dd4da69e": 1022151, "0d2ee665be": 1136402}
SCENES = ["27dd4da69e", "0d2ee665be"]          # smallest first


def gpu_free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    used, total = [int(x) for x in out.splitlines()[0].split(",")]
    return total - used


def main():
    for s in SCENES:
        out_dir = os.path.join(PF, "output", f"spp_pf_matched_{s}")
        if os.path.exists(os.path.join(out_dir, "metrics.txt")):
            print(f"[skip] {s} already done", flush=True); continue
        src_cfg = os.path.join(SRC, f"spp_pf_unfroz_{s}", "config.yaml")
        if not os.path.exists(src_cfg):
            print(f"[MISS] {s}", flush=True); continue
        free = gpu_free_mb()
        if free < 30_000:
            print(f"[ABORT] {s}: only {free} MB free on the GPU; refusing to contend", flush=True)
            return 1
        cfg = os.path.join(PF, "configs", f"spp_pf_matched_{s}.yaml")
        lines = []
        for ln in open(src_cfg):
            if ln.startswith("final_points:"):
                ln = f"final_points: {TARGET[s]}\n"
            elif ln.startswith("experiment_name:"):
                ln = f"experiment_name: spp_pf_matched_{s}\n"
            elif ln.startswith("data_path:"):
                ln = f"data_path: {DATA}\n"
            lines.append(ln)
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        open(cfg, "w").writelines(lines)
        # all-numeric scene names come back YAML-quoted; accept both forms
        got = [l.strip() for l in lines if l.startswith("scene:")]
        if not got or got[0].split(":", 1)[1].strip().strip("'\"") != s:
            print(f"[BADCFG] {s}: {got}", flush=True); continue
        print(f"[start] {s} final={TARGET[s]} (was 700000) free={free}MB "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        log = os.path.join(PF, f"logs_matched_local_{s}.log")
        rc = subprocess.run([PY, "train.py", "-c", cfg], cwd=PF,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        met = os.path.join(out_dir, "metrics.txt")
        if os.path.exists(met):
            print(f"[OK] {s} {mins:.0f}min {open(met).read().strip()}", flush=True)
        else:
            print(f"[FAIL] {s} rc={rc} {mins:.0f}min", flush=True)
            print("".join(open(log).readlines()[-6:]), flush=True)
    print("=== LOCAL MATCHED DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
