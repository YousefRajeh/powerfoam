"""Local GPU queue for the two arms radfoam cannot take.

SPLIT RATIONALE. radfoam is remote-only -- it has no compiled CUDA extension on this Windows box
(importing it from D:\\Downloads only appears to work because a `radfoam/` DIRECTORY is there and
Python treats it as a namespace package; from inside the repo it is ModuleNotFoundError). So the
15 radfoam jobs are pinned to the 3 remote GPUs, ~39 GPU-h => ~13 h.

Everything else is flexible, and the local RTX 6000 Ada can run all of it:
  PowerFoam truefrozen  x10  (~1.2 h each)
  gaussian  unfrozen    x10  (~0.2 h each)
  ~= 13.4 h, which balances the remote side almost exactly.

Ordered SHORTEST FIRST here, opposite to the remote pool's longest-first. The remote pool is
minimising makespan across three workers, where long-jobs-first is optimal. This is a single
worker, so ordering cannot change total time -- but finishing the cheap gaussian arm early means
a complete extra arm is available for evaluation hours sooner.
"""
import json
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
GS = r"D:\Downloads\splat-distiller\gaussian_splatting\simple_trainer.py"
DATA = r"D:\Downloads\powerfoam\data\scannet"

N = {"scene0062_00": 51610, "scene0347_00": 67984, "scene0097_00": 72007,
     "scene0000_00": 81369, "scene0200_00": 83291, "scene0070_00": 109380,
     "scene0400_00": 155959, "scene0590_00": 222957, "scene0645_00": 352477,
     "scene0140_00": 372941}

STATUS = "artifacts/local_ablation_status.json"


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}

    jobs = []
    # gaussian first: 10 min each, so the whole arm lands within ~2 h and can be evaluated
    # while the slower PowerFoam arm is still running.
    for s in N:
        jobs.append(("gsunfroz", s))
    for s in N:
        jobs.append(("pftfroz", s))

    base = open("configs/scannet_truefrozen.yaml", encoding="utf-8").read()

    for arm, scene in jobs:
        key = f"{arm}_{scene}"
        if st.get(key, {}).get("rc") == 0:
            print(f"[SKIP] {key}", flush=True)
            continue

        if arm == "pftfroz":
            out = f"output/scannet_{scene}_truefrozen"
            if os.path.exists(f"{out}/model.pt"):
                print(f"[SKIP] {key} (model.pt exists)", flush=True)
                st[key] = {"rc": 0, "note": "pre-existing"}
                continue
            cfg = f"configs/_tf_{scene}.yaml"
            lines = []
            for ln in base.splitlines():
                k = ln.split(":")[0].strip()
                if k == "scene":
                    ln = f"scene: {scene}_colmap"
                elif k == "init_points":
                    ln = f"init_points: {N[scene]}"
                elif k == "final_points":
                    ln = f"final_points: {N[scene]}"
                lines.append(ln)
            open(cfg, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            cmd = [PY, "train.py", "-c", cfg,
                   "--experiment_name", f"scannet_{scene}_truefrozen"]
        else:
            out = rf"D:\Downloads\gaussian_unfrozen_scannet\{scene}"
            if os.path.exists(rf"{out}\ckpts\ckpt_29999_rank0.pt"):
                print(f"[SKIP] {key} (ckpt exists)", flush=True)
                st[key] = {"rc": 0, "note": "pre-existing"}
                continue
            # Identical to the frozen gaussian arm except the two freeze knobs are released:
            # means_lr 0.0 -> 1.6e-4 (positions move), refine_stop_iter 0 -> 15000 (densify on).
            cmd = [PY, GS, "default",
                   "--data_dir", rf"{DATA}\{scene}_colmap",
                   "--data_factor", "1", "--result_dir", out,
                   "--init_type", "sfm", "--max_steps", "30000",
                   "--means_lr", "1.6e-4", "--strategy.refine-stop-iter", "15000",
                   "--eval_steps", "1000000000", "--ply_steps", "30000",
                   "--disable_viewer", "--disable_video"]

        t0 = time.time()
        print(f"[START] {key} {time.strftime('%H:%M:%S')}", flush=True)
        rc = run(cmd, f"logs_{key}.log")
        dt = (time.time() - t0) / 60
        print(f"[DONE ] {key} rc={rc} {dt:.1f}min", flush=True)
        st[key] = {"rc": rc, "minutes": round(dt, 1)}
        json.dump(st, open(STATUS, "w"), indent=2)

    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
