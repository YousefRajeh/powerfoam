"""Re-run all 10 PowerFoam scenes with a point set that is ACTUALLY frozen.

The existing `output/scannet_*_frozen` checkpoints are kept -- they are not deleted, because they
are the "broken freeze" arm and are worth having for comparison. These write to `_truefrozen`.

Why they were broken: train.py:507 ran resample() + sort_points() outside the densify guard, so
the point COUNT stayed at N while the point SET churned (19.6% GT survival at 30k). Fixed by
gating that block on `freeze_points`, verified at 1500 iters: 51610/51610 = 100.00%.

Protocol is otherwise identical to the existing `_nonfrozen` arm (eval: false, 30k iterations),
so frozen-vs-unfrozen is a clean paired comparison within PowerFoam.
"""
import json
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
SCENES = {
    "scene0062_00": 51610, "scene0347_00": 67984, "scene0097_00": 72007,
    "scene0000_00": 81369, "scene0200_00": 83291, "scene0070_00": 109380,
    "scene0400_00": 155959, "scene0590_00": 222957, "scene0645_00": 352477,
    "scene0140_00": 372941,
}


def main():
    base = open("configs/scannet_truefrozen.yaml", encoding="utf-8").read()
    log = {}
    for scene, n in SCENES.items():
        out = f"output/scannet_{scene}_truefrozen"
        if os.path.exists(f"{out}/model.pt"):
            print(f"[SKIP] {scene} already done", flush=True)
            continue
        cfg = f"configs/_tf_{scene}.yaml"
        lines = []
        for ln in base.splitlines():
            k = ln.split(":")[0].strip()
            if k == "scene":
                ln = f"scene: {scene}_colmap"
            elif k == "init_points":
                ln = f"init_points: {n}"
            elif k == "final_points":
                ln = f"final_points: {n}"
            lines.append(ln)
        open(cfg, "w", encoding="utf-8").write("\n".join(lines) + "\n")

        t0 = time.time()
        print(f"[START] {scene} pts={n} {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run(
            [PY, "train.py", "-c", cfg, "--experiment_name", f"scannet_{scene}_truefrozen"],
            stdout=open(f"logs_tf_{scene}.log", "w"), stderr=subprocess.STDOUT)
        dt = time.time() - t0
        print(f"[DONE ] {scene} rc={r.returncode} {dt/60:.1f} min", flush=True)
        log[scene] = {"rc": r.returncode, "minutes": dt / 60}
        json.dump(log, open("artifacts/truefrozen_queue_status.json", "w"), indent=2)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
