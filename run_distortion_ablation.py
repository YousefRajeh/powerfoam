"""Distortion-channel ablation: the power-diagram-native claim, tested.

CLAIM. VoroTracing routes the distortion gradient to DENSITY ONLY. On a plain Voronoi
diagram that is nearly forced: a midpoint bisector cannot move without moving a site, which
drags every other boundary of that cell with it. On a POWER diagram the weight r_i
translates cell i's planes without moving its center, so the same loss can also mean "make
this cell thinner" -- and the (1/3) sum w^2 ds self-term is exactly a thickness penalty.

CONDITIONS (all else identical, densification off, 30k iters):
  off      distortion disabled                      -- control
  density  gradient to density only                 -- reproduces VoroTracing
  radii    gradient to the power weights only       -- the channel they cannot have
  both     unrestricted                             -- full power-diagram formulation

HARDEST SCENE FIRST (scene0347_00): coherence-gated geodesic growing collapsed to 1.84 mIoU
there against a ~40 baseline, so it is the scene most likely to falsify an idea early.

KILL CRITERION, stated before launching: if neither `radii` nor `both` moves the median
dist-to-GT/radius below the `off` control by at least 0.5 radii while holding PSNR within
0.3 dB, the channel claim is not supported and the direction is dropped rather than tuned.
Position learning drives that metric from 0.06 to 3.49 radii, so a real geometry effect has
room to show; anything smaller is noise against the ~1 mIoU run-to-run floor already
measured on this pipeline.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = r"D:\conda\envs\powerfoam\python.exe"
SCENE_POINTS = {
    "scene0347_00": 67984, "scene0070_00": 109380, "scene0140_00": 372941,
    "scene0645_00": 352477, "scene0590_00": 222957, "scene0200_00": 83291,
    "scene0097_00": 72007, "scene0400_00": 155959, "scene0062_00": 51610,
    "scene0000_00": 81369,
}


def train_one(scene, tag, extra, iterations):
    out = Path("output") / tag
    if (out / "metrics.txt").exists():
        print(f"[skip] {tag} already done", flush=True)
        return True
    n = SCENE_POINTS[scene]
    cmd = [PY, "train.py", "-c", "configs/scannet.yaml", "--scene", f"{scene}_colmap",
           "--init_points", str(n), "--final_points", str(n),
           "--densify_from", "0", "--densify_until", "0",
           "--iterations", str(iterations), "--experiment_name", tag] + extra
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace")
    mins = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"[FAIL] {tag} rc={r.returncode} after {mins:.1f}min", flush=True)
        print("\n".join(r.stdout.splitlines()[-15:]), flush=True)
        return False
    print(f"[ok] {tag} ({mins:.1f}min)", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--weight", default="2e-3")
    p.add_argument("--quantiles", type=int, default=16)
    p.add_argument("--channels", default="off,both,density,radii")
    args = p.parse_args()

    done = []
    for ch in args.channels.split(","):
        tag = f"dist_{args.scene}_{ch}"
        if ch == "off":
            extra = ["--distortion_weight", "0.0"]
        else:
            extra = ["--distortion_mode", "exact",
                     "--distortion_weight", args.weight,
                     "--distortion_num_quantiles", str(args.quantiles),
                     "--distortion_channel", ch]
        if train_one(args.scene, tag, extra, args.iterations):
            done.append(tag)

    print("\n=== photometric summary ===")
    rows = {}
    for tag in done:
        f = Path("output") / tag / "metrics.txt"
        if not f.exists():
            continue
        vals = {}
        for line in f.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    vals[k.strip().replace("Average ", "").lower()] = float(v)
                except ValueError:
                    pass
        rows[tag] = vals
        print(f"{tag:<34} PSNR={vals.get('psnr', 0):.4f} SSIM={vals.get('ssim', 0):.4f} "
              f"LPIPS={vals.get('lpips', 0):.4f}")
    json.dump(rows, open(f"artifacts/scannet/distortion_ablation_{args.scene}.json", "w"),
              indent=2)
    print("\nNext: run diagnose_sigma_radius_joint.py on these runs for the geometry side "
          "(median dist-to-GT / radius). The photometric number alone cannot decide this -- "
          "the claim is about geometry, and PSNR is only the guard against buying thin "
          "surfaces by wrecking appearance.")


if __name__ == "__main__":
    main()
