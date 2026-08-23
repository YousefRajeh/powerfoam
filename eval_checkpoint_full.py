"""Full downstream evaluation of an arbitrary PowerFoam checkpoint.

The distortion ablation produced new GEOMETRIES, so nothing downstream can be reused: the
lifted features, the facet adjacency and the surface mesh are all functions of the cells.
This runs the whole chain per checkpoint --

    SAM+CLIP accumulation (L3 only)  ->  geometric-median solve  ->  facet adjacency
      ->  3D point-level mIoU/mAcc + semantic-surface metrics
      ->  reconstruction surface metrics (median depth -> TSDF -> Chamfer)

-- so a training-side change can be judged on what the paper actually reports, not on PSNR.
That matters here specifically: the distortion loss traded 0.26 dB of PSNR for 0.47 radii of
surface drift, and neither number says whether the SEGMENTATION got better or worse.

Paths are explicit rather than derived from a `variant` string, because these checkpoints do
not follow the output/scannet_{scene}_{variant} convention the older scripts assume.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

PY = sys.executable


def sh(cmd, tag):
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace")
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"  [{tag}] FAILED rc={r.returncode} ({dt:.1f}min)", flush=True)
        print("\n".join(r.stdout.splitlines()[-18:]), flush=True)
        return False
    print(f"  [{tag}] ok ({dt:.1f}min)", flush=True)
    return True


def prepare(ckpt_dir, scene, feat_dir, work):
    """Lift features onto THIS checkpoint's cells and build its facet graph."""
    stats = work / "stats_l3.pt"
    solved = work / "solved_l3.pt"
    adj = work / "adjacency.pt"
    cfg = f"{ckpt_dir}/config.yaml"
    if not solved.exists():
        if not stats.exists():
            if not sh([PY, "accumulate_feature_stats_sam.py", "--scene", scene,
                       "--config", cfg, "--feature-folder", str(feat_dir),
                       "--output", str(stats), "--sam-level", "3"], "accumulate"):
                return None
        if not sh([PY, "solve_geometric_median.py", "--stats", str(stats),
                   "--output", str(solved)], "solve"):
            return None
    if not adj.exists():
        if not sh([PY, "export_adjacency_graph.py", "-c", cfg, "--output", str(adj)],
                  "adjacency"):
            return None
    return stats, solved, adj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dirs", nargs="+", required=True)
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--feature-dir", default=None)
    p.add_argument("--keep-stats", action="store_true",
                   help="keep the ~1.9GB accumulator stats (needed for true reliability)")
    p.add_argument("--skip-surface", action="store_true")
    p.add_argument("--output", default="artifacts/scannet/checkpoint_full_eval.json")
    args = p.parse_args()

    feat_dir = Path(args.feature_dir or f"artifacts/scannet/{args.scene}/openclip_features_sam")
    results = {}
    for ckpt_dir in args.ckpt_dirs:
        name = Path(ckpt_dir).name
        print(f"\n=== {name} ===", flush=True)
        work = Path("artifacts/scannet/ckpt_eval") / name
        work.mkdir(parents=True, exist_ok=True)
        got = prepare(ckpt_dir, args.scene, feat_dir, work)
        if got is None:
            continue
        stats, solved, adj = got

        # semantic: point mIoU/mAcc + per-class semantic Chamfer, champion stack
        sem_out = work / "semantic.json"
        ok = sh([PY, "eval_semantic_surface.py", "--scenes", args.scene,
                 "--gt-root", args.gt_root, "--output", str(sem_out),
                 "--ckpt-dir", ckpt_dir, "--solved", str(solved),
                 "--stats", str(stats), "--adjacency", str(adj)], "semantic")
        entry = {"ckpt": ckpt_dir}
        if ok and sem_out.exists():
            entry["semantic"] = json.load(open(sem_out))["summary"]

        if not args.skip_surface:
            surf_out = work / "surface.json"
            if sh([PY, "eval_surface_chamfer.py", "--scene", args.scene,
                   "--gt-root", args.gt_root, "--ckpt-dir", ckpt_dir,
                   "--output", str(surf_out)], "surface"):
                if surf_out.exists():
                    entry["surface"] = json.load(open(surf_out)).get("full")

        results[name] = entry
        if not args.keep_stats and stats.exists():
            stats.unlink()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")
    for k, v in results.items():
        s = v.get("semantic", {})
        line = f"{k:<34}"
        for cs in ("opengaussian19", "opengaussian15", "opengaussian10"):
            if cs in s:
                line += f" {cs[-2:]}cls mIoU={s[cs]['mIoU']*100:5.2f}"
        if "surface" in v and v["surface"]:
            line += f"  CD-L1={v['surface'].get('chamfer_l1', 0)*100:.2f}cm"
        print(line)


if __name__ == "__main__":
    main()
