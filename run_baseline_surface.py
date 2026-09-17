"""Semantic SURFACE metrics for every baseline, under the same protocol as Table 4.

This is the measurement the baselines were actually run for. Point mIoU says how often a label is
right; the surface metrics say WHERE the predicted class boundaries land in 3D relative to the GT
mesh -- which is the claim the foam representation is making, and the one a Gaussian mixture with a
pruned support set is expected to struggle with.

Reuses eval_semantic_surface_gaussian.py unchanged so the numbers are directly comparable to the
existing Table 4 rows (same tau, same opacity threshold, same class sets, same
semantic_surface_metrics implementation). That script exposes single-scene --ckpt/--features
overrides, so it is driven once per manifest entry and the per-scene results are aggregated here.

Emitted per row, alongside the metrics:
    kept_frac   the fraction of the reconstruction's Gaussians the method retained.
Occam keeps ~37% and VALA ~9-14%, so a surface metric computed over their surviving support is
measuring a materially thinner geometry than LUDVIG's or LangSplat's 100%. That is precisely the
effect worth reporting, but it must be read WITH the coverage column, not instead of it -- so the
column travels with every row rather than being averaged away.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
GT_ROOT = r"D:\Downloads\scannet_pointcept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "artifacts", "baseline_eval", "manifest.json"))
    ap.add_argument("--output", default=os.path.join(HERE, "artifacts", "baseline_eval", "surface.json"))
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--tags", default=None)
    a = ap.parse_args()

    entries = json.load(open(a.manifest))
    if a.tags:
        want = set(a.tags.split(","))
        entries = [e for e in entries if e["tag"] in want]

    done = []
    if os.path.exists(a.output):
        done = json.load(open(a.output))
    seen = {(r["tag"], r["scene"]) for r in done}

    for e in entries:
        if (e["tag"], e["scene"]) in seen:
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        cmd = [PY, os.path.join(HERE, "eval_semantic_surface_gaussian.py"),
               "--scenes", e["scene"], "--gt-root", GT_ROOT,
               "--ckpt", e["ckpt"], "--features", e["features"],
               "--class-sets", a.class_sets, "--tau", str(a.tau),
               "--output", tmp.name]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        try:
            res = json.load(open(tmp.name))
        except Exception:
            print(f"  {e['tag']:22s} {e['scene']}: FAIL", flush=True)
            tail = (out.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            print(f"    {tail}", flush=True)
            os.unlink(tmp.name)
            continue
        os.unlink(tmp.name)

        for cs, per in res.get("per_scene", {}).items():
            m = per.get(e["scene"])
            if not m:
                continue
            done.append(dict(tag=e["tag"], baseline=e["baseline"], arm=e["arm"], level=e["level"],
                             scene=e["scene"], class_set=cs,
                             kept_frac=e["kept_frac"], n_kept=e["n_kept"], n_original=e["n_original"],
                             **{k: m[k] for k in ("scd", "mae_pred2gt", "mae_gt2pred", "hd95",
                                                  "boundary_f1", "mIoU", "mAcc",
                                                  "n_missed", "n_classes_present")}))
        with open(a.output, "w") as fh:
            json.dump(done, fh, indent=1)
        s = [r for r in done if r["tag"] == e["tag"] and r["scene"] == e["scene"]
             and r["class_set"] == "opengaussian19"]
        if s:
            r = s[0]
            print(f"  {e['tag']:22s} {e['scene']} scd={r['scd']*100:.2f}cm "
                  f"hd95={r['hd95']*100:.2f} bF1={r['boundary_f1']:.3f} "
                  f"kept={100*(e['kept_frac'] or 0):.1f}%", flush=True)

    print(f"\n{len(done)} rows -> {a.output}")


if __name__ == "__main__":
    main()
