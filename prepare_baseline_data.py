"""Point the baseline repositories at the ScanNet data we already have, without duplicating it.

WHY THIS IS SMALL. OpenGaussian's reader wants plain COLMAP (`sparse/0/{cameras,images,points3D}.bin`,
`images/`) plus `language_features/{stem}_f.npy` + `{stem}_s.npy` -- and that is exactly what
`data/scannet/{scene}_colmap/` already is, with the features under a different directory name. The
stems match too (`0.jpg` <-> `0_f.npy`, 279 of each on scene0000_00). So the "adapter" is a
directory alias, not a dataloader.

Verified before writing, because the failure mode of guessing here is a run that trains for hours
on silently-empty features:
  * scene/dataset_readers.py:121-122 builds  <images_folder minus 'images'>/language_features/<stem>_{s,f}.npy
  * its scan_list is our exact ten scenes
  * `--sam_level 0` matches our single-level extraction (asking for 3 would select nothing)

DIRECTORY JUNCTIONS, not copies. The SAM features are ~2 GB per scene; copying them per baseline
would be tens of GB for no reason. A junction is resolved by every reader transparently and costs
nothing. If junction creation is unavailable the script falls back to copying and says so, rather
than silently proceeding with a missing directory.

RECONSTRUCTION ARM PER METHOD. OpenGaussian passes `--frozen_init_pts` (one Gaussian per GT vertex,
no densification), so it belongs against our `gs_froz` arm; LangSplat/LaGa/THGS/VALA do not freeze
and belong against `gs_unfroz`. Recorded here so the pairing is stated once rather than rediscovered
per method.
"""
import argparse
import os
import subprocess
import sys

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
ROOT = r"D:\Downloads\powerfoam\data\scannet"
FEATURE_DIR = "openclip_features_sam_l3"

# which of OUR reconstruction arms each method's protocol corresponds to
ARM = {"OpenGaussian": "gs_froz",      # --frozen_init_pts
       "LangSplat": "gs_unfroz", "LaGa": "gs_unfroz",
       "THGS": "gs_unfroz", "VALA": "gs_unfroz"}


def junction(src, dst):
    """Windows directory junction; falls back to a copy, loudly."""
    if os.path.exists(dst):
        return "exists"
    try:
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True,
                       capture_output=True, text=True)
        return "junction"
    except Exception:                                        # noqa: BLE001
        import shutil
        shutil.copytree(src, dst)
        return "COPIED (junction unavailable -- this used real disk)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    ok = miss = 0
    for s in a.scenes.split(","):
        base = os.path.join(ROOT, f"{s}_colmap")
        src = os.path.join(base, FEATURE_DIR)
        dst = os.path.join(base, "language_features")
        if not os.path.isdir(src):
            print(f"  [MISS] {s}: no {FEATURE_DIR}")
            miss += 1
            continue
        how = junction(src, dst)
        n_img = len(os.listdir(os.path.join(base, "images")))
        n_f = len([f for f in os.listdir(src) if f.endswith("_f.npy")])
        flag = "ok" if n_img == n_f else f"MISMATCH images={n_img} feats={n_f}"
        print(f"  {s:14s} {how:12s} images={n_img:4d} features={n_f:4d}  {flag}")
        ok += 1
    print(f"\n{ok} scenes prepared, {miss} missing")
    print("\nreconstruction arm per method (frozen vs not, from each paper/script):")
    for m, arm in ARM.items():
        print(f"  {m:14s} -> {arm}")


if __name__ == "__main__":
    main()
