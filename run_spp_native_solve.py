"""Lift CLIP features onto the NATIVE-RESOLUTION ScanNet++ reconstructions and solve, locally.

WHY LOCAL. This is a short GPU job per scene, not a multi-hour training run -- scheduling it through
SLURM cost three failed submissions (wrong CLI, CRLF line endings, a package missing from the remote
env) before producing anything. The reconstructions are ~2 GB each to copy; the machine that already
has the features, the lifting package and a free GPU is the right place to run it.

THE LIFT RUNS AT 1600 px, NOT 1752. load_image_feature_from_SAMOpenCLIP builds the per-pixel feature
map from the shape of the stored _s.npy (1066x1600). A native 1168x1752 camera indexes it out of
bounds -- the failure its own docstring documents. SAM+CLIP features only exist at 1600 px, so the
config keeps max_image_width: 1600 while the checkpoint is the native-trained geometry. Cell
positions are resolution-independent, so the lift is valid; this isolates the RECONSTRUCTION change
(native training + refbench-matched count, ~700k -> 1.0-2.25M primitives) and leaves the feature
pipeline byte-identical to the runs that produced the existing numbers. A fully-native lift would
need SAM+CLIP re-extracted at 1752 px.

SCENES RUN SEQUENTIALLY. An unrelated job (validate_bound.py) holds ~20 GB of the 49 GB card, and
d755b3d9d8 is 2.25M primitives over 568 views; running scenes in parallel risks OOMing someone
else's work as much as our own.
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\Downloads\spp_data_1600"
CKPTS = {
    "0d2ee665be": r"artifacts\spp_native_ckpts\spp_native_0d2ee665be",
    "27dd4da69e": r"artifacts\spp_native_ckpts\spp_native_27dd4da69e",
    "3864514494": r"artifacts\spp_native_ckpts\spp_native_3864514494",
    "578511c8a9": r"artifacts\spp_native_ckpts\spp_native_578511c8a9",
    "c50d2d1d42": r"artifacts\spp_native_ckpts\spp_native_c50d2d1d42",
    "d755b3d9d8": r"output\spp_native_d755b3d9d8",          # trained here, never left
}
OUT = os.path.join(HERE, "artifacts", "spp_native")


def local_config(scene, ckpt):
    """The config the run resolved, with the data root and lift resolution set for 1600 px.

    accumulate_feature_stats_sam derives the checkpoint directory from experiment_name relative to
    the CWD, so experiment_name is left alone and the caller runs from a directory where
    output/<experiment_name> resolves -- here that is a staged copy next to the checkpoint.
    """
    # accumulate_feature_stats_sam derives the checkpoint dir as
    #   config_path.replace("\\config.yaml", "")
    # so the config must be named EXACTLY config.yaml and live beside model.pt. Stage a directory
    # per scene with the cleaned config and a HARDLINK to model.pt: no original is modified
    # (d755b3d9d8's dir is the real training output) and no 2 GB file is duplicated.
    stage = os.path.join(OUT, "ckpt_%s" % scene)
    os.makedirs(stage, exist_ok=True)
    link = os.path.join(stage, "model.pt")
    if not os.path.exists(link):
        real = os.path.join(ckpt, "model.pt")
        try:
            os.link(real, link)
        except OSError:
            import shutil
            shutil.copy2(real, link)
    src = os.path.join(ckpt, "config.yaml")
    dst = os.path.join(stage, "config.yaml")
    lines = open(src).read().replace("\r\n", "\n").split("\n")
    out = []
    for ln in lines:
        # train.py dumps EVERY resolved arg, including its own argparse-only flags. configargparse
        # re-emits those as --ckpt_every=1000 --resume=true, which accumulate_feature_stats_sam does
        # not declare, so it dies at parse time before loading anything. train.py survives the
        # round-trip only because it declares both spellings itself; nothing else does.
        if ln.split(":")[0].strip() in ("ckpt_every", "resume"):
            continue
        if ln.startswith("data_path:"):
            ln = "data_path: %s" % DATA.replace("\\", "/")
        elif ln.startswith("max_image_width:"):
            ln = "max_image_width: 1600"
        out.append(ln)
    open(dst, "w", newline="\n").write("\n".join(out))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(CKPTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for scene in [s.strip() for s in a.scenes.split(",") if s.strip()]:
        ckpt = os.path.join(HERE, CKPTS[scene])
        feat = os.path.join(DATA, scene, "openclip_features_sam_l3")
        imgs = os.path.join(DATA, scene, "images")
        n_img = len(os.listdir(imgs))
        n_feat = len(os.listdir(feat)) if os.path.isdir(feat) else 0
        # Two files per image (_f.npy + _s.npy). A partial feature set would silently produce a
        # wrong number rather than an error, so refuse instead of guessing.
        if n_feat < 2 * n_img:
            print("[SKIP ] %s: features %d/%d incomplete" % (scene, n_feat, 2 * n_img), flush=True)
            continue
        if not os.path.exists(os.path.join(ckpt, "model.pt")):
            print("[SKIP ] %s: no model.pt" % scene, flush=True)
            continue

        stats = os.path.join(OUT, "stats_native_%s.pt" % scene)
        solved = os.path.join(OUT, "solved_gm_native_%s.pt" % scene)
        if os.path.exists(solved) and not a.force:
            print("[SKIP ] %s already solved" % scene, flush=True)
            continue

        cfg = local_config(scene, ckpt)
        t0 = time.time()
        if not os.path.exists(stats) or a.force:
            print("[LIFT ] %s (%d views)" % (scene, n_img), flush=True)
            log = os.path.join(OUT, "lift_%s.log" % scene)
            with open(log, "w") as fh:
                r = subprocess.run(
                    [PY, os.path.join(HERE, "accumulate_feature_stats_sam.py"),
                     "--scene", scene, "--config", cfg, "--feature-folder", feat,
                     "--output", stats, "--sam-level", "0"],
                    cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print("[FAIL ] %s lift rc=%d (%s)" % (scene, r.returncode, log), flush=True)
                continue

        print("[SOLVE] %s" % scene, flush=True)
        log = os.path.join(OUT, "solve_%s.log" % scene)
        with open(log, "w") as fh:
            r = subprocess.run([PY, os.path.join(HERE, "solve_geometric_median.py"),
                                "--stats", stats, "--output", solved],
                               cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print("[FAIL ] %s solve rc=%d (%s)" % (scene, r.returncode, log), flush=True)
            continue
        print("[OK   ] %s in %.1f min -> %s" % (scene, (time.time() - t0) / 60.0,
                                                os.path.basename(solved)), flush=True)


if __name__ == "__main__":
    main()
