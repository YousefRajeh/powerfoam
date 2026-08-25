"""Re-extract SAM+CLIP features with the pad colour matched to the mask fill.

THE FIX. `get_seg_img` fills outside-mask/inside-bbox with WHITE 255 (an upstream splat-distiller
deviation from LangSplat, which uses BLACK 0), while `pad_img` squares the crop with `np.zeros` =
BLACK. Every crop therefore carries two different artificial backgrounds with a hard boundary at
the bbox edge. Setting LANGSPLAT_PAD_VALUE=255 makes the padding match the fill.

MEASURED JUSTIFICATION (real SAM masks, OpenCLIP ViT-B-16/laion2b_s34b_b88k, two scenes), share
of all crops whose nearest class text embedding is `floor` -- the dominant attractor:
    black fill + black pad (LangSplat)      78.9%
    white fill + black pad (what we have)   71.9%
    white fill + white pad (THIS RUN)       37.3%
per-class mean-cosine span narrows 0.123 -> 0.074, and argmax wins spread across 8-10 classes
instead of 3. This was the largest single effect in a full line-by-line audit of our extraction
against LangSplat's `preprocess.py` and OpenGaussian's consumption path; the encoder, SAM
parameters, NMS thresholds, level ordering and ID convention were all verified IDENTICAL.

WHAT THIS DOES NOT CLAIM. The attractor is present in both fill variants and is only mitigated,
not removed, so this is not established as a fix for the benchmark gap -- it is the one
extraction-side change with a measured effect large enough to be worth the GPU time. Output goes
to a SEPARATE folder so the existing artifacts stay intact and the two can be A/B'd.

SCHEDULING. Extraction is GPU-heavy (SAM ViT-H over every view). This script therefore WAITS for
the reconstruction queues to drain before starting, so it never contends with them.
"""
import argparse
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
EXTRACT = r"D:\Downloads\splat-distiller\feature_extractor.py"
SAM_CKPT = r"D:\Downloads\powerfoam\checkpoints\sam_vit_h_4b8939.pth"
DATA = r"D:\Downloads\powerfoam\data\scannet"
OUT_NAME = "openclip_features_sam_whitepad"

# ALL 10 SCENES. Ordered SMALLEST FIRST so the cheap scenes land early and can be evaluated
# while the large ones are still extracting -- scene0347_00 (54 imgs) has every comparison
# artifact already, so it gives the first read on whether the white-pad fix moves mIoU.
# Cost note: a 1-image shard measured 178 s, but that INCLUDES the one-time SAM ViT-H (2.4 GB)
# + CLIP model load and ran against a busy GPU, so it is an overestimate of the marginal rate.
# 1,158 images total across the 10 scenes.
SCENES = ["scene0062_00", "scene0097_00", "scene0200_00", "scene0347_00", "scene0400_00",
          "scene0070_00", "scene0590_00", "scene0140_00", "scene0645_00", "scene0000_00"]


def local_gpu_busy():
    """True while a PowerFoam training run still holds the local GPU."""
    try:
        # NOT --query-compute-apps: on Windows that lists desktop/graphics processes too (22 of
        # them here, all with [N/A] memory), so it never reports idle and the wait never ends.
        # Total used memory is the reliable signal -- an idle desktop sits near 1.7 GB, while a
        # PowerFoam training run holds 4-8 GB.
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out.split()[0]) > 3000
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait", action="store_true",
                   help="Block until the local GPU is free before extracting.")
    p.add_argument("--poll", type=int, default=300)
    a = p.parse_args()

    # SINGLE INSTANCE. Three copies of this script were once alive at the same time, all
    # walking the same scene list into the same folder and each truncating the others'
    # per-scene logs -- which is how a run that died at 85% got recorded as `rc=0` in 6.1 min.
    # An exclusive create fails if another runner holds the lock.
    lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".reextract_whitepad.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[ABORT] another runner holds {lock}; delete it if that runner is dead.")
        return 1
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    import atexit
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))

    if a.wait:
        print(f"[wait] holding until the local GPU frees up ...", flush=True)
        while local_gpu_busy():
            time.sleep(a.poll)
        print(f"[wait] GPU free at {time.strftime('%H:%M:%S')}", flush=True)

    env = dict(os.environ)
    env["LANGSPLAT_PAD_VALUE"] = "255"          # <- the fix
    env["PYTHONPATH"] = (r"D:\Downloads\splat-distiller;"
                          r"D:\Downloads\splat-distiller\submodules\segment-anything-langsplat")

    for s in SCENES:
        src = os.path.join(DATA, f"{s}_colmap")
        done = os.path.join(src, OUT_NAME)
        # The extractor emits TWO files per image (`{idx}_f.npy` features, `{idx}_s.npy` seg),
        # so completeness is 2*n_images. The old `> 4` guard would have accepted a run that
        # died at 85% as finished and silently skipped re-doing it.
        n_img = len(os.listdir(os.path.join(src, "images")))
        have = len(os.listdir(done)) if os.path.isdir(done) else 0
        if have >= 2 * n_img:
            print(f"[SKIP] {s} (complete: {have}/{2 * n_img})", flush=True)
            continue
        if have:
            print(f"[REDO] {s} (partial: {have}/{2 * n_img})", flush=True)
        t0 = time.time()
        print(f"[START] {s} {time.strftime('%H:%M:%S')}", flush=True)
        # ABSOLUTE path. `feature_extractor.py:241` uses `--ouput-dir` verbatim as the output
        # path, so a bare folder name resolves against the CWD -- every scene then writes into
        # ONE shared directory, and because the files are named by frame index (`4840_f.npy`)
        # and ScanNet frame indices repeat across scenes, the scenes silently overwrite each
        # other. The first run of this script lost its output exactly that way.
        r = subprocess.run(
            [PY, EXTRACT, "-s", src, "--model", "SAMOpenCLIP",
             "--ouput-dir", done, "--sam_ckpt_path", SAM_CKPT, "--device", "cuda"],
            env=env, stdout=open(f"logs_reextract_{s}.log", "w"), stderr=subprocess.STDOUT)
        print(f"[DONE ] {s} rc={r.returncode} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
