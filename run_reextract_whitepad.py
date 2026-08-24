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
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return bool(out)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait", action="store_true",
                   help="Block until the local GPU is free before extracting.")
    p.add_argument("--poll", type=int, default=300)
    a = p.parse_args()

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
        if os.path.isdir(done) and len(os.listdir(done)) > 4:
            print(f"[SKIP] {s} ({len(os.listdir(done))} files present)", flush=True)
            continue
        t0 = time.time()
        print(f"[START] {s} {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run(
            [PY, EXTRACT, "-s", src, "--model", "SAMOpenCLIP",
             "--ouput-dir", OUT_NAME, "--sam_ckpt_path", SAM_CKPT, "--device", "cuda"],
            env=env, stdout=open(f"logs_reextract_{s}.log", "w"), stderr=subprocess.STDOUT)
        print(f"[DONE ] {s} rc={r.returncode} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
