"""Re-extract SAM+CLIP features under a chosen mask-fill / crop-pad colour pair.

DEFAULTS ARE THE PROTOCOL. --fill 0 --pad 0 is LangSplat's configuration, which is what
OpenGaussian consumes and therefore what their published ScanNet numbers come from. Our
splat-distiller checkout deviates: `get_seg_img` fills outside-mask/inside-bbox with WHITE
255 (the black line survives commented out), while `pad_img` pads BLACK -- so each crop
carries two disagreeing artificial backgrounds. Matching OpenGaussian means BOTH black.

WHAT THE RAW-CROP STATISTIC SAID, AND WHY IT WAS NOT ENOUGH. Share of crops whose nearest
class text embedding is `floor`, the dominant attractor (real SAM masks, OpenCLIP
ViT-B-16/laion2b_s34b_b88k):
    black fill + black pad (OpenGaussian/LangSplat)  78.9%
    white fill + black pad (our previous default)    71.9%
    white fill + white pad                           37.3%
That ranking argued for white+white, so it was tried first. It LOST on the actual metric:
on scene0062_00, mIoU fell 27.27 -> 26.31 (19cls) and 38.61 -> 37.27 (10cls), with mAcc
down too -- 6/6 cells negative. So the attractor share does NOT predict mIoU, and the
argument that once justified deviating from black no longer stands. This runs the protocol.

SCHEDULING. Extraction is GPU-heavy (SAM ViT-H over every view); --wait holds until the
local GPU frees, and --shards runs disjoint image subsets concurrently (~23 s/image at
3 shards vs ~34 single-process; the residual bottleneck is CPU-side mask post-processing).
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

# ALL 10 SCENES, HARDEST FIRST. scene0062_00 leads only because it is already extracted and
# is the reproduction check against the measured 34.71/34.71/50.65. After it come the scenes
# that would FALSIFY the black-fill result soonest: scene0347/0070/0140 are where
# coherence-gated geodesic growing collapsed (1.84 / 0.42 / 3.67 mIoU against a ~40 baseline),
# and 0645/0590 carry the lowest baseline mIoU (28.40 / 35.54) with the largest cell counts
# (352k / 223k), stressing memory and clustering together.
#
# Smallest-first was the previous order and it is actively misleading here: it would finish
# the four cheapest scenes first and build a running average out of the easiest cases. A
# +1.75 mIoU pilot on scene0000_00 once reversed to -12.3 over ten scenes.
#
# KILL CRITERION for the black-fill (protocol) change: it is winning on scene0062_00 by +7.4
# to +12.2 with 6/6 cells positive. If the running mean over the first FOUR hardest scenes is
# not clearly above the white-fill baseline, stop and re-examine rather than extracting the
# remaining six. 1,158 images total.
SCENES = ["scene0062_00", "scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00",
          "scene0590_00", "scene0000_00", "scene0097_00", "scene0200_00", "scene0400_00"]


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
    p.add_argument("--fill", type=int, default=0,
                   help="Pixels inside the bbox but outside the mask. 0 = LangSplat/OpenGaussian.")
    p.add_argument("--pad", type=int, default=0,
                   help="Square-padding around the crop. 0 = LangSplat/OpenGaussian.")
    p.add_argument("--scenes", default="",
                   help="Comma-separated subset to extract. Use this to PARTITION work "
                        "between the local box and the remote GPUs -- two runners walking the "
                        "same list would both start the same scene into the same folder.")
    p.add_argument("--only-level", default="l",
                   help="Generate ONLY this SAM granularity ('l'=whole-object, the level we "
                        "score). Skips postprocessing for the levels we never read; measured "
                        "3.16x faster with the kept level byte-identical. Empty = all four. "
                        "NOTE: the artifact then holds ONE level, stored at index 0, so the "
                        "lift must be run with --sam-level 0, not 3.")
    p.add_argument("--out-name", default="openclip_features_sam_l3",
                   help="Per-scene output folder name.")
    p.add_argument("--shards", type=int, default=3,
                   help="Concurrent extractor processes per scene, over disjoint image "
                        "subsets. Each holds ~9.8 GB of the 48 GB GPU and ~3-6 cores.")
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
    env["LANGSPLAT_FILL_VALUE"] = str(a.fill)
    env["LANGSPLAT_PAD_VALUE"] = str(a.pad)
    if a.only_level:
        env["SAM_ONLY_LEVEL"] = a.only_level
    else:
        env.pop("SAM_ONLY_LEVEL", None)
    env["PYTHONPATH"] = (r"D:\Downloads\splat-distiller;"
                          r"D:\Downloads\splat-distiller\submodules\segment-anything-langsplat")

    todo = [x for x in (a.scenes.split(",") if a.scenes else SCENES) if x]
    print(f"[plan] {len(todo)} scene(s): {', '.join(todo)}", flush=True)
    for s in todo:
        src = os.path.join(DATA, f"{s}_colmap")
        done = os.path.join(src, a.out_name)
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
        print(f"[START] {s} {time.strftime('%H:%M:%S')} shards={a.shards}", flush=True)
        # ABSOLUTE path. `feature_extractor.py:241` uses `--ouput-dir` verbatim as the output
        # path, so a bare folder name resolves against the CWD -- every scene then writes into
        # ONE shared directory, and because the files are named by frame index (`4840_f.npy`)
        # and ScanNet frame indices repeat across scenes, the scenes silently overwrite each
        # other. The first run of this script lost its output exactly that way.
        # SHARDED. Measured single-process cost is ~34 s/image and it is NOT a GPU wall:
        # GPU utilisation averages 16% (bursty, peaking ~68%) while the process spreads
        # ~3-6 cores' worth of CPU across SAM's mask post-processing. On this box (48 GB
        # RTX 6000 Ada using 9.8 GB per process, 32 logical cores) several extractions fit
        # side by side, so the throughput fix is to run disjoint image subsets in parallel.
        #
        # Deliberately NOT a rewrite of the mask/NMS code. That path was verified bit-exact
        # against LangSplat (encoder identical, mask_nms bit-exact on 44 cases); sharding
        # splits *which images* each process handles and leaves the arithmetic untouched,
        # so the verification still holds.
        procs = []
        for k in range(a.shards):
            log = f"logs_reextract_{a.out_name[-9:]}_{s}" + (f"_shard{k}.log" if a.shards > 1 else ".log")
            cmd = [PY, EXTRACT, "-s", src, "--model", "SAMOpenCLIP",
                   "--ouput-dir", done, "--sam_ckpt_path", SAM_CKPT, "--device", "cuda"]
            if a.shards > 1:
                cmd += ["--num-shards", str(a.shards), "--shard-index", str(k)]
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                                          stderr=subprocess.STDOUT))
        rcs = [p.wait() for p in procs]
        rc = max(rcs) if rcs else 1
        have_now = len(os.listdir(done)) if os.path.isdir(done) else 0
        ok = "OK" if have_now >= 2 * n_img else f"INCOMPLETE {have_now}/{2 * n_img}"
        print(f"[DONE ] {s} rc={rc} rcs={rcs} {ok} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
