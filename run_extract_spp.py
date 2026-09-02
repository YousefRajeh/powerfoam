"""SAM+CLIP extraction for ScanNet++, matching the ScanNet `_ogl3` protocol exactly.

WHY THIS EXISTS. `run_reextract_whitepad.py` hardcodes ScanNet's `data/scannet/{scene}_colmap`
layout and scene list. ScanNet++ lives at `spp_data_1600/{scene}` with the same `images/` + `sparse/`
structure, so only the path construction differs -- the extractor invocation, the environment, and
the mask/NMS arithmetic are byte-identical to what produced the ScanNet features.

PROTOCOL IS FROZEN AND MUST NOT BE "IMPROVED" HERE. fill=0, pad=0 (LangSplat/OpenGaussian black on
both, which is what their published ScanNet numbers consume) and SAM_ONLY_LEVEL=l, output folder
`openclip_features_sam_l3`. The whole point of the ScanNet++ run is to test the frozen constants
(lam=0.3, CSLS k=1000, s=200, alpha=0.95, iters=100) on a DIFFERENT domain. Any change to the
extraction protocol would confound the transfer result with a feature change.

SHARDING. Measured single-process cost is ~34 s/image and it is not a GPU wall: utilisation averages
~16% while SAM's mask post-processing spreads across several cores. Disjoint image subsets run in
parallel; sharding changes only WHICH images each process handles, leaving the arithmetic untouched,
so the bit-exact verification against LangSplat still holds.
"""
import argparse
import os
import subprocess
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
EXTRACT = r"D:\Downloads\splat-distiller\feature_extractor.py"
SAM_CKPT = r"D:\Downloads\powerfoam\checkpoints\sam_vit_h_4b8939.pth"
DATA = r"D:\Downloads\spp_data_1600"

# Image counts, largest first. The 12 scenes with a COMPLETED PowerFoam-unfrozen reconstruction.
SPP = ["09c1414f1b", "e7af285f7d", "9071e139d9", "3db0a1c8f3", "d755b3d9d8", "f9f95681fd",
       "5942004064", "578511c8a9", "c50d2d1d42", "27dd4da69e", "3864514494", "0d2ee665be"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="", help="comma list; default all 12")
    p.add_argument("--shards", type=int, default=3)
    p.add_argument("--out-name", default="openclip_features_sam_l3")
    p.add_argument("--fill", type=int, default=0)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--only-level", default="l")
    p.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES for this worker")
    a = p.parse_args()

    env = dict(os.environ)
    env["LANGSPLAT_FILL_VALUE"] = str(a.fill)
    env["LANGSPLAT_PAD_VALUE"] = str(a.pad)
    env["SAM_ONLY_LEVEL"] = a.only_level
    env["PYTHONPATH"] = (r"D:\Downloads\splat-distiller;"
                         r"D:\Downloads\splat-distiller\submodules\segment-anything-langsplat")
    if a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu

    todo = [x for x in (a.scenes.split(",") if a.scenes else SPP) if x]
    print(f"[plan] {len(todo)} scene(s): {', '.join(todo)}", flush=True)
    for s in todo:
        src = os.path.join(DATA, s)
        done = os.path.join(src, a.out_name)
        img_dir = os.path.join(src, "images")
        if not os.path.isdir(img_dir):
            print(f"[MISS] {s}: no images/ at {img_dir}", flush=True); continue
        n_img = len(os.listdir(img_dir))
        # TWO files per image ({idx}_f.npy, {idx}_s.npy), so completeness is 2*n_img. A loose
        # guard once accepted a run that died at 85% as finished.
        have = len(os.listdir(done)) if os.path.isdir(done) else 0
        if have >= 2 * n_img:
            print(f"[SKIP] {s} (complete {have}/{2*n_img})", flush=True); continue
        if have:
            print(f"[REDO] {s} (partial {have}/{2*n_img})", flush=True)
        t0 = time.time()
        print(f"[START] {s} n_img={n_img} shards={a.shards} {time.strftime('%H:%M:%S')}", flush=True)
        procs = []
        for k in range(a.shards):
            log = f"logs_spp_{s}_shard{k}.log"
            cmd = [PY, EXTRACT, "-s", src, "--model", "SAMOpenCLIP",
                   "--ouput-dir", done, "--sam_ckpt_path", SAM_CKPT, "--device", "cuda"]
            if a.shards > 1:
                cmd += ["--num-shards", str(a.shards), "--shard-index", str(k)]
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                                          stderr=subprocess.STDOUT))
        rcs = [q.wait() for q in procs]
        have_now = len(os.listdir(done)) if os.path.isdir(done) else 0
        ok = "OK" if have_now >= 2 * n_img else f"INCOMPLETE {have_now}/{2*n_img}"
        print(f"[DONE ] {s} rcs={rcs} {ok} {(time.time()-t0)/60:.1f} min", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
