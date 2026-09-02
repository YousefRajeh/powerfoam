"""A8: exact primitives-per-ray, PowerFoam vs 3DGS, 10 ScanNet scenes x 4 reconstructions.

THE CLAIM UNDER TEST. Paper A asserts a foam ray touches fewer primitives and saturates sooner than
a Gaussian ray. That has never been measured -- the only evidence was one-sided (foam rarely exceeds
64 hits; nothing was known about 3DGS). This measures the exact integer count on both.

FOUR ARMS, ALL 10 SCENES:
    pf_truefrozen   one cell per GT vertex          | budget-MATCHED to gs_froz by construction
    pf_nonfrozen    3x that                         |
    gs_froz         one Gaussian per GT point       | budget-MATCHED to pf_truefrozen
    gs_unfroz       densified, ~2.6M                |
The frozen pair is the only like-for-like comparison in the project: identical primitive counts
(81,369 / 109,380 / 67,984 ...), identical scenes, identical GT.

max_hits_per_pixel = 512, NOT the default 64. At 64 the garden 3DGS operator hit exactly 64 as its
maximum with a median of 32 -- i.e. it was CENSORED, while foam (max 32) was not. Measuring under a
cap that binds on one arm and not the other manufactures a foam advantage. 512 is chosen to sit far
above both distributions; the script REPORTS the fraction at the cap so censoring stays visible.

ALL views, native resolution, for both representations -- the two must traverse the same rays or the
comparison is between view sets rather than representations.

STREAMING, NOT MATERIALISED. Only the histogram of per-ray hit counts is kept. The full operator at
512 hits x all views would be hundreds of millions of nonzeros per scene (garden 3DGS was already
225M at a 64 cap on 24 views), and A8 needs the distribution, not the matrix.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
MAX_HITS = 512
OUT = "artifacts/scannet/a8"


def summarise(counts, cap):
    """counts: 1-D int array of per-ray primitive counts (rays with zero hits excluded)."""
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return {"rays": 0}
    qs = np.percentile(c, [1, 25, 50, 75, 95, 99])
    return {"rays": int(c.size), "mean": float(c.mean()), "median": float(np.median(c)),
            "p1": float(qs[0]), "p25": float(qs[1]), "p75": float(qs[3]),
            "p95": float(qs[4]), "p99": float(qs[5]), "max": int(c.max()),
            "at_cap_frac": float((c >= cap).mean()),
            "hist": np.bincount(counts[counts > 0].astype(np.int64),
                                minlength=cap + 1)[:cap + 1].tolist()}


def run_foam(scene, recon, cap):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import export_operator_for_views
    from data_loader import DataHandler

    wp.init()
    cfg = f"output/scannet_{scene}_{recon}/config.yaml"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", cfg])
    # exact pattern from export_feature_operator.py: initialize_from_dataset BEFORE load_pt, and
    # deliberately NO sort_points()/resample(), so primitive indices stay stable.
    data = DataHandler(args)
    data.reload("train", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data, device="cuda")
    model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
    cams = data.cameras
    view_ids = list(range(len(cams)))
    # ONE VIEW AT A TIME. Building all 244 views at 512 hits/pixel in a single call allocates the
    # whole operator and OOM'd at 30.9 GiB -- and A8 only needs the per-ray histogram, so each
    # view's COO is reduced to counts and discarded immediately.
    total, nnz, P = None, 0, None
    for i, cam in enumerate(cams):
        op = export_operator_for_views(model, [cam], [view_ids[i]], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        c = torch.bincount(op.row_indices, minlength=op.num_rows).cpu().numpy()
        total = c if total is None else np.concatenate([total, c])
        nnz += int(op.values.numel())
        P = int(op.num_primitives)
        del op
        if i % 40 == 0:
            torch.cuda.empty_cache()
    return summarise(total, cap), {"P": P, "views": len(cams), "nnz": nnz}


def run_gs(scene, arm, cap):
    sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
    import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import (DLL search path)
    from gsplat_baseline.export_gsplat_operator import export_view_operator
    from camera_bridge import K_from_ray_dirs
    from data_loader import DataHandler
    import configargparse
    from configs import Params, add_group

    ck = torch.load(f"recon_remote/{arm}/{scene}/ckpt.pt", map_location="cuda", weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    # ACTIVATIONS ARE REQUIRED. The checkpoint stores scales in LOG space and opacities as LOGITS;
    # passing them raw gives tiny, near-transparent Gaussians and the ray counts collapse to
    # mean 1.07 / max 2 instead of the true mean 27.73 / max 90.
    means, quats = sp["means"].cuda(), sp["quats"].cuda()
    scales = torch.exp(sp["scales"].cuda())
    opac = torch.sigmoid(sp["opacities"].cuda().reshape(-1))

    # identical rays to the foam arm: SAME views, SAME resolution, from the same loader and the
    # same downsample the foam config uses -- otherwise this compares view sets, not representations
    cfg = f"output/scannet_{scene}_nonfrozen/config.yaml"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", cfg])
    data = DataHandler(args)
    data.reload("train", downsample=args.downsample[-1])

    # CAMERA BRIDGE, verified visually and against COLMAP:
    #  K       derived from the ray grid foam actually traversed -- matches cameras.bin to 4 dp
    #          (fx=1170.188, cx=647.75) with a 4.3e-05 px pinhole residual.
    #  viewmat inverse of the loader's own c2ws (3x4, padded). c2ws[:, :3, 2] is forward, i.e. the
    #          OpenCV convention gsplat expects -- no basis reconstruction, no sign guessing.
    # Reconstructing either by hand gave a render that scored 5 dB against a BLACK ground truth;
    # with both taken from the loader it is 18.7-21.2 dB and visually correct.
    colors = torch.zeros((means.shape[0], 1), device="cuda")
    total, nnz = None, 0
    for vi, cam in enumerate(data.cameras):
        K, _ = K_from_ray_dirs(cam)
        K = K.cuda()
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :4] = data.c2ws[vi].double()
        viewmat = torch.linalg.inv(c2w).float().cuda()
        W, H = int(cam.width), int(cam.height)
        # transmittance_floor MUST match foam's transmittance_threshold (1e-3). The gsplat
        # exporter defaults to 1/255 = 3.9e-3, ~4x higher, so it stops counting a contributor
        # EARLIER and reports fewer primitives per ray for threshold reasons rather than
        # representation reasons. Measured effect of the mismatch on gs_froz/scene0000_00:
        # mean 1.07 at the 1/255 default.
        ri, ci, v, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                               viewmat, K, W, H, max_hits_per_pixel=cap,
                                               transmittance_floor=1e-3)
        c = torch.bincount(ri, minlength=H * W).cpu().numpy()
        total = c if total is None else np.concatenate([total, c])
        nnz += int(v.numel())
        del ri, ci, v
    return summarise(total, cap), {"P": int(means.shape[0]), "views": len(data.cameras), "nnz": nnz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="pf_truefrozen,pf_nonfrozen,gs_froz,gs_unfroz")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--cap", type=int, default=MAX_HITS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for scene in a.scenes.split(","):
        for arm in a.arms.split(","):
            dst = f"{OUT}/{arm}_{scene}.json"
            if os.path.exists(dst):
                print(f"[skip] {arm}/{scene}", flush=True)
                continue
            try:
                if arm.startswith("pf_"):
                    s, meta = run_foam(scene, arm[3:], a.cap)
                else:
                    s, meta = run_gs(scene, arm, a.cap)
            except Exception as e:
                print(f"[FAIL] {arm}/{scene}: {type(e).__name__}: {e}", flush=True)
                continue
            json.dump({"scene": scene, "arm": arm, "cap": a.cap, **meta, **s},
                      open(dst, "w"), indent=1)
            print(f"[ok] {arm:14s} {scene}  P={meta['P']:>9,}  rays={s['rays']:>10,}  "
                  f"mean={s['mean']:6.2f}  median={s['median']:5.0f}  p95={s['p95']:6.1f}  "
                  f"max={s['max']:4d}  at_cap={s['at_cap_frac']*100:.2f}%", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
