"""Lift RAW (un-normalised) CLIP features and compare against lifting normalised ones.

THE QUESTION. The forward model A x = B is alpha-compositing: linear superposition. It is the right
model exactly when the observable composes and its magnitude is meaningful -- which is why the RGB
round-trip verified Theorem 1(ii) to 1.55e-06. But the feature loader normalises every pixel
(`accumulate_feature_stats_sam.load_image_feature_from_SAMOpenCLIP`, `skip_normalize=False` by
default), so ||B_i|| = 1 while ||(A x)_i|| <= 1: A X = B becomes structurally unsatisfiable, and
least squares can only chase the gap by inflating ||x_j|| (measured: iterates overshoot the sphere,
1.07x at k=10).

So every negative solver result this session -- Richardson, projected Richardson, surface-block --
was obtained on an operator whose observations had been made inconsistent with the forward model
BEFORE the solve. This asks whether the solver line was dead or merely mis-fed.

WHAT IT DOES. One pass over the views, accumulating BOTH numerators from the SAME loaded feature
map so the comparison is exactly matched (same views, same rays, same weights):

    Atb_raw  = sum_i A_ij * f_pix_raw[i]           (magnitude preserved -- composes linearly)
    Atb_norm = sum_i A_ij * normalize(f_pix)[i]    (what the pipeline does today)

Both are then divided by the support and written as solved features. S = A^T A is untouched by
either -- it depends only on geometry -- so no gram rebuild is needed.

WHAT WOULD COUNT. If raw lifting scores higher, the forward model was being violated and the
solver line deserves a second look on a consistent operator. If it does not, the normalisation was
never the problem and the negatives stand as measured.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import configargparse
import torch
import torch.nn.functional as F
import warp as wp

from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

CH = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="nonfrozen")
    # The FOLDER name and the LEVEL INDEX inside each file are independent, and conflating them
    # cost two failed launches: `openclip_features_sam_l3` is a single-level extraction of SAM
    # level 3 that stores it at index 0, so the correct pair is folder=..._l3, sam-level=0.
    ap.add_argument("--feature-folder", default="openclip_features_sam_l3")
    ap.add_argument("--sam-level", default="0", help="index WITHIN the file, not the folder name")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--tag", default="", help="suffix for the output names, e.g. _alllevels")
    a = ap.parse_args()

    dev = "cuda"
    wp.init()
    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt}/model.pt")

    cameras = dh.cameras
    stems = sorted(q.stem for q in (Path(args.data_path) / args.scene / "images").iterdir())
    assert len(stems) == len(cameras), (len(stems), len(cameras))
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    feat_dir = Path(args.data_path) / args.scene / a.feature_folder
    assert feat_dir.is_dir(), (
        f"{feat_dir} not found. --feature-folder names the DIRECTORY; --sam-level is the index "
        f"inside each _s.npy. Available: "
        f"{[q.name for q in (Path(args.data_path) / args.scene).iterdir() if q.is_dir() and 'openclip' in q.name]}")

    P = model.points.shape[0]
    D = 512
    Atb_raw = torch.zeros(P, D, device=dev)
    Atb_norm = torch.zeros(P, D, device=dev)
    support = torch.zeros(P, device=dev)
    used = 0
    t0 = time.time()
    ar = torch.arange(a.max_hits, device=dev)
    norm_stats = []

    for vi in range(n_views):
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        # skip_normalize=True gives the RAW summed CLIP feature; normalising it here reproduces
        # exactly what the default loader returns, so both arms come from one load.
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H_, W_,
                                                   sam_level=a.sam_level, skip_normalize=True)
        if float(fmap.abs().max()) == 0.0:
            continue
        f_raw = fmap.reshape(-1, D)
        nrm = f_raw.norm(dim=-1, keepdim=True)
        f_nrm = f_raw / (nrm + 1e-6)
        live = nrm.squeeze(-1) > 1e-6
        if live.any():
            norm_stats.append(float(nrm.squeeze(-1)[live].median()))

        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=a.max_hits)
        slots_used = slots.reshape(-1).clamp(max=a.max_hits)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.arange(slots_used.numel(), device=dev).repeat_interleave(slots_used)

        support.index_add_(0, cols, vals)
        for s in range(0, cols.numel(), CH):
            e = min(s + CH, cols.numel())
            Atb_raw.index_add_(0, cols[s:e], vals[s:e, None] * f_raw[rows[s:e]])
            Atb_norm.index_add_(0, cols[s:e], vals[s:e, None] * f_nrm[rows[s:e]])
        used += 1
        del fmap, f_raw, f_nrm, nrm, out_col, out_val, cols, vals, rows, keep, slots_used
        torch.cuda.empty_cache()
        if used % 10 == 0:
            print(f"  {used}/{n_views} views ({time.time()-t0:.0f}s)", flush=True)

    med = sorted(norm_stats)[len(norm_stats) // 2] if norm_stats else float("nan")
    print(f"\n{a.scene}: {used} views, P={P:,}, median raw pixel-feature norm {med:.4f}")

    valid = (support > 0).cpu()
    d = support.clamp_min(1e-30).unsqueeze(1)
    for tag, num in (("rawlift", Atb_raw), ("normlift", Atb_norm)):
        x = num / d
        out = f"artifacts/scannet/{a.scene}/solved_{tag}{a.tag}.pt"
        torch.save({"primitive_features": x.cpu(), "valid_mask": valid}, out)
        print(f"  wrote {os.path.basename(out)}  ||x|| median {float(x[valid.to(dev)].norm(dim=1).median()):.4f}")

    # how different are they in DIRECTION? scale alone is invisible to the evaluation.
    xa = F.normalize((Atb_raw / d)[valid.to(dev)], dim=-1)
    xb = F.normalize((Atb_norm / d)[valid.to(dev)], dim=-1)
    cos = (xa * xb).sum(-1)
    print(f"  cos(raw, norm): median {float(cos.median()):.6f}  mean {float(cos.mean()):.6f}  "
          f"frac<0.99 {float((cos < 0.99).float().mean()):.4f}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
