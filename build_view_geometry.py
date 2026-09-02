"""Per-cell VIEW GEOMETRY: how well-distributed are the directions each cell was seen from?

THE MISSING-WEDGE ANALOGY
-------------------------
In cryo-ET, a limited tilt range leaves part of Fourier space unsampled and produces systematic,
anisotropic artefacts that no amount of averaging removes -- the missing wedge. A foam cell has the
same problem: it is observed from whatever directions its rays happened to arrive from, and a cell
seen only from one side carries a view-dependent bias that averaging over those views cannot fix,
because every one of them shares the bias.

Unlike the CLIP cone (which is intrinsic to the encoder: raw 2D features already sit at cone share
0.822 before any ray is cast), this quantity is PURELY GEOMETRIC and foam-computable. It is the
natural place to look for a bias term that geometry can actually address.

WHAT IS COMPUTED
    for each view v:  w_v[j] = total render weight cell j receives  (the same quantity the lift
                               accumulates into `support`)
                      d_v[j] = normalize(camera_eye_v - center_j)   viewing direction
    R_j = || sum_v w_v[j] d_v[j] || / sum_v w_v[j]        mean resultant length, in [0,1]

R is the standard directional-statistics concentration measure. R -> 0 means the cell was seen from
well-spread directions (good coverage); R -> 1 means every ray arrived from essentially one
direction (a maximal missing wedge). Also stored: the mean direction itself, the weighted view
count, and the angular spread about the mean, so downstream work can use the full anisotropy rather
than just the scalar.

WHY THIS IS CHEAP. A full gram-cache rebuild re-accumulates S = A^T A (267M edges on the large
scenes) and is the expensive part. Nothing here needs S: only the per-view per-cell weights, which
`export_feature_operator` already returns. Feature maps are not even loaded -- this is geometry
only -- so a scene streams in minutes rather than hours.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\powerfoam")

import configargparse
import torch
import torch.nn.functional as F
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the run's config.yaml")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda"
    wp.init()

    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", a.config])
    ckpt = a.config.replace("/config.yaml", "").replace("\\config.yaml", "")

    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))

    centers = model.points.detach().float().to(device)
    P = centers.shape[0]
    acc_dir = torch.zeros(P, 3, device=device)
    acc_w = torch.zeros(P, device=device)
    n_seen = torch.zeros(P, device=device)
    print(f"[{a.scene}] P={P:,} views={n_views}", flush=True)

    t0 = time.time()
    for vi in range(n_views):
        cam = cameras[vi]
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        w_cell = torch.zeros(P, device=device).index_add_(0, cols, vals)

        eye = cam.eye.to(device).float().reshape(1, 3)
        d = F.normalize(eye - centers, dim=-1)                 # (P,3) direction to the camera
        acc_dir += w_cell[:, None] * d
        acc_w += w_cell
        n_seen += (w_cell > 0).float()
        del out_col, out_val, slots, cols, vals, w_cell, d
        if vi % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  view {vi}/{n_views} ({time.time()-t0:.0f}s)", flush=True)

    ok = acc_w > 0
    R = torch.zeros(P, device=device)
    R[ok] = acc_dir[ok].norm(dim=-1) / acc_w[ok]               # mean resultant length in [0,1]
    mean_dir = torch.zeros_like(acc_dir)
    mean_dir[ok] = F.normalize(acc_dir[ok], dim=-1)

    out = a.out or f"artifacts/scannet/{a.scene}/view_geometry.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"R": R.cpu(), "mean_dir": mean_dir.cpu(), "weight": acc_w.cpu(),
                "n_views_seen": n_seen.cpu(), "P": P, "n_views": n_views}, out)
    q = torch.quantile(R[ok].float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device))
    print(f"[{a.scene}] R (1 = single direction, 0 = well spread): "
          f"p5={q[0]:.3f} p25={q[1]:.3f} med={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f}", flush=True)
    print(f"[{a.scene}] cells with R>0.9 (severe wedge): "
          f"{float((R[ok] > 0.9).float().mean())*100:.1f}%", flush=True)
    print(f"[{a.scene}] wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
