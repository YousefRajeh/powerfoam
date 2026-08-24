"""Independent reference implementation of back-projection, for the vMF kappa=0 smoke test.

Streams the views once and computes THREE estimators from the same accumulation:

  diag        f_j = normalize( sum_r A_rj b_r )                 <- back-projection (the incumbent)
  diag_nobg   same, with background rays (b_r = 0) dropped      <- what solved_nobg_diag.pt is
  rownorm     f_j = normalize( sum_r (A_rj / sum_l A_rl) b_r )  <- what solve_vmf_assignment does at kappa=0

The third exists because the E-step in solve_vmf_assignment.py RENORMALISES over the cells of a
ray. That is z_rj = A_rj / rowsum_r, not A_rj, so the M-step is a row-rescaled back-projection.
They coincide only if rowsum_r is constant over rays. This script measures whether it is.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP

D = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--sam-level", default="3")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--compare", default=None, help="a solved .pt to compare all three against")
    p.add_argument("--chunk", type=int, default=1_000_000)
    a = p.parse_args()
    device = "cuda"

    wp.init()
    ckpt = f"output/scannet_{a.scene}_{a.variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    stems = sorted(q.stem for q in (Path(args.data_path) / args.scene / "images").iterdir())
    feat_dir = Path(f"artifacts/scannet/{a.scene}/openclip_features_sam")
    P = model.points.shape[0]

    num_d = torch.zeros(P, D, device=device)
    den_d = torch.zeros(P, device=device)
    num_b = torch.zeros(P, D, device=device)
    den_b = torch.zeros(P, device=device)
    num_r = torch.zeros(P, D, device=device)
    den_r = torch.zeros(P, device=device)
    rowsum_hist = []
    t0 = time.time()
    n_used = 0

    for vi in range(len(cameras)):
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H_, W_, sam_level=a.sam_level)
        if float(fmap.abs().max()) == 0.0:
            continue
        n_used += 1
        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=64)
        npix = H_ * W_
        slots_used = slots.reshape(-1).clamp(max=64)
        ar = torch.arange(64, device=device)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.repeat_interleave(torch.arange(npix, device=device), slots_used.long())
        f_pix = fmap.reshape(-1, D)
        del out_col, out_val, fmap
        torch.cuda.empty_cache()

        rowsum = torch.zeros(npix, device=device).index_add_(0, rows, vals)
        bg = f_pix.norm(dim=-1) <= 1e-6
        if n_used <= 5:
            live_rows = ~bg
            rs = rowsum[live_rows]
            rowsum_hist.append(torch.quantile(
                rs.float(), torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=device)).cpu())

        z = vals / rowsum[rows].clamp_min(1e-30)
        for s in range(0, cols.numel(), a.chunk):
            e = min(s + a.chunk, cols.numel())
            fp = f_pix[rows[s:e]]
            num_d.index_add_(0, cols[s:e], vals[s:e, None] * fp)
            num_r.index_add_(0, cols[s:e], z[s:e, None] * fp)
            nb = ~bg[rows[s:e]]
            num_b.index_add_(0, cols[s:e][nb], (vals[s:e, None] * fp)[nb])
            del fp
        den_d.index_add_(0, cols, vals)
        den_r.index_add_(0, cols, z)
        nbg = ~bg[rows]
        den_b.index_add_(0, cols[nbg], vals[nbg])
        del cols, vals, rows, f_pix, z, rowsum, bg, nbg
        torch.cuda.empty_cache()

    print(f"[ref] {n_used} views used, {time.time()-t0:.0f}s", flush=True)
    print("[ref] per-ray rowsum (sum_j A_rj) quantiles 1/25/50/75/99 on the first 5 views:")
    for q in rowsum_hist:
        print("   ", "  ".join(f"{float(x):.4f}" for x in q))

    outs = {}
    for nm, num, den in (("diag", num_d, den_d), ("diag_nobg", num_b, den_b),
                         ("rownorm", num_r, den_r)):
        v = den > 0
        f = torch.zeros(P, D, device=device)
        f[v] = F.normalize(num[v], dim=-1)
        outs[nm] = (f, v)
        torch.save({"primitive_features": f.cpu(), "valid_mask": v.cpu()},
                   f"{a.out_prefix}_{nm}.pt")
        print(f"[ref] {nm}: {int(v.sum()):,}/{P:,} valid -> {a.out_prefix}_{nm}.pt", flush=True)

    def cmp(n1, f1, v1, n2, f2, v2):
        v = v1 & v2
        idx = torch.where(v)[0]
        cs = []
        for s in range(0, idx.numel(), 200_000):
            ii = idx[s:s + 200_000]
            cs.append(F.cosine_similarity(f1[ii], f2[ii], dim=-1))
        c = torch.cat(cs)
        print(f"[cmp] cos({n1}, {n2}): median {float(c.median()):.6f}  mean {float(c.mean()):.6f}  "
              f"min {float(c.min()):.4f}  frac<0.999 {float((c < 0.999).float().mean())*100:.2f}%  "
              f"frac<0.99 {float((c < 0.99).float().mean())*100:.2f}%  "
              f"(over {idx.numel():,} cells; valid1 {int(v1.sum()):,} valid2 {int(v2.sum()):,})",
              flush=True)

    cmp("diag", *outs["diag"], "diag_nobg", *outs["diag_nobg"])
    cmp("diag", outs["diag"][0], outs["diag"][1], "rownorm", outs["rownorm"][0], outs["rownorm"][1])

    if a.compare:
        t = torch.load(a.compare, map_location=device, weights_only=True)
        fc = F.normalize(t["primitive_features"].to(device).float(), dim=-1)
        vc = t["valid_mask"].to(device)
        for nm in ("diag", "diag_nobg", "rownorm"):
            cmp(a.compare.split("/")[-1], fc, vc, nm, outs[nm][0], outs[nm][1])


if __name__ == "__main__":
    main()
