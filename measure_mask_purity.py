"""Per-primitive MASK-purity, and the angular bound under the correct observation model.

WHY THIS REPLACES THE BLENDING MODEL. `foamyfoam/scripts/test_angular_bound.py` derives
tan(theta_j) <= 2*Omega*rho_j / (1 - 2*Omega*rho_j) assuming B_i = normalize(sum_k A_ik f_k), i.e.
a ray's feature is an alpha-weighted blend of the surfaces it crosses. This pipeline does no such
thing: `accumulate_feature_stats_sam.load_image_feature_from_SAMOpenCLIP` does
`F.embedding(segment, features_pad)`, so every pixel gets the CLIP embedding of the SAM MASK it
belongs to -- piecewise-constant over segments, unit-norm, no blending.

THE CORRECTED DERIVATION. With B_i = e_{m(i)}, group the estimator's rays by mask:

    x_j = normalize( sum_i A_ij e_{m(i)} ) = normalize( sum_m W_jm e_m ),
    W_jm = sum_{i in mask m} A_ij ,   p_jm = W_jm / D_jj ,   p*_j = max_m p_jm

Taking f_j = e_{m*(j)} (the dominant mask's embedding) and Omega = max ||e_m - e_m'|| over masks
co-assigned to the same primitive:

    tan(theta_j)  <=  (1 - p*_j) Omega / (1 - (1 - p*_j) Omega)

The factor 2 of the blending version disappears -- no projection-to-sphere step is needed, because
the e_m are already unit vectors. p*_j = 1 (every ray of j lands in one mask) gives exact recovery:
the mask-level analogue of Cor. 3.

So the quantity controlling lift error is NOT rho_j but **1 - p*_j, the share of a primitive's ray
weight arriving from the WRONG mask**.

CAVEAT ON THE VALIDATION. f_j := e_{m*(j)} makes the measured theta_j a SELF-CONSISTENCY figure --
how far the lift drifts from its own dominant mask -- not a correctness figure. If the dominant
mask itself straddles two objects, theta_j is small while the feature is wrong. That is the
separate mask-quality question and is not measured here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import configargparse
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variant", default="nonfrozen")
    ap.add_argument("--feature-folder", default="openclip_features_sam_l3")
    ap.add_argument("--sam-level", type=int, default=0)
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--out", default=None)
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
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))
    feat_dir = Path(args.data_path) / args.scene / a.feature_folder
    P = model.points.shape[0]

    # (primitive, global mask id) weight accumulator, built sparse then coalesced per view
    keys_all, vals_all = [], []
    emb_all = []
    mask_base = 0
    support = torch.zeros(P, device=dev)
    ar = torch.arange(a.max_hits, device=dev)
    t0 = time.time()
    used = 0

    for vi in range(n_views):
        fp = feat_dir / f"{stems[vi]}_f.npy"
        sp = feat_dir / f"{stems[vi]}_s.npy"
        if not (fp.exists() and sp.exists()):
            continue
        emb = torch.from_numpy(np.load(fp)).float().to(dev)          # (n_masks, 512)
        seg = torch.from_numpy(np.load(sp)).long().to(dev)           # (levels, H, W)
        seg = seg[a.sam_level]                                       # (H, W); -1 = background
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        if seg.shape != (H_, W_):
            seg = F.interpolate(seg[None, None].float(), size=(H_, W_), mode="nearest")[0, 0].long()

        out_col, out_val, slots, _, _ = model.export_feature_operator(
            cam, max_intersections=1024, max_hits_per_pixel=a.max_hits)
        slots_used = slots.reshape(-1).clamp(max=a.max_hits)
        keep = (ar[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep]
        rows = torch.arange(slots_used.numel(), device=dev).repeat_interleave(slots_used)

        support.index_add_(0, cols, vals)
        pix_mask = seg.reshape(-1)                                   # mask id per pixel
        mid = pix_mask[rows]
        live = mid >= 0                                              # drop background pixels
        if live.any():
            gm = mid[live] + mask_base
            # (primitive, global mask) as a pair of int64 columns rather than a packed key, so no
            # assumption is made about mask counts fitting a fixed radix. Coalesced PER VIEW:
            # holding every view's raw triples to the end is what pushed this over 40 GiB.
            part = torch.sparse_coo_tensor(
                torch.stack([cols[live], gm]), vals[live],
                (P, mask_base + emb.shape[0]), device=dev).coalesce()
            keys_all.append(part.indices().cpu())
            vals_all.append(part.values().cpu())
            del part
        emb_all.append(emb)
        mask_base += emb.shape[0]
        used += 1
        del emb, seg, out_col, out_val, cols, vals, rows, keep, slots_used, pix_mask, mid, live
        torch.cuda.empty_cache()
        if used % 20 == 0:
            print(f"  {used}/{n_views} views ({time.time()-t0:.0f}s)", flush=True)

    idx = torch.cat(keys_all, dim=1).to(dev)
    w = torch.cat(vals_all).to(dev)
    E = torch.cat(emb_all, 0)
    E = F.normalize(E, dim=-1)
    M = E.shape[0]
    print(f"\n{a.scene}: {used} views, P={P:,}, {M:,} masks, {w.numel():,} (primitive,mask) entries")

    Wsp = torch.sparse_coo_tensor(idx, w, (P, M), device=dev).coalesce()
    D = torch.sparse.sum(Wsp, dim=1).to_dense()
    live = D > 0

    # p*_j: dominant-mask share.  scatter-max over the coalesced entries
    si, sv = Wsp.indices(), Wsp.values()
    pmax = torch.zeros(P, device=dev).scatter_reduce_(0, si[0], sv, reduce="amax", include_self=True)
    pstar = torch.where(live, pmax / D.clamp_min(1e-30), torch.zeros_like(D))

    # the lift under this model, and its dominant-mask reference
    # Chunked over feature columns: the dense product is P x 512 and the intermediate blows past
    # the card when anything else is resident. Same pattern as covis_graph.spmm.
    Xnum = torch.cat([torch.sparse.mm(Wsp, E[:, c:c + 64]) for c in range(0, E.shape[1], 64)], 1)
    x = F.normalize(Xnum, dim=-1)
    # Deterministic argmax. `scatter_` with DUPLICATE indices is explicitly nondeterministic in
    # PyTorch -- "last write wins" is not guaranteed -- so the previous version often picked a mask
    # that was NOT the dominant one. That corrupts f_j, hence both the measured angle and Omega_j,
    # and produced 17.5k spurious bound "violations". Select entries whose value equals the
    # per-primitive max, then take the lowest mask id among them for reproducibility.
    is_max = sv >= pmax[si[0]] - 1e-12
    BIG = int(M) + 1
    cand = torch.full((P,), BIG, dtype=torch.long, device=dev)
    cand.scatter_reduce_(0, si[0][is_max], si[1][is_max], reduce="amin", include_self=True)
    argmax_m = torch.where(cand < BIG, cand, torch.zeros_like(cand))
    f = E[argmax_m]
    # assert the selection really is the dominant mask
    chk = torch.zeros(P, device=dev)
    hit = si[1] == argmax_m[si[0]]
    chk.scatter_reduce_(0, si[0][hit], sv[hit], reduce="amax", include_self=True)
    bad = int(((chk - pmax).abs() > 1e-9)[live].sum())
    assert bad == 0, f"argmax selection wrong on {bad} primitives"

    cos = (x[live] * f[live]).sum(-1).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    tan_meas = torch.tan(theta.clamp(max=np.pi / 2 - 1e-9))

    # Omega_j: PER-PRIMITIVE max ||e_m - f_j|| over that primitive's own masks. The derivation
    # is per-primitive; an earlier version used a single global max estimated by SAMPLING 2000
    # primitives, which is simultaneously loose for most primitives and an under-estimate of a
    # maximum -- it reported thousands of "violations" that were an artefact of the estimate.
    # This is exact and O(nnz): one scatter-max over the (primitive, mask) entries.
    cnt = torch.bincount(si[0], minlength=P)          # masks per primitive
    dist = (E[si[1]] - f[si[0]]).norm(dim=-1)
    om_j = torch.zeros(P, device=dev).scatter_reduce_(0, si[0], dist, reduce="amax",
                                                      include_self=True)
    om_global = float(dist.max())
    z = (1 - pstar[live]) * om_j[live]
    bound = torch.where(z < 1, z / (1 - z).clamp_min(1e-12), torch.full_like(z, float("inf")))
    fin = torch.isfinite(bound)
    # float32 tolerance. Features and weights are fp32 and the stored embeddings fp16, so the
    # achievable floor on tan is ~1e-3; measured worst excess with everything correct was 5.98e-04.
    # At 1e-6 the check reports ~1% "violations" that are pure precision.
    TOL = 1e-3
    viol = int((tan_meas[fin] > bound[fin] + TOL).sum())
    exc = (tan_meas[fin] - bound[fin])
    exc = exc[exc > 0]
    if exc.numel():
        print(f"  violation excess (tan units): n={exc.numel():,}  median {float(exc.median()):.2e}  "
              f"p95 {float(exc.quantile(0.95)):.2e}  max {float(exc.max()):.2e}", flush=True)
    om = float(om_j[live].mean())

    # PER-PRIMITIVE DUMP. The summary quantiles below characterise the bound; testing whether
    # p*_j is USABLE (does it predict correctness? does gating on it beat gating on agreement?)
    # needs the per-primitive vector, so write it alongside.
    ps_out = f"artifacts/scannet/{a.scene}/pstar_{a.variant}.npz"
    np.savez(ps_out,
             pstar=pstar.detach().cpu().numpy(),
             live=live.detach().cpu().numpy(),
             n_masks=cnt.detach().cpu().numpy(),
             omega_j=om_j.detach().cpu().numpy(),
             D=D.detach().cpu().numpy())
    print(f"  wrote {os.path.basename(ps_out)} (per-primitive p*, live mask, mask count)",
          flush=True)

    q = lambda t, p: float(torch.quantile(t, p))
    row = {
        "scene": a.scene, "variant": a.variant, "views": used, "P": P, "masks": M,
        "pstar_mean": float(pstar[live].mean()), "pstar_p05": q(pstar[live], 0.05),
        "pstar_p50": q(pstar[live], 0.50), "pstar_p95": q(pstar[live], 0.95),
        "frac_pstar_gt_0.9": float((pstar[live] > 0.9).float().mean()),
        "masks_per_primitive_mean": float(cnt[live].float().mean()),
        "omega_mean_per_primitive": om, "omega_global_max": om_global, "bound_live_frac": float(fin.float().mean()),
        "theta_deg_p50": float(np.degrees(float(theta.median()))),
        "theta_deg_p95": float(np.degrees(q(theta, 0.95))),
        "bound_violations": viol,
    }
    print(f"  p*_j (dominant-mask share): mean {row['pstar_mean']:.4f}  "
          f"p05 {row['pstar_p05']:.4f}  p50 {row['pstar_p50']:.4f}  p95 {row['pstar_p95']:.4f}")
    print(f"  frac p* > 0.9: {row['frac_pstar_gt_0.9']:.4f}   masks/primitive {row['masks_per_primitive_mean']:.2f}")
    print(f"  Omega_j mean {om:.4f} (global max {om_global:.4f})   bound live on {row['bound_live_frac']:.2%} of primitives   "
          f"violations {viol}")
    print(f"  measured angle to dominant mask: p50 {row['theta_deg_p50']:.2f} deg  "
          f"p95 {row['theta_deg_p95']:.2f} deg")
    out = a.out or f"artifacts/scannet/{a.scene}/mask_purity_{a.variant}.json"
    json.dump(row, open(out, "w"), indent=1)
    print(f"  wrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
