"""Measure beta, the cross-talk quantity Splat Feature Solver published but never interpreted.

SFS (arXiv:2508.12216) proves its back-projection estimator x' is near-optimal:

    Eq. 18   L(x') <= (1 + beta) L(x_hat)
    Eq. 13-14   Delta_ij = ||x_hat_j - B_i||          (norm, NOT a scalar residual)
                mu_i     = sum_j A_ij Delta_ij
                sigma_i^2= sum_j A_ij (Delta_ij^2 - mu_i^2)
                beta     = max_i beta_i,  beta_i = sigma_i^2 / mu_i^2
    Eq. 7    x_j = sum_i A_ij B_i / sum_i A_ij        (denominator is the SUM OF WEIGHTS)

beta_i is the weighted variance of ||x_hat_j - B_i|| over the primitives ON RAY i, normalised by
its mean. It is zero exactly when every primitive sharing a ray sits at the same distance from
that ray's observation -- i.e. when there is no disagreement to resolve. **beta IS the cross-talk
term.** SFS bound the damage it can do and never measure it or name it as such.

Two reasons to measure it here.

1. It is the field's own quantity, so a per-scene value is a citable statement about how much
   coupling there is to exploit, independent of whether our solver exploits it well.
2. It closes a loop: we measured the coupled solve to LOSE by 2.45 mIoU across 10 scenes. If beta
   is small, SFS's bound says back-projection was already near-optimal and there was little to
   win -- which would explain the negative result rather than merely restate it.

Delta_ij needs x_hat, the true minimiser of ||A f - b||^2. We have it: the unconstrained coupled
solve is exactly that (converged to relative residual 9.6e-05). So this measurement is only
possible BECAUSE of the solver infrastructure, even though that solver lost.

Note Eq. 7's denominator is sum_i A_ij, the sum of weights -- NOT sum_i A_ij^2. This confirms from
the primary source that describing back-projection as "the diagonal of the normal equations" was
wrong; it is a weighted average, hence a convex combination, hence inside the hull of observed
embeddings for free.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP

D = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--xhat", required=True,
                   help="the least-squares optimum: an unconstrained coupled solve")
    p.add_argument("--feature-folder", default=None)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--out", default=None, help="optional JSON dump of the summary")
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
    feat_dir = Path(a.feature_folder or f"artifacts/scannet/{a.scene}/openclip_features_sam")
    n_views = len(cameras) if a.max_views is None else min(a.max_views, len(cameras))

    sol = torch.load(a.xhat, map_location=device, weights_only=True)
    xhat = sol["primitive_features"].to(device).float()
    print(f"[beta] x_hat from {a.xhat}: {tuple(xhat.shape)}", flush=True)

    betas = []          # beta_i on rays that carry a real CLIP observation (B_i != 0)
    betas_bg = []       # beta_i on rays whose pixel has no mask at this SAM level (B_i == 0)
    nprim = []          # number of primitives on each kept ray
    rowsums = []        # sum_j A_ij on each kept ray -- evidence for the substochastic fix
    n_rays_total = 0
    for vi in range(n_views):
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H_, W_,
                                                   sam_level=a.sam_level)
        if float(fmap.abs().max()) == 0.0:
            continue
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

        # per-ray accumulators for the three SFS quantities. fp64: beta_i is a ratio of a
        # VARIANCE to a mean^2, and E[d^2]-E[d]^2 in fp32 cancels catastrophically whenever the
        # spread is small relative to the mean -- which is exactly the regime under test, so a
        # fp32 accumulator could manufacture the answer.
        w_sum = torch.zeros(npix, device=device, dtype=torch.float64)
        m_num = torch.zeros(npix, device=device, dtype=torch.float64)   # sum_j A_ij Delta_ij
        d2_num = torch.zeros(npix, device=device, dtype=torch.float64)  # sum_j A_ij Delta_ij^2
        cnt = torch.zeros(npix, device=device, dtype=torch.float64)
        # CHUNKED over nonzeros -- (nnz, 512) is the trap that has bitten four times here
        for s in range(0, cols.numel(), a.chunk):
            e = min(s + a.chunk, cols.numel())
            delta = (xhat[cols[s:e]] - f_pix[rows[s:e]]).norm(dim=-1).double()  # SFS Eq. 13
            v = vals[s:e].double()
            w_sum.index_add_(0, rows[s:e], v)
            m_num.index_add_(0, rows[s:e], v * delta)
            d2_num.index_add_(0, rows[s:e], v * delta * delta)
            cnt.index_add_(0, rows[s:e], torch.ones_like(v))
            del delta

        live = w_sum > 0
        # SFS normalise by the ray's weight sum: their A is row-STOCHASTIC by construction
        # (paper Property 2, "sum omega_p = 1"). Ours is row-SUBstochastic -- cells are disjoint
        # and transmittance telescopes, so the row sum is 1 - T_background <= 1. Dividing each
        # ray's accumulators by its own weight sum is exactly the row-renormalisation
        # A_ij -> A_ij / sum_j A_ij that restores their assumption; without it mu scales like s
        # and mu^2 like s^2, and beta_i would not even be well defined.
        mu = torch.zeros(npix, device=device, dtype=torch.float64)
        mu[live] = m_num[live] / w_sum[live]
        d2 = torch.zeros(npix, device=device, dtype=torch.float64)
        d2[live] = d2_num[live] / w_sum[live]
        var = (d2 - mu * mu).clamp_min(0.0)
        ok = live & (mu > 1e-12)
        b = torch.zeros(npix, device=device, dtype=torch.float64)
        b[ok] = var[ok] / (mu[ok] * mu[ok])

        # A pixel with no SAM mask at this level gets B_i = 0 from the loader (the embedding
        # table's background row), i.e. it carries NO observation. Delta_ij would collapse to
        # ||x_hat_j||, so beta_i there measures the spread of primitive feature NORMS, not
        # cross-talk. Split rather than silently mix: headline is the observed rays.
        has_obs = f_pix.norm(dim=-1) > 0
        sel = ok & has_obs
        betas.append(b[sel].float().cpu())
        nprim.append(cnt[sel].float().cpu())
        rowsums.append(w_sum[sel].float().cpu())
        betas_bg.append(b[ok & ~has_obs].float().cpu())
        n_rays_total += npix

        del out_col, out_val, fmap, f_pix, rows, cols, vals, has_obs, sel
        torch.cuda.empty_cache()
        if (vi + 1) % 20 == 0:
            print(f"  view {vi+1}/{n_views}", flush=True)

    # numpy, not torch.quantile: torch.quantile silently caps at 2^24 = 16.7M elements and
    # a full ScanNet scene has more rays than that.
    def qs(t):
        x = t.numpy().astype(np.float64)
        p = np.percentile(x, [50, 90, 99, 99.9])
        return dict(median=float(p[0]), p90=float(p[1]), p99=float(p[2]), p999=float(p[3]),
                    max=float(x.max()), mean=float(x.mean()), n=int(x.size))

    allb = torch.cat(betas)
    S = qs(allb)
    npr = torch.cat(nprim)
    rs = torch.cat(rowsums)
    multi = torch.cat(nprim) > 1
    Smulti = qs(allb[multi])
    Sbg = qs(torch.cat(betas_bg)) if sum(t.numel() for t in betas_bg) else None

    print(f"\n=== {a.scene}: SFS cross-talk beta ===")
    print(f"  rays with a real CLIP observation: {S['n']:,} of {n_rays_total:,} pixels rendered")
    print(f"  primitives per ray: median {float(npr.median()):.0f}  mean {float(npr.mean()):.1f}"
          f"   frac with >1 primitive {float(multi.float().mean())*100:.1f}%")
    print(f"  row sums sum_j A_ij (SUBstochastic, hence the renormalisation): "
          f"median {float(rs.median()):.4f}  mean {float(rs.mean()):.4f}  max {float(rs.max()):.4f}")
    print(f"\n  beta_i over observed rays:")
    print(f"    median {S['median']:.4f}   p90 {S['p90']:.4f}   p99 {S['p99']:.4f}   "
          f"p99.9 {S['p999']:.4f}   MAX {S['max']:.4f}   mean {S['mean']:.4f}")
    print(f"  beta_i restricted to rays touching >1 primitive (cross-talk is undefined on a "
          f"single-primitive ray, where beta_i == 0 by construction):")
    print(f"    median {Smulti['median']:.4f}   p90 {Smulti['p90']:.4f}   p99 {Smulti['p99']:.4f}"
          f"   p99.9 {Smulti['p999']:.4f}   MAX {Smulti['max']:.4f}   mean {Smulti['mean']:.4f}")
    if Sbg is not None:
        print(f"  [control] beta_i on the {Sbg['n']:,} rays with NO observation (B_i = 0, so "
              f"Delta_ij = ||x_hat_j||): median {Sbg['median']:.4f}  max {Sbg['max']:.4f}")
    print(f"\n  SFS bound: L(x') <= (1 + beta) L(x_hat), beta = MAX_i beta_i")
    print(f"  -> worst-case looseness of back-projection on this scene: "
          f"{S['max']*100:.1f}% above optimal")
    print(f"  -> but for the MEDIAN ray only {S['median']*100:.2f}%, so the bound is driven by "
          f"a thin tail")
    print(f"\n  beta_i = 0 exactly when every primitive on ray i sits at the same distance from")
    print(f"  that ray's observation, i.e. when there is no cross-talk to resolve.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"scene": a.scene, "xhat": a.xhat, "observed": S, "multi_primitive": Smulti,
                   "no_observation_control": Sbg, "n_pixels_rendered": n_rays_total,
                   "prims_per_ray_median": float(npr.median()),
                   "prims_per_ray_mean": float(npr.mean()),
                   "rowsum_median": float(rs.median()), "rowsum_mean": float(rs.mean())},
                  open(a.out, "w"), indent=2)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
