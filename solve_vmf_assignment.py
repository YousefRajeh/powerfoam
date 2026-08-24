"""Training-free EM for the ASSIGNMENT problem, on the sphere.

WHY ASSIGNMENT AND NOT INVERSION

The coupled solve was falsified (-2.45 mIoU over 10 scenes x 3 seeds) and the diagnosis is that
its generative model is false. `b_r = sum_j A_rj f_j` is exactly right for RGB, where a pixel
colour genuinely IS the alpha-composite of what the ray passes through. It is wrong for CLIP
features: b_r is the embedding of a SAM MASK, a whole-object descriptor stamped identically onto
every pixel of that mask. It was never formed by compositing. Solving A f = b exactly therefore
forces the field to reproduce a process that never happened, and the solver satisfies it by
pushing compensating differences onto whatever cells share the ray -- including occluded cells
behind the surface, whose recovered values are meaningless. The better the fit, the more of them
it manufactures.

The honest model is an ASSIGNMENT:

    b_r  =  f_{j*(r)}  +  noise,      j*(r) = the cell that actually owns pixel r

with j*(r) LATENT. Under this model the maximum-likelihood estimate of f_j is a robust average of
the observations assigned to it -- which is why plain back-projection is hard to beat, and why our
geometric-median solver beat weighted least squares on room_0 (0.6095 vs 0.4649), and why VALA's
cosine median is the strongest constraint in the literature. All three are robust central
tendencies, i.e. the right family. What none of them do is question the ASSIGNMENT.

WHY THE POWER DIAGRAM MAKES THIS WELL-POSED

The ~12 cells a ray crosses are DISJOINT, so exactly one owns the pixel and the responsibilities
along a ray live on a simplex, sum_j z_rj = 1. That is an assignment, not a mixture, and it is
only writable because the cells partition space. A 3DGS method has 50+ OVERLAPPING primitives per
ray and no notion of "the" owner, so this formulation has no analogue there.

THE ITERATION (no learned parameters, no gradients, no training)

    E:   z_rj  proportional to  A_rj * exp( kappa * <b_r, f_j> ),   renormalised over j on ray r
    M:   f_j   =  normalize( sum_r z_rj b_r )

The likelihood is von Mises-Fisher, the natural density on the unit sphere where CLIP embeddings
live, so the M-step is a robust spherical mean rather than a least-squares projection.

THREE PROPERTIES THAT MAKE THIS THE RIGHT BET

1. It STRICTLY GENERALISES the incumbent. At kappa = 0 the E-step returns z proportional to A and
   the M-step is exactly back-projection with weight normalisation. The rule everyone uses is the
   kappa=0 member of this family, so kappa=0 is always available as a fallback and the method can
   only be judged against a baseline it contains.
2. It attacks the MEASURED gap. Mean aggregation reaches 55.08% per-cell accuracy; an oracle that
   picks the best OBSERVED view per cell reaches 78.59%. That 23.5-point gap is selection, not
   inversion, and it is reachable only by a responsibility that DISAGREES with the render weight
   -- which we know is a bad proxy for trustworthiness, since argmax_w was the worst rule tested
   (33.01 vs 39.10 mIoU).
3. It stays in the CLIP hull by construction: z >= 0 and the M-step is a normalised non-negative
   combination of observed embeddings, so every iterate is a scaled convex combination. This is
   exactly the property the coupled solve destroyed (35.67% of cells with ||f|| > 1, max 7.15).

THE FAILURE MODE TO DESIGN AGAINST, from our own results

Sharpening assignments has failed twice here, both times on COVERAGE rather than on the idea:
`top1` hard assignment (-6.09 mIoU) and surface truncation (-2.74, coverage 90.2% -> 70.1%). A
cell that loses responsibility on every ray ends up with no feature at all. So kappa is swept from
0 upward, coverage is reported at every step, and any cell whose total responsibility collapses
reverts to its kappa=0 answer -- the same fallback that turned -0.98 into +0.04 for the cone solve.

WHAT TO MEASURE FIRST. Per-cell accuracy against the 78.59% oracle, NOT mIoU. Voting improved
per-cell accuracy by +2.6 on all four scenes and moved mIoU by zero, because mIoU here is limited
by class balance. Screening on mIoU would discard a real improvement in the thing this targets.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from accumulate_feature_stats_sam import load_image_feature_from_SAMOpenCLIP

D = 512


def stream_views(model, cameras, stems, feat_dir, sam_level, n_views, device):
    """Yield one view's (cols, vals, rows, f_pix, npix). One render + one feature load per view."""
    for vi in range(n_views):
        cam = cameras[vi]
        H_, W_ = int(cam.height), int(cam.width)
        if not (feat_dir / f"{stems[vi]}_f.npy").exists():
            continue
        fmap = load_image_feature_from_SAMOpenCLIP(feat_dir, stems[vi], H_, W_, sam_level=sam_level)
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
        yield cols, vals, rows, fmap.reshape(-1, D), npix
        del out_col, out_val, fmap
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--feature-folder", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--sam-level", default="3")
    p.add_argument("--kappa", type=float, default=10.0,
                   help="vMF concentration; 0 recovers back-projection EXACTLY")
    p.add_argument("--em-iters", type=int, default=5)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--save-kappa0", default=None,
                   help="also write the kappa=0 answer (= back-projection) from the same pass")
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
    P = model.points.shape[0]

    def sweep(kappa, f_init, iters):
        """EM at a given kappa. Returns (features, total responsibility per cell)."""
        f = f_init.clone()
        resp = torch.zeros(P, device=device)
        for it in range(iters):
            num = torch.zeros(P, D, device=device)
            resp = torch.zeros(P, device=device)
            t0 = time.time()
            for cols, vals, rows, f_pix, npix in stream_views(
                    model, cameras, stems, feat_dir, a.sam_level, n_views, device):
                nnz = cols.numel()
                sc = torch.empty(nnz, device=device)
                # CHUNKED: (nnz, 512) is the gather trap that has caused four OOMs here
                for s in range(0, nnz, a.chunk):
                    e = min(s + a.chunk, nnz)
                    if kappa == 0.0:
                        sc[s:e] = vals[s:e]
                    else:
                        cos = (f[cols[s:e]] * f_pix[rows[s:e]]).sum(-1)
                        sc[s:e] = vals[s:e] * torch.exp(kappa * cos)
                # renormalise over the cells of each ray -> z_r lives on the simplex
                denom = torch.zeros(npix, device=device).index_add_(0, rows, sc)
                z = sc / denom[rows].clamp_min(1e-30)
                for s in range(0, nnz, a.chunk):
                    e = min(s + a.chunk, nnz)
                    num.index_add_(0, cols[s:e], z[s:e, None] * f_pix[rows[s:e]])
                resp.index_add_(0, cols, z)
                del sc, z, denom, cols, vals, rows, f_pix
                torch.cuda.empty_cache()
            live = resp > 0
            f_new = f.clone()
            f_new[live] = F.normalize(num[live], dim=-1)
            drift = float((1.0 - (f_new[live] * f[live]).sum(-1)).abs().mean())
            f = f_new
            print(f"  [em] kappa={kappa:g} iter {it}: mean 1-cos drift {drift:.5f}  "
                  f"live {int(live.sum()):,}/{P:,}  ({time.time()-t0:.0f}s)", flush=True)
            del num
        return f, resp

    # kappa = 0 IS back-projection; one pass gives the incumbent and the initialisation together
    print(f"[init] kappa=0 pass (this reproduces back-projection exactly)", flush=True)
    f0, resp0 = sweep(0.0, torch.zeros(P, D, device=device), 1)
    valid0 = resp0 > 0
    if a.save_kappa0:
        torch.save({"primitive_features": f0.cpu(), "valid_mask": valid0.cpu()}, a.save_kappa0)
        print(f"[init] kappa=0 answer -> {a.save_kappa0}", flush=True)

    if a.kappa == 0.0:
        f, resp = f0, resp0
    else:
        f, resp = sweep(a.kappa, f0, a.em_iters)

    # COVERAGE FALLBACK: a cell whose responsibility collapsed keeps its kappa=0 answer. Learned
    # the hard way twice -- top1 (-6.09) and surface truncation (-2.74) both died on coverage.
    dead = valid0 & (resp <= 1e-12)
    if int(dead.sum()):
        f[dead] = f0[dead]
        print(f"[fallback] {int(dead.sum()):,} cells with collapsed responsibility reverted to "
              f"kappa=0 ({float(dead.sum())/max(int(valid0.sum()),1)*100:.2f}% of valid)", flush=True)

    cos = F.cosine_similarity(f[valid0], f0[valid0], dim=-1)
    print(f"[compare] cosine(kappa={a.kappa:g}, kappa=0): median {float(cos.median()):.4f}  "
          f"frac<0.99 {float((cos<0.99).float().mean())*100:.2f}%", flush=True)
    torch.save({"primitive_features": f.cpu(), "valid_mask": valid0.cpu()}, a.output)
    print(f"[solve_vmf_assignment] {int(valid0.sum())}/{P} valid -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
