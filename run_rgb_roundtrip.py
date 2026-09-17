"""I2 from BETA_BOUND.md: the RGB round-trip -- a label-free probe of one-shot lift quality.

THE IDEA. Property 3 of the SFS paper is the assumption that the observations lie in the range of
the rendering operator, `B in range(A)`. Test it on a signal whose ground truth is known and needs
no annotation at all: **lift the image colours through the same solver, then re-render them.**

    x' = D^-1 A^T C          (the SFS closed form, Eq. 6, applied to RGB)
    x_hat = argmin ||A x - C||_F^2                       (what the lift should have returned)

    PSNR(A x', C)  vs  PSNR(A x_hat, C)

The gap between those two IS the excess `L(x') - L(x_hat)` of Theorem 2(ii), read out in dB, with no
segmentation labels, no CLIP, and no downstream metric standing between the theory and the number.

WHY `A x_hat` IS THE REFERENCE AND NOT THE MODEL'S OWN RENDER. The excess we are measuring is a
property of the SOLVER, so the reference has to be the best any per-primitive field can do under the
SAME operator -- that is `A x_hat` exactly. The model's trained render would additionally fold in
spherical harmonics, view dependence and the training objective, none of which the solver is
responsible for, so a gap against it would not isolate the quantity Theorem 2 is about.

WHAT ELSE COMES OUT OF THE SAME PASS, for free, because A is already built:

  * `rho_j = 1 - (sum_i A_ij^2)/(sum_i A_ij)` (I3) for the **3DGS** arms as well as the foam. The
    reduced accumulators we store only exist for the foam, so this is the only route to a
    foam-vs-3DGS rho comparison, and it comes from `colsum` and `diag` which the solve needs anyway.
  * `gamma = ||A(x' - x_hat)||^2 / L(x_hat)`, the EXACT suboptimality of the closed form (not a
    bound). Since x_hat solves the normal equations the cross term vanishes and Pythagoras is exact.
  * Theorem 2(ii) as an IDENTITY CHECK: `L(x') - L(x_hat)` must equal `||A D^-1 L_G x_hat||^2`
    computed independently. If those two disagree beyond float noise, the theorem is not describing
    this operator and nothing else in the output is trustworthy.
  * I1's Richardson iteration `x_{t+1} = x_t + D^-1 (A^T C - A^T A x_t)`, x_0 = 0, whose k = 1 step
    IS the closed form. PSNR against k is the label-free version of I1's artifact and costs k
    products with an operator that is already resident.

SCOPE, STATED NOT HIDDEN. Caching A for every view of a ScanNet scene is not affordable (the 3DGS
arm reaches ~5e8 nonzeros for 12 views), so this evaluates a well-defined subproblem: a fixed,
evenly spaced subset of `--views` views at native resolution, exactly as compute_beta.py does. On
that subproblem x_hat is solved to the reported CG residual and every number is exact. RGB is 3
channels rather than 512, so the same view budget is ~170x cheaper here than for features.

Row-stochasticity is checked and reported, not assumed: `sum_j A_ij = 1` is what makes rho a
fraction and D the row-sum lumping of G.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import gsplat_env_gsview  # noqa: F401  MUST precede any gsplat import

import configargparse
import numpy as np
import torch

from camera_bridge import K_from_ray_dirs
from compute_beta import cg_normal_equations
from configs import Params, add_group
from data_loader import DataHandler

OUT = "artifacts/scannet/roundtrip"
KS = (1, 2, 3, 5, 10, 20)


def psnr(pred, target):
    mse = float(((pred - target) ** 2).mean())
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--arm", default="pf_truefrozen",
                    choices=("pf_truefrozen", "pf_nonfrozen", "gs_froz", "gs_unfroz"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cg-iters", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    recon = "nonfrozen" if a.arm.startswith("gs_") else a.arm[3:]
    cfg = f"output/scannet_{a.scene}_{recon}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()

    if a.arm.startswith("pf_"):
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{a.scene}_{recon}/model.pt")
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{a.arm}/{a.scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].cuda(), sp["quats"].cuda()
        gs_ = torch.exp(sp["scales"].cuda())
        go = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)

    rows, cols, vals, rgbs, offs = [], [], [], [], 0
    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if a.arm.startswith("pf_"):
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            P = op.num_primitives
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[vi].double()
            vmat = torch.linalg.inv(c2w).float().cuda()
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vmat, K.cuda(), W, H,
                                                    max_hits_per_pixel=a.cap,
                                                    transmittance_floor=1e-3)
            P = gm.shape[0]
        rows.append(ri.to(torch.int64) + offs)
        cols.append(ci.to(torch.int64))
        vals.append(vv.float())
        rgbs.append(dh.rgbs[vi].reshape(-1, 3).to(dev).float())
        offs += H * W
        del ri, ci, vv

    row = torch.cat(rows); col = torch.cat(cols); val = torch.cat(vals)
    C = torch.cat(rgbs)                                   # (R, 3) observed colours
    del rows, cols, vals, rgbs
    R, D = offs, 3
    nnz = val.numel()

    rowsum = torch.zeros(R, device=dev).index_add_(0, row, val)
    live = rowsum > 0
    q = torch.quantile(rowsum[live].double(),
                       torch.tensor([.01, .5, .99], device=dev, dtype=torch.float64))
    print(f"[{a.arm}/{a.scene}] views={a.views} rays={R:,} nnz={nnz:,} P={P:,}", flush=True)
    print(f"  row sums 1/50/99%: {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}", flush=True)

    if not bool((row[1:] >= row[:-1]).all()):
        order = torch.argsort(row)
        row, col, val = row[order].contiguous(), col[order].contiguous(), val[order].contiguous()
        del order
    starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
    NNZ_BUDGET = max(1, int(3e8 // max(D, 1)))
    tgts = torch.arange(0, nnz + NNZ_BUDGET, NNZ_BUDGET, device=dev)
    bnd = torch.unique(torch.cat([torch.searchsorted(starts.contiguous(), tgts).clamp(0, R),
                                  torch.tensor([R], device=dev)]))
    blocks = [(int(x), int(y)) for x, y in zip(bnd[:-1], bnd[1:]) if int(y) > int(x)]

    def Ax(x):
        """(R, D) render of a per-primitive field."""
        o = torch.zeros((R, x.shape[1]), device=dev)
        for s in range(0, nnz, 40_000_000):
            e = min(s + 40_000_000, nnz)
            o.index_add_(0, row[s:e], val[s:e, None] * x[col[s:e]])
        return o

    def AtA(x):
        o = torch.zeros((P, x.shape[1]), device=dev)
        for r0, r1 in blocks:
            s, e = int(starts[r0]), int(starts[r1])
            if e <= s:
                continue
            lr = row[s:e] - r0
            ap = torch.zeros((r1 - r0, x.shape[1]), device=dev)
            ap.index_add_(0, lr, val[s:e, None] * x[col[s:e]])
            o.index_add_(0, col[s:e], val[s:e, None] * ap[lr])
            del ap
        return o

    rhs = torch.zeros((P, D), device=dev)                 # A^T C
    for s in range(0, nnz, 40_000_000):
        e = min(s + 40_000_000, nnz)
        rhs.index_add_(0, col[s:e], val[s:e, None] * C[row[s:e]])
    colsum = torch.zeros(P, device=dev).index_add_(0, col, val)          # D_jj = sum_i A_ij
    diag = torch.zeros(P, device=dev).index_add_(0, col, val * val)      # diag(G) = sum_i A_ij^2

    seen = colsum > 0
    rho = 1.0 - diag[seen] / colsum[seen]
    rq = torch.quantile(rho.double(), torch.tensor([.05, .5, .95], device=dev, dtype=torch.float64))
    print(f"  rho: mean {float(rho.mean()):.4f}  p5/50/95 {rq[0]:.3f}/{rq[1]:.3f}/{rq[2]:.3f}  "
          f"frac<0.1 {float((rho < 0.1).float().mean()):.4f}  observed {int(seen.sum()):,}/{P:,}",
          flush=True)

    x_prime = rhs / colsum.clamp_min(torch.finfo(rhs.dtype).eps)[:, None]
    x_hat, info = cg_normal_equations(lambda z: z, AtA, rhs, diag, iters=a.cg_iters)
    print(f"  CG: {info['iterations']} iters, rel residual {info['final_rel_residual']:.3e}",
          flush=True)

    r_prime = Ax(x_prime) - C
    r_hat = Ax(x_hat) - C
    L_prime, L_hat = float((r_prime ** 2).sum()), float((r_hat ** 2).sum())
    # Theorem 2(ii): the excess equals ||A D^-1 L_G x_hat||^2 computed independently.
    # L_G x_hat = D x_hat - G x_hat, and G x_hat = A^T A x_hat.
    lg = colsum[:, None] * x_hat - AtA(x_hat)
    excess_thm = float((Ax(lg / colsum.clamp_min(1e-30)[:, None]) ** 2).sum())
    excess_meas = L_prime - L_hat
    rel_mismatch = abs(excess_thm - excess_meas) / max(excess_meas, 1e-30)
    print(f"  Theorem 2(ii): measured excess {excess_meas:.6e} vs predicted {excess_thm:.6e}  "
          f"(rel {rel_mismatch:.2e})", flush=True)

    out = {
        "scene": a.scene, "arm": a.arm, "views": a.views, "rays": R, "nnz": nnz, "P": P,
        "row_sum_median": float(q[1]),
        "rho_mean": float(rho.mean()), "rho_p50": float(rq[1]),
        "frac_rho_lt_0.1": float((rho < 0.1).float().mean()),
        "primitives_observed": int(seen.sum()),
        "cg_iterations": info["iterations"], "cg_residual": info["final_rel_residual"],
        "psnr_closed_form": psnr(Ax(x_prime), C), "psnr_least_squares": psnr(Ax(x_hat), C),
        "L_closed_form": L_prime, "L_least_squares": L_hat,
        "gamma": (L_prime - L_hat) / max(L_hat, 1e-30),
        "excess_measured": excess_meas, "excess_theorem2ii": excess_thm,
        "theorem2ii_rel_mismatch": rel_mismatch,
    }
    out["psnr_gap_db"] = out["psnr_least_squares"] - out["psnr_closed_form"]
    print(f"  PSNR closed-form {out['psnr_closed_form']:.3f} dB   "
          f"least-squares {out['psnr_least_squares']:.3f} dB   "
          f"gap {out['psnr_gap_db']:.3f} dB   gamma {out['gamma']:.4f}", flush=True)

    # I1: preconditioned Richardson. x_1 IS the closed form, so the k=1 row must reproduce it.
    inv = colsum.clamp_min(torch.finfo(rhs.dtype).eps).reciprocal()[:, None]
    x, ks = torch.zeros_like(rhs), {}
    for k in range(1, max(KS) + 1):
        x = x + inv * (rhs - AtA(x))
        if k in KS:
            ks[k] = psnr(Ax(x), C)
            print(f"    richardson k={k:>2}: {ks[k]:.3f} dB", flush=True)
    out["richardson_psnr"] = ks

    path = os.path.join(OUT, f"{a.arm}_{a.scene}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
