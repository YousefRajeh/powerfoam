"""Split NormLift's reliability into within-view and between-view agreement, exactly.

NormLift defines per-Gaussian reliability from the back-projected norm and observes that it
"algebraically factors into intra-view and inter-view consistency", but on Gaussians that
factorisation is nominal: an unbounded kernel has no definite image footprint, so within-view spread
cannot be separated from the tail reaching across the picture. On a bounded disjoint partition each
cell HAS a footprint, so the split is identifiable. That is the foam-native claim this measures.

THE IDENTITY. For unit observations B_i, weights A_ij, and views v, write the per-view resultant
    M_jv = sum_{i in v} A_ij B_i ,      S_jv = sum_{i in v} A_ij ,      S_j = sum_v S_jv
and define
    R_j = ||sum_v M_jv|| / S_j                      total reliability (NormLift's quantity)
    W_j = sum_v ||M_jv|| / S_j                      WITHIN-view agreement
    B_j = ||sum_v M_jv|| / sum_v ||M_jv||           BETWEEN-view agreement
Then

    R_j = W_j * B_j        EXACTLY, by construction -- no approximation, no equal-variance assumption

because both sides equal ||sum_v M_jv|| / S_j. W_j is 1 iff every view is internally unanimous about
cell j; B_j is 1 iff the per-view mean directions all point the same way. Choosing the ||M_jv||
weighting for B_j (rather than the more obvious S_jv) is what makes the product exact -- with S_jv
weights the identity only holds when the per-view resultants are equal.

WHY IT IS WORTH THE PASS. The two factors call for different fixes, per cell:
  * W_j low  -> the cell's own footprint straddles a semantic boundary. A geometric problem: the
                cell is too big, or the surface is in the wrong place. A better ESTIMATOR cannot help.
  * B_j low  -> the views disagree with each other. A robust accumulator (geometric median,
                mode-vote) is exactly the right fix; more geometry is not.
Falsifiable prediction: SAM-mask features are view-unstable, so B should dominate the loss there --
which is what would retro-explain the geometric median's SAM-round win (0.6095 vs 0.4649) while it
barely moved the dense round (0.5198 vs 0.5177).

MEMORY. Only one (P, D) accumulator is held: sum_v M_jv. The within-view term needs just the SCALAR
||M_jv|| per view, so per-view resultants are never all resident.
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

OUT = "artifacts/scannet/reliability"


def selftest():
    """Prove R = W*B on hand-computable inputs before trusting it on a real operator."""
    torch.manual_seed(0)
    ok = True

    def rwb(Ms, S):
        norms = torch.stack([m.norm() for m in Ms])
        tot = torch.stack(Ms).sum(0)
        R = float(tot.norm() / S)
        W = float(norms.sum() / S)
        B = float(tot.norm() / norms.sum())
        return R, W, B

    # (a) One view, perfectly unanimous: W = 1 (all mass aligned), B = 1 (only one view).
    b = torch.tensor([1.0, 0.0, 0.0])
    Ms = [3.0 * b]
    R, W, B = rwb(Ms, 3.0)
    ok &= abs(R - 1) < 1e-6 and abs(W - 1) < 1e-6 and abs(B - 1) < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  unanimous single view: R={R:.4f} W={W:.4f} B={B:.4f}")

    # (b) Two views, each internally unanimous but pointing 90 deg apart:
    #     W must be 1 (no within-view spread), B must carry all the loss.
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    Ms = [2.0 * e1, 2.0 * e2]
    R, W, B = rwb(Ms, 4.0)
    c = abs(W - 1.0) < 1e-6 and abs(B - (2 ** 0.5) / 2) < 1e-6 and abs(R - W * B) < 1e-6
    ok &= c
    print(f"  {'PASS' if c else 'FAIL'}  orthogonal views: R={R:.4f} W={W:.4f} B={B:.4f} "
          f"(expect W=1, B=0.7071)")

    # (c) One view, internally split 50/50 between two orthogonal directions:
    #     the loss is entirely WITHIN-view; B must be 1.
    Ms = [1.0 * e1 + 1.0 * e2]
    R, W, B = rwb(Ms, 2.0)
    c = abs(B - 1.0) < 1e-6 and abs(W - (2 ** 0.5) / 2) < 1e-6 and abs(R - W * B) < 1e-6
    ok &= c
    print(f"  {'PASS' if c else 'FAIL'}  split single view: R={R:.4f} W={W:.4f} B={B:.4f} "
          f"(expect B=1, W=0.7071)")

    # (d) Random: the product identity must hold exactly, always.
    worst = 0.0
    for _ in range(500):
        nv = int(torch.randint(1, 8, (1,)))
        Ms = [torch.randn(16) * float(torch.rand(1) * 3 + 0.1) for _ in range(nv)]
        S = float(torch.rand(1) * 5 + 1)
        R, W, B = rwb(Ms, S)
        worst = max(worst, abs(R - W * B) / max(abs(R), 1e-12))
    c = worst < 1e-5
    ok &= c
    print(f"  {'PASS' if c else 'FAIL'}  random: R == W*B, worst rel err {worst:.2e}")
    return ok


def load_view_features(feat_dir, stem, H, W):
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[-1] if s.shape[0] <= 4 else s[..., -1]
    t = torch.from_numpy(np.ascontiguousarray(f)).float()
    t = torch.nn.functional.normalize(t, dim=-1)          # unit B_i, as the identity assumes
    seg = torch.from_numpy(np.ascontiguousarray(s)).long()
    if seg.shape != (H, W):
        seg = torch.nn.functional.interpolate(
            seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0347_00")
    ap.add_argument("--arm", default="pf_truefrozen",
                    choices=("pf_truefrozen", "pf_nonfrozen", "gs_froz", "gs_unfroz"))
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("self-test of the R = W*B identity:")
        raise SystemExit(0 if selftest() else 1)

    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    from camera_bridge import K_from_ray_dirs
    from configs import Params, add_group
    from data_loader import DataHandler

    recon = "nonfrozen" if a.arm.startswith("gs_") else a.arm[3:]
    cfg = f"output/scannet_{a.scene}_{recon}/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    sel = np.linspace(0, len(dh.cameras) - 1, a.views).astype(int).tolist()
    feat_dir = os.path.join(args.data_path, args.scene, "openclip_features_sam_l3")
    stems = sorted(os.path.splitext(f)[0][:-2] for f in os.listdir(feat_dir) if f.endswith("_f.npy"))

    if a.arm.startswith("pf_"):
        import warp as wp
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        wp.init()
        model = PowerfoamScene(args)
        model.initialize_from_dataset(dh, device=dev)
        model.load_pt(f"output/scannet_{a.scene}_{recon}/model.pt")
        P = int(model.points.shape[0])
    else:
        from gsplat_baseline.export_gsplat_operator import export_view_operator
        ck = torch.load(f"recon_remote/{a.arm}/{a.scene}/ckpt.pt", map_location=dev,
                        weights_only=False)
        sp = ck["splats"] if "splats" in ck else ck
        gm, gq = sp["means"].cuda(), sp["quats"].cuda()
        gs_ = torch.exp(sp["scales"].cuda())
        go = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
        gc = torch.zeros((gm.shape[0], 1), device=dev)
        P = int(gm.shape[0])

    D = None
    M_tot = None                                   # sum_v M_jv          (P, D)
    norm_sum = torch.zeros(P, device=dev)          # sum_v ||M_jv||      (P,)
    S = torch.zeros(P, device=dev)                 # sum_v S_jv          (P,)

    for vi in sel:
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        if a.arm.startswith("pf_"):
            op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=a.cap,
                                           max_intersections=4096)
            ri, ci, vv = op.row_indices, op.col_indices, op.values
            del op
        else:
            K, _ = K_from_ray_dirs(cam)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :4] = dh.c2ws[vi].double()
            vm = torch.linalg.inv(c2w).float().cuda()
            ri, ci, vv, _, _ = export_view_operator(gm, gq, gs_, go, gc, vm, K.cuda(), W, H,
                                                    max_hits_per_pixel=a.cap,
                                                    transmittance_floor=1e-3)
        seg, tab = load_view_features(feat_dir, stems[vi % len(stems)], H, W)
        seg, tab = seg.to(dev), tab.to(dev)
        if D is None:
            D = tab.shape[1]
            M_tot = torch.zeros((P, D), device=dev)
        ri = ri.to(torch.int64)
        ci = ci.to(torch.int64)
        vv = vv.float()

        M_v = torch.zeros((P, D), device=dev)
        CH = 4_000_000
        for s0 in range(0, vv.numel(), CH):
            e0 = min(s0 + CH, vv.numel())
            b = tab[seg[ri[s0:e0]].clamp(0, tab.shape[0] - 1)]
            M_v.index_add_(0, ci[s0:e0], vv[s0:e0, None] * b)
        S.index_add_(0, ci, vv)
        norm_sum += M_v.norm(dim=1)
        M_tot += M_v
        del M_v, ri, ci, vv, seg, tab

    live = S > 1e-8
    tot_norm = M_tot.norm(dim=1)
    R = torch.zeros(P, device=dev)
    Wt = torch.zeros(P, device=dev)
    Bt = torch.zeros(P, device=dev)
    R[live] = tot_norm[live] / S[live]
    Wt[live] = norm_sum[live] / S[live]
    Bt[live] = tot_norm[live] / norm_sum[live].clamp_min(1e-12)
    ident = float((R[live] - Wt[live] * Bt[live]).abs().max())

    def q(t):
        return [float(x) for x in torch.quantile(t[live].double(),
                torch.tensor([.1, .25, .5, .75, .9], device=dev, dtype=torch.float64))]

    # Which factor carries the disagreement? Compare the two deficits per cell.
    dW = 1 - Wt[live]
    dB = 1 - Bt[live]
    res = {"scene": a.scene, "arm": a.arm, "views": a.views, "P": P, "P_live": int(live.sum()),
           "identity_max_abs_err": ident,
           "R_mean": float(R[live].mean()), "R_q": q(R),
           "W_mean": float(Wt[live].mean()), "W_q": q(Wt),
           "B_mean": float(Bt[live].mean()), "B_q": q(Bt),
           "deficit_within_mean": float(dW.mean()),
           "deficit_between_mean": float(dB.mean()),
           "frac_cells_within_dominant": float((dW > dB).float().mean())}
    json.dump(res, open(f"{OUT}/{a.arm}_{a.scene}.json", "w"), indent=1)
    print(f"[{a.arm}/{a.scene}] P_live={res['P_live']:,}  identity max|R-WB| = {ident:.2e}",
          flush=True)
    print(f"  R (total reliability) mean {res['R_mean']:.4f}   median {res['R_q'][2]:.4f}",
          flush=True)
    print(f"  W (within-view)       mean {res['W_mean']:.4f}   median {res['W_q'][2]:.4f}",
          flush=True)
    print(f"  B (between-view)      mean {res['B_mean']:.4f}   median {res['B_q'][2]:.4f}",
          flush=True)
    print(f"  deficit: within {res['deficit_within_mean']:.4f}  between "
          f"{res['deficit_between_mean']:.4f}   within-dominant on "
          f"{res['frac_cells_within_dominant']*100:.1f}% of cells", flush=True)


if __name__ == "__main__":
    main()
