"""kappa_j -- the coefficient in OUR bound -- measured on foam and on 3DGS, same scene, same rays.

WHY THIS NUMBER AND NOT THE EARLIER ONE. The exact identity is

    x*_j - x'_j = (1/d_j) sum_{k!=j} G_jk (x*_j - x*_k),
    ||x*_j - x'_j|| <= kappa_j * spread_j ,   kappa_j = 1 - (sum_i A_ij^2)/(sum_i A_ij)

so kappa_j is a COLUMN-side quantity: it asks how much of primitive j's total received weight is
shared with other primitives. The 0.144 vs 0.850 figures reported earlier in this project are the
ROW-side purity 1 - sum_j Ahat_ij^2, which asks how concentrated a single RAY is. Both measure
co-visibility, from opposite sides, and they are NOT interchangeable -- foam's rays are nearly
one-hot (row-side 0.144) while its cells still share rays with neighbours (column-side kappa 0.24),
because a cell is hit by many rays each of which also grazes a different neighbour.

Only the column-side number appears in the theorem, and it had been measured on foam alone. This
closes that gap: if kappa(foam) << kappa(3DGS), then the additive bound is tighter on a disjoint
bounded partition as a matter of measurement, and the theory and the foam story are one result.
Cameras come from the shared bridge npz so both arms see identical rays.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))


def kappa_stats(sq, d, live_floor=1e-9):
    """kappa_j and its summary, from the column sums of A^2 and of A."""
    live = d > live_floor
    k = np.zeros_like(d)
    k[live] = 1.0 - sq[live] / d[live]
    k = np.clip(k, 0.0, 1.0)
    return k, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", choices=["foam", "gs", "both"], default="gs")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--out", default="artifacts/kappa_{scene}_{arm}.npz")
    a = ap.parse_args()

    from determinism import enable_determinism
    enable_determinism()
    camf = f"artifacts/participation/{a.scene}_cams_all.npz"
    cz = np.load(camf)
    nviews = cz["viewmats"].shape[0]
    print(f"{a.scene}: {nviews} shared cameras")

    if a.arm in ("gs", "both"):
        from gsplat import rasterization  # noqa: F401  (ensures the CUDA ext is live)
        from export_gsplat_operator import export_view_operator
        sp = torch.load(f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt", map_location="cuda",
                        weights_only=False)
        sp = sp["splats"] if "splats" in sp else sp
        means, quats = sp["means"], sp["quats"]
        scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]).reshape(-1)
        colors = sp["sh0"].reshape(len(means), 3)
        P = means.shape[0]
        Kt = torch.as_tensor(cz["K"], dtype=torch.float32, device="cuda")
        vmt = torch.as_tensor(cz["viewmats"], dtype=torch.float32, device="cuda")
        W, H = (int(x) for x in cz["wh"])
        d = np.zeros(P); sq = np.zeros(P); nnz = np.zeros(P)
        for k in range(nviews):
            r, c, v, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                                 vmt[k], Kt, W, H,
                                                 max_hits_per_pixel=a.max_hits)
            cn = c.cpu().numpy(); vn = v.cpu().numpy().astype(np.float64)
            d += np.bincount(cn, weights=vn, minlength=P)
            sq += np.bincount(cn, weights=vn ** 2, minlength=P)
            nnz += np.bincount(cn, minlength=P)
            if k % 10 == 0:
                print(f"  view {k}/{nviews}", flush=True)
        kap, live = kappa_stats(sq, d)
        print(f"\n[3DGS {a.gs_arm}] {P:,} primitives, {int(live.sum()):,} with support")
        print(f"  kappa_j : median {np.median(kap[live]):.4f}  mean {kap[live].mean():.4f}  "
              f"p90 {np.percentile(kap[live],90):.4f}")
        print(f"  frac kappa<0.05 : {(kap[live]<0.05).mean():.2%}   "
              f"mean rays/primitive {nnz[live].mean():.1f}")
        np.savez_compressed(a.out.format(scene=a.scene, arm="gs"),
                            kappa=kap.astype(np.float32), live=live)

    if a.arm in ("foam", "both"):
        import configargparse
        import warp as wp
        from configs import Params, add_group
        from data_loader import DataHandler
        from powerfoam.feature_operator import export_operator_for_views
        from powerfoam.scene import PowerfoamScene
        ck = f"output/scannet_{a.scene}_{a.variant}"
        wp.init()
        pr = configargparse.ArgParser(); add_group(pr, Params)
        pr.add_argument("-c", "--config", is_config_file=True)
        cargs = pr.parse_args(["-c", f"{ck}/config.yaml"])
        dh = DataHandler(cargs); dh.reload("all", downsample=cargs.downsample[-1])
        model = PowerfoamScene(cargs)
        model.initialize_from_dataset(dh, device="cuda")
        model.load_pt(f"{ck}/model.pt")
        P = model.points.shape[0]
        d = np.zeros(P); sq = np.zeros(P); nnz = np.zeros(P)
        for k in range(len(dh.cameras)):
            op = export_operator_for_views(model, [dh.cameras[k]], [k])
            cn = op.col_indices.cpu().numpy()
            vn = op.values.cpu().numpy().astype(np.float64)
            d += np.bincount(cn, weights=vn, minlength=P)
            sq += np.bincount(cn, weights=vn ** 2, minlength=P)
            nnz += np.bincount(cn, minlength=P)
        kap, live = kappa_stats(sq, d)
        print(f"\n[foam {a.variant}] {P:,} primitives, {int(live.sum()):,} with support")
        print(f"  kappa_j : median {np.median(kap[live]):.4f}  mean {kap[live].mean():.4f}  "
              f"p90 {np.percentile(kap[live],90):.4f}")
        print(f"  frac kappa<0.05 : {(kap[live]<0.05).mean():.2%}   "
              f"mean rays/primitive {nnz[live].mean():.1f}")
        np.savez_compressed(a.out.format(scene=a.scene, arm="foam"),
                            kappa=kap.astype(np.float32), live=live)


if __name__ == "__main__":
    main()
