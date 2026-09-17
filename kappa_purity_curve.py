"""kappa versus ray purity: the bias/variance curve our bound predicts, measured.

THE INTERVENTION THE BOUND IMPLIES. The exact identity says the closed form's error is

    x*_j - x'_j = (1/d_j) sum_{k!=j} G_jk (x*_j - x*_k),
    ||x*_j - x'_j|| <= kappa_j * spread_j,   kappa_j = 1 - (sum_i A_ij^2)/(sum_i A_ij)

so the error is created by OFF-DIAGONAL mass, and off-diagonal mass is created by rays that graze
several primitives. That is a per-ray property we already measure -- the row purity

    p_i = sum_j Ahat_ij^2            (1 = the ray reads one primitive, 1/k = it splits over k)

Down-weighting or dropping impure rays therefore shrinks kappa directly. Unlike Tikhonov Guidance --
which pursues the same goal by polarising the trained opacities, changing the reconstruction itself
-- this touches only the estimator, and the bound predicts the direction of the effect rather than
hoping for it.

WHAT THIS MEASURES, AND WHY IT IS A CURVE NOT A NUMBER. Dropping rays trades bias for variance:
kappa falls (the closed form moves toward least squares) but each primitive keeps less evidence, and
some lose support entirely. Both sides are reported at every threshold so the trade is visible:

    kappa            the bound's coefficient (bias)
    weight kept      fraction of total rendering weight surviving (variance)
    cells alive      primitives that still have any support at all
    rays kept        fraction of observations surviving

HARD vs SOFT. `thresh` keeps rays with p_i >= tau. `soft` keeps every ray but weights it by p_i,
which cannot orphan a primitive -- the analogue of trimming vs reweighting from the contamination
work, where trimming won because deletion removes weight outright while attenuation only reduces it.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))


def accumulate(rows, cols, vals, P, taus, d, sq, nray, dS, sqS):
    """Fold one view into the per-threshold accumulators (and the soft-weighted one)."""
    tot = np.bincount(rows, weights=vals, minlength=int(rows.max()) + 1)
    sqr = np.bincount(rows, weights=vals ** 2, minlength=int(rows.max()) + 1)
    live = tot > 1e-12
    p = np.zeros_like(tot)
    p[live] = sqr[live] / (tot[live] ** 2)          # row purity, in (0, 1]
    pv = p[rows]
    for t, tau in enumerate(taus):
        k = pv >= tau
        if not k.any():
            continue
        d[t] += np.bincount(cols[k], weights=vals[k], minlength=P)
        sq[t] += np.bincount(cols[k], weights=vals[k] ** 2, minlength=P)
        nray[t] += np.bincount(cols[k], minlength=P)
    w = vals * pv                                    # soft: weight each ray by its purity
    dS += np.bincount(cols, weights=w, minlength=P)
    sqS += np.bincount(cols, weights=w ** 2, minlength=P)
    return float(vals.sum()), float(np.bincount(rows, minlength=1).sum()), p[live]


def summarise(d, sq, base_d, label, kept_w, kept_r):
    live = d > 1e-12
    kap = np.zeros_like(d)
    kap[live] = np.clip(1.0 - sq[live] / d[live], 0, 1)
    alive = live.sum()
    print(f"{label:>14} {np.median(kap[live]):>9.4f} {kap[live].mean():>8.4f} "
          f"{(kap[live] < 0.05).mean():>9.2%} {kept_w:>9.2%} {kept_r:>9.2%} "
          f"{alive / max((base_d > 1e-12).sum(), 1):>9.2%}")
    return float(np.median(kap[live])), float(alive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--arm", choices=["foam", "gs"], default="foam")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--taus", nargs="*", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--out", default="artifacts/kappa_purity_{scene}_{arm}.npz")
    a = ap.parse_args()

    from determinism import enable_determinism
    enable_determinism()
    taus = list(a.taus)

    if a.arm == "foam":
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
        nviews = len(dh.cameras)

        def op(k):
            o = export_operator_for_views(model, [dh.cameras[k]], [k])
            return (o.row_indices.cpu().numpy(), o.col_indices.cpu().numpy(),
                    o.values.cpu().numpy().astype(np.float64))
    else:
        from export_gsplat_operator import export_view_operator
        cz = np.load(f"artifacts/participation/{a.scene}_cams_all.npz")
        Kt = torch.as_tensor(cz["K"], dtype=torch.float32, device="cuda")
        vmt = torch.as_tensor(cz["viewmats"], dtype=torch.float32, device="cuda")
        W, H = (int(x) for x in cz["wh"])
        sp = torch.load(f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt", map_location="cuda",
                        weights_only=False)
        sp = sp["splats"] if "splats" in sp else sp
        means, quats = sp["means"], sp["quats"]
        scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]).reshape(-1)
        colors = sp["sh0"].reshape(len(means), 3)
        P = means.shape[0]
        nviews = vmt.shape[0]

        def op(k):
            r, c, v, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                                 vmt[k], Kt, W, H, max_hits_per_pixel=64)
            return (r.cpu().numpy(), c.cpu().numpy(), v.cpu().numpy().astype(np.float64))

    d = np.zeros((len(taus), P)); sq = np.zeros((len(taus), P))
    nray = np.zeros((len(taus), P))
    dS = np.zeros(P); sqS = np.zeros(P)
    tot_w = 0.0
    purities = []
    for k in range(nviews):
        rows, cols, vals = op(k)
        if len(rows) == 0:
            continue
        w_, _, pl = accumulate(rows, cols, vals, P, taus, d, sq, nray, dS, sqS)
        tot_w += w_
        if k % 10 == 0:
            purities.append(pl[::37])
            print(f"  view {k}/{nviews}", flush=True)
    pall = np.concatenate(purities) if purities else np.array([1.0])
    print(f"\n[{a.arm} {a.scene}] {P:,} primitives, {nviews} views")
    print(f"ray purity p_i: median {np.median(pall):.4f}  mean {pall.mean():.4f}  "
          f"frac p>0.9 {np.mean(pall > 0.9):.2%}")
    print(f"\n{'threshold':>14} {'median k':>9} {'mean k':>8} {'k<0.05':>9} "
          f"{'weight':>9} {'rays':>9} {'cells':>9}")
    base_d = d[0].copy()
    base_r = nray[0].sum()
    res = []
    for t, tau in enumerate(taus):
        res.append(summarise(d[t], sq[t], base_d, f"p >= {tau:g}",
                             d[t].sum() / max(base_d.sum(), 1e-12),
                             nray[t].sum() / max(base_r, 1)))
    summarise(dS, sqS, base_d, "soft (w*p)", dS.sum() / max(base_d.sum(), 1e-12), 1.0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out.format(scene=a.scene, arm=a.arm),
                        taus=np.array(taus), d=d.astype(np.float32),
                        sq=sq.astype(np.float32), dS=dS.astype(np.float32),
                        sqS=sqS.astype(np.float32))
    print(f"wrote {a.out.format(scene=a.scene, arm=a.arm)}")


if __name__ == "__main__":
    main()
