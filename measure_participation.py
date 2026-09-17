"""How many primitives does a single ray actually depend on? Foam vs 3DGS, same points, same views.

WHY THIS QUANTITY. Splat Feature Solver's closed form (Eq. 9) is a contribution-weighted mean, and
it equals the least-squares solution exactly when the columns of A are orthogonal -- i.e. when no
two primitives are ever seen together by the same ray. The gap between the two is governed by the
OFF-DIAGONAL mass of A^T A, which is what our corrected (additive) bound is written in terms of.

For a single ray i, normalise its row to sum 1 and write

    purity_i   = sum_j Ahat_ij^2                (Simpson index of the row)
    PR_i       = 1 / purity_i                   (participation ratio: effective #primitives)
    offdiag_i  = 1 - purity_i                   (this row's contribution to the off-diagonal mass)

A one-hot row has PR = 1 and contributes NOTHING off-diagonal, so a representation whose rays each
depend on essentially one primitive makes the fast solver exact. That is the precise sense in which
foam's disjoint, bounded cells should beat Gaussians' overlapping unbounded support -- and it is
measurable rather than asserted.

WHY THE FROZEN ARMS. PowerFoam-frozen and 3DGS-frozen are initialised one primitive per GT vertex,
so on scene0062_00 they hold the IDENTICAL 51,610 positions. Cameras are shared through
camera_bridge. Holding points and views fixed means any difference in PR is attributable to the
kernel -- bounded power cell vs unbounded Gaussian -- and to nothing else.

The two sides cannot share an interpreter (warp+fpsample vs a compiled gsplat), so each writes an
npz and `--side report` compares them.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))


def row_stats(rows, vals, n_rows, min_weight=1e-6):
    """Per-row purity/PR from COO triples, ignoring rows with negligible total weight.

    Rows are normalised to sum 1 first: an empty-space ray that accumulates almost nothing is not
    'pure', it is unobserved, and letting it through would flatter whichever method leaks less.
    """
    rows = np.asarray(rows, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    vals = np.maximum(vals, 0.0)
    tot = np.bincount(rows, weights=vals, minlength=n_rows)
    sq = np.bincount(rows, weights=vals ** 2, minlength=n_rows)
    nnz = np.bincount(rows, minlength=n_rows)
    live = tot > min_weight
    purity = np.zeros(n_rows)
    purity[live] = sq[live] / (tot[live] ** 2)
    return {
        "n_rows_total": int(n_rows),
        "n_rows_live": int(live.sum()),
        "frac_live": float(live.mean()),
        "mean_row_sum": float(tot[live].mean()) if live.any() else 0.0,
        "mean_nnz": float(nnz[live].mean()) if live.any() else 0.0,
        "mean_purity": float(purity[live].mean()) if live.any() else 0.0,
        "mean_PR": float((1.0 / np.maximum(purity[live], 1e-12)).mean()) if live.any() else 0.0,
        "median_PR": float(np.median(1.0 / np.maximum(purity[live], 1e-12))) if live.any() else 0.0,
        "mean_offdiag_frac": float((1.0 - purity[live]).mean()) if live.any() else 0.0,
        "frac_rows_PR_under_1p5": float((1.0 / np.maximum(purity[live], 1e-12) < 1.5).mean())
        if live.any() else 0.0,
        "_purity_live": purity[live],
    }


def dump(path, st, extra):
    keep = {k: v for k, v in st.items() if not k.startswith("_")}
    keep.update(extra)
    np.savez_compressed(path, purity=st["_purity_live"].astype(np.float32),
                        meta=np.array(str(keep)))
    return keep


def side_foam(a):
    import configargparse
    import warp as wp

    from camera_bridge import K_from_ray_dirs, viewmat_from_camera
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    ckpt_dir = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ckpt_dir}/model.pt")
    model.update_vis_cache()

    ids = list(range(0, len(dh.cameras), max(1, len(dh.cameras) // a.views)))[:a.views]
    cams = [dh.cameras[i] for i in ids]
    op = export_operator_for_views(model, cams, ids, max_hits_per_pixel=a.max_hits)
    rows = op.row_indices.detach().cpu().numpy()
    vals = op.values.detach().cpu().numpy()
    n_rows = int(op.num_rows)
    st = row_stats(rows, vals, n_rows)
    print(f"[foam] {a.variant}: {model.points.shape[0]:,} primitives, {len(ids)} views")
    for k, v in st.items():
        if not k.startswith("_"):
            print(f"    {k:24s} {v}")

    # hand the shared cameras to the gs side so both measure the identical rays
    K, info = K_from_ray_dirs(cams[0])
    np.savez_compressed(f"{a.outdir}/{a.scene}_foam_{a.variant}.npz",
                        purity=st["_purity_live"].astype(np.float32))
    np.savez_compressed(f"{a.outdir}/{a.scene}_cams.npz",
                        K=np.asarray(K, dtype=np.float64),
                        viewmats=np.stack([viewmat_from_camera(c).cpu().numpy() for c in cams]),
                        wh=np.array([int(cams[0].width), int(cams[0].height)]),
                        view_ids=np.array(ids))
    import json
    json.dump({k: v for k, v in st.items() if not k.startswith("_")},
              open(f"{a.outdir}/{a.scene}_foam_{a.variant}.json", "w"), indent=1)


def side_gs(a):
    from export_gsplat_operator import export_view_operator

    cams = np.load(f"{a.outdir}/{a.scene}_cams.npz")
    K = torch.as_tensor(cams["K"], dtype=torch.float32, device="cuda")
    vms = torch.as_tensor(cams["viewmats"], dtype=torch.float32, device="cuda")
    W, H = (int(x) for x in cams["wh"])

    ck = torch.load(f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt", map_location="cuda",
                    weights_only=False)["splats"]
    means, quats = ck["means"], ck["quats"]
    scales, opac = torch.exp(ck["scales"]), torch.sigmoid(ck["opacities"]).reshape(-1)
    colors = ck["sh0"].reshape(len(means), 3)

    all_rows, all_vals, base = [], [], 0
    for v in range(vms.shape[0]):
        r, c, val, _, _ = export_view_operator(
            means, quats, scales, opac, colors, vms[v], K, W, H,
            max_hits_per_pixel=a.max_hits)
        all_rows.append(r.detach().cpu().numpy().astype(np.int64) + base)
        all_vals.append(val.detach().cpu().numpy().astype(np.float64))
        base += H * W
        print(f"    view {v}: {len(r):,} nonzeros", flush=True)
    rows = np.concatenate(all_rows); vals = np.concatenate(all_vals)
    st = row_stats(rows, vals, base)
    print(f"[gs] {a.gs_arm}: {len(means):,} primitives, {vms.shape[0]} views")
    for k, v in st.items():
        if not k.startswith("_"):
            print(f"    {k:24s} {v}")
    np.savez_compressed(f"{a.outdir}/{a.scene}_gs_{a.gs_arm}.npz",
                        purity=st["_purity_live"].astype(np.float32))
    import json
    json.dump({k: v for k, v in st.items() if not k.startswith("_")},
              open(f"{a.outdir}/{a.scene}_gs_{a.gs_arm}.json", "w"), indent=1)


def side_report(a):
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for tag, label in [(f"foam_{a.variant}", "PowerFoam (frozen)"),
                       (f"gs_{a.gs_arm}", "3DGS (frozen)")]:
        p = f"{a.outdir}/{a.scene}_{tag}"
        if not os.path.exists(p + ".npz"):
            print(f"[miss] {p}.npz")
            continue
        pur = np.load(p + ".npz")["purity"].astype(np.float64)
        meta = json.load(open(p + ".json"))
        entries.append((label, pur, meta))
    if len(entries) < 2:
        print("need both sides"); return

    print(f"\n{'arm':22s} {'mean PR':>9s} {'median PR':>10s} {'off-diag frac':>14s} "
          f"{'rows PR<1.5':>12s}")
    for label, pur, meta in entries:
        print(f"{label:22s} {meta['mean_PR']:9.3f} {meta['median_PR']:10.3f} "
              f"{meta['mean_offdiag_frac']:14.4f} {meta['frac_rows_PR_under_1p5']:11.1%}")
    r = entries[1][2]["mean_offdiag_frac"] / max(entries[0][2]["mean_offdiag_frac"], 1e-12)
    print(f"\n3DGS carries {r:.2f}x the off-diagonal (co-visibility) mass of foam per ray.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), dpi=160)
    for label, pur, meta in entries:
        pr = 1.0 / np.maximum(pur, 1e-12)
        ax[0].hist(np.clip(pr, 1, 20), bins=np.linspace(1, 20, 76), alpha=0.55, label=label,
                   density=True)
        ax[1].hist(1.0 - pur, bins=np.linspace(0, 1, 76), alpha=0.55, label=label, density=True)
    ax[0].set_xlabel("participation ratio  PR$_i$ = effective #primitives per ray")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=9)
    ax[0].set_title("How many primitives does one ray depend on?", fontsize=11)
    ax[1].set_xlabel(r"per-ray off-diagonal fraction  $1-\sum_j \hat{A}_{ij}^2$")
    ax[1].set_title("Co-visibility mass (0 = solver is exact)", fontsize=11)
    ax[1].legend(fontsize=9)
    for x in ax:
        x.grid(alpha=0.18)
    fig.suptitle(f"Same {entries[0][2]['n_rows_live']:,} rays, same points, different kernel "
                 f"- {a.scene}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{a.outdir}/{a.scene}_participation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["foam", "gs", "report"], required=True)
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--outdir", default="artifacts/participation")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    from determinism import enable_determinism
    enable_determinism()
    {"foam": side_foam, "gs": side_gs, "report": side_report}[a.side](a)


if __name__ == "__main__":
    main()
