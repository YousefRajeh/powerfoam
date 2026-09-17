"""Is within-view conflict just a cell-size artefact?

`c_intra` (mean per-view feature norm) fell out as the strongest single predictor of whether a cell
is classified correctly. Two very different readings:

  (a) MASK QUALITY -- a cell's pixels land in genuinely different SAM masks, so the per-view mean
      blends unrelated objects. Interesting, and fixable by hard per-view mask assignment.
  (b) CELL SIZE -- a cell that projects to many pixels simply spans more of the image and is more
      likely to straddle a mask boundary for purely geometric reasons. Then `c_intra` is a proxy for
      "big cell", and any gain from it is a gain from down-weighting big cells, which we could get
      without any of this machinery.

Reading (b) would make the finding much less interesting, so it is worth ruling in or out before
building on it. This measures each cell's actual projected footprint -- pixels it contributes to,
per view, from the renderer's own operator, not an analytic radius/depth estimate -- and correlates
it with c_intra using rank statistics (the relationship need not be linear).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--stats", default="artifacts/adaptive/s0062_stats_l3bb.pt")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--weight-floor", type=float, default=1e-3,
                    help="a pixel counts toward a cell's footprint above this contribution")
    a = ap.parse_args()

    import configargparse
    import warp as wp
    from scipy.stats import spearmanr

    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    model.update_vis_cache()
    P = model.points.shape[0]

    ids = list(range(0, len(dh.cameras), max(1, len(dh.cameras) // a.views)))[:a.views]
    px_sum = np.zeros(P)          # total pixels contributed to, across views
    views_seen = np.zeros(P)      # number of views in which the cell appears at all
    for vi in ids:
        op = export_operator_for_views(model, [dh.cameras[vi]], [vi])
        cols = op.col_indices.detach().cpu().numpy()
        vals = op.values.detach().cpu().numpy()
        keep = vals > a.weight_floor
        c = cols[keep]
        cnt = np.bincount(c, minlength=P).astype(np.float64)
        px_sum += cnt
        views_seen += (cnt > 0)
    seen = views_seen > 0
    area = np.zeros(P)
    area[seen] = px_sum[seen] / views_seen[seen]        # mean projected pixels per view seen
    print(f"{int(seen.sum()):,} of {P:,} cells appear in the {len(ids)} sampled views")
    print(f"projected area (px/view): median {np.median(area[seen]):.1f}  "
          f"mean {area[seen].mean():.1f}  p95 {np.percentile(area[seen], 95):.1f}")

    d = torch.load(a.stats, map_location="cpu", weights_only=False)
    g = lambda k: (d[k] if isinstance(d, dict) else getattr(d, k)).float().numpy()
    sup = g("support").reshape(-1)
    intra = g("intra_sum").reshape(-1)
    num = g("numerator")
    svw = g("sum_view_weight_sq").reshape(-1)
    valid = sup > 0
    c_intra = np.where(valid, intra / np.maximum(sup, 1e-12), 0.0)
    nn = np.linalg.norm(num, axis=1)
    c_inter = np.where(valid, nn / np.maximum(intra, 1e-12), 0.0)
    n_eff = np.where(svw > 1e-12, sup ** 2 / np.maximum(svw, 1e-12), 0.0)

    m = valid & seen
    print(f"\ncells with both stats and a footprint: {int(m.sum()):,}")
    for name, y in [("c_intra (within-view agreement)", c_intra),
                    ("c_inter (between-view agreement)", c_inter),
                    ("n_eff  (effective views)", n_eff)]:
        r_area = spearmanr(area[m], y[m]).statistic
        r_sup = spearmanr(sup[m], y[m]).statistic
        print(f"  spearman({name:34s}, projected area) = {r_area:+.4f}"
              f"    (vs support: {r_sup:+.4f})")

    # Decile table: if (b) held, c_intra would fall monotonically as area grows.
    q = np.quantile(area[m], np.linspace(0, 1, 11))
    q[-1] += 1e-9
    print(f"\n{'area decile':>12} {'px/view':>16} {'n':>8} {'c_intra':>9} {'c_inter':>9}")
    for i in range(10):
        k = m & (area >= q[i]) & (area < q[i + 1])
        if not k.any():
            continue
        print(f"{i + 1:>12} {q[i]:>7.1f}-{q[i + 1]:<8.1f} {int(k.sum()):>8,} "
              f"{c_intra[k].mean():>9.4f} {c_inter[k].mean():>9.4f}")

    # The decisive control: WITHIN a narrow size band, does c_intra still vary and still track
    # anything? If c_intra were purely a size proxy it would be near-constant inside the band.
    lo, hi = np.percentile(area[m], [45, 55])
    band = m & (area >= lo) & (area <= hi)
    print(f"\nnarrow size band {lo:.1f}-{hi:.1f} px/view, {int(band.sum()):,} cells: "
          f"c_intra mean {c_intra[band].mean():.4f}  std {c_intra[band].std():.4f}  "
          f"range {c_intra[band].min():.3f}-{c_intra[band].max():.3f}")


if __name__ == "__main__":
    main()
