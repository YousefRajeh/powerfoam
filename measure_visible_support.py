"""PIXEL-level accounting: of what a camera actually sees, how much was ever lifted onto?

`diagnose_holes.py` counts primitives, and by that denominator 41% of the foam is dead
(D_jj = 0). That number is misleading -- most of those cells are interior and never render.
The honest denominator is the PIXEL, so this drives the rasteriser and uses two outputs it
already returns: `alpha_out` (coverage, so alpha ~ 0 means no surface at all was hit) and
`front_prim_idx_out` (the front-most primitive per pixel, an exact pixel -> primitive map).

Reported per scene, over evenly spaced held-out-free train views:
  empty        alpha < eps                -- a genuine geometric hole, nothing to shade
  dead         front primitive has D = 0  -- rendered, but the lift never saw it
  single view  n_eff < 2                  -- lifted from effectively one observation
  healthy      everything else
plus the pixel-weighted distribution of n_eff, which is the quantity the disjointness
theorem leaves uncontrolled.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
import numpy as np, torch
import configargparse, warp as wp
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

SCENES = ["scene0000_00","scene0062_00","scene0070_00","scene0097_00","scene0140_00",
          "scene0200_00","scene0347_00","scene0400_00","scene0590_00","scene0645_00"]


def one_scene(scene, recon, n_views, alpha_eps, neff_min, dev="cuda"):
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device=dev)
    m.load_pt(f"{ck}/model.pt"); m.update_vis_cache()

    st = torch.load(f"artifacts/scannet/{scene}/stats_{recon}_ogl3.pt", map_location="cpu",
                    weights_only=False)
    D = st["support"].numpy().astype(np.float64)
    svw = st["sum_view_weight_sq"].numpy().astype(np.float64)
    n_eff = np.where(svw > 0, D ** 2 / np.maximum(svw, 1e-12), 0.0)
    n_eff_t = torch.from_numpy(n_eff).to(dev).float()
    dead_t = torch.from_numpy(D <= 0).to(dev)

    c = m._vis_cache
    pts, rad = c["points"], c["radii"]
    rgb = torch.zeros(pts.shape[0], m.args.num_texel_sites, 3, device=dev)
    idx = np.linspace(0, len(dh.cameras) - 1, min(n_views, len(dh.cameras))).astype(int)
    tot = dict(px=0, empty=0, dead=0, single=0, healthy=0)
    neff_hist = []
    for vi in idx:
        with torch.no_grad():
            out = m.rasterizer.visualize(dh.cameras[int(vi)], pts, rad, c["density"],
                                         c["normals"], c["texel_sites"], rgb,
                                         c["texel_height"], c["adjacency"],
                                         c["adjacency_offsets"])
        alpha, fpi = out[3].reshape(-1), out[7].reshape(-1).long()
        empty = (alpha < alpha_eps) | (fpi < 0)
        j = fpi.clamp_min(0)
        dead = ~empty & dead_t[j]
        single = ~empty & ~dead & (n_eff_t[j] < neff_min)
        tot["px"] += alpha.numel(); tot["empty"] += int(empty.sum())
        tot["dead"] += int(dead.sum()); tot["single"] += int(single.sum())
        tot["healthy"] += int((~empty & ~dead & ~single).sum())
        neff_hist.append(n_eff_t[j[~empty]].cpu().numpy())
    nh = np.concatenate(neff_hist) if neff_hist else np.zeros(0)
    return tot, nh, len(D), float((D <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--n-views", type=int, default=20)
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--neff-min", type=float, default=2.0)
    ap.add_argument("--out", default="artifacts/scannet/visible_support.json")
    a = ap.parse_args()
    wp.init()
    rows, pool = [], dict(px=0, empty=0, dead=0, single=0, healthy=0)
    qs = []
    for sc in a.scenes.split(","):
        try:
            t, nh, P, fd = one_scene(sc, a.recon, a.n_views, a.alpha_eps, a.neff_min)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}"); continue
        q = np.quantile(nh, [0.01, 0.05, 0.25, 0.5]) if len(nh) else [np.nan] * 4
        qs.append(q)
        r = dict(scene=sc, P=P, frac_dead_prims=fd,
                 **{k: t[k] / t["px"] for k in ["empty", "dead", "single", "healthy"]},
                 neff_p01=float(q[0]), neff_p05=float(q[1]), neff_p25=float(q[2]),
                 neff_med=float(q[3]))
        rows.append(r)
        for k in pool: pool[k] += t[k]
        print(f"[{sc}] prims dead {fd:6.1%} | PIXELS empty {r['empty']:6.2%} dead {r['dead']:6.2%} "
              f"single {r['single']:6.2%} healthy {r['healthy']:6.2%} | visible n_eff "
              f"p01 {q[0]:.1f} p05 {q[1]:.1f} p25 {q[2]:.1f} med {q[3]:.1f}")
    print(f"\n=== pooled over {pool['px']:,} pixels ===")
    for k in ["empty", "dead", "single", "healthy"]:
        print(f"  {k:<8} {pool[k]/pool['px']:7.3%}")
    if qs:
        q = np.array(qs)
        print("  visible n_eff, mean of per-scene quantiles: "
              f"p01 {q[:,0].mean():.1f}  p05 {q[:,1].mean():.1f}  p25 {q[:,2].mean():.1f}  "
              f"med {q[:,3].mean():.1f}")
    json.dump({"rows": rows, "pooled": pool}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
