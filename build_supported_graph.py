"""Supported-surface graph: keep only facet edges that are OBSERVED to join one visible surface.

WHY. Reweighting cannot fix an edge that should not exist. Two independent measurements say the
facet graph admits such edges:
  * 15.9% of `adjacency_true_facet` edges enclose a face of exactly zero area (facet_areas.py).
  * Regions grown on it are spatially incoherent -- projected footprints have a median of 8
    connected components and 0.325 bbox fill, i.e. a "region" is scattered across the image.
A power-diagram facet is a geometric fact, not a surface fact: two cells on opposite sides of a
thin wall share a perfectly good face, and so do two cells separated by an air gap. Neither pair
lies on the same supported surface, and an l0 partition run over those edges will happily merge
across them.

THE TEST. For each view, rasterize the front surface (`fpi` = frontmost primitive per pixel, at
alpha >= eps). Two cells are evidence of a shared surface when they own ADJACENT PIXELS -- that is
a direct observation that their surfaces touch in the image. Accumulate, per facet edge:
  * `n_obs`   -- how many times the pair was seen pixel-adjacent on the front surface
  * `depth`   -- |z_i - z_j| at those boundaries, in camera space, summed for a mean
An edge is SUPPORTED when it was observed at least `--min-obs` times AND its mean depth jump is
below `--tau` times the local primitive spacing. Everything else is dropped.

DEPTH SOURCE. The renderer's `depth_out` is all zeros (a known defect in the renderer spec), so
depth is the z of the owning cell's CENTRE in camera space -- exact for the cell, and the quantity
we want anyway, since we are asking whether two cells sit at the same range, not where the visible
surface within a cell lies.

OUTPUT is an adjacency in the same CSR layout as the input (`offsets`, `adjacent`, `dist`), so it
drops straight into cutpursuit_facet.py with no other change; the only difference is which edges
exist. `--keep-zero-area 0` additionally requires a real face, folding in the area test.
"""
import argparse
import os
import sys

import configargparse
import numpy as np
import torch
import warp as wp

sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")
sys.path.insert(0, "D:/Downloads/powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene


def undirected_keys(src, dst, P):
    lo = torch.minimum(src, dst).to(torch.int64)
    hi = torch.maximum(src, dst).to(torch.int64)
    return lo * P + hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--recon", default="nonfrozen")
    ap.add_argument("--adjacency", required=True)
    ap.add_argument("--areas", default=None, help="facet_area_*.pt; enables the zero-area test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--alpha-eps", type=float, default=0.5)
    ap.add_argument("--min-obs", type=int, default=1, help="pixel-adjacency observations required")
    ap.add_argument("--tau", type=float, default=3.0, help="max mean depth jump, in local spacings")
    ap.add_argument("--keep-zero-area", type=int, default=0, help="1 keeps zero-area faces")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    dev = "cuda"
    wp.init()

    ck = "output/scannet_%s_%s" % (a.scene, a.recon)
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", "%s/config.yaml" % ck])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args)
    m.initialize_from_dataset(dh, device=dev)
    m.load_pt("%s/model.pt" % ck)
    m.update_vis_cache()

    ad = torch.load(a.adjacency, map_location="cpu", weights_only=False)
    P = int(ad["num_primitives"])
    off = ad["offsets"].to(torch.int64)
    adj = ad["adjacent"].to(torch.int64)
    E = adj.numel()
    deg = (off[1:] - off[:-1])
    src = torch.repeat_interleave(torch.arange(P, dtype=torch.int64), deg)

    # one sorted key per UNDIRECTED edge, so a pixel observation maps to both directions at once
    keys_dir = undirected_keys(src, adj, P)
    uniq, inv = torch.unique(keys_dir, return_inverse=True)
    U = uniq.numel()
    uniq_d = uniq.to(dev)
    n_obs = torch.zeros(U, dtype=torch.float32, device=dev)
    dsum = torch.zeros(U, dtype=torch.float32, device=dev)
    print("[graph] P=%d directed E=%d undirected=%d" % (P, E, U), flush=True)

    c = m._vis_cache
    centers = c["points"]
    rgbz = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)
    n_views = len(dh.cameras)

    for vi in range(n_views):
        cam = dh.cameras[vi]
        with torch.no_grad():
            out = m.rasterizer.visualize(cam, centers, c["radii"], c["density"], c["normals"],
                                         c["texel_sites"], rgbz, c["texel_height"],
                                         c["adjacency"], c["adjacency_offsets"])
        alpha, fpi = out[3], out[7].long()
        H, W = alpha.shape[-2], alpha.shape[-1]
        alpha = alpha.reshape(H, W)
        fpi = fpi.reshape(H, W)
        ok = (alpha >= a.alpha_eps) & (fpi >= 0)
        if not bool(ok.any()):
            continue

        # depth of each cell centre in THIS camera (renderer depth_out is all zeros)
        w2c = cam.w2c().to(dev).float()          # 4x4, row 2 is the camera's forward axis
        z = centers @ w2c[2, :3] + w2c[2, 3]
        zmap = torch.zeros(H, W, device=dev)
        zmap[ok] = z[fpi[ok]]

        for (ra, ca, rb, cb) in ((slice(0, H), slice(0, W - 1), slice(0, H), slice(1, W)),
                                 (slice(0, H - 1), slice(0, W), slice(1, H), slice(0, W))):
            i = fpi[ra, ca]
            j = fpi[rb, cb]
            good = ok[ra, ca] & ok[rb, cb] & (i != j)
            if not bool(good.any()):
                continue
            ii = i[good]
            jj = j[good]
            dz = (zmap[ra, ca][good] - zmap[rb, cb][good]).abs()
            k = undirected_keys(ii, jj, P)
            pos = torch.searchsorted(uniq_d, k)
            pos = pos.clamp(max=U - 1)
            hit = uniq_d[pos] == k                      # only pairs that are real facet edges
            if not bool(hit.any()):
                continue
            ph = pos[hit]
            n_obs.index_add_(0, ph, torch.ones(ph.numel(), device=dev))
            dsum.index_add_(0, ph, dz[hit])
        if vi % 40 == 0:
            print("  view %d/%d" % (vi, n_views), flush=True)

    mean_dz = torch.where(n_obs > 0, dsum / n_obs.clamp_min(1), torch.full_like(dsum, float("inf")))

    # local spacing: median nearest-neighbour distance between adjacent cell centres
    if "dist" in ad:
        d_all = ad["dist"].float()
        spacing = float(d_all[d_all > 0].median())
    else:
        cc = centers.cpu()
        spacing = float((cc[src] - cc[adj]).norm(dim=-1).median())

    keep_u = (n_obs >= a.min_obs) & (mean_dz <= a.tau * spacing)
    if a.areas and not a.keep_zero_area:
        ar = torch.load(a.areas, map_location="cpu", weights_only=False)["area"].to(dev)
        if ar.numel() != E:
            raise SystemExit("areas %d != directed edges %d" % (ar.numel(), E))
        area_u = torch.zeros(U, device=dev).index_add_(0, inv.to(dev), ar)
        keep_u &= area_u > 0

    keep_dir = keep_u[inv.to(dev)].cpu()
    new_deg = torch.zeros(P, dtype=torch.int64)
    new_deg.index_add_(0, src[keep_dir], torch.ones(int(keep_dir.sum()), dtype=torch.int64))
    new_off = torch.zeros(P + 1, dtype=torch.int64)
    new_off[1:] = torch.cumsum(new_deg, 0)
    out = {"num_primitives": P,
           "offsets": new_off.to(torch.int32),
           "adjacent": adj[keep_dir].to(torch.int32)}
    if "dist" in ad:
        out["dist"] = ad["dist"][keep_dir]
    torch.save(out, a.output)

    stats = {"scene": a.scene, "recon": a.recon, "P": P, "directed_in": E,
             "directed_out": int(keep_dir.sum()), "kept_frac": float(keep_dir.float().mean()),
             "undirected_in": U, "undirected_kept": int(keep_u.sum()),
             "never_observed": int((n_obs == 0).sum()),
             "spacing_m": spacing, "min_obs": a.min_obs, "tau": a.tau,
             "isolated_vertices": int((new_deg == 0).sum())}
    print("\n[supported] kept %d/%d directed edges (%.1f%%)"
          % (stats["directed_out"], E, 100 * stats["kept_frac"]))
    print("            never observed pixel-adjacent: %d/%d undirected (%.1f%%)"
          % (stats["never_observed"], U, 100 * stats["never_observed"] / U))
    print("            isolated vertices after filter: %d (%.2f%%)"
          % (stats["isolated_vertices"], 100 * stats["isolated_vertices"] / P))
    print("            local spacing %.4f m, depth-jump cap %.4f m" % (spacing, a.tau * spacing))
    if a.report:
        import json
        json.dump(stats, open(a.report, "w"), indent=2)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
