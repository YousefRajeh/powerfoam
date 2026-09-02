"""OBJECT-ALIGNED GROUPING from mask centroids (OpenVoxel arXiv 2601.09575 section 4.1), foam version.

WHY THIS IS THE BLOCKING PREREQUISITE. Three separate ideas failed today for ONE shared reason -- they
aggregate evidence over a group that is not an object:

  * hierarchical query ([[Four-ideas-view-hub-hierarchy-quorum-2026-08-31]] section 3): spatial
    k-means nodes are 0.398 pure at the coarsest level, so a pooled "coarse scale" embedding is
    mostly other objects. H_descend scored -16.53.
  * view quorum (section 4): voting per CELL is too fine -- each view's argmax over 100 classes is
    noisy, so votes do not concentrate (median purity 0.427).
  * feature-similarity region growing (earlier): adjacent cells agree at median cosine 0.996, so no
    threshold has a natural scale -- either one 632k-cell component or 2-15 cell dust.

Meanwhile two measured facts say the per-view evidence genuinely disagrees and is worth exploiting:
42% of errors flip class between disjoint view halves, and vote purity is 0.427.

THE MECHANISM. OpenVoxel accumulates, per voxel, the 3D CENTROID of the 2D instance mask that each
pixel belongs to. Two cells on the SAME object receive the same centroid from every view; two cells
on different objects receive different ones. Unlike feature similarity this has a real scale (metres),
and unlike spatial k-means it is derived from actual object extent.

FOAM SPECIFICS, and why the attribution is exact here. OpenVoxel needs a rendered point map to get
each pixel's 3D position. We get it directly from the operator: `export_feature_operator` returns the
exact ray-cell incidence (`out_col`) with exact segment weights (`out_val`), so

    pointmap[pixel] = sum_slots val * centre[col] / sum_slots val

is the expected hit position computed from the partition itself, with no interpolation and no
ambiguity about which primitive a pixel belongs to. In an overlapping Gaussian mixture that
attribution is genuinely ill-posed -- many primitives contribute at the same depth.

THE GATE. Purity of the resulting groups at matched group COUNT against the spatial k-means baseline
that failed. If mask-centroid groups are not markedly purer at coarse granularity, the whole
"aggregate over objects" programme is closed and the three ideas above stay dead.

    D:/conda/envs/powerfoam/python.exe run_mask_centroid_grouping.py --stage accumulate --scene X
    python run_mask_centroid_grouping.py --stage analyze --scene X
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

RECON = os.path.join("D:" + os.sep, "Downloads", "spp_results", "full")
FEAT_ROOT = Path("D:" + os.sep) / "Downloads" / "spp_data_1600"


def log(m):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def mask_centroids_for_view(out_col, out_val, slots_used, seg_flat, centres, max_hits):
    """Per-cell (numerator, denominator) contribution of one view's mask centroids.

    Steps, all exact given the operator:
      1. pointmap[p] = sum_s val * centre[col] / sum_s val      -- expected hit position per pixel
      2. centroid[k]  = weighted mean of pointmap over pixels of mask k
      3. scatter centroid[mask[p]] back onto the cells the pixel's rays traverse, weighted by val
    """
    npix = seg_flat.shape[0]
    col = out_col.reshape(npix, max_hits)
    val = out_val.reshape(npix, max_hits)
    ar = torch.arange(max_hits, device=col.device)
    keep = ar[None, :] < slots_used[:, None]
    # UNUSED SLOTS CARRY GARBAGE COLUMN INDICES. Zeroing only `val` is not enough -- `centres[col]`
    # and the scatter still dereference those indices and trip a device-side assert. Point them at
    # cell 0 with zero weight so every gather is in-bounds and contributes nothing.
    col = torch.where(keep, col, torch.zeros_like(col)).clamp_(0, centres.shape[0] - 1)
    val = val * keep
    w_pix = val.sum(1)                                            # (npix,)
    good = w_pix > 1e-8

    pm = torch.zeros(npix, 3, device=col.device)
    pm[good] = (centres[col[good]] * val[good][..., None]).sum(1) / w_pix[good][:, None]

    n_masks = int(seg_flat.max().item()) + 1
    cnum = torch.zeros(n_masks, 3, device=col.device)
    cden = torch.zeros(n_masks, device=col.device)
    cnum.index_add_(0, seg_flat[good], pm[good] * w_pix[good][:, None])
    cden.index_add_(0, seg_flat[good], w_pix[good])
    cent = cnum / cden.clamp_min(1e-8)[:, None]
    cent[0] = 0.0                                                 # slot 0 = unmasked, casts nothing

    payload = cent[seg_flat]                                      # (npix, 3)
    live = good & (seg_flat > 0)
    P = centres.shape[0]
    num = torch.zeros(P, 3, device=col.device)
    den = torch.zeros(P, device=col.device)
    c_ok, v_ok = col[live], val[live]
    pay = payload[live]
    num.index_add_(0, c_ok.reshape(-1),
                   (pay[:, None, :] * v_ok[..., None]).reshape(-1, 3))
    den.index_add_(0, c_ok.reshape(-1), v_ok.reshape(-1))
    return num, den


def stage_accumulate(scene, device="cuda", max_hits=64):
    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ck}/model.pt")
    images_dir = Path(args.data_path) / args.scene / "images"
    names = sorted(p_.stem for p_ in images_dir.iterdir())
    folder = FEAT_ROOT / scene / "openclip_features_sam_l3"
    centres = model.points.detach().float()
    P = centres.shape[0]
    NUM = torch.zeros(P, 3, device=device)
    DEN = torch.zeros(P, device=device)

    for vid, cam in enumerate(dh.cameras):
        sp = folder / f"{names[vid]}_s.npy"
        if not sp.exists():
            continue
        seg = torch.from_numpy(np.load(sp)).to(device).long().reshape(-1) + 1
        oc, ov, sc, _, _ = model.export_feature_operator(
            cam, transmittance_threshold=1e-3, max_intersections=1024,
            max_hits_per_pixel=max_hits)
        n, d = mask_centroids_for_view(oc, ov, sc.clamp(max=max_hits), seg, centres, max_hits)
        NUM += n; DEN += d
        del oc, ov, sc, n, d, seg
        if vid % 50 == 0:
            log(f"    view {vid}/{len(dh.cameras)}")
    cent = NUM / DEN.clamp_min(1e-8)[:, None]
    out = f"artifacts/scannetpp/{scene}/mask_centroid.npz"
    np.savez(out, centroid=cent.cpu().numpy(), weight=DEN.cpu().numpy())
    seen = (DEN > 0).sum().item()
    log(f"  saved {out}: {seen:,}/{P:,} cells received a centroid")


def stage_analyze(scene, out_json, device="cuda"):
    from build_true_facet_graph import load_points_radii
    from evaluate_point_cloud_miou import remap_gt_labels
    from point_cloud_query import assign_points_to_power_cells
    from run_simplex_diffusion_eval import csr_to_edges
    from run_macro_iou_gap import cell_histograms
    from run_hierarchy_eval import build_hierarchy
    from run_spp_eval import benchmark_map, load_gt, coverage_filter
    from feature_foam_lifting.operator import AccumulatedFeatureStats

    art = f"artifacts/scannetpp/{scene}"
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    centers, radii = load_points_radii(ck)
    z = np.load(f"{art}/mask_centroid.npz")
    cent = torch.from_numpy(z["centroid"]).to(device).float()
    wgt = torch.from_numpy(z["weight"]).to(device).float()
    vmn = torch.load(f"{art}/solved_geometric_median_nonfrozen_ogl3.pt",
                     map_location="cpu", weights_only=True)["valid_mask"].numpy()
    vm = torch.from_numpy(vmn).to(device)
    P = len(centers)

    adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
    src, dst, _ = csr_to_edges(adj["adjacent"].to(device).long(),
                               adj["offsets"].to(device).long(), P, device)
    ok = vm[src] & vm[dst] & (wgt[src] > 0) & (wgt[dst] > 0)
    src, dst = src[ok], dst[ok]

    top, r2b = benchmark_map()
    gt_pts, lab0, _ = load_gt(scene, top, r2b)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
    keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
    lab = np.where(keepc, lab0, -1)
    pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
    H, _ = cell_histograms(assigned, torch.from_numpy(remap_gt_labels(lab, pres)).long(),
                           P, len(pres))
    has_gt = H.sum(1) > 0
    cell_lab = H.argmax(1)

    # scene scale for a dimensionless threshold: median facet length
    pos = torch.from_numpy(centers).to(device).float()
    scale = float((pos[src] - pos[dst]).norm(dim=-1).median())
    d_cent = (cent[src] - cent[dst]).norm(dim=-1)
    log(f"  {scene}: facet scale {scale:.4f} m; centroid-gap quantiles "
        + "/".join(f"{float(q):.3f}" for q in torch.quantile(
            d_cent[torch.randperm(d_cent.numel(), device=device)[:200_000]].float(),
            torch.tensor([0.25, 0.5, 0.75, 0.95], device=device))))

    def purity_of(labels_np, n_groups):
        m = has_gt & (labels_np >= 0)
        if m.sum() == 0:
            return 0.0, 0
        g, c = labels_np[m], cell_lab[m]
        order = np.argsort(g)
        g, c = g[order], c[order]
        bnd = np.flatnonzero(np.diff(g)) + 1
        ps, tot = [], 0
        for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(g)]):
            s = c[a:b]
            if s.size >= 5:
                ps.append(np.bincount(s).max() / s.size); tot += 1
        return float(np.mean(ps)) if ps else 0.0, tot

    def components(thresh):
        keep = d_cent <= thresh * scale
        s, d = src[keep], dst[keep]
        labl = torch.arange(P, device=device)
        for _ in range(200):
            upd = labl.clone()
            upd.scatter_reduce_(0, s, labl[d], reduce="amin")
            upd.scatter_reduce_(0, d, labl[s], reduce="amin")
            upd = torch.minimum(upd, labl)
            if torch.equal(upd, labl):
                break
            labl = upd
        labl[~(vm & (wgt > 0))] = -1
        return labl

    res = {"scene": scene, "scale": scale, "rows": []}
    print(f"\n{'thresh':>8}{'groups':>10}{'purity':>9}{'kmeans@same':>13}{'kmeans purity':>15}")
    valid_idx = torch.nonzero(vm & (wgt > 0)).squeeze(1)
    for thr in (0.5, 1.0, 2.0, 4.0, 8.0):
        labl = components(thr)
        ln = labl.cpu().numpy()
        n_groups = int(len(np.unique(ln[ln >= 0])))
        pur, used = purity_of(ln, n_groups)
        # spatial k-means baseline at the SAME group count -- the arm that failed
        km_pur = float("nan")
        if 2 <= n_groups <= 200_000:
            lv = build_hierarchy(pos[valid_idx], wgt[valid_idx],
                                 branch=8, min_size=max(2, int(valid_idx.numel() / max(n_groups, 1))),
                                 max_levels=6)
            asg, nn_ = min(lv, key=lambda t: abs(t[1] - n_groups))
            full = np.full(P, -1, dtype=np.int64)
            full[valid_idx.cpu().numpy()] = asg.cpu().numpy()
            km_pur, _ = purity_of(full, nn_)
            km_n = nn_
        else:
            km_n = 0
        print(f"{thr:>8.1f}{n_groups:>10,}{pur:>9.3f}{km_n:>13,}{km_pur:>15.3f}")
        res["rows"].append({"thresh": thr, "n_groups": n_groups, "purity": pur,
                            "kmeans_n": int(km_n), "kmeans_purity": km_pur})
    print("\n  GATE: mask-centroid groups must be markedly PURER than spatial k-means at a matched")
    print("  group count. If not, aggregating over 'objects' is closed and the three ideas stay dead.")
    json.dump(res, open(out_json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["accumulate", "analyze"], required=True)
    p.add_argument("--scene", default="f9f95681fd")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.stage == "accumulate":
        stage_accumulate(a.scene)
    else:
        stage_analyze(a.scene, a.out or f"artifacts/scannetpp/maskcentroid_{a.scene}.json")


if __name__ == "__main__":
    main()
