"""LIFTING-STAGE RAY DIAGNOSTICS -- statistics that are NOT per-view reweighting.

One export pass per scene (same kernel the real lift uses,
`rasterize.py::export_operator_kernel` via `PowerfoamScene.export_feature_operator`),
accumulating per-CELL and per-RAY statistics that the retained
`AccumulatedFeatureStats` cannot express because it has already reduced over rays:

  RAY COUNT / COVERAGE
    n_rays[c]     number of (pixel, view) nonzeros depositing into cell c  (nnz of column c)
    n_views[c]    number of views in which c received ANY weight
    w_sum[c]      = support = (A^T 1)[c]                      (sanity-checked vs stats file)

  DEPTH ORDER (impossible for a Gaussian cloud: cells along a ray are DISJOINT and ORDERED)
    w_slot[c]     sum_r w * k          -> weighted mean traversal index
    w_trans[c]    sum_r w * T_k        -> weighted mean transmittance-on-arrival
    w_first[c]    sum_r w * 1[k == 0]
    w_firstsig[c] sum_r w * 1[k == first slot with w >= SIG]

  CONTAMINATION (never measured): a ray deposits into an ordered run of cells; do those
    cells belong to DIFFERENT objects?  Per ray we form the weight-weighted histogram over
    the GT class of each cell it hits (cell GT class = majority GT label of the points the
    power-cell owns), giving the ray's dominant class and its share.  Per cell we then
    accumulate how much of its incoming weight came from rays whose DOMINANT object is not
    this cell's own object.

  SAM-MASK PURITY per cell: within one view a cell is hit by many pixels; those pixels may
    lie in different SAM masks (mask boundary straddle, or occlusion).  Per (view, cell) we
    take the weighted share of the dominant SAM mask id and accumulate.

Writes one .npz per scene; `report.py`-style analysis lives in analyze_lifting_rays.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

DCACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"
SIG = 0.1          # "significant alpha" threshold for the first-significant-cell statistic
SHARE = 0.05       # a class counts as "crossed" by a ray if it holds >= 5% of the ray's weight
MH = 64


def cell_gt_labels(scene, num_prim, suffix="_ogl3", recon="nonfrozen"):
    """(P,) int64, 0 = no GT point owned, 1..C = majority GT class of the owned points.

    Uses exactly the decision cache built by build_decision_cache.py (power-cell assignment
    of the ScanNet GT cloud), so the labelling matches every mIoU number in this project.
    """
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{recon}{suffix}.pt",
                        map_location="cpu", weights_only=True)
    valid_idx = np.where(solved["valid_mask"].cpu().numpy())[0]
    c = torch.load(os.path.join(DCACHE, f"{scene}{suffix}.pt"), map_location="cpu", weights_only=False)
    raw = c["raw_labels"].numpy()
    prow = c["point_row"].numpy()
    names = c["all_names"]
    n2i = {n: i for i, n in enumerate(names)}
    present = set(np.unique(raw).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt = remap_gt_labels(raw, [i for i, _ in kept])          # 0 = ignore, 1..C
    nC = len(kept) + 1
    hist = np.zeros((valid_idx.shape[0], nC), dtype=np.int32)
    ok = (prow >= 0) & (gt > 0)
    np.add.at(hist, (prow[ok], gt[ok]), 1)
    maj = hist.argmax(1)
    maj[hist.sum(1) == 0] = 0
    full = np.zeros(num_prim, dtype=np.int64)
    full[valid_idx] = maj
    return full, nC, valid_idx, [n for _, n in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-views", type=int, default=0)
    # A8: primitives-per-ray must be measured on BOTH reconstructions. `truefrozen` is one cell per
    # GT vertex, which on ScanNet is EXACTLY the 3DGS frozen count (81,369 / 109,380 / 67,984 per
    # scene), so it is the only budget-matched foam-vs-Gaussian comparison we have. `nonfrozen` is
    # 3x that and is NOT matched to the 3DGS unfrozen arm (244k vs 2.6M), so its ray statistics
    # describe foam at its own density, not a like-for-like contrast.
    ap.add_argument("--recon", default="nonfrozen", choices=("nonfrozen", "truefrozen"))
    args_cli = ap.parse_args()
    scene = args_cli.scene
    device = "cuda"
    wp.init()

    cfg = f"output/scannet_{scene}_{args_cli.recon}/config.yaml"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", cfg])
    ckpt = cfg.replace("/config.yaml", "")
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt}/model.pt")
    cameras = dh.cameras
    P = model.points.shape[0]

    images_dir = Path(args.data_path) / args.scene / "images"
    image_names = sorted(p.stem for p in images_dir.iterdir())
    assert len(image_names) == len(cameras), (len(image_names), len(cameras))
    feat_dir = Path(args.data_path) / args.scene / "openclip_features_sam_l3"

    gtlab_np, nC, valid_idx, class_names = cell_gt_labels(scene, P, recon=args_cli.recon)
    gtlab = torch.from_numpy(gtlab_np).to(device)
    print(f"[rays] {scene} P={P} views={len(cameras)} classes={nC-1} "
          f"cells_with_gt={(gtlab_np>0).sum()}", flush=True)

    z = lambda: torch.zeros(P, device=device, dtype=torch.float64)
    acc = {k: z() for k in ["n_rays", "n_views", "w_sum", "w_slot", "w_trans", "w_first",
                            "w_firstsig", "w_clean", "w_contam", "w_labelled",
                            "sam_top", "sam_tot", "sam_groups", "w_sq", "w_front05",
                            "n_rays_firstsig"]}
    # ray-side histograms
    ray_ncls = torch.zeros(8, device=device, dtype=torch.float64)     # #distinct classes >=SHARE
    ray_domshare_hist = torch.zeros(20, device=device, dtype=torch.float64)
    ray_nnz_hist = torch.zeros(MH + 1, device=device, dtype=torch.float64)
    ray_nlab_hist = torch.zeros(16, device=device, dtype=torch.float64)
    slot_w_hist = torch.zeros(MH, device=device, dtype=torch.float64)
    n_rays_total = 0.0

    views = list(range(len(cameras)))
    if args_cli.max_views:
        views = views[:args_cli.max_views]
    t0 = time.time()
    for vi in views:
        cam = cameras[vi]
        H, W = cam.height, cam.width
        NP = H * W
        out_col, out_val, slot_counter, _, _ = model.export_feature_operator(
            cam, transmittance_threshold=1e-3, max_intersections=1024, max_hits_per_pixel=MH)
        vmat = out_val.view(NP, MH)
        cmat = out_col.view(NP, MH).long().clamp_(0, P - 1)
        slots = slot_counter.clamp(max=MH)
        karange = torch.arange(MH, device=device)
        keep = karange[None, :] < slots[:, None]
        vmat = vmat * keep                                   # unused slots -> 0
        trans_before = 1.0 - (vmat.cumsum(1) - vmat)
        # first slot with significant alpha
        sig = vmat >= SIG
        has_sig = sig.any(1)
        first_sig = torch.where(has_sig, sig.float().argmax(1), torch.full_like(slots, -1))
        is_firstsig = keep & (karange[None, :] == first_sig[:, None])

        rowidx = torch.arange(NP, device=device)[:, None].expand(NP, MH)

        c = cmat[keep]
        v = vmat[keep].double()
        r = rowidx[keep]
        k = karange[None, :].expand(NP, MH)[keep]
        tb = trans_before[keep].double()

        acc["n_rays"].index_add_(0, c, torch.ones_like(v))
        acc["w_sum"].index_add_(0, c, v)
        acc["w_sq"].index_add_(0, c, v * v)
        acc["w_slot"].index_add_(0, c, v * k.double())
        acc["w_trans"].index_add_(0, c, v * tb)
        acc["w_first"].index_add_(0, c, v * (k == 0).double())
        fs = is_firstsig[keep].double()
        acc["w_firstsig"].index_add_(0, c, v * fs)
        acc["n_rays_firstsig"].index_add_(0, c, fs)
        acc["w_front05"].index_add_(0, c, v * (tb >= 0.5).double())
        touched = torch.zeros(P, device=device, dtype=torch.bool)
        touched[c] = True
        acc["n_views"] += touched.double()

        slot_w_hist.index_add_(0, k, v)
        rn = keep.sum(1)
        ray_nnz_hist.index_add_(0, rn, torch.ones_like(rn, dtype=torch.float64))

        # ---- GT-class contamination along the ray ----
        lab = gtlab[c]                                        # 0..C
        Hh = torch.zeros(NP * nC, device=device, dtype=torch.float64)
        Hh.index_add_(0, r * nC + lab, v)
        Hh = Hh.view(NP, nC)
        lab_w = Hh[:, 1:].sum(1)
        top_w, top_c = Hh[:, 1:].max(1)
        dom = top_c + 1
        has_lab = lab_w > 0
        dom_share = torch.zeros(NP, device=device, dtype=torch.float64)
        dom_share[has_lab] = top_w[has_lab] / lab_w[has_lab]
        ncls = ((Hh[:, 1:] >= SHARE * lab_w[:, None]) & (Hh[:, 1:] > 0)).sum(1)
        ray_ncls.index_add_(0, ncls[has_lab].clamp(max=7), torch.ones(int(has_lab.sum()), device=device, dtype=torch.float64))
        ray_domshare_hist.index_add_(0, (dom_share[has_lab] * 19.999).long(),
                                     torch.ones(int(has_lab.sum()), device=device, dtype=torch.float64))
        n_rays_total += float(has_lab.sum())

        labelled = lab > 0
        # How many GT-LABELLED cells does a ray actually touch?  Without this the
        # "97.5% of rays cross one class" number is unreadable: a ray that touches ONE
        # labelled cell is single-class by construction, not by cleanliness.
        nlab = torch.zeros(NP, device=device, dtype=torch.float64)
        nlab.index_add_(0, r, labelled.double())
        ray_nlab_hist.index_add_(0, nlab.long().clamp(max=15),
                                 torch.ones(NP, device=device, dtype=torch.float64))
        agree = (dom[r] == lab) & labelled
        acc["w_labelled"].index_add_(0, c, v * labelled.double())
        acc["w_clean"].index_add_(0, c, v * agree.double())
        acc["w_contam"].index_add_(0, c, v * (labelled & ~agree).double())

        # ---- per-(view, cell) SAM mask purity ----
        s = np.load(feat_dir / f"{image_names[vi]}_s.npy")
        seg = torch.from_numpy(s.astype(np.int64)).to(device)
        if seg.ndim == 3:
            seg = seg[0]
        if seg.shape != (H, W):
            seg = torch.nn.functional.interpolate(
                seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
        seg = (seg + 1).reshape(-1)                      # 0 = background/no mask
        M = int(seg.max().item()) + 1
        key = c * M + seg[r]
        order = torch.argsort(key)
        ks, vs = key[order], v[order]
        uniq, inv, cnt = torch.unique_consecutive(ks, return_inverse=True, return_counts=True)
        gw = torch.zeros(uniq.shape[0], device=device, dtype=torch.float64)
        gw.index_add_(0, inv, vs)
        gcell = uniq // M
        top = torch.zeros(P, device=device, dtype=torch.float64)
        top.scatter_reduce_(0, gcell, gw, reduce="amax", include_self=True)
        tot = torch.zeros(P, device=device, dtype=torch.float64)
        tot.index_add_(0, gcell, gw)
        ngrp = torch.zeros(P, device=device, dtype=torch.float64)
        ngrp.index_add_(0, gcell, torch.ones_like(gw))
        acc["sam_top"] += top
        acc["sam_tot"] += tot
        acc["sam_groups"] += ngrp

        del out_col, out_val, vmat, cmat, keep, trans_before, Hh
        if vi % 10 == 0:
            print(f"  view {vi}/{len(views)} {time.time()-t0:.1f}s", flush=True)

    out = {k: v.cpu().numpy() for k, v in acc.items()}
    out["gtlab"] = gtlab_np
    out["valid_idx"] = valid_idx
    out["ray_ncls"] = ray_ncls.cpu().numpy()
    out["ray_domshare_hist"] = ray_domshare_hist.cpu().numpy()
    out["ray_nnz_hist"] = ray_nnz_hist.cpu().numpy()
    out["ray_nlab_hist"] = ray_nlab_hist.cpu().numpy()
    out["slot_w_hist"] = slot_w_hist.cpu().numpy()
    out["n_rays_total"] = np.array([n_rays_total])
    out["n_views_used"] = np.array([len(views)])
    out["class_names"] = np.array(class_names)
    np.savez_compressed(args_cli.out, **out)
    print(f"[rays] wrote {args_cli.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
