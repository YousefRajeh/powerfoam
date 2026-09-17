"""Cache the primitive <- region linkage once, so upstream experiments cost seconds, not minutes.

Every upstream experiment (different text prompts, different crop treatments, ensembles across
extractions) changes only the REGION FEATURES or the TEXT EMBEDDINGS. None of them changes
which primitive sees which region, or what the GT says. That linkage is the expensive part --
it needs the model loaded and every view rasterised -- so it is computed once here and reused.

What is cached, per (scene, recon, feature dir):

  W          sparse (primitive, global region) -> accumulated alpha weight. `global region` is
             this view's region index offset into a per-scene concatenation, so a region in
             view 7 and a region in view 8 are distinct entries even at the same local index.
  F          (total_regions, 512) the concatenated region features, in that same global order.
  region_gt  per global region: majority GT class, pixel count, purity. Lets region-level CLIP
             accuracy be recomputed for any text embedding with one matmul.
  prim_gt    per primitive: majority GT class, for primitive-level checks.
  assigned / gt_lab / kept  everything the reported point protocol needs to score.

With this, a score field is just  S = W @ (F @ T^T)  -- and every arm below becomes a matmul.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F
import configargparse, warp as wp

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, remap_gt_labels,
                                       apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import SCENES, GT_ROOT


def build(scene, recon, class_set, feat_dir, alpha_eps, opacity_threshold, gt_opacity_mask,
          levels=None, dev="cuda"):
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device=dev)
    m.load_pt(f"{ck}/model.pt"); m.update_vis_cache()

    sol = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{recon}_ogl3.pt",
                     map_location="cpu", weights_only=True)
    valid = sol["valid_mask"].numpy()
    Xlift = sol["primitive_features"].float().numpy()
    centers = m.points.detach().cpu().numpy()
    radii = m.get_radii().detach().cpu().numpy()
    density = m.get_density().detach().float().cpu().numpy().reshape(-1)

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    if gt_opacity_mask:
        alpha_p = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
        gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, alpha_p, opacity_threshold, scene)

    P = centers.shape[0]
    v = np.zeros((P, C + 1), np.int32)
    okm = (assigned >= 0) & (gt_lab > 0)
    np.add.at(v, (assigned[okm], gt_lab[okm]), 1)
    prim_gt = v.argmax(1).astype(np.int16); prim_gt[v.max(1) == 0] = 0
    prim_gt_t = torch.from_numpy(prim_gt.astype(np.int64)).to(dev)

    names = sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))
    fdir = f"data/scannet/{scene}_colmap/{feat_dir}"
    c = m._vis_cache
    pts, rad = c["points"], c["radii"]
    rgb = torch.zeros(P, m.args.num_texel_sites, 3, device=dev)

    rows, cols, wts = [], [], []
    feats, rgt, rnpx, rpur = [], [], [], []
    off = 0
    nviews = 0
    for vi, nm in enumerate(names):
        stem = os.path.splitext(nm)[0]
        fp, sp = f"{fdir}/{stem}_f.npy", f"{fdir}/{stem}_s.npy"
        if not (os.path.exists(fp) and os.path.exists(sp)):
            continue
        nviews += 1
        with torch.no_grad():
            out = m.rasterizer.visualize(dh.cameras[vi], pts, rad, c["density"], c["normals"],
                                         c["texel_sites"], rgb, c["texel_height"],
                                         c["adjacency"], c["adjacency_offsets"])
        alpha, fpi = out[3], out[7].long()
        H, W = alpha.shape[-2], alpha.shape[-1]
        alpha, fpi = alpha.reshape(H, W), fpi.reshape(H, W)
        fvv = np.load(fp).astype(np.float32)
        N = fvv.shape[0]
        feats.append(fvv)
        segs = np.load(sp).astype(np.int64)
        if segs.ndim == 2:
            segs = segs[None]
        want = range(segs.shape[0]) if levels is None else [L for L in levels if L < segs.shape[0]]

        hist_v = torch.zeros(N, C + 1, device=dev)
        base = (alpha >= alpha_eps) & (fpi >= 0)
        for L in want:
            seg = F.interpolate(torch.from_numpy(segs[L])[None, None].float(), size=(H, W),
                                mode="nearest")[0, 0].to(dev).long()
            keep = base & (seg >= 0)
            if not bool(keep.any()):
                continue
            j, s, w = fpi[keep], seg[keep], alpha[keep]
            # collapse duplicate (primitive, region) pixels within this view before storing
            key = j * N + s
            uk, inv = torch.unique(key, return_inverse=True)
            acc = torch.zeros(uk.shape[0], device=dev).index_add_(0, inv, w)
            rows.append((uk // N).cpu().numpy().astype(np.int32))
            cols.append(((uk % N) + off).cpu().numpy().astype(np.int32))
            wts.append(acc.cpu().numpy().astype(np.float32))
            g = prim_gt_t[j]
            hm = g > 0
            if bool(hm.any()):
                hist_v.index_put_((s[hm], g[hm]),
                                  torch.ones(int(hm.sum()), device=dev), accumulate=True)
        tot = hist_v.sum(1)
        mx = hist_v.max(1)
        rgt.append(torch.where(tot > 0, mx.indices, torch.zeros_like(mx.indices)
                               ).cpu().numpy().astype(np.int16))
        rnpx.append(tot.cpu().numpy().astype(np.float32))
        rpur.append((mx.values / tot.clamp_min(1)).cpu().numpy().astype(np.float32))
        off += N

    return dict(
        rows=np.concatenate(rows), cols=np.concatenate(cols), wts=np.concatenate(wts),
        F=np.concatenate(feats), region_gt=np.concatenate(rgt),
        region_npix=np.concatenate(rnpx), region_purity=np.concatenate(rpur),
        prim_gt=prim_gt, assigned=assigned.astype(np.int32), gt_lab=gt_lab.astype(np.int16),
        kept=np.array(kept), P=np.int64(P), nviews=np.int64(nviews),
        lifted=Xlift.astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--feat-dirs", default="openclip_features_sam_l3")
    ap.add_argument("--levels", default=None, help="comma-separated level indices; default all")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha-eps", type=float, default=0.05)
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")] if a.levels else None
    wp.init()
    os.makedirs("artifacts/region_cache", exist_ok=True)
    for fd in a.feat_dirs.split(","):
        for sc in a.scenes.split(","):
            tag = f"{sc}_{a.recon}_{a.class_set}_{fd}" + (f"_L{a.levels}" if a.levels else "")
            out = f"artifacts/region_cache/{tag}.npz"
            if os.path.exists(out) and not a.force:
                print(f"[{tag}] cached"); continue
            if not os.path.isdir(f"data/scannet/{sc}_colmap/{fd}"):
                print(f"[{tag}] no such feature dir, skipping"); continue
            try:
                d = build(sc, a.recon, a.class_set, fd, a.alpha_eps, a.opacity_threshold,
                          not a.no_gt_opacity_mask, levels)
            except Exception as e:
                print(f"[{tag}] SKIP {type(e).__name__}: {e}"); continue
            np.savez_compressed(out, **d)
            print(f"[{tag}] {d['nviews']} views, {len(d['F']):,} regions, "
                  f"{len(d['rows']):,} links -> {out}")


if __name__ == "__main__":
    main()
