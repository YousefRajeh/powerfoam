"""N4: mask-boundary facet counting.

Foam-only boundary detector that bypasses the measured over-smoothness of lifted
features (facet feature-cosine is a weak boundary predictor, AUC 0.65). For each view we
assign every cell its dominant SAM l-mask id via the exact rendering operator, then for
each FACET count views where the two adjacent cells fall in DIFFERENT masks:

    boundary_score(facet) = n_diff / (n_diff + n_same)

SAM mask boundaries are sharp by construction, so multi-view agreement about "these two
cells are in different objects" should separate true object boundaries far better than
blended CLIP features do.

Gate before any downstream use: this score must beat the 0.65 feature AUC by a wide
margin (target >0.85) at predicting GT class boundaries across owner-owner facets.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, remap_gt_labels,
    load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES

MAX_HITS = 64


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--min-pixel-weight", type=float, default=0.3)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scene = args.scene
    split = SCENES[scene]
    ckpt_dir = f"output/scannet_{scene}_{args.variant}"

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt_dir}/model.pt")
    cameras = dh.cameras
    images_dir = Path(cargs.data_path) / cargs.scene / "images"
    image_names = sorted(q.stem for q in images_dir.iterdir())
    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()
    P = centers.shape[0]

    adj = torch.load(f"artifacts/scannet/{scene}/adjacency_{args.variant}.pt",
                     map_location=device, weights_only=True)
    adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    E = adjacent.numel()
    src = torch.repeat_interleave(torch.arange(P, device=device), offsets[1:] - offsets[:-1])

    n_same = torch.zeros(E, device=device)
    n_diff = torch.zeros(E, device=device)
    feature_dir = Path(f"artifacts/scannet/{scene}/openclip_features_sam")

    for view_id, camera in enumerate(cameras):
        spath = feature_dir / f"{image_names[view_id]}_s.npy"
        if not spath.exists():
            continue
        H, W = camera.height, camera.width
        num_pixels = H * W
        out_col, out_val, slot_counter, _, _ = model.export_feature_operator(
            camera, transmittance_threshold=1e-3, max_intersections=1024,
            max_hits_per_pixel=MAX_HITS)
        slots_used = slot_counter.clamp(max=MAX_HITS)
        keep = (torch.arange(MAX_HITS, device=device)[None, :] < slots_used[:, None]).reshape(-1)
        cols = out_col.reshape(-1)[keep].long()
        vals = out_val.reshape(-1)[keep].float()
        pix = torch.arange(num_pixels, device=device).repeat_interleave(MAX_HITS)[keep]
        del out_col, out_val, slot_counter

        seg = torch.from_numpy(np.load(spath)).to(device).long()
        if seg.dim() == 3:
            seg = seg[3]  # l-level only, matching the feature pipeline
        seg = seg.reshape(-1)
        M = int(seg.max().item()) + 2  # +1 for shift, +1 for -1 -> 0

        # per-cell dominant mask this view: argmax over masks of summed rendering weight
        m_of_pix = seg[pix] + 1  # 0 = no mask
        ok = m_of_pix > 0
        c_ok, m_ok, v_ok = cols[ok], m_of_pix[ok], vals[ok]
        key = c_ok * M + m_ok
        order = torch.argsort(key)
        key_s, val_s = key[order], v_ok[order]
        uniq, inv = torch.unique_consecutive(key_s, return_inverse=True)
        seg_sum = torch.zeros(uniq.numel(), device=device)
        seg_sum.index_add_(0, inv, val_s)
        u_cell = (uniq // M).long()
        u_mask = (uniq % M).long()
        best = torch.zeros(P, device=device)
        best.scatter_reduce_(0, u_cell, seg_sum, reduce="amax")
        win = seg_sum >= best[u_cell] - 1e-12
        dom = torch.zeros(P, dtype=torch.long, device=device)
        dom[u_cell[win]] = u_mask[win]
        tot = torch.zeros(P, device=device)
        tot.index_add_(0, cols, vals)
        dom[tot < args.min_pixel_weight] = 0

        both = (dom[src] > 0) & (dom[adjacent] > 0)
        same = both & (dom[src] == dom[adjacent])
        n_same += same.float()
        n_diff += (both & ~same).float()
        del seg, dom, cols, vals, pix

    total = n_same + n_diff
    boundary = torch.where(total > 0, n_diff / total.clamp_min(1), torch.zeros_like(total))
    print(f"[{scene}] facets with >=1 co-observation: {(total > 0).float().mean()*100:.1f}% of {E}")

    # --- AUC gate vs GT class boundaries (owner-owner facets) ---
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(f"{args.gt_root}/{split}/{scene}", "segment20")
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}_l3.pt",
                        map_location=device, weights_only=True)
    vm = solved["valid_mask"].cpu().numpy()
    feats = solved["primitive_features"].to(device).float()
    vi = torch.where(torch.from_numpy(vm).to(device))[0]
    unit = torch.zeros_like(feats)
    unit[vi] = F.normalize(feats[vi], dim=-1)
    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)

    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    tids = [i for i, _ in kept]
    gt19 = remap_gt_labels(raw_labels, tids)
    owned = assigned >= 0
    sel = gt19[owned] > 0
    cell_of_pt = assigned[owned][sel]
    lbl_of_pt = gt19[owned][sel]
    K = len(tids)
    vote = np.zeros((P, K + 1), dtype=np.int64)
    np.add.at(vote, (cell_of_pt, lbl_of_pt), 1)
    cell_label = vote.argmax(1)
    owner = cell_label > 0

    s_np, d_np = src.cpu().numpy(), adjacent.cpu().numpy()
    m = owner[s_np] & owner[d_np] & (total.cpu().numpy() > 0)
    same_lbl = cell_label[s_np[m]] == cell_label[d_np[m]]
    bscore = boundary.cpu().numpy()[m]
    fsim = np.einsum("ec,ec->e", unit.cpu().numpy()[s_np[m]], unit.cpu().numpy()[d_np[m]])

    def auc_same(score):  # higher score should mean SAME label
        o = np.argsort(score)
        lab = same_lbl[o]
        npos, nneg = lab.sum(), (~lab).sum()
        r = np.arange(1, len(lab) + 1)
        return (r[lab].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)

    auc_mask = auc_same(-bscore)   # low boundary score => same label
    auc_feat = auc_same(fsim)
    print(f"[{scene}] AUC(same-label): mask-boundary={auc_mask:.3f}  feature-cosine={auc_feat:.3f}  "
          f"(n={m.sum()} owner-owner facets, {same_lbl.mean()*100:.0f}% same-label)")
    print(f"  median boundary_score: same={np.median(bscore[same_lbl]):.3f}  diff={np.median(bscore[~same_lbl]):.3f}")
    print(f"  GATE (target >0.85 and clearly > feature AUC): "
          f"{'PASS' if auc_mask > 0.85 and auc_mask > auc_feat + 0.1 else 'FAIL'}")

    if args.output:
        torch.save({"boundary": boundary.cpu(), "n_same": n_same.cpu(), "n_diff": n_diff.cpu(),
                    "auc_mask": float(auc_mask), "auc_feat": float(auc_feat)}, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
