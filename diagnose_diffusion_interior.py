"""Does simplex diffusion fix INTERIOR errors, or only boundary ones?

83% of our errors are INTERIOR cells of coherent regions.  A 1-hop copy/vote (NormLift
mode-voting, Potts, any boundary-only method) caps at the remaining 17%.  Multi-step
diffusion has no such cap in principle: information travels ~sqrt(iters) hops.  This
measures it directly.

For every GT point we have base pred, diffused pred, and GT.  We classify the owning CELL
as INTERIOR iff all of its true-facet neighbours carry the same BASE label as it does
(i.e. the cell sits inside a label-coherent block and no 1-hop method can move it), and
BOUNDARY otherwise.  Then we count fixes/breaks in each stratum.
"""
import argparse, json, os, sys, time
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges, diffuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--scale", type=float, default=1000.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--iters", type=int, default=60)
    a = p.parse_args()

    enable_determinism()
    device = "cuda"
    art = f"artifacts/scannet/{a.scene}"
    centers, radii = load_points_radii(f"output/scannet_{a.scene}_{a.variant}")
    P = centers.shape[0]
    solved = torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(valid_mask).to(device)
    unit = torch.zeros_like(feats); unit[vm_t] = F.normalize(feats[vm_t], dim=-1)
    del feats, solved

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{SCENES[a.scene]}/{a.scene}", "segment20")
    assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                            valid=valid_mask, k=64)
    owned = assigned >= 0

    adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device,
                     weights_only=True)
    src, dst, deg = csr_to_edges(adj["adjacent"].to(device).long(),
                                 adj["offsets"].to(device).long(), P, device)
    keep = vm_t[src] & vm_t[dst]
    src, dst = src[keep], dst[keep]
    deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(
        0, src, torch.ones_like(src))

    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set]
            if name_to_id[n] in present]
    tids = [i for i, _ in kept]
    gt = remap_gt_labels(raw_labels, tids)                 # 0 = ignore, 1..K
    text = embed_class_names([n for _, n in kept], device)
    cos = unit @ text.T
    p0 = torch.softmax(a.scale * cos, dim=-1); p0[~vm_t] = 0.0
    base_cell = cos.argmax(-1)
    diff_cell = diffuse(p0, src, dst, deg, a.alpha, a.iters).argmax(-1)

    # INTERIOR = every facet neighbour agrees with the cell's BASE label
    disagree = torch.zeros(P, dtype=torch.long, device=device).index_add_(
        0, src, (base_cell[src] != base_cell[dst]).long())
    interior = (disagree == 0) & (deg > 0)

    bc = base_cell.cpu().numpy(); dc = diff_cell.cpu().numpy()
    intr = interior.cpu().numpy()
    m = owned & (gt > 0)
    ai = assigned[m]; g = gt[m] - 1
    b_ok = bc[ai] == g; d_ok = dc[ai] == g; is_int = intr[ai]

    out = {"scene": a.scene, "class_set": a.class_set,
           "n_points": int(m.sum()),
           "cells_interior_frac": float(intr[valid_mask].mean()),
           "base_acc": float(b_ok.mean()), "diff_acc": float(d_ok.mean())}
    for name, sel in (("interior", is_int), ("boundary", ~is_int)):
        n = int(sel.sum())
        err = int((~b_ok & sel).sum())
        fixed = int((~b_ok & d_ok & sel).sum())
        broke = int((b_ok & ~d_ok & sel).sum())
        out[name] = {"points": n, "share_of_points": n / max(int(m.sum()), 1),
                     "base_errors": err, "share_of_all_errors": err / max(int((~b_ok).sum()), 1),
                     "fixed": fixed, "fixed_frac_of_stratum_errors": fixed / max(err, 1),
                     "broke": broke, "net": fixed - broke}
    print(json.dumps(out, indent=2))
    with open(f"{art}/diffusion_interior_{a.class_set}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
