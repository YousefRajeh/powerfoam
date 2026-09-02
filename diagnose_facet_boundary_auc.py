"""Is the FACET FEATURE CUE actually weak, or was that a Cech artifact?

The vault closed the whole boundary-based branch on "facet cues are too weak (feature AUC
0.65, mask AUC 0.688)".  That AUC was measured on the CECH graph, in which 62% of edges
are FABRICATED -- they join cells that do NOT share a boundary.  A fabricated edge has an
essentially random GT-boundary label, so it injects pure noise into the AUC and drags any
real separability toward 0.5.  This re-measures the cue on the TRUE power-diagram facet
dual and on the Cech graph, paired, on the same scene and same features.

Edge label: the two endpoint cells' MAJORITY GT class differs  ->  positive (boundary).
Score:      1 - cos(f_i, f_j).
Only edges where BOTH endpoints own >= --min-pts GT points are used.

FALSIFIER: if true-facet AUC is within ~0.02 of Cech AUC, the closed branch stays closed
and the weakness is real, not a graph artifact.
"""
import argparse, json, os, sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, remap_gt_labels,
                                       load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import csr_to_edges


def auc(score, pos):
    """Mann-Whitney AUC, tie-corrected, via rank sums."""
    order = torch.argsort(score)
    s = score[order]; y = pos[order].double()
    n = s.numel()
    rank = torch.arange(1, n + 1, device=s.device, dtype=torch.float64)
    # average ranks over ties
    uniq, inv, cnt = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    csum = torch.cumsum(cnt, 0).double()
    avg = (csum - cnt.double() / 2.0 + 0.5)
    rank = avg[inv]
    n1 = y.sum(); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(((rank * y).sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--suffix", default="_ogl3")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--min-pts", type=int, default=1)
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
    positions = torch.from_numpy(centers).to(device).float()

    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{a.gt_root}/{SCENES[a.scene]}/{a.scene}", "segment20")
    assigned = assign_points_to_power_cells(gt_points, centers, radii,
                                            valid=valid_mask, k=64)
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set]
            if name_to_id[n] in present]
    gt = remap_gt_labels(raw_labels, [i for i, _ in kept])   # 0 ignore, 1..K
    K = len(kept)

    ok = (assigned >= 0) & (gt > 0)
    ai = torch.from_numpy(assigned[ok]).to(device).long()
    gl = torch.from_numpy(gt[ok] - 1).to(device).long()
    hist = torch.zeros(P, K, device=device)
    hist.index_put_((ai, gl), torch.ones_like(ai, dtype=torch.float), accumulate=True)
    npts = hist.sum(1)
    cell_lab = hist.argmax(1)
    has = npts >= a.min_pts
    print(f"[{a.scene}] cells with >= {a.min_pts} GT pts: {int(has.sum())} / {P}")

    out = {"scene": a.scene, "class_set": a.class_set, "graphs": {}}
    for g in ("true_facet", "cech"):
        path = (f"{art}/adjacency_true_facet.pt" if g == "true_facet"
                else f"{art}/adjacency_{a.variant}.pt")
        adj = torch.load(path, map_location=device, weights_only=True)
        src, dst, _ = csr_to_edges(adj["adjacent"].to(device).long(),
                                   adj["offsets"].to(device).long(), P, device)
        m = (src < dst) & has[src] & has[dst] & vm_t[src] & vm_t[dst]
        s, d = src[m], dst[m]
        pos = (cell_lab[s] != cell_lab[d])
        cosv = (unit[s] * unit[d]).sum(-1)
        A = auc(1.0 - cosv, pos)
        # distance cue as a control (what a purely geometric method would get)
        dist = (positions[s] - positions[d]).norm(dim=-1)
        Ad = auc(dist, pos)
        out["graphs"][g] = {
            "edges_scored": int(m.sum()), "boundary_frac": float(pos.float().mean()),
            "auc_feature_cosine": A, "auc_center_distance": Ad,
            "mean_cos_same": float(cosv[~pos].mean()),
            "mean_cos_diff": float(cosv[pos].mean())}
        print(f"  {g:<11} edges={int(m.sum()):>9}  boundary={float(pos.float().mean())*100:5.2f}%  "
              f"AUC(1-cos)={A:.4f}  AUC(dist)={Ad:.4f}  "
              f"cos same={float(cosv[~pos].mean()):.4f} diff={float(cosv[pos].mean()):.4f}")
        del adj, src, dst
        torch.cuda.empty_cache()

    with open(f"{art}/facet_boundary_auc_{a.class_set}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["graphs"], indent=2))


if __name__ == "__main__":
    main()
