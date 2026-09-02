"""Sufficiency test for per-facet features -- without touching the renderer.

CAPACITY was already shown (`run_facet_subdivision_gate.py`): inside cells holding more than one GT
class, splitting by facet raises purity 0.75 -> 0.85 (+10.6) using ~3.9 sub-cells per cell. So the
facet partition ALIGNS with the label boundary.

That is necessary but not sufficient: the 2D observations attributed to each facet-side must also
DIFFER in CLIP space, or a per-facet solve would just replicate one feature across all facets.

THE PROXY THAT AVOIDS THE KERNEL CHANGE. A GT point on facet-side j lies spatially adjacent to
neighbour cell j, and that neighbour ALREADY carries its own independently-lifted feature. If the
neighbour's feature classifies the point better than the owner cell's does, then the observations on
that side of the facet genuinely differ -- which is exactly what a per-facet solve would exploit.
Full per-facet attribution needs `export_feature_operator` to return entry/exit facet IDs; this needs
nothing new.

It is also directly deployable if it wins: "for a point near a facet, prefer the feature of the cell
across that facet" is a reassignment rule, not a new representation.

ARMS (scored on the points inside mixed cells only, which is where any of this can matter):
  own          the owner cell's feature                      -- current behaviour
  neighbour    the feature of the cell across the nearest facet
  blend        distance-weighted mix of the two, w by relative power distance
  best_oracle  min(own, neighbour) error -- the ceiling of any own/neighbour choice rule
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
from build_true_facet_graph import load_points_radii
from point_cloud_query import assign_points_to_power_cells
from run_overnight import RECON, LAM
from run_spp_eval import benchmark_map, load_gt, coverage_filter
from run_facet_subdivision_gate import second_nearest_power

SCENES = ["f9f95681fd", "c50d2d1d42", "0d2ee665be", "3864514494"]


def main():
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    res = {}
    print(f"{'scene':<13}{'pts':>9}{'own':>9}{'neigh':>9}{'blend':>9}{'oracle':>9}")
    for scene in SCENES:
        ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
        sp = f"artifacts/scannetpp/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        if not (os.path.isdir(ck) and os.path.exists(sp)):
            continue
        centers, radii = load_points_radii(ck)
        centers = np.asarray(centers, dtype=np.float64); radii = np.asarray(radii, dtype=np.float64)
        sv = torch.load(sp, map_location=device, weights_only=True)
        feats = sv["primitive_features"].to(device).float()
        vm = sv["valid_mask"].to(device); vmn = sv["valid_mask"].cpu().numpy()
        unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
        mu = F.normalize(unit[vm].mean(0, keepdim=True), dim=-1)
        cen = unit.clone(); cen[vm] = F.normalize(unit[vm] - LAM * mu, dim=-1)  # same centring as the stack
        del feats, sv

        gt, lab0, _ = load_gt(scene, top, r2b)
        a = assign_points_to_power_cells(gt, centers, radii, valid=vmn, k=64)
        keepc, _, _ = coverage_filter(gt, a, centers, vmn, 20.0)
        ok = (a >= 0) & keepc & (lab0 >= 0)
        pts, own, lab = gt[ok].astype(np.float64), a[ok], lab0[ok]

        order = np.argsort(own, kind="stable"); o_s, l_s = own[order], lab[order]
        bnd = np.flatnonzero(np.diff(o_s)) + 1
        mixed = set()
        for s_, e_ in zip(np.r_[0, bnd], np.r_[bnd, len(o_s)]):
            if e_ - s_ >= 2 and np.unique(l_s[s_:e_]).size > 1:
                mixed.add(int(o_s[s_]))
        sel = np.fromiter((c in mixed for c in own), bool, len(own))
        if sel.sum() < 50:
            print(f"{scene:<13}{'(too few)':>45}"); continue
        p, o, y = pts[sel], own[sel], lab[sel]
        nb = second_nearest_power(p, centers, radii, o)
        valid_nb = vmn[nb]
        p, o, y, nb = p[valid_nb], o[valid_nb], y[valid_nb], nb[valid_nb]

        pres = sorted(set(np.unique(y).tolist()) & set(range(100)))
        names = [top[:100][i] for i in pres]
        txt = embed_class_names(names, device)
        remap = {c: i for i, c in enumerate(pres)}
        yy = np.array([remap[v] for v in y])

        co = (cen[torch.from_numpy(o).to(device)] @ txt.T)          # owner-cell logits
        cn = (cen[torch.from_numpy(nb).to(device)] @ txt.T)         # neighbour-cell logits
        # relative power distance to each centre -> blend weight
        d_o = ((p - centers[o]) ** 2).sum(-1) - radii[o] ** 2
        d_n = ((p - centers[nb]) ** 2).sum(-1) - radii[nb] ** 2
        w = torch.from_numpy(
            (d_n / np.maximum(d_o + d_n, 1e-12))).to(device).float().unsqueeze(1)
        cb = w * co + (1 - w) * cn

        yt = torch.from_numpy(yy).to(device)
        acc_o = (co.argmax(-1) == yt).float().mean().item()
        acc_n = (cn.argmax(-1) == yt).float().mean().item()
        acc_b = (cb.argmax(-1) == yt).float().mean().item()
        acc_or = ((co.argmax(-1) == yt) | (cn.argmax(-1) == yt)).float().mean().item()
        res[scene] = {"n": int(len(yy)), "own": acc_o, "neighbour": acc_n,
                      "blend": acc_b, "oracle": acc_or}
        print(f"{scene:<13}{len(yy):>9,}{acc_o:>9.4f}{acc_n:>9.4f}{acc_b:>9.4f}{acc_or:>9.4f}",
              flush=True)
        del unit, cen, txt, co, cn, cb
        torch.cuda.empty_cache()
    if res:
        k = lambda f: np.mean([v[f] for v in res.values()])
        print(f"\nMEAN  own {k('own'):.4f} | neighbour {k('neighbour'):.4f} "
              f"| blend {k('blend'):.4f} | oracle {k('oracle'):.4f}")
        print(f"neighbour - own = {k('neighbour')-k('own'):+.4f}   "
              f"oracle headroom = {k('oracle')-k('own'):+.4f}")
    json.dump(res, open("artifacts/scannetpp/facet_sufficiency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
