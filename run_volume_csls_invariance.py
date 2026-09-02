"""Is the CSLS radius RECONSTRUCTION-INVARIANT? Counting cells says no; integrating volume says yes.

THE ARGUMENT. `r_K(t_c) = mean of the top-K cosines over CELLS` estimates the local density of the
feature distribution around class c, and implicitly treats the K cells as i.i.d. draws. They are
not: cells are a spatially smoothed field, so the effective sample size is far below K -- the same
over-counting NormLift's Kish `N_eff = (sum W)^2 / sum W^2` corrects for views.

Thresholded region-growing was the naive fix and it FAILED for a diagnosable reason: adjacent cells
agree at median cosine 0.996, so any threshold either keeps everything (one 632k-cell component of
700k) or shatters into 2-15 cell dust. The graph has no natural region scale.

THE WELL-POSED VERSION. The feature field is a function on SPACE. A class's crowding is a property
of that field, so it should be measured with the VOLUME measure, not the counting measure the
reconstruction happens to induce:

    counting:  r_K(t_c) = (1/K) sum_{i in topK} cos(f_i, t_c)
    volume  :  r_V(t_c) = sum_i V_i cos(f_i,t_c) 1[i in topV] / sum_i V_i 1[i in topV]

with `topV` the cells forming the top V-fraction of VOLUME rather than the top K by count. Foam can
do this exactly -- disjoint bounded partition, so V_i is well defined. A Gaussian mixture cannot:
overlapping, unbounded support, no partition of unity.

THE PROVABLE PROPERTY. Two reconstructions of the same scene at different point budgets must report
the SAME crowding for a class, because crowding is a property of the scene, not of the mesh
resolution. Refining the reconstruction splits cells: a region previously held by one cell of volume
V becomes n cells of volume V/n each. Under the counting measure that region's contribution grows
n-fold; under the volume measure it is unchanged. So r_V is invariant and r_K is not.

THE TEST. The matched-budget reconstruction is not lifted yet, so the discretisation change is
SIMULATED on a single reconstruction, which is the cleaner controlled experiment anyway: resample
the cell set with retention probability proportional to V_i^alpha. alpha=0 is uniform (no
discretisation bias); alpha>0 mimics a coarser partition that keeps large cells; alpha<0 mimics a
finer one that splits dense regions. A reconstruction-invariant statistic must be stable in alpha.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import embed_class_names
from build_true_facet_graph import load_points_radii
from run_lambda_derivation_eval import mc_cell_volumes
from run_overnight import RECON, LAM, CSLS_K
from run_spp_eval import benchmark_map

SCENE = "27dd4da69e"
MC = 24_000_000


def r_count(cos_v, k):
    return cos_v.topk(min(k, cos_v.shape[0]), dim=0).values.mean(0)


def r_volume(cos_v, vol, frac):
    """Top `frac` of VOLUME, averaged with volume weights."""
    C = cos_v.shape[1]
    out = torch.zeros(C, device=cos_v.device)
    target = frac * vol.sum()
    for c in range(C):
        order = torch.argsort(cos_v[:, c], descending=True)
        cv = torch.cumsum(vol[order], 0)
        n = int(torch.searchsorted(cv, target).item()) + 1
        sel = order[:n]
        w = vol[sel]
        out[c] = (cos_v[sel, c] * w).sum() / w.sum().clamp_min(1e-12)
    return out


def main():
    enable_determinism()
    device = "cuda"
    top, r2b = benchmark_map()
    ck = os.path.join(RECON, f"spp_pf_unfroz_{SCENE}")
    centers, radii = load_points_radii(ck)
    sv = torch.load(f"artifacts/scannetpp/{SCENE}/solved_geometric_median_nonfrozen_ogl3.pt",
                    map_location=device, weights_only=True)
    feats = sv["primitive_features"].to(device).float()
    vmn = sv["valid_mask"].cpu().numpy(); vm = sv["valid_mask"].to(device)
    u = torch.zeros_like(feats); u[vm] = F.normalize(feats[vm], dim=-1)
    mu = F.normalize(u[vm].mean(0, keepdim=True), dim=-1)
    u[vm] = F.normalize(u[vm] - LAM * mu, dim=-1)
    del feats, sv

    print(f"computing MC cell volumes ({MC:,} samples)...", flush=True)
    volc = mc_cell_volumes(centers, radii, vmn, MC, "cpu").numpy()
    hit = volc > 0
    print(f"  cells receiving >=1 sample: {hit.mean()*100:.1f}%  "
          f"volume concentration: top-10% of cells hold "
          f"{np.sort(volc)[::-1][:int(0.1*len(volc))].sum()/max(volc.sum(),1)*100:.1f}% of volume")

    names = top[:60]
    txt = embed_class_names(names, device)
    keep = vmn & hit
    cos_v = (u[torch.from_numpy(keep).to(device)] @ txt.T)
    vol = torch.from_numpy(volc[keep]).to(device).float()
    N = cos_v.shape[0]
    print(f"  {N:,} cells with volume, {len(names)} classes")

    rng = np.random.default_rng(0)
    base_c = r_count(cos_v, CSLS_K)
    base_v = r_volume(cos_v, vol, CSLS_K / N)
    res = {}
    print(f"\n{'alpha':>7}{'kept':>9}{'drift r_count':>16}{'drift r_volume':>16}")
    for alpha in (-0.5, -0.25, 0.0, 0.25, 0.5):
        w = volc[keep] ** alpha
        w = w / w.max()
        m = rng.random(N) < (0.5 * w / w.mean()).clip(0, 1)
        if m.sum() < 5000:
            continue
        mt = torch.from_numpy(m).to(device)
        cvs, vs = cos_v[mt], vol[mt]
        # same NOMINAL neighbourhood: K scaled to the retained count, V-fraction unchanged
        rc = r_count(cvs, max(1, int(CSLS_K * m.mean())))
        rv = r_volume(cvs, vs, CSLS_K / N)
        dc = float((rc - base_c).abs().mean())
        dv = float((rv - base_v).abs().mean())
        res[str(alpha)] = {"kept": float(m.mean()), "drift_count": dc, "drift_volume": dv}
        print(f"{alpha:>7.2f}{m.mean()*100:>8.1f}%{dc:>16.5f}{dv:>16.5f}")
    if res:
        mc_ = np.mean([v["drift_count"] for v in res.values()])
        mv_ = np.mean([v["drift_volume"] for v in res.values()])
        print(f"\nmean drift  count {mc_:.5f}   volume {mv_:.5f}   ratio {mc_/max(mv_,1e-9):.2f}x")
        print("A reconstruction-invariant statistic drifts LESS as the discretisation bias alpha "
              "changes. Ratio > 1 supports the volume measure.")
    json.dump(res, open("artifacts/scannetpp/volume_csls_invariance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
