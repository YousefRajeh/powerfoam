"""What per-primitive confidence signals does the LIFTING itself already contain, and do any
of them predict correctness?

All three are recoverable from the retained stats with no re-accumulation:
  Rbar   = ||gm_z|| / gm_weight        spherical concentration = VIEW AGREEMENT in direction
  fnorm  = ||numerator|| / support     mean feature MAGNITUDE  (NormLift's reliability input --
                                       the quantity `solve_geometric_median` throws away)
  n_eff  = support^2 / sum_view_weight_sq       effective number of contributing views
  support                                       total ray mass (already known non-predictive)
"""
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels, embed_class_names

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"


def confidences(scene, suffix="_ogl3", device="cuda"):
    st = torch.load(f"artifacts/scannet/{scene}/stats_nonfrozen{suffix}.pt",
                    map_location=device, weights_only=False)
    sup = st["support"].clamp_min(1e-12)
    nn = st["numerator"].norm(dim=-1)
    intra = st["intra_sum"].clamp_min(1e-12)
    n_eff = st["support"] ** 2 / st["sum_view_weight_sq"].clamp_min(1e-12)
    norm_f = nn / sup
    out = {
        "norm_f": norm_f,                 # = c_intra * c_inter, NormLift's magnitude term
        "c_intra": intra / sup,           # within-view pixel agreement
        "c_inter": nn / intra,            # across-view directional agreement
        "n_eff": n_eff,
        "R_beta1": norm_f * n_eff / (n_eff + 1.0),   # NormLift Eq. 8
        "support": st["support"],
    }
    return {k: v.float() for k, v in out.items()}


def main():
    device = "cuda"
    scene = os.environ.get("SCENE", "scene0347_00")
    suffix = "_ogl3"
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen{suffix}.pt",
                        map_location="cpu", weights_only=True)
    valid_idx = np.where(solved["valid_mask"].cpu().numpy())[0]
    c = torch.load(os.path.join(CACHE, f"{scene}{suffix}.pt"), map_location="cpu", weights_only=False)
    unit = F.normalize(c["unit"].to(device).float(), dim=-1)
    assert unit.shape[0] == valid_idx.shape[0], (unit.shape, valid_idx.shape)

    conf = {k: v[torch.from_numpy(valid_idx).to(device)] for k, v in confidences(scene).items()}

    # per-primitive correctness at 19cls: does the primitive's own argmax match the majority GT
    # label of the points it owns?
    raw = c["raw_labels"].numpy()
    prow = c["point_row"].numpy()
    names = c["all_names"]
    n2i = {n: i for i, n in enumerate(names)}
    present = set(np.unique(raw).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt = remap_gt_labels(raw, [i for i, _ in kept])
    text = embed_class_names([n for _, n in kept], device)
    pred = (unit @ text.T).argmax(-1).cpu().numpy() + 1

    N = unit.shape[0]
    nC = len(kept) + 1
    hist = np.zeros((N, nC), dtype=np.int32)
    ok = (prow >= 0) & (gt > 0)
    np.add.at(hist, (prow[ok], gt[ok]), 1)
    scorable = hist.sum(1) > 0
    maj = hist.argmax(1)
    correct = (pred == maj)[scorable]
    print(f"{scene}: {N} valid prims, {scorable.sum()} scorable, per-cell acc {correct.mean():.4f}")

    for k, v in conf.items():
        x = v[torch.from_numpy(scorable).to(device)].cpu().numpy()
        q = np.quantile(x, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(q[1:-1], x), 0, 9)
        accs = [correct[dec == d].mean() for d in range(10)]
        rho = np.corrcoef(x, correct.astype(float))[0, 1]
        print(f"\n{k:<8} rho={rho:+.4f}  range {x.min():.4g}..{x.max():.4g}")
        print("  decile acc: " + " ".join(f"{a:.3f}" for a in accs))


if __name__ == "__main__":
    main()
