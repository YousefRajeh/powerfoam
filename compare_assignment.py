"""Does the exp reconstruction make GT->primitive assignment cleaner, or the features better-posed?

THE QUESTION. The exp run has the same primitive count and near-identical PSNR, but median density
50.3 -> 12.9 with the mass concentrated in the tail: more genuinely empty cells, less low-but-nonzero
density smeared through the interior. If that is real, three things MIGHT follow, and they are
separable:

  1. assignment is better localised -- the cell that owns a GT point sits closer to it
  2. fewer cells are "solid" but interior, so the lifting weight concentrates on surface cells
  3. mIoU improves

Only (3) matters, and it does NOT follow from (1) or (2). This script measures (1) and (2) only,
because they are cheap and need no re-lift. (3) requires accumulating SAM+CLIP against this
checkpoint and re-solving -- every existing feature file was lifted against the SOFTPLUS model, so
there is no shortcut.

WHAT IS MEASURED
  coverage        fraction of GT points that land in some cell at all
  d(point,owner)  distance from a GT point to its owning cell's CENTRE -- tighter is better
                  localised, and it is the quantity that decides whether a per-cell label is a
                  sensible label for that point
  owners          how many distinct cells own at least one GT point (a proxy for how much of the
                  primitive budget is doing GT-facing work)
  alpha at L=2r   the per-cell opacity proxy, to check the "more truly empty cells" claim survives
                  the same activation on both sides

Assignment is the exact power-diagram query (argmin ||x-c||^2 - r^2), identical for both models, so
any difference is a property of the reconstruction rather than of the query.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")

from point_cloud_query import assign_points_to_power_cells
from run_percell_masked import SPLIT

SCENE = "scene0347_00"
RUNS = [("softplus baseline", f"output/scannet_{SCENE}_nonfrozen/model.pt", "softplus"),
        ("exp 0.1->0.01",     f"output/scannet_{SCENE}_nonfrozen_voro/model.pt", "exp")]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = np.load(rf"D:\Downloads\scannet_pointcept\{SPLIT[SCENE]}\{SCENE}\coord.npy").astype(np.float64)
    print(f"{SCENE}: {len(gt):,} GT points\n")
    print(f"{'run':20s} {'n_prim':>9s} {'cover%':>7s} {'d(pt,owner) cm':>15s} "
          f"{'owners':>9s} {'own%':>6s} {'alpha<0.1%':>11s}")
    for name, path, act in RUNS:
        m = torch.load(path, map_location="cpu", weights_only=False)
        pts = m["points"].float()
        radii = F.softplus(m["radii"].float(), beta=100)
        dens = torch.exp(m["density"].float()) if act == "exp" \
            else F.softplus(m["density"].float(), beta=100)
        alpha = 1.0 - torch.exp(-dens * 2.0 * radii)

        assigned = assign_points_to_power_cells(gt, pts.numpy().astype(np.float64),
                                                radii.numpy().astype(np.float64))
        owned = assigned >= 0
        d = np.linalg.norm(gt[owned] - pts.numpy()[assigned[owned]], axis=1)
        n_owners = len(np.unique(assigned[owned]))
        print(f"{name:20s} {pts.shape[0]:9,d} {100*owned.mean():6.2f}% "
              f"{np.median(d)*100:15.2f} {n_owners:9,d} "
              f"{100*n_owners/pts.shape[0]:5.1f}% {100*(alpha < 0.1).float().mean():10.2f}%")


if __name__ == "__main__":
    main()
