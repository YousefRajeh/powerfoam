"""Per-primitive opacity/alpha, and OpenGaussian's low-opacity GT masking as an ablation axis.

WHY THIS IS AN AXIS AND NOT A DEFAULT. OpenGaussian's eval (`scripts/eval_scannet.py:127-129`)
zeroes the GT label of any point whose Gaussian has sigmoid(opacity) < 0.1, which DELETES those
points from the metric rather than predicting for them. NormLift inherits it. Our numbers are
therefore scored on points the published baselines never scored, which understates us against
them -- so the masked number is the one that is comparable.

But the mask is not neutral across representations, and that is the whole point of measuring it
rather than just switching it on. Measured on scene0062_00 at threshold 0.1:

    recon        GT masked   mIoU raw -> masked
    pf_nonfroz       6.9%    33.89 -> 34.97  (+1.08)
    gs_froz         22.3%    (not yet scored)
    rf_unfroz       60.7%    23.49 -> 34.58  (+11.09)
    rf_froz         66.2%    24.19 -> 31.14  (+6.94)

The mask nearly erases a 10-point gap -- by scoring RadFoam on 39% of GT points and PowerFoam on
93%. Those are not the same task. It converts "this method left transparent geometry covering a
real surface" from a penalty into an exemption, which is exactly the representational difference
worth reporting. So BOTH numbers are recorded, always, with the kept fraction beside them:
masked for comparability with published baselines, unmasked as the honest representation
comparison.

ALPHA PER REPRESENTATION. There is no single "opacity" across the three; each gets the quantity
that plays that role in its own renderer:
  gaussian   sigmoid(opacity)              -- exactly OpenGaussian's quantity
  radfoam    1 - exp(-sigma * dt)          -- sigma = activation_scale * softplus(density, beta=10)
  powerfoam  1 - exp(-sigma * dt)          -- sigma = softplus(density, beta=100)
dt is a fixed reference step (default 1 cm), so the foams' alpha is "opacity accrued crossing one
centimetre of this cell". That constant is a convention and is recorded with the result; only the
RANKING of primitives matters for a threshold test, and dt does not change the ranking.
"""
import os

import numpy as np
import torch

REFERENCE_DT = 0.01          # metres; see docstring -- affects the threshold's meaning, not the ranking
DEFAULT_THRESHOLD = 0.1      # OpenGaussian's value


def primitive_alpha(recon, scene, recon_root="recon_remote", output_root="output",
                    dt=REFERENCE_DT):
    """-> (N,) float alpha per primitive, or None if the checkpoint is unavailable."""
    if recon.startswith("gs"):
        p = os.path.join(recon_root, recon, scene, "ckpt.pt")
        if not os.path.exists(p):
            return None
        sd = torch.load(p, map_location="cpu", weights_only=False)
        sp = sd.get("splats", sd)
        return torch.sigmoid(sp["opacities"].float()).numpy()

    if recon.startswith("rf"):
        p = os.path.join(recon_root, recon, scene, "model.pt")
        if not os.path.exists(p):
            return None
        sd = torch.load(p, map_location="cpu", weights_only=False)
        sigma = torch.nn.functional.softplus(sd["density"].float().squeeze(-1), beta=10)
        scale = float(sd.get("activation_scale", 1.0)) if not torch.is_tensor(
            sd.get("activation_scale", 1.0)) else float(sd["activation_scale"])
        # radfoam has no radii (r == 0), so "crossing the cell" needs a length from the site
        # spacing instead. Nearest-neighbour distance is the natural Voronoi cell scale, and it
        # makes the foam alphas comparable: both become "opacity accrued crossing this cell".
        from scipy.spatial import cKDTree
        xyz = sd["xyz"].float().numpy().astype(np.float64)
        dnn, _ = cKDTree(xyz).query(xyz, k=2, workers=-1)
        return (1 - torch.exp(-sigma * scale * torch.from_numpy(dnn[:, 1]).float())).numpy()

    variant = "truefrozen" if recon == "pf_tfroz" else "nonfrozen"
    p = os.path.join(output_root, f"scannet_{scene}_{variant}", "model.pt")
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sigma = torch.nn.functional.softplus(sd["density"].float().squeeze(), beta=100)
    # PowerFoam silences a cell by collapsing its RADIUS, not its density: measured on
    # scene0062_00 pf_tfroz, invalid cells have median radius 0.00011 m against 0.01904 m for
    # valid ones (a 173x shrink) while only 1.35% of them have collapsed density. A fixed-dt
    # alpha therefore calls those cells opaque, which is why visibility assignment scored
    # pf_tfroz at 23.92 vs 35.22 geometric -- the proxy was wrong, not the method.
    # The physically meaningful quantity is opacity accrued CROSSING THE CELL: sigma * 2r.
    radii = torch.nn.functional.softplus(sd["radii"].float().squeeze(), beta=100)
    return (1 - torch.exp(-sigma * 2.0 * radii)).numpy()


def mask_low_opacity(gt_labels, assigned, alpha, threshold=DEFAULT_THRESHOLD):
    """-> (masked_gt, n_masked, kept_fraction).

    Faithful to OpenGaussian: the point's ASSIGNED primitive is the one tested, and masking sets
    the GT label to 0, which `calculate_metrics` treats as ignore (it gates on gt != 0).

    A point owned by NO primitive keeps its label and scores as a miss. Masking those too would
    let a method delete the points it fails to cover, a far larger loophole than the threshold.
    """
    if alpha is None:
        return gt_labels, 0, 1.0
    gt = gt_labels.copy()
    owned = assigned >= 0
    low = np.zeros(len(gt), dtype=bool)
    low[owned] = alpha[assigned[owned]] < threshold
    gt[low] = 0
    return gt, int(low.sum()), float((~low).mean())
