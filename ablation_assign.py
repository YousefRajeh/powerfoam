"""Load any of the six reconstructions, and compute the GT-point -> primitive assignment.

EACH REPRESENTATION GETS ITS OWN NATURAL QUERY -- no reimplemented math on either side:

  powerfoam  power cell:      argmin_i |x - c_i|^2 - r_i^2   (exact, disjoint by construction;
                              the same formula the ray-cell traversal kernel uses)
  radfoam    nearest centre:  radfoam's foam is UNWEIGHTED, so its cells are ordinary Voronoi
                              cells and nearest-centre IS the exact cell membership. Proven in
                              test_unweighted_delaunay_adjacency.py T3 (0/5000 disagreements
                              against the power-cell query at r=0).
  gaussian   Mahalanobis:     argmin_i (x-mu_i)^T Sigma_i^-1 (x-mu_i)   -- see ablation_maha
                              A Gaussian has no disjoint cell, so nearest-CENTRE ignores that
                              an anisotropic splat extends much further along its long axis
                              than its short one. Computed EXACTLY (no kNN candidate set: on
                              real checkpoints kNN disagrees with the exhaustive argmin on
                              11-50% of points) and WITHOUT a +log|Sigma| term (real
                              checkpoints contain zero-scale Gaussians, whose log|Sigma| is
                              -inf, and one such sliver would win every point in the scene).

The assignment is written ONCE per (scene, recon) and every downstream ablation cell reads
the stored array, so the correspondence cannot drift between methods within a scene.
"""
import os
import time

import numpy as np
import torch

RECON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_remote")

# recon tag -> (kind, path template relative to the powerfoam dir)
RECONS = {
    "pf_tfroz":   ("powerfoam", "output/scannet_{scene}_truefrozen/model.pt"),
    "pf_nonfroz": ("powerfoam", "output/scannet_{scene}_nonfrozen/model.pt"),
    "rf_froz":    ("radfoam",   "recon_remote/rf_froz/{scene}/model.pt"),
    "rf_unfroz":  ("radfoam",   "recon_remote/rf_unfroz/{scene}/model.pt"),
    "gs_froz":    ("gaussian",  "recon_remote/gs_froz/{scene}/ckpt.pt"),
    "gs_unfroz":  ("gaussian",  "recon_remote/gs_unfroz/{scene}/ckpt.pt"),
    # 20k arms: radfoam's OWN schedule (iterations 20_000, freeze_points 18_000), against the
    # 30k we imposed to "match PowerFoam". Measured on scene0062_00, the extra 10k iterations
    # cost 14.6 points of vacuum fraction (sigma<1e-6: 22.7% -> 37.3%) for 1.5 dB of PSNR,
    # which propagated to -4.10 mIoU. Naming note: the remote experiment tag was rf20k_match_*
    # ("matched to PowerFoam"); it is renamed rf20k_froz_* so one word means frozen at every
    # layer -- remote tag, config, local directory, and ablation arm.
    "rf20k_froz":   ("radfoam", "recon_remote/rf20k_froz/{scene}/model.pt"),
    "rf20k_unfroz": ("radfoam", "recon_remote/rf20k_unfroz/{scene}/model.pt"),
}


def ckpt_path(recon, scene):
    kind, tmpl = RECONS[recon]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), tmpl.format(scene=scene))


def load_primitives(recon, scene, device="cuda"):
    """-> dict(kind, centers (N,3) f32, radii (N,) f32 or None, cov (N,3,3) or None).

    radii is None for gaussians; cov is None for foams."""
    kind, _ = RECONS[recon]
    p = ckpt_path(recon, scene)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    sd = torch.load(p, map_location="cpu", weights_only=False)

    if kind in ("powerfoam", "radfoam"):
        # radfoam checkpoints carry `primal_points`/`points`; powerfoam uses `points`+`radii`.
        # radfoam checkpoints store sites under `xyz`; powerfoam uses `points`.
        pts = None
        for k in ("points", "xyz", "primal_points", "means"):
            if k in sd and sd[k] is not None:
                pts = sd[k]
                break
        if pts is None:
            raise KeyError(f"no point tensor in checkpoint; keys={list(sd)}")
        pts = torch.as_tensor(pts).float()
        if kind == "powerfoam":
            # PowerfoamScene.get_radii() is softplus(raw, beta=100)
            radii = torch.nn.functional.softplus(torch.as_tensor(sd["radii"]).float(), beta=100)
        else:
            # radfoam is UNWEIGHTED: every site has zero radius, so the power diagram
            # degenerates to the ordinary Voronoi diagram.
            radii = torch.zeros(pts.shape[0])
        return {"kind": kind, "centers": pts.to(device), "radii": radii.to(device), "cov": None}

    # gsplat checkpoint: {"splats": {"means","scales","quats","opacities",...}}
    sp = sd.get("splats", sd)
    means = torch.as_tensor(sp["means"]).float()
    scales = torch.exp(torch.as_tensor(sp["scales"]).float())        # stored as log
    quats = torch.as_tensor(sp["quats"]).float()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return {"kind": "gaussian", "centers": means.to(device), "radii": None,
            "scales": scales.to(device), "quats": quats.to(device), "cov": None}


def _quat_to_R(q):
    """(N,4) wxyz -> (N,3,3). Matches gsplat's quat_to_rotmat convention."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def compute_assignment(recon, scene, gt_points, device="cuda", progress=None):
    """-> (assignment int64 (P,), method str, seconds float). -1 means unowned."""
    from point_cloud_query import assign_points_to_power_cells

    from ablation_maha import assign_exact, prepare

    t0 = time.time()
    prim = load_primitives(recon, scene, device=device)
    if prim["kind"] == "gaussian":
        # prepare() re-exponentiates, so hand it the LOG scales the checkpoint stored.
        means, scales, R, _extent, _ndeg = prepare(
            prim["centers"].cpu(), torch.log(prim["scales"].cpu().clamp_min(1e-30)),
            prim["quats"].cpu())
        idx, _ = assign_exact(torch.as_tensor(gt_points).float(), means.to(device),
                              scales.to(device), R.to(device), device=device,
                              progress=progress)
        return idx.cpu().numpy(), "mahalanobis", time.time() - t0

    centers = prim["centers"].cpu().numpy().astype(np.float64)
    radii = prim["radii"].cpu().numpy().astype(np.float64)
    a = assign_points_to_power_cells(gt_points, centers, radii, valid=None, k=64)
    method = "power_cell" if prim["kind"] == "powerfoam" else "nearest_center"
    return np.asarray(a), method, time.time() - t0
