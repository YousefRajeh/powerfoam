"""Load any of the six reconstructions, and compute the GT-point -> primitive assignment.

EACH REPRESENTATION GETS ITS OWN NATURAL QUERY -- no reimplemented math on either side:

  powerfoam  power cell:      argmin_i |x - c_i|^2 - r_i^2   (exact, disjoint by construction;
                              the same formula the ray-cell traversal kernel uses)
  radfoam    nearest centre:  radfoam's foam is UNWEIGHTED, so its cells are ordinary Voronoi
                              cells and nearest-centre IS the exact cell membership. Proven in
                              test_unweighted_delaunay_adjacency.py T3 (0/5000 disagreements
                              against the power-cell query at r=0).
  gaussian   Mahalanobis:     argmin_i (x-mu_i)^T Sigma_i^-1 (x-mu_i) + log|Sigma_i|
                              A Gaussian has no disjoint cell, so nearest-CENTRE ignores that
                              an anisotropic splat extends much further along its long axis
                              than its short one. Mahalanobis asks "which Gaussian most
                              plausibly generated this point", which is the natural query for
                              the representation. The +log|Sigma| term is what makes it a
                              likelihood comparison rather than a per-Gaussian-rescaled
                              distance: without it, a Gaussian can always win simply by being
                              large, since inflating Sigma shrinks every Mahalanobis distance.
                              (Equivalently: argmax of the un-normalised Gaussian log-density,
                              dropping the constant 3*log(2*pi).)

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
        pts = sd.get("points", sd.get("primal_points"))
        if pts is None:
            for k in ("primal_points", "points", "means"):
                if k in sd:
                    pts = sd[k]
                    break
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


def assign_mahalanobis(points, means, scales, quats, k=32, chunk=200_000, device="cuda"):
    """argmin_i (x-mu_i)^T Sigma_i^-1 (x-mu_i) + log|Sigma_i|, restricted to the k nearest
    centres per point.

    The kNN restriction is what makes this tractable at 10^5 points x 10^6 Gaussians. It is a
    genuine approximation: a distant, hugely elongated Gaussian could in principle win on
    Mahalanobis distance while not being among the k nearest by Euclidean distance. k=32 is
    generous relative to the anisotropy actually present (gsplat's scales rarely span more
    than ~1-2 orders of magnitude), and the chosen k is recorded so the assumption is
    auditable rather than hidden.

    Sigma = R diag(s^2) R^T, so Sigma^-1 = R diag(s^-2) R^T and the quadratic form is
    sum_j ((R^T d)_j / s_j)^2 -- computed without ever forming a 3x3 inverse.
    """
    P = points.shape[0]
    R = _quat_to_R(quats)                                    # (N,3,3)
    logdet = 2.0 * torch.log(scales.clamp_min(1e-12)).sum(-1)  # log|Sigma| = 2*sum log s
    out = torch.full((P,), -1, dtype=torch.long, device=device)

    for s0 in range(0, P, chunk):
        x = points[s0:s0 + chunk].to(device)                 # (C,3)
        d = torch.cdist(x, means)                            # (C,N) euclidean
        kk = min(k, means.shape[0])
        nn = d.topk(kk, largest=False).indices               # (C,k)
        diff = x[:, None, :] - means[nn]                     # (C,k,3)
        # local = R_i^T diff, then scale-normalise
        local = torch.einsum("ckij,cki->ckj", R[nn], diff)    # (C,k,3)
        m2 = ((local / scales[nn].clamp_min(1e-12)) ** 2).sum(-1)   # (C,k)
        score = m2 + logdet[nn]
        out[s0:s0 + chunk] = nn.gather(1, score.argmin(1, keepdim=True)).squeeze(1)
    return out


def compute_assignment(recon, scene, gt_points, device="cuda", knn_k=32):
    """-> (assignment int64 (P,), method str, seconds float). -1 means unowned."""
    from point_cloud_query import assign_points_to_power_cells

    t0 = time.time()
    prim = load_primitives(recon, scene, device=device)
    if prim["kind"] == "gaussian":
        a = assign_mahalanobis(torch.as_tensor(gt_points).float(), prim["centers"],
                               prim["scales"], prim["quats"], k=knn_k, device=device)
        return a.cpu().numpy(), "mahalanobis", time.time() - t0

    centers = prim["centers"].cpu().numpy().astype(np.float64)
    radii = prim["radii"].cpu().numpy().astype(np.float64)
    a = assign_points_to_power_cells(gt_points, centers, radii, valid=None, k=64)
    method = "power_cell" if prim["kind"] == "powerfoam" else "nearest_center"
    return np.asarray(a), method, time.time() - t0
