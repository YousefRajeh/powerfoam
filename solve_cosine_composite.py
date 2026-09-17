"""Sphere-consistent AND overlap-coupled lifting: minimise a COSINE composite residual.

The solver table has one empty cell. SFS Eq. 6/18 and NormLift Eq. 4/5 are sphere-consistent but
LINEAR in u, so they decouple across primitives and cannot see G's off-diagonal. The
sphere-deconvolution of A26 was coupled but minimised a EUCLIDEAN residual, mixing a spherical
constraint with a metric the cosine readout ignores -- and it lost 0/10. This is the missing
combination: coupled through the renderer, and measured by angle.

    min_{||u_j|| = 1}   J(U) = sum_i ( 1 - < r_i / ||r_i|| , b_i > ),    r_i = sum_j A_ij u_j

WHY THIS IS NOT REFUTED BY A27. A27 showed the closed form BEATS the least-squares optimum Xhat
on 20/20 scene-arms, so any objective whose optimum is Xhat is aiming below where we already
stand. J is NOT that objective: it is scale-invariant in each r_i, so it never spends effort on
the ray magnitudes whose near-null directions drive Xhat to ||x|| ~ 1e7. Its stationary point is
a different point, and whether it is better is an empirical question this script answers.

GRADIENT.  dJ/dr_i = -(1/||r_i||) ( b_i - cos_i * r_i/||r_i|| ),  cos_i = <r_i/||r_i||, b_i>
           dJ/du_j = sum_i A_ij dJ/dr_i
followed by a projection back to the sphere. `--check-grad` verifies this against finite
differences on a small random problem before any scene is touched, because a sign error here
would look exactly like "the idea does not work".

COST. Unlike everything else in this queue, J needs per-ray residuals, so the cached Gram is not
enough -- A and B are required every iteration. A is built once and cached in memory; rays are
processed view by view so the (rays, F) intermediate never exceeds one view.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from diagnose_holes import SCENES


def check_grad(seed=0, R=40, P=12, Fd=6):
    """Finite-difference check of dJ/dU on a small dense problem."""
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(R, P, generator=g).double()
    A = A / A.sum(1, keepdim=True)
    B = F.normalize(torch.randn(R, Fd, generator=g).double(), dim=-1)
    U = F.normalize(torch.randn(P, Fd, generator=g).double(), dim=-1)

    def J(Um):
        r = A @ Um
        return float((1.0 - (F.normalize(r, dim=-1) * B).sum(-1)).sum())

    r = A @ U
    nr = r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    rh = r / nr
    cos = (rh * B).sum(-1, keepdim=True)
    dJdr = -(B - cos * rh) / nr
    ana = A.T @ dJdr

    eps = 1e-6
    num = torch.zeros_like(U)
    for j in range(P):
        for k in range(Fd):
            Up = U.clone(); Up[j, k] += eps
            Um = U.clone(); Um[j, k] -= eps
            num[j, k] = (J(Up) - J(Um)) / (2 * eps)
    err = (ana - num).abs().max() / num.abs().max().clamp_min(1e-30)
    print(f"[grad check] max relative error {float(err):.3e}  -> "
          f"{'OK' if err < 1e-5 else 'FAIL'}")
    return float(err) < 1e-5


def load_view_B(feat_dir, stem, H, W, dev):
    """Per-pixel unit CLIP feature, reconstructed from the SAM segment table."""
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[0]
    tab = F.normalize(torch.from_numpy(np.ascontiguousarray(f)).float().to(dev), dim=-1)
    seg = torch.from_numpy(np.ascontiguousarray(s)).long().to(dev)
    if tuple(seg.shape) != (H, W):
        seg = F.interpolate(seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), tab


def run(scene, recon, n_views, iters, cap, feat_dir_name, dev="cuda"):
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    wp.init()
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]

    names = sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))
    feat_dir = f"data/scannet/{scene}_colmap/{feat_dir_name}"
    sel = np.linspace(0, len(dh.cameras) - 1, min(n_views, len(dh.cameras))).astype(int).tolist()

    views = []
    for vi in sel:
        stem = os.path.splitext(names[vi])[0]
        if not os.path.exists(os.path.join(feat_dir, f"{stem}_f.npy")):
            continue
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        row = op.row_indices.to(torch.int64)
        col = op.col_indices.to(torch.int64)
        val = op.values.float()
        del op
        seg, tab = load_view_B(feat_dir, stem, H, W, dev)
        views.append((row, col, val, seg, tab, H * W))
    if not views:
        raise RuntimeError("no views with features")
    Fd = views[0][4].shape[1]

    src = torch.load(f"artifacts/scannet/{scene}/solved_weighted_{recon}_ogl3.pt",
                     map_location=dev, weights_only=True)
    valid = src["valid_mask"].to(dev)

    # MATCHED-BUDGET BASELINE. The shipped `solved_weighted_*` is accumulated over EVERY view of
    # the scene (37-279 of them); this solver sees only `n_views`. Scoring one against the other
    # compares data budgets, not solvers -- and initialising from the full-view estimate and then
    # iterating to fit a 12-view subset actively drags a better field toward a worse one, which is
    # also why more iterations made mIoU fall. So the closed form is rebuilt HERE from exactly the
    # views this solver uses, and that is the baseline the result must be read against.
    AtB12 = torch.zeros(P, Fd, device=dev)
    D12 = torch.zeros(P, device=dev)
    for row, col, val, seg, tab, nray in views:
        b = tab[seg.clamp_min(0)]
        b[seg < 0] = 0.0
        AtB12.index_add_(0, col, val.unsqueeze(-1) * b[row])
        D12.index_add_(0, col, val)
        del b
    live12 = D12 > 0
    W12 = torch.zeros(P, Fd, device=dev)
    W12[live12] = AtB12[live12] / D12[live12].unsqueeze(-1)
    torch.save({"primitive_features": F.normalize(W12, dim=-1).cpu().half(),
                "valid_mask": (valid & live12).cpu()},
               f"artifacts/scannet/{scene}/solved_w{n_views}v_{recon}_ogl3.pt")

    U = F.normalize(W12, dim=-1)          # start from the MATCHED baseline, not the full-view one
    valid = valid & live12
    U[~valid] = 0.0

    def obj_and_grad(Um, want_grad=True):
        tot = 0.0
        G = torch.zeros_like(Um) if want_grad else None
        for row, col, val, seg, tab, nray in views:
            r = torch.zeros(nray, Fd, device=dev)
            r.index_add_(0, row, val.unsqueeze(-1) * Um[col])
            nr = r.norm(dim=-1, keepdim=True)
            live = (nr.squeeze(-1) > 1e-12) & (seg >= 0)
            rh = r / nr.clamp_min(1e-30)
            b = tab[seg.clamp_min(0)]
            cos = (rh * b).sum(-1, keepdim=True)
            tot += float((1.0 - cos.squeeze(-1))[live].sum())
            if want_grad:
                dJdr = -(b - cos * rh) / nr.clamp_min(1e-30)
                dJdr[~live] = 0.0
                G.index_add_(0, col, val.unsqueeze(-1) * dJdr[row])
            del r, rh, b, cos
        return tot, G

    j0, _ = obj_and_grad(U, False)
    best, jb, best_it = U.clone(), j0, 0
    # step from the gradient scale: eta = s / ||G||_inf, halved on any non-improving step
    _, G = obj_and_grad(U)
    eta = 1.0 / G.norm(dim=-1).max().clamp_min(1e-30)
    for it in range(iters):
        _, G = obj_and_grad(U)
        Un = U - eta * G
        Un = F.normalize(Un, dim=-1)
        Un[~valid] = 0.0
        j, _ = obj_and_grad(Un, False)
        if j < jb:
            jb, best, best_it, U = j, Un.clone(), it + 1, Un
        else:
            eta = eta * 0.5
            U = Un
    return best, dict(scene=scene, recon=recon, P=int(P), views=len(views), Fd=int(Fd),
                      obj_init=j0, obj_best=jb, best_iter=best_it,
                      obj_drop=float(1.0 - jb / max(j0, 1e-30)), iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0097_00")
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--feature-folder", default="openclip_features_sam_l3")
    ap.add_argument("--tag", default="coscomp")
    ap.add_argument("--check-grad", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/cosine_composite.json")
    a = ap.parse_args()
    if a.check_grad and not check_grad():
        raise SystemExit("gradient check failed -- not running")
    rows = []
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                U, info = run(sc, rec, a.views, a.iters, a.cap, a.feature_folder)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            src = torch.load(f"artifacts/scannet/{sc}/solved_weighted_{rec}_ogl3.pt",
                             map_location="cpu", weights_only=True)
            torch.save({"primitive_features": U.cpu().half(),
                        "valid_mask": src["valid_mask"]},
                       f"artifacts/scannet/{sc}/solved_{a.tag}_{rec}_ogl3.pt")
            rows.append(info)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] views {info['views']} obj {info['obj_init']:.4e} -> "
                  f"{info['obj_best']:.4e} ({info['obj_drop']:+.2%}) best@{info['best_iter']}",
                  flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
