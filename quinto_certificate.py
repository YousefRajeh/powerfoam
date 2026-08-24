"""Quinto's stable-recovery certificate, applied per primitive.

THE THEOREM (verified against the PDF, not paraphrased from notes).

  Eric Todd Quinto, "Singularities of the X-ray Transform and Limited Data Tomography in R^2
  and R^3", SIAM J. Math. Anal. 24(5):1215-1225, 1993.

  THEOREM 4.1. Let gamma be a smooth curve in R^3 and f in E'(R^3 \ gamma). Let x0 in supp f and
  xi0 in T*_{x0}(R^3) \ 0. Then any wavefront set of f at (x0; xi0) is stably detected from data
  with sources on gamma IF AND ONLY IF the plane pi through x0 conormal to xi0 intersects gamma
  TRANSVERSALLY.

  The global version -- every plane meeting supp f intersects gamma transversally -- is the
  Kirillov-Tuy condition, which is what Tuy's inversion formula requires.

WHY THIS IS AN IDENTIFIABILITY STATEMENT AND NOT A SOLVER CAVEAT. The abstract is explicit that
it determines which singularities can be stably recovered "no matter how good the inversion
algorithm". So a cell that fails this test is not badly conditioned, it is UNRECOVERABLE from
this camera set, and no preconditioner, regulariser or better solver changes that. Our own
measurements agree from the other direction: 9.44% of cells are never touched by any ray, and
cond(G) >= 1.1653e21 is set entirely by that tail -- "rank deficiency, not ill-conditioning".

WHAT WE HAD WRITTEN DOWN WAS THE 2D COROLLARY, AND IT IS NOT OUR CASE. The note in
Solver-Research-Agent-Findings recorded "stably recoverable iff some ray passes through x
perpendicular to xi". That is corollary (3.3), the R^2 statement about a single line. Our setting
is R^3 with cameras on a trajectory, which is Theorem 4.1: a PLANE-versus-CURVE transversality
test, not a ray-versus-point test. The distinction matters because the 3D test is a property of
the whole camera path, and it is the one that is cheap to evaluate exactly.

WHY SCANNET SHOULD DO WELL. Quinto notes that sources on a single CIRCLE leave many singularities
undetected -- "the more undetected singularities, the farther x0 is from the plane of C" -- and
that inversion is "more stable for nonplanar curves such as two parallel circles or curves
oscillating on a cylinder". Handheld ScanNet trajectories are strongly nonplanar, so this
predicts a high certified fraction, and predicts that scenes with flatter camera paths certify
worse. That is a falsifiable per-scene prediction, which is why this script also reports a
planarity measure for the trajectory.

THE CONORMAL WE USE. The singularity of interest is the surface itself, whose conormal is the
surface normal. Under a genuinely frozen reconstruction each primitive sits exactly on a GT mesh
vertex, and ScanNet ships a per-vertex `normal.npy`, so we can use the TRUE surface normal rather
than estimating one from the foam. That removes a whole class of error from the certificate --
but it is only legitimate when the frozen bijection actually holds, so this script REFUSES to run
unless it does (see `--require-exact`).
"""
import argparse
import json
import os

import numpy as np
import torch


def signed_plane_distances(centers, normals, cam_centers, chunk=20000, device="cuda"):
    """s[c, v] = (C_v - x_c) . n_c  -- the signed distance of camera v from cell c's conormal plane.

    Chunked over cells: the full (P, V) matrix is fine for V ~ 300 but the intermediate
    (chunk, V, 3) is not, so we contract with einsum instead of materialising it. The (N,512)
    gather trap has bitten this project four times; this is the same shape of mistake.
    """
    P = centers.shape[0]
    V = cam_centers.shape[0]
    out = torch.empty((P, V), dtype=torch.float32, device=device)
    C = cam_centers.to(device)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        x = centers[s:e].to(device)
        n = normals[s:e].to(device)
        # (C_v . n_c) - (x_c . n_c), broadcast without forming (chunk, V, 3)
        out[s:e] = (n @ C.T) - (x * n).sum(dim=-1, keepdim=True)
    return out


def certify(signed, seg_len, min_margin=0.0):
    """Transversal crossing test along the ORDERED camera trajectory.

    A sign change in the signed distance between consecutive cameras means the polyline gamma
    crosses the plane. Transversality is the requirement that it crosses rather than grazes: the
    component of the trajectory step along the plane normal is exactly |s_{v+1} - s_v|, so
    dividing by the step length gives |cos(angle between step and plane normal)|, which is 0 for
    a tangential graze and 1 for a perpendicular stab. `min_margin` thresholds that.

    Returns (certified, n_crossings, best_margin).
    """
    a, b = signed[:, :-1], signed[:, 1:]
    crossing = (torch.sign(a) * torch.sign(b)) < 0          # strict sign change
    margin = (b - a).abs() / seg_len.clamp_min(1e-12)        # |cos| of crossing angle
    ok = crossing & (margin >= min_margin)
    return ok.any(dim=1), ok.sum(dim=1), torch.where(crossing, margin, torch.zeros_like(margin)).amax(dim=1)


def trajectory_planarity(cam_centers):
    """How planar is the camera path? Quinto's failure case is sources on a single circle.

    Ratio of the smallest to largest singular value of the centred camera centres: 0 = exactly
    planar (the degenerate case), 1 = isotropic. Reported so the per-scene prediction -- flatter
    trajectory certifies worse -- can be tested rather than asserted.
    """
    X = cam_centers - cam_centers.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(X.double())
    return float(sv[2] / sv[0]), [float(v) for v in sv]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--centers-npz", required=True, help="npz with 'centers' (P,3) and optionally 'orig_indices'")
    p.add_argument("--gt-dir", required=True, help="Pointcept scene dir with coord.npy + normal.npy")
    p.add_argument("--cameras-npz", required=True, help="npz with 'centers' (V,3) ORDERED along the trajectory")
    p.add_argument("--output", required=True)
    p.add_argument("--min-margin", type=float, default=0.0,
                   help="Minimum |cos| of the crossing angle. 0 = any strict sign change counts.")
    p.add_argument("--require-exact", action="store_true",
                   help="Refuse to run unless every primitive sits bitwise on its GT vertex. The "
                        "true-normal shortcut is only valid under a genuine frozen bijection.")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.centers_npz)
    centers = torch.from_numpy(d["centers"].astype(np.float32))
    coord = np.load(os.path.join(a.gt_dir, "coord.npy")).astype(np.float32)
    normal = np.load(os.path.join(a.gt_dir, "normal.npy")).astype(np.float32)

    if "orig_indices" in d:
        oi = d["orig_indices"].astype(np.int64)
        gt_xyz, gt_n = coord[oi], normal[oi]
        # Exact set check, NOT torch.cdist: cdist's default mm-based formula reports up to 2.4e-3
        # error for BITWISE IDENTICAL points at ScanNet coordinate magnitudes in float32, which
        # already produced one false test failure in this project.
        exact = np.array_equal(centers.numpy(), gt_xyz)
        maxdev = float(np.abs(centers.numpy().astype(np.float64) - gt_xyz.astype(np.float64)).max())
        print(f"[bijection] primitives={len(centers)} gt={len(coord)} bitwise_exact={exact} maxdev={maxdev:.3e}")
        if a.require_exact and not exact:
            raise SystemExit("--require-exact: primitives are not bitwise on GT vertices, so the "
                             "GT-normal shortcut is not valid. Use an estimated-normal path instead.")
        normals = torch.from_numpy(gt_n)
    else:
        raise SystemExit("no orig_indices in centers npz: cannot map primitives to GT normals")

    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cams = torch.from_numpy(np.load(a.cameras_npz)["centers"].astype(np.float32))
    seg = (cams[1:] - cams[:-1]).norm(dim=-1).to(device)
    planarity, svals = trajectory_planarity(cams)
    print(f"[trajectory] V={len(cams)} planarity(sv3/sv1)={planarity:.4f} svals={['%.3f'%v for v in svals]}")

    signed = signed_plane_distances(centers, normals, cams, device=device)
    cert, ncross, margin = certify(signed, seg, a.min_margin)
    frac = float(cert.float().mean())
    print(f"[certificate] certified {int(cert.sum())}/{len(cert)} = {frac*100:.2f}% "
          f"at min_margin={a.min_margin}")
    print(f"[certificate] crossings per cell: median {int(ncross.median())} mean {float(ncross.float().mean()):.2f}")
    m = margin[cert]
    if m.numel():
        q = torch.quantile(m[:16_000_000].float(), torch.tensor([0.1,0.5,0.9], device=m.device))
        print(f"[certificate] crossing margin |cos| q10/q50/q90: {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}")

    # margin sweep: how fast does the certified fraction fall as we demand a cleaner crossing?
    sweep = {}
    for t in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3):
        c, _, _ = certify(signed, seg, t)
        sweep[t] = float(c.float().mean())
        print(f"    min_margin={t:4.2f} -> certified {sweep[t]*100:6.2f}%")

    np.savez(a.output, certified=cert.cpu().numpy(), n_crossings=ncross.cpu().numpy(),
             margin=margin.cpu().numpy())
    json.dump({"certified_frac": frac, "planarity": planarity, "svals": svals,
               "n_views": int(len(cams)), "n_primitives": int(len(cert)),
               "margin_sweep": {str(k): v for k, v in sweep.items()}},
              open(a.output.replace(".npz", ".json"), "w"), indent=2)
    print(f"[certificate] wrote {a.output}")


if __name__ == "__main__":
    main()
