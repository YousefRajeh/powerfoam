"""Sphere-deconvolved lifting on REAL features, from cached stats -- no rendering, no A.

WHAT THE SOLVER IS. Every published lifting solver here optimises something LINEAR in u:
SFS Eq. 6/18 is a weighted mean, and NormLift Eq. 4 maximises sum_i A_ij <u, B_i> over the unit
sphere, which by linearity of the inner product is the same weighted mean renormalised. A linear
objective decouples across primitives no matter how much overlap there is, so none of them can
see the off-diagonal of G. The sphere-constrained LEAST SQUARES objective can:

    min_{||u_j|| = 1}  J(U) = || A U - B ||^2 = tr(U^T G U) - 2 <U, A^T B> + const

The quadratic term is the Gram. When G = D it equals sum_j D_jj ||u_j||^2 = sum_j D_jj, constant
on the sphere, so it drops out and the problem collapses EXACTLY to their Eq. 4: their solver is
optimal in the disjoint limit and only there.

WHY THIS IS FAST. The iteration needs only G and A^T B, and both are already cached:
    G off-diagonal   covis_<arm>.pt          (S_keys / S_vals, kmax=64)
    G_jj             stats support2          = diag(A^T A)
    A^T B            stats numerator
So no view is rendered and A is never formed. Each step is ONE sparse matvec over G's edges
rather than two scatters over A's nonzeros (~9x fewer for the foam arms), and the objective
    J = <GW, W> - const,   W = U - Uhat
falls out of that same matvec instead of costing a third pass.

STEP SIZE, UNTUNED. Projected gradient on J has gradient 2(GU - A^T B) and is stable for
eta < 2/lambda_max(G). G is entrywise non-negative, so Gershgorin gives lambda_max(G) <=
max_j sum_k G_jk for free; applied PER ROW as eta_j = 1/sum_k G_jk it is a safe diagonal
preconditioner with no hand-set damping and an unchanged fixed point.

FLOOR, GUARANTEED. The iterate is initialised at the back-projection and accepted only when it
lowers J, so the returned field is never worse than the estimator it replaces on the quantity the
bound is about. Throttling the correction instead (a per-row alpha) was tried and is WRONG: it
moves the fixed point rather than the path, and cost 19 mIoU on the high-overlap arm.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from diagnose_holes import SCENES

ARMS = {
    "truefrozen": ("stats_truefrozen_ogl3.pt", "covis_truefrozen.pt"),
    "nonfrozen": ("stats_nonfrozen_ogl3.pt", "covis_nf_h64.pt"),
}


def load(scene, arm, dev):
    st_name, cv_name = ARMS[arm]
    ap = f"artifacts/scannet/{scene}"
    st = torch.load(f"{ap}/{st_name}", map_location="cpu", weights_only=False)
    cv = torch.load(f"{ap}/{cv_name}", map_location="cpu", weights_only=False)
    if int(cv.get("kmax", 0)) != 64:
        raise RuntimeError(f"{cv_name} kmax={cv.get('kmax')} != 64; truncated grams undercount "
                           f"the off-diagonal mass and are not comparable")
    D = st["support"].to(dev).double()
    Gd = st["support2"].to(dev).double()
    AtB = st["numerator"].to(dev).float()
    P = D.numel()
    k = cv["S_keys"].to(dev)
    v = cv["S_vals"].to(dev).double()
    lo = torch.div(k, P, rounding_mode="floor")
    hi = k % P
    r = torch.cat([lo, hi]).to(torch.int32)
    c = torch.cat([hi, lo]).to(torch.int32)
    w = torch.cat([v, v])
    del k, v, lo, hi
    torch.cuda.empty_cache()
    return D, Gd, AtB, r, c, w, P


def run(scene, arm, iters, method="sirt", relax=1.0, dev="cuda", edge_budget_bytes=1.0e9):
    D, Gd, AtB, r, c, w, P = load(scene, arm, dev)
    Fdim = AtB.shape[1]
    eb = max(1, int(edge_budget_bytes // (4 * Fdim)))

    def Gmv(X):
        """G @ X, chunked so the (edges, F) temporary stays bounded."""
        y = Gd.float().unsqueeze(-1) * X
        for s0 in range(0, r.numel(), eb):
            e = slice(s0, s0 + eb)
            y.index_add_(0, r[e].long(), w[e].float().unsqueeze(-1) * X[c[e].long()])
        return y

    Gs = Gd.clone()                                    # sum_k G_jk = G_jj + off-diagonal row sum
    for s0 in range(0, r.numel(), 1 << 24):
        e = slice(s0, s0 + (1 << 24))
        Gs.index_add_(0, r[e].long(), w[e])
    live = D > 0

    U = torch.zeros(P, Fdim, device=dev)
    U[live] = F.normalize(AtB[live] / D[live].float().unsqueeze(-1), dim=-1)   # back-projection
    eta = (1.0 / Gs.clamp_min(1e-30)).float().unsqueeze(-1)                    # Gershgorin, per row

    def obj(X):
        return float((X * Gmv(X)).sum()) - 2.0 * float((X * AtB).sum())

    j0 = obj(U)
    best, jb, best_it = U.clone(), j0, 0

    if method == "sirt":
        # SIRT with relaxation. The classical scheme is
        #     x <- x + lam * C A^T R (b - A x)
        # and the closed form everyone ships is its FIRST iterate from x = 0 (under (P2) the row
        # normaliser R is the identity, so x_1 = D^-1 A^T b). lam in (0,2) is the standard
        # relaxation range; lam = 1 is plain SIRT. Our per-row Gershgorin eta_j = 1/sum_k G_jk
        # coincides with SIRT's column normaliser C to 0.1% here (rowsum(G)/D = 0.999 measured),
        # so this is SIRT with a sphere projection rather than an analogy to it.
        for it in range(iters):
            GU = Gmv(U)
            Un = U - relax * eta * (GU - AtB)
            U = torch.zeros_like(U)
            U[live] = F.normalize(Un[live], dim=-1)
            j = obj(U)
            if j < jb:
                jb, best, best_it = j, U.clone(), it + 1
    elif method == "cgls":
        # CGLS on the normal equations G X = A^T B, with the SPHERE PROJECTION applied only when
        # scoring the iterate. Projecting inside the recursion would break conjugacy, so the
        # Krylov sequence is left unconstrained and each iterate is projected, scored, and kept if
        # it improves. Early stopping is the point: CT calls this semi-convergence -- the iterates
        # improve, then degrade as the near-null directions amplify, which is exactly the
        # blow-up measured in A22.3 (||xhat|| -> 1e7). The guard turns that into a stopping rule.
        X = torch.zeros_like(U)
        Rr = AtB.clone()
        Pk = Rr.clone()
        rs = float((Rr * Rr).sum())
        for it in range(iters):
            GP = Gmv(Pk)
            denom = float((Pk * GP).sum())
            if denom <= 0:
                break
            al = rs / denom
            X = X + al * Pk
            Rr = Rr - al * GP
            rs_new = float((Rr * Rr).sum())
            Pk = Rr + (rs_new / max(rs, 1e-30)) * Pk
            rs = rs_new
            U = torch.zeros_like(X)
            U[live] = F.normalize(X[live], dim=-1)
            j = obj(U)
            if j < jb:
                jb, best, best_it = j, U.clone(), it + 1
    else:
        raise ValueError(method)

    return best, dict(scene=scene, arm=arm, method=method, relax=relax,
                      P=int(P), live=int(live.sum()),
                      obj_init=j0, obj_best=jb, best_iter=best_it,
                      obj_drop=float(1.0 - jb / j0) if j0 < 0 else float("nan"),
                      iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default="truefrozen,nonfrozen")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--method", default="sirt", choices=["sirt", "cgls"])
    ap.add_argument("--relax", type=float, default=1.0,
                    help="SIRT relaxation lambda; (0,2) is the classical stable range")
    ap.add_argument("--tag", default="spheredeconv")
    ap.add_argument("--out", default="artifacts/scannet/sphere_deconv_solve.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                U, info = run(sc, arm, a.iters, a.method, a.relax)
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            outp = f"artifacts/scannet/{sc}/solved_{a.tag}_{arm}_ogl3.pt"
            # valid_mask mirrors the closed-form file's so scoring compares like with like
            src = torch.load(f"artifacts/scannet/{sc}/solved_geometric_median_{arm}_ogl3.pt",
                             map_location="cpu", weights_only=True)
            torch.save({"primitive_features": U.cpu().half(),
                        "valid_mask": src["valid_mask"]}, outp)
            rows.append(info)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] P {info['P']:,}  obj {info['obj_init']:.4e} -> "
                  f"{info['obj_best']:.4e}  ({info['obj_drop']:+.2%})  "
                  f"best@{info['best_iter']}/{a.iters}  -> {outp}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
