"""Is solving in CLASS space (C~7-19 dims) identical to solving in full CLIP space (512 dims)?

`oracle_projected.py` exploits B = S T, with S the one-hot per-pixel class and T the text
embeddings, to run every solver in C dimensions instead of 512. That is an ASSERTION about three
different solvers and it is not equally obvious for all of them, so it is tested here against a
brute-force 512-dimensional reference before any scene is touched.

WHAT SHOULD HOLD, AND WHY

  Eq. 6   X = D^-1 A^T B = D^-1 A^T S T = (D^-1 A^T S) T = W T.
          Readout argmax_c <x_j/||x_j||, t_c> = argmax_c (W T T^T)_{jc}, because ||x_j|| is a
          per-primitive POSITIVE SCALAR and cannot reorder an argmax over c.
  Eq. 18  identical with A_ij -> A_ij^2 in both numerator and denominator.
  GeoMed  NOT obvious. The streaming Riemannian update normalises and takes tangent steps in the
          512-d sphere. It is equivalent only because span(T) is INVARIANT under that update --
          z_init = f_v lies in span(T), and the step z + eta (f - <f,z> z) is a linear combination
          of two vectors already in span(T) -- and only if inner products carry the metric T T^T.
          T's rows are unit-norm but NOT orthogonal, so using a plain Euclidean norm on the
          C-vector would silently compute a DIFFERENT statistic. That is the failure this test
          exists to catch.

The reference implementation below is deliberately dumb: it materialises B at (rays x 512) and
runs each solver there with no algebraic shortcut whatsoever.
"""
from __future__ import annotations
import argparse

import torch
import torch.nn.functional as F


def make_problem(R, P, C, Fd, nnz_per_ray, seed, dev, ortho_T=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    rows = torch.arange(R, device=dev).repeat_interleave(nnz_per_ray)
    cols = torch.randint(0, P, (R * nnz_per_ray,), generator=g, device=dev)
    vals = torch.rand(R * nnz_per_ray, generator=g, device=dev).double() + 1e-3
    cls = torch.randint(0, C, (R,), generator=g, device=dev)          # one class per ray
    T = torch.randn(C, Fd, generator=g, device=dev).double()
    if ortho_T:                                                      # sanity: orthonormal case
        T, _ = torch.linalg.qr(T.T); T = T.T
    T = F.normalize(T, dim=-1)
    return rows, cols, vals, cls, T


def reference(rows, cols, vals, cls, T, P, squared=False):
    """Full 512-d solve. B is materialised; no class-space structure is used."""
    C, Fd = T.shape
    B = T[cls]                                                       # (R, Fd)
    w = vals * vals if squared else vals
    X = torch.zeros(P, Fd, dtype=torch.float64, device=T.device)
    X.index_add_(0, cols, w.unsqueeze(-1) * B[rows])
    d = torch.zeros(P, dtype=torch.float64, device=T.device).index_add_(0, cols, w)
    live = d > 0
    X[live] = X[live] / d[live].unsqueeze(-1)
    sim = F.normalize(X, dim=-1) @ T.T                               # the actual readout
    pred = torch.full((P,), -1, dtype=torch.long, device=T.device)
    pred[live] = sim[live].argmax(1)
    return X, pred, live


def class_space(rows, cols, vals, cls, T, P, squared=False):
    C, Fd = T.shape
    TT = T @ T.T
    w = vals * vals if squared else vals
    S = torch.zeros(P, C, dtype=torch.float64, device=T.device)
    S.index_put_((cols, cls[rows]), w, accumulate=True)
    d = torch.zeros(P, dtype=torch.float64, device=T.device).index_add_(0, cols, w)
    live = d > 0
    W = torch.zeros(P, C, dtype=torch.float64, device=T.device)
    W[live] = S[live] / d[live].unsqueeze(-1)
    pred = torch.full((P,), -1, dtype=torch.long, device=T.device)
    pred[live] = (W[live] @ TT).argmax(1)
    return W @ T, pred, live


def gm_reference(views, T, P):
    """Streaming Riemannian median in FULL 512-d space."""
    Fd = T.shape[1]
    z = torch.zeros(P, Fd, dtype=torch.float64, device=T.device)
    gw = torch.zeros(P, dtype=torch.float64, device=T.device)
    for rows, cols, vals, cls in views:
        B = T[cls]
        y = torch.zeros(P, Fd, dtype=torch.float64, device=T.device)
        y.index_add_(0, cols, vals.unsqueeze(-1) * B[rows])
        wv = torch.zeros(P, dtype=torch.float64, device=T.device).index_add_(0, cols, vals)
        seen = (wv > 0) & (y.norm(dim=-1) > 1e-20)
        fv = torch.zeros_like(y)
        fv[seen] = F.normalize(y[seen], dim=-1)
        init = seen & (gw <= 0); upd = seen & (gw > 0)
        z[init] = fv[init]; gw[init] = wv[init]
        if upd.any():
            zp, wp, wn = z[upd], gw[upd], wv[upd]
            eta = (wn / (wp + wn)).clamp_max(1.0)
            cos = (fv[upd] * zp).sum(-1, keepdim=True)
            zn = zp + eta[:, None] * (fv[upd] - cos * zp)
            z[upd] = F.normalize(zn, dim=-1); gw[upd] = wp + wn
    live = gw > 0
    pred = torch.full((P,), -1, dtype=torch.long, device=T.device)
    pred[live] = (z[live] @ T.T).argmax(1)
    return z, pred, live


def gm_class_space(views, T, P):
    C, Fd = T.shape
    TT = T @ T.T
    def sph(Y):
        return ((Y @ TT) * Y).sum(-1).clamp_min(0).sqrt()
    z = torch.zeros(P, C, dtype=torch.float64, device=T.device)
    gw = torch.zeros(P, dtype=torch.float64, device=T.device)
    for rows, cols, vals, cls in views:
        y = torch.zeros(P, C, dtype=torch.float64, device=T.device)
        y.index_put_((cols, cls[rows]), vals, accumulate=True)
        wv = torch.zeros(P, dtype=torch.float64, device=T.device).index_add_(0, cols, vals)
        seen = (wv > 0) & (sph(y) > 1e-20)
        fv = torch.zeros_like(y)
        fv[seen] = y[seen] / sph(y[seen]).unsqueeze(-1)
        init = seen & (gw <= 0); upd = seen & (gw > 0)
        z[init] = fv[init]; gw[init] = wv[init]
        if upd.any():
            zp, wp, wn = z[upd], gw[upd], wv[upd]
            eta = (wn / (wp + wn)).clamp_max(1.0)
            cos = ((fv[upd] @ TT) * zp).sum(-1, keepdim=True)
            zn = zp + eta[:, None] * (fv[upd] - cos * zp)
            z[upd] = zn / sph(zn).clamp_min(1e-30).unsqueeze(-1); gw[upd] = wp + wn
    live = gw > 0
    pred = torch.full((P,), -1, dtype=torch.long, device=T.device)
    pred[live] = (z[live] @ TT).argmax(1)
    return z @ T, pred, live


def gm_class_space_EUCLIDEAN(views, T, P):
    """DELIBERATELY WRONG control: same code, but normalising the C-vector with a plain Euclidean
    norm instead of the T T^T metric. If the test cannot tell this apart from the correct version,
    the test is not actually checking anything."""
    C, Fd = T.shape
    TT = T @ T.T
    z = torch.zeros(P, C, dtype=torch.float64, device=T.device)
    gw = torch.zeros(P, dtype=torch.float64, device=T.device)
    for rows, cols, vals, cls in views:
        y = torch.zeros(P, C, dtype=torch.float64, device=T.device)
        y.index_put_((cols, cls[rows]), vals, accumulate=True)
        wv = torch.zeros(P, dtype=torch.float64, device=T.device).index_add_(0, cols, vals)
        seen = (wv > 0) & (y.norm(dim=-1) > 1e-20)
        fv = torch.zeros_like(y)
        fv[seen] = F.normalize(y[seen], dim=-1)
        init = seen & (gw <= 0); upd = seen & (gw > 0)
        z[init] = fv[init]; gw[init] = wv[init]
        if upd.any():
            zp, wp, wn = z[upd], gw[upd], wv[upd]
            eta = (wn / (wp + wn)).clamp_max(1.0)
            cos = (fv[upd] * zp).sum(-1, keepdim=True)
            zn = zp + eta[:, None] * (fv[upd] - cos * zp)
            z[upd] = F.normalize(zn, dim=-1); gw[upd] = wp + wn
    live = gw > 0
    pred = torch.full((P,), -1, dtype=torch.long, device=T.device)
    pred[live] = (z[live] @ TT).argmax(1)
    return z @ T, pred, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--rays", type=int, default=4000)
    ap.add_argument("--prims", type=int, default=300)
    ap.add_argument("--classes", type=int, default=19)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--hits", type=int, default=5)
    ap.add_argument("--views", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    worst = {"eq7_vec": 0.0, "eq7_pred": 0, "eq24_vec": 0.0, "eq24_pred": 0,
             "gm_vec": 0.0, "gm_pred": 0}
    eucl_disagree = 0
    n_live = 0
    for t in range(a.trials):
        rows, cols, vals, cls, T = make_problem(a.rays, a.prims, a.classes, a.dim,
                                                a.hits, 1234 + t, dev)
        for sq, tag in ((False, "eq7"), (True, "eq24")):
            Xr, pr, lr = reference(rows, cols, vals, cls, T, a.prims, squared=sq)
            Xc, pc, lc = class_space(rows, cols, vals, cls, T, a.prims, squared=sq)
            assert bool((lr == lc).all()), "live masks differ"
            rel = float(((Xr - Xc).norm(dim=-1) / Xr.norm(dim=-1).clamp_min(1e-30))[lr].max())
            worst[f"{tag}_vec"] = max(worst[f"{tag}_vec"], rel)
            worst[f"{tag}_pred"] += int((pr[lr] != pc[lr]).sum())

        views = []
        for v in range(a.views):
            r2, c2, v2, k2, _ = make_problem(a.rays // a.views, a.prims, a.classes, a.dim,
                                             a.hits, 9000 + 37 * t + v, dev)
            views.append((r2, c2, v2, k2))
        Zr, pr, lr = gm_reference(views, T, a.prims)
        Zc, pc, lc = gm_class_space(views, T, a.prims)
        _, pe, _ = gm_class_space_EUCLIDEAN(views, T, a.prims)
        assert bool((lr == lc).all()), "gm live masks differ"
        rel = float(((Zr - Zc).norm(dim=-1) / Zr.norm(dim=-1).clamp_min(1e-30))[lr].max())
        worst["gm_vec"] = max(worst["gm_vec"], rel)
        worst["gm_pred"] += int((pr[lr] != pc[lr]).sum())
        eucl_disagree += int((pr[lr] != pe[lr]).sum())
        n_live += int(lr.sum())

    print(f"trials {a.trials}  R {a.rays}  P {a.prims}  C {a.classes}  F {a.dim}  "
          f"live primitives compared {n_live:,}")
    print(f"  Eq6    max rel vector error {worst['eq7_vec']:.3e}   label disagreements {worst['eq7_pred']}")
    print(f"  Eq18   max rel vector error {worst['eq24_vec']:.3e}   label disagreements {worst['eq24_pred']}")
    print(f"  GeoMed max rel vector error {worst['gm_vec']:.3e}   label disagreements {worst['gm_pred']}")
    print(f"  [control] GeoMed with EUCLIDEAN C-norm instead of TT^T: "
          f"{eucl_disagree:,} disagreements ({eucl_disagree/max(n_live,1):.2%}) "
          f"-- must be NONZERO or this test proves nothing")
    ok = (worst["eq7_vec"] < 1e-10 and worst["eq24_vec"] < 1e-10 and worst["gm_vec"] < 1e-10
          and worst["eq7_pred"] == 0 and worst["eq24_pred"] == 0 and worst["gm_pred"] == 0
          and eucl_disagree > 0)
    print("RESULT:", "EQUIVALENT (and the test is sensitive)" if ok else "*** FAILED ***")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
