"""Validate the overlap bound on real scenes: is the closed form actually exact on foam?

The theory says the price of the cheap lift X' = D^-1 A^T B, relative to the least-squares
optimum Xhat = G^-1 A^T B, is controlled by the OVERLAP MASS

    o_i = (sum_j A_ij)^2 - sum_j A_ij^2 ,      sum_i o_i = sum_{j != k} G_jk = 2 * sum_{j<k} G_jk

-- twice the off-diagonal mass of the Gram, and therefore a property of the OPERATOR alone: no
solve, no features, no ground truth. Under (P2) it also equals `sum_j (D_jj - G_jj)`, i.e.
`sum(support) - sum(support2)` straight out of the accumulated stats, which is how arms without
a cached co-visibility graph are reached. The two forms are computed independently here and
compared, which doubles as a check on (P2).

The EXACT excess needs no ray data either. From the appendix,

    L(X') - L(Xhat) = || A D^-1 L Xhat ||_F^2 = tr(U^T G U),   U = D^-1 (D - G) Xhat

so G and A^T B (`numerator`) suffice -- both cached. It is reported relative to
tr(Xhat^T G Xhat) = ||A Xhat||^2 so scenes of different size are comparable.

WHAT WOULD FALSIFY THE STORY. (a) foam showing a large relative excess, (b) `sum_i o_i` failing
to separate foam from 3DGS, or (c) rowsum(G) departing badly from D, which would mean (P2) does
not hold on this data and the Laplacian step of the proof does not apply. All three are
measured and printed rather than assumed.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from diagnose_holes import SCENES

ARMS = {
    "foam_truefrozen": ("stats_truefrozen_ogl3.pt", "covis_truefrozen.pt"),
    # covis_nonfrozen.pt is kmax=6 and UNDERCOUNTS the off-diagonal mass by ~30% (sum 1.08e7 vs
    # 1.55e7 on scene0062); every truefrozen cache is kmax=64, so the h64 build is the only one
    # comparable across arms. Mixing them would have made the low-overlap arm look better than it
    # is for purely numerical reasons.
    "foam_nonfrozen": ("stats_nonfrozen_ogl3.pt", "covis_nf_h64.pt"),
    "gs_frozen": ("stats_gs_froz_ogl3.pt", None),
}


def load_arm(scene, stats_name, covis_name):
    ap = f"artifacts/scannet/{scene}"
    st = torch.load(f"{ap}/{stats_name}", map_location="cpu", weights_only=False)
    if st.get("lean", False) or st["support2"].numel() == 0:
        raise RuntimeError(
            f"{stats_name} was accumulated LEAN: support2 = diag(A^T A) is empty, so neither "
            f"sum_i o_i nor the excess can be formed. Re-accumulate with squared stats enabled.")
    D = st["support"].double()
    Gd = st["support2"].double()
    AtB = st["numerator"].float()
    cov = None
    if covis_name and os.path.exists(f"{ap}/{covis_name}"):
        cov = torch.load(f"{ap}/{covis_name}", map_location="cpu", weights_only=False)
        if int(cov.get("kmax", 0)) != 64:
            raise RuntimeError(
                f"{covis_name} has kmax={cov.get('kmax')}, not 64: truncated grams undercount the "
                f"off-diagonal mass and are not comparable across arms. Rebuild with "
                f"--max-hits 64.")
    return D, Gd, AtB, cov


def sym_offdiag(cov, P, dev):
    """(rows, cols, vals) of the SYMMETRISED off-diagonal Gram from the upper-triangular COO."""
    k = cov["S_keys"].to(dev)
    v = cov["S_vals"].to(dev).double()
    lo, hi = torch.div(k, P, rounding_mode="floor"), k % P
    r = torch.cat([lo, hi])
    c = torch.cat([hi, lo])
    return r, c, torch.cat([v, v])


def cg(matvec, B, diag, iters, tol=1e-10):
    """Jacobi-preconditioned CG on multiple right-hand sides. Returns (X, relative residual)."""
    X = torch.zeros_like(B)
    R = B - matvec(X)
    Z = R / diag
    Pk = Z.clone()
    rz = (R * Z).sum()
    b0 = B.norm()
    for _ in range(iters):
        AP = matvec(Pk)
        a = rz / (Pk * AP).sum().clamp_min(1e-30)
        X += a * Pk
        R -= a * AP
        Z = R / diag
        rz_new = (R * Z).sum()
        Pk = Z + (rz_new / rz.clamp_min(1e-30)) * Pk
        rz = rz_new
        if R.norm() / b0.clamp_min(1e-30) < tol:
            break
    return X, float(R.norm() / b0.clamp_min(1e-30))


def power_iter(matvec, n, F, dev, iters=30):
    """Largest eigenvalue of G, for the projected-gradient step size."""
    v = torch.randn(n, 1, device=dev, dtype=torch.float64)
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)
        lam = float(w.norm())
        if lam <= 0:
            return 1.0
        v = w / lam
    return lam


def one(scene, arm, iters, proj_iters, do_ball=False, tol=1e-8, dev="cuda"):
    stats_name, covis_name = ARMS[arm]
    D, Gd, AtB, cov = load_arm(scene, stats_name, covis_name)
    P = D.numel()
    out = dict(scene=scene, arm=arm, P=int(P), live=int((D > 0).sum()))

    # --- sum_i o_i, the operator-only quantity, both ways ---
    out["sum_o_stats"] = float((D - Gd).clamp_min(0).sum())
    out["mass"] = float(D.sum())                      # = sum_i s_i, the total ray weight
    out["mean_o_stats"] = out["sum_o_stats"] / max(out["mass"], 1e-12)

    if cov is None:
        return out
    r, c, v = sym_offdiag(cov, P, dev)
    out["sum_o_gram"] = float(v.sum())                # = sum_{j!=k} G_jk (already symmetrised)
    out["mean_o_gram"] = out["sum_o_gram"] / max(out["mass"], 1e-12)
    out["o_form_ratio"] = out["sum_o_gram"] / max(out["sum_o_stats"], 1e-12)

    # --- (P2) check: does rowsum(G) equal D? ---
    Dg = torch.zeros(P, dtype=torch.float64, device=dev).index_add_(0, r, v) + Gd.to(dev)
    Dt = D.to(dev)
    live = Dt > 0
    ratio = (Dg[live] / Dt[live].clamp_min(1e-12))
    out["rowsumG_over_D_median"] = float(ratio.median())
    out["rowsumG_over_D_p05"] = float(torch.quantile(ratio.float(), 0.05))
    out["rowsumG_over_D_p95"] = float(torch.quantile(ratio.float(), 0.95))
    del Dg, ratio
    torch.cuda.empty_cache()

    # --- exact excess, on the live subgraph ---
    idx = torch.nonzero(live, as_tuple=True)[0]
    remap = torch.full((P,), -1, dtype=torch.long, device=dev)
    remap[idx] = torch.arange(idx.numel(), device=dev)
    keep = live[r] & live[c]
    # int32 indices and an explicit free of the pre-remap arrays. On the largest nonfrozen scene
    # the symmetrised edge list is 512M entries: int64 pairs + f64 weights is 12.3 GB, and
    # holding BOTH the pre- and post-remap copies alongside 23.6 GB of CG vectors is what OOM'd
    # a 48 GB card. int32 is safe here -- P < 2^31 on every scene -- and is cast back per chunk,
    # which index_add_ requires anyway.
    rr = remap[r[keep]].to(torch.int32)
    cc = remap[c[keep]].to(torch.int32)
    vv = v[keep]
    del r, c, v, keep
    torch.cuda.empty_cache()
    n = idx.numel()
    dg = Gd.to(dev)[idx]
    dd = Dt[idx]
    B = AtB.to(dev).double()[idx]

    # CHUNKED over edges: the naive form materialises (nnz, F), which is 103 GB on the largest
    # scene. Chunking bounds the temporary to `edge_budget` rows at a time and changes nothing
    # numerically (index_add_ accumulates).
    Fdim = AtB.shape[1]
    edge_budget = max(1, int(5.0e8 // (8 * Fdim)))

    def Gmv(X):
        y = dg.unsqueeze(-1) * X
        for s0 in range(0, rr.numel(), edge_budget):
            e = slice(s0, s0 + edge_budget)
            y.index_add_(0, rr[e].long(), vv[e].unsqueeze(-1) * X[cc[e].long()])
        return y

    def quad(X):
        """tr(X^T G X) without holding a second (n, F) temporary longer than necessary."""
        return float((X * Gmv(X)).sum())

    Xh, res = cg(Gmv, B, dg.clamp_min(1e-12).unsqueeze(-1), iters, tol)
    out["cg_residual"] = res
    # U = D^-1 (D - G) Xhat = Xhat - D^-1 G Xhat
    U = Xh - Gmv(Xh) / dd.unsqueeze(-1)
    out["excess"] = float((U * Gmv(U)).sum())
    out["signal"] = float((Xh * Gmv(Xh)).sum())          # = ||A Xhat||^2
    out["rel_excess"] = out["excess"] / max(out["signal"], 1e-30)

    # Two forms of the bound. The Dirichlet form is the one the proof actually establishes
    # before Omega is introduced; Omega is the last step, and it is the step that destroys it.
    # Chunked for the same reason as Gmv: (Xh[rr] - Xh[cc]) is (nnz, F), which is 103 GB on the
    # largest scene. Only the reduction is kept, never the full difference matrix.
    dir_sum, om2 = 0.0, 0.0
    for s0 in range(0, rr.numel(), edge_budget):
        e = slice(s0, s0 + edge_budget)
        d2c = (Xh[rr[e].long()] - Xh[cc[e].long()]).pow(2).sum(-1)
        dir_sum += float((vv[e] * d2c).sum())
        om2 = max(om2, float(d2c.max()))
    d2 = None
    out["dirichlet"] = float(0.5 * dir_sum)                # 1/2 sum_{j!=k} G_jk ||xh_j-xh_k||^2
    out["dirichlet_over_excess"] = out["dirichlet"] / max(out["excess"], 1e-30)
    out["omega"] = float(om2 ** 0.5)
    out["bound"] = 0.5 * out["omega"] ** 2 * out["sum_o_gram"]
    out["bound_over_excess"] = out["bound"] / max(out["excess"], 1e-30)
    out["bound_holds"] = bool(out["bound"] >= out["excess"] - 1e-6 * abs(out["bound"]))
    out["dirichlet_holds"] = bool(out["dirichlet"] >= out["excess"] - 1e-6 * abs(out["dirichlet"]))

    # CONDITIONING. Xhat is the target the bound is stated against, but on real data G is badly
    # conditioned and a small tail of primitives runs away, which is what makes Omega -- a MAX
    # over co-visible pairs -- astronomically large. X' cannot do this: it is a convex average of
    # unit-norm per-view features, so ||x'_j|| <= 1 by construction. Recorded because it decides
    # whether Xhat is even the right target, not merely how well X' approximates it.
    Xp_ = B / dd.unsqueeze(-1)
    nh, npp = Xh.norm(dim=-1), Xp_.norm(dim=-1)
    out["norm_Xh"] = float(Xh.norm()); out["norm_Xp"] = float(Xp_.norm())
    out["xh_norm_med"] = float(nh.median()); out["xh_norm_max"] = float(nh.max())
    out["xp_norm_max"] = float(npp.max())
    out["frac_blowup"] = float((nh > 100 * npp.clamp_min(1e-12)).float().mean())
    out["rel_dist_Xp_Xh"] = float((Xp_ - Xh).norm() / Xh.norm().clamp_min(1e-30))

    # --- the SAME comparison against the optimum RESTRICTED TO THE UNIT BALL ---
    # OPT-IN, because VALIDITY of the ball bound does not require solving for X_ball at all:
    #     L(X_ball) >= L(Xhat)  =>  excess_ball <= excess  =>  2 sum_o >= excess  suffices.
    # The measured 2*sum_o/excess is 150-600 on every scene, so the ball bound is already
    # verified by the cheap unconstrained excess. The projected-gradient solve is needed ONLY to
    # report how tight the bound is, not whether it holds -- and it was 400 of the ~634 matvecs
    # this function issued per scene, on an operator that streams nnz x 512 x 8 bytes each time
    # (330 GB per matvec on a median nonfrozen scene). That is why the first full run did not
    # finish overnight.
    out["bound_ball_holds_cheap"] = bool(2.0 * out["sum_o_gram"] >= out["excess"])
    if not do_ball:
        out["bound_ball"] = 2.0 * out["sum_o_gram"]
        out["bound_ball_over_excess_unconstrained"] = (
            out["bound_ball"] / max(out["excess"], 1e-30))
        return out
    # Omega <= 2 is what makes the bound operator-only and finite, and it needs ||x_j|| <= 1.
    # That is false for the unconstrained optimum (its norms run to 1e6 on these scenes), but it
    # is exactly the feasible set the cheap estimator already lives in: X' is a convex
    # combination of unit-norm per-view features, so ||x'_j|| <= 1 always. Restricting the target
    # to that ball therefore costs the estimator nothing, keeps X' feasible, and makes
    #     L(X') - L(Xc)  <=  (Omega^2 / 2) sum_i o_i  <=  2 sum_i o_i
    # a legitimate bound with no solve and no max over a heavy tail.
    #
    # Differences of the objective need no ray data: L(X) - L(Y) = tr(X'GX) - tr(Y'GY)
    # - 2 tr((X-Y)^T A^T B), and ||B||^2 cancels.
    # Gershgorin: lam_max(G) <= max_j sum_k |G_jk| = max_j rowsum(G), and G >= 0 entrywise, so
    # this is free. The previous power iteration cost 30 extra matvecs for the same purpose.
    lip = float((dg + torch.zeros(n, dtype=torch.float64, device=dev).index_add_(
        0, rr.long(), vv)).max())
    Xc = Xp_.clone()
    step = 1.0 / max(lip, 1e-30)
    for _ in range(proj_iters):
        Xc -= step * 2.0 * (Gmv(Xc) - B)
        nrm = Xc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        Xc = Xc * torch.clamp(1.0 / nrm, max=1.0)

    def obj_diff(X, Y):
        return quad(X) - quad(Y) - 2.0 * float(((X - Y) * B).sum())

    out["excess_ball"] = obj_diff(Xp_, Xc)
    out["bound_ball"] = 2.0 * out["sum_o_gram"]
    out["bound_ball_over_excess"] = out["bound_ball"] / max(out["excess_ball"], 1e-30)
    out["bound_ball_holds"] = bool(out["bound_ball"] >= out["excess_ball"])
    out["xc_norm_max"] = float(Xc.norm(dim=-1).max())
    # Xc must not be WORSE than X' -- if it is, the projected solve did not converge and the
    # excess is meaningless rather than small.
    out["ball_converged"] = bool(out["excess_ball"] >= -1e-6 * abs(out["signal"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--ball", action="store_true",
                    help="also solve for the unit-ball optimum (expensive; validity does not "
                         "need it, only the tightness number does)")
    ap.add_argument("--proj-iters", type=int, default=400,
                    help="projected-gradient steps for the unit-ball-constrained optimum")
    ap.add_argument("--out", default="artifacts/scannet/bound_validation.json")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, arm, a.iters, a.proj_iters, a.ball, a.tol)
            except FileNotFoundError as e:
                print(f"[{arm}/{sc}] missing: {os.path.basename(str(e))}"); continue
            except Exception as e:
                print(f"[{arm}/{sc}] SKIP {type(e).__name__}: {e}"); continue
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)     # written per scene, not at the end
            if "rel_excess" in r:
                print(f"[{arm}/{sc}] mean o {r['mean_o_gram']:.4f}  rel excess "
                      f"{r['rel_excess']:.3e}  dirichlet/excess {r['dirichlet_over_excess']:.2f}"
                      f"  omega-bound/excess {r['bound_over_excess']:.2e}  |xh|max "
                      f"{r['xh_norm_max']:.1e}  blowup {r['frac_blowup']:.2%}  rowsumG/D "
                      f"{r['rowsumG_over_D_median']:.3f}  cg {r['cg_residual']:.1e}  "
                      + (f"|| BALL: 2*sum_o/excess {r['bound_ball_over_excess']:.1f} "
                         f"holds={r['bound_ball_holds']} conv={r['ball_converged']}"
                         if "bound_ball_over_excess" in r else
                         f"|| 2*sum_o/excess "
                         f"{r['bound_ball_over_excess_unconstrained']:.1f} "
                         f"ball-holds={r['bound_ball_holds_cheap']}"))
            else:
                print(f"[{arm}/{sc}] mean o {r['mean_o_stats']:.4f}  (no gram cache: "
                      f"stats form only)")
    json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== mean over scenes, by arm ===")
    print(f"{'arm':<18}{'mean o_i':>10}{'rel excess':>13}{'dirich/exc':>12}"
          f"{'omega-b/exc':>13}{'ball-b/exc':>12}{'blowup':>9}{'rowsumG/D':>11}{'n':>4}")
    for arm in a.arms.split(","):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        mo = np.mean([r.get("mean_o_gram", r["mean_o_stats"]) for r in rs])
        has = [r for r in rs if "rel_excess" in r]
        f = lambda k: np.mean([r[k] for r in has]) if has else float("nan")
        print(f"{arm:<18}{mo:>10.4f}{f('rel_excess'):>13.3e}"
              f"{f('dirichlet_over_excess'):>12.2f}{f('bound_over_excess'):>13.2e}"
              f"{f('bound_ball_over_excess') if any('bound_ball_over_excess' in r for r in rs) else f('bound_ball_over_excess_unconstrained'):>12.1f}{f('frac_blowup'):>9.2%}{f('rowsumG_over_D_median'):>11.3f}{len(rs):>4}")
    bad = [r for r in rows if r.get("bound_holds") is False]
    print(f"\nbound violations: {len(bad)} / {sum('bound_holds' in r for r in rows)}")
    agree = [r["o_form_ratio"] for r in rows if "o_form_ratio" in r]
    if agree:
        print(f"sum_o gram/stats agreement: mean {np.mean(agree):.4f}  "
              f"min {np.min(agree):.4f}  max {np.max(agree):.4f}   (1.0 confirms (P2))")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
