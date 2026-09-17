"""Test the Laplacian error identity on a REAL scene, and check that it predicts.

THE THEOREM (verified synthetically in this project, see the curriculum note). With A row-stochastic,
G = A^T A, d = A^T 1 the column sums, D = diag(d), and L = D - G the co-visibility Laplacian:

    x'  =  (I - D^-1 L) x*          exactly, where  x* = G^-1 A^T B  and  x' = D^-1 A^T B

    x*_j - x'_j  =  (1/d_j) sum_{k != j} G_jk ( x*_j - x*_k )

    ||x*_j - x'_j||  <=  kappa_j * spread_j ,
        kappa_j = 1 - (sum_i A_ij^2)/(sum_i A_ij),   spread_j = max_k ||x*_j - x*_k||

An identity is not a contribution until it says something true about data. Three things are checked
here that synthetic tests cannot:

  1. the DISTRIBUTION of kappa_j on a real reconstruction -- is the coefficient actually small?
  2. whether the bound is TIGHT or vacuous, i.e. the ratio of the true error to kappa*spread;
  3. whether kappa_j * spread_j PREDICTS the per-primitive error (rank correlation), which is what
     would let it be used to decide where the fast solver may be trusted.

kappa_j is a COLUMN-side quantity. The 0.144 / 0.850 figures reported earlier in this project are
the ROW-side purity 1 - sum_j Ahat_ij^2, a different number measuring the same phenomenon from the
ray side. They are not interchangeable and this script computes the one the theorem uses.

COST. A^T A is P x P but SPARSE -- a foam ray touches ~2 cells, so co-visibility is local. It is
accumulated one view at a time and never densified. x* is obtained by sparse CG on a handful of
random projections of B rather than all F=512 columns: the identity is linear in B, so any
projection is a valid test, and the per-primitive error norms are preserved up to
Johnson-Lindenstrauss distortion.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--stats", default="artifacts/adaptive/s0062_stats_l3bb.pt")
    ap.add_argument("--proj", type=int, default=48, help="random projections of B to solve for")
    ap.add_argument("--views", type=int, default=0, help="0 = all views (must match the stats)")
    ap.add_argument("--ridge", type=float, default=1e-9,
                    help="ridge as a fraction of mean(d); floors G's null directions")
    ap.add_argument("--out", default="artifacts/laplacian_bound_scene0062.npz")
    a = ap.parse_args()

    import configargparse
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    import warp as wp

    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]

    ids = range(len(dh.cameras)) if a.views == 0 else range(min(a.views, len(dh.cameras)))
    G = sp.csr_matrix((P, P), dtype=np.float64)
    d = np.zeros(P)
    sq = np.zeros(P)                                  # sum_i A_ij^2, for kappa
    for vi in ids:
        op = export_operator_for_views(model, [dh.cameras[vi]], [vi])
        r = op.row_indices.cpu().numpy()
        c = op.col_indices.cpu().numpy()
        v = op.values.cpu().numpy().astype(np.float64)
        nrows = int(r.max()) + 1 if len(r) else 1
        Av = sp.csr_matrix((v, (r, c)), shape=(nrows, P))
        G = G + (Av.T @ Av).tocsr()
        d += np.asarray(Av.sum(0)).ravel()
        sq += np.asarray(Av.multiply(Av).sum(0)).ravel()
        if vi % 10 == 0:
            print(f"  view {vi}: G nnz {G.nnz:,}", flush=True)

    live = d > 1e-9
    print(f"\n{P:,} primitives, {int(live.sum()):,} with support")
    print(f"G: {G.nnz:,} nonzeros = {G.nnz / max(int(live.sum()), 1):.2f} per live primitive")

    # kappa_j: the theorem's coefficient
    kappa = np.zeros(P)
    kappa[live] = 1.0 - sq[live] / d[live]
    print(f"kappa_j (column-side co-visibility): median {np.median(kappa[live]):.4f}  "
          f"mean {kappa[live].mean():.4f}  p90 {np.percentile(kappa[live], 90):.4f}")
    print(f"  fraction of primitives with kappa < 0.05: {(kappa[live] < 0.05).mean():.1%}")

    # L is a Laplacian on the live block; verify before using it
    Gl = G[live][:, live].tocsr()
    dl = d[live]
    rowsum_G = np.asarray(Gl.sum(1)).ravel()
    print(f"\nrow sums of G == column sums of A? max abs diff "
          f"{np.abs(rowsum_G - dl).max():.3e}   <- Property 2 on real data")

    # A^T B comes free from the accumulator (numerator), so no second feature pass is needed
    st = torch.load(a.stats, map_location="cpu", weights_only=False)
    g = lambda k: (st[k] if isinstance(st, dict) else getattr(st, k)).float().numpy()
    AtB = g("numerator")[live]
    rng = np.random.default_rng(0)
    Rp = rng.normal(size=(AtB.shape[1], a.proj)) / np.sqrt(a.proj)
    Y = AtB @ Rp                                            # projected A^T B

    Dl = sp.diags(dl)
    Ll = (Dl - Gl).tocsr()
    xp = Y / dl[:, None]                                     # x' = D^-1 A^T B
    # x* must be solved to convergence or nothing below means anything. Plain CG hits the iteration
    # cap on every projection here -- G is badly conditioned, which is itself the reason a one-shot
    # diagonal approximation was attractive in the first place. Two changes: JACOBI PRECONDITIONING
    # with the very matrix D the paper lumps to (a preconditioner used the way preconditioners are
    # meant to be used -- inside an iteration that still converges to x*), and a ridge floor for the
    # rank-deficient directions, since primitives never separated by any ray leave G singular
    # (Module 2's degenerate case, which real scenes do contain).
    ridge = a.ridge * float(dl.mean())
    Gr = (Gl + sp.diags(np.full(len(dl), ridge))).tocsr()
    Minv = spl.LinearOperator(Gr.shape, matvec=lambda z: z / (dl + ridge), dtype=np.float64)
    print(f"\nsolving {a.proj} sparse systems for x* "
          f"(Jacobi-preconditioned CG, ridge {ridge:.3e}) ...", flush=True)
    xs = np.zeros_like(Y)
    n_bad, res = 0, []
    for k in range(a.proj):
        sol, info = spl.cg(Gr, Y[:, k], rtol=1e-12, maxiter=20000, M=Minv)
        r = np.linalg.norm(Gr @ sol - Y[:, k]) / max(np.linalg.norm(Y[:, k]), 1e-30)
        res.append(r)
        n_bad += int(info != 0 or r > 1e-8)
        xs[:, k] = sol
    print(f"  converged {a.proj - n_bad}/{a.proj}   max relative residual {max(res):.2e}")
    if n_bad:
        print("  [ABORT] x* not solved to tolerance; every number below would be an artefact.")
        return

    # CLAIM: x' = (I - D^-1 L) x*
    Lr = (sp.diags(dl + ridge) - Gr).tocsr()
    pred = xs - (Lr @ xs) / (dl + ridge)[:, None]
    xp = Y / (dl + ridge)[:, None]
    rel = np.linalg.norm(pred - xp) / max(np.linalg.norm(xp), 1e-12)
    print(f"identity  x' == (I - D^-1 L) x*  : relative residual {rel:.3e}")

    err = np.linalg.norm(xs - xp, axis=1)
    kap_l = kappa[live]

    # spread_j = max_k ||x*_j - x*_k|| over co-visible k
    Gc = Gl.tocoo()
    spread = np.zeros(len(dl))
    diff = np.linalg.norm(xs[Gc.row] - xs[Gc.col], axis=1)
    np.maximum.at(spread, Gc.row, np.where(Gc.row != Gc.col, diff, 0.0))
    bound = kap_l * spread
    ok = err <= bound + 1e-9
    # THE PROPERTY-2 DEFECT. rowsum(G)_j = sum_i A_ij * s_i with s_i the row sum of A, so
    #   defect_j := d_j - rowsum(G)_j = sum_i A_ij (1 - s_i)  >= 0
    # which is exactly the light that leaked past every primitive to the background, re-expressed
    # per primitive. When it is nonzero, L = D - G is NOT a Laplacian (rows do not sum to zero) and
    # the elementwise form gains a term:
    #   x*_j - x'_j = (1/d_j) sum_{k!=j} G_jk (x*_j - x*_k)  +  (defect_j/d_j) * x*_j
    # so the honest bound is  kappa_j*spread_j + (defect_j/d_j)*||x*_j||. The pure-difference form
    # is the idealisation; this is what a real reconstruction obeys.
    defect = (dl + ridge) - np.asarray(Gr.sum(1)).ravel()
    reld = defect / (dl + ridge)
    print(f"\nProperty-2 defect (1 - rowsum(G)/d): median {np.median(reld):.4f}  "
          f"mean {reld.mean():.4f}  p99 {np.percentile(reld, 99):.4f}")
    bound2 = bound + np.abs(reld) * np.linalg.norm(xs, axis=1)
    print(f"bound holds for {ok.mean():.4%} of primitives   (pure-difference form)")
    print(f"bound holds for {(err <= bound2 + 1e-9).mean():.4%} of primitives   "
          f"(with the defect term)")
    nz = bound > 1e-12
    print(f"tightness  err / (kappa*spread): median {np.median(err[nz] / bound[nz]):.4f}  "
          f"mean {(err[nz] / bound[nz]).mean():.4f}   (1.0 = tight, 0 = vacuous)")
    from scipy.stats import spearmanr
    print(f"does it PREDICT?  spearman(kappa*spread, err) = "
          f"{spearmanr(bound[nz], err[nz]).statistic:+.4f}")
    print(f"                  spearman(kappa alone,  err) = "
          f"{spearmanr(kap_l[nz], err[nz]).statistic:+.4f}")
    print(f"                  spearman(spread alone, err) = "
          f"{spearmanr(spread[nz], err[nz]).statistic:+.4f}")
    print(f"\nmean ||x*-x'|| {err.mean():.5f}   vs mean ||x*|| "
          f"{np.linalg.norm(xs, axis=1).mean():.5f}  "
          f"-> relative error {err.mean() / max(np.linalg.norm(xs, axis=1).mean(), 1e-12):.4%}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, kappa=kap_l.astype(np.float32), err=err.astype(np.float32),
                        spread=spread.astype(np.float32))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
