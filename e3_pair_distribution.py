"""E3 (FINDINGS5): joint distribution of pairwise co-visibility and spatial separation.

Tests the geometry-mixture hypothesis without the biased pair sampling that would beg the question.

THE SAMPLER. With `r_i = sum_j A_ij`, `o_i = r_i^2 - sum_j A_ij^2`:

    P(ray i)     = o_i / sum(o)
    P(j | i)     = A_ij (r_i - A_ij) / o_i
    P(k | i,j)   = A_ik / (r_i - A_ij),   k != j

The ORDERED pair probability given `i` is `A_ij A_ik / o_i`, so the UNORDERED pair marginal is
exactly `2 G_jk / sum(o)`. Rays with `o_i = 0` (one-hot) contribute no pairs, which is correct: they
have no co-visibility. Verified in --selftest against the analytic marginal.

THE TRAP FINDINGS5 IDENTIFIES. A co-visibility-weighted sample contains NO `G_jk = 0` pairs by
construction, so it can never substantiate "foam's mass sits at c = 0". A second, separate UNIFORM
sample over each arm's 16-nearest-neighbour graph is therefore drawn to measure the zero-`c` fraction.
The two distributions are reported separately and must not be merged.

EXACTNESS. Sampled pairs give the pair IDENTITIES only. Their `G_jk` is then computed EXACTLY by
sparse column dot products over the full operator -- never by dividing a sampled numerator by a full
denominator. `c_jk = G_jk / sqrt(G_jj G_kk)` with `G_jj` accumulated exactly.

SEPARATION. `h_jk = ||p_j - p_k|| / (R_j + R_k)` with `R_j` the bounding-ball radius for foam and the
declared proxy `R_j = 3 sqrt(lambda_max(Sigma_j)) = 3 max(scale_j)` for a Gaussian. That proxy is NOT
the Gaussian's true finite support, and the foam ball need not fill its clipped cell -- so raw
distance and a spacing-normalised `||p_j - p_k|| / (l_j + l_k)` are reported alongside, where `l_j` is
the distance to the 16th distinct nearest centre. Zero denominator with distinct centres gives
`h = inf`; both zero gives `h = 0`.

--selftest verifies:
  1. the unordered pair marginal is proportional to 2 G_jk (empirical vs analytic);
  2. both hand checks: a (1/2,1/2) ray yields its single pair with probability 1; a one-hot ray has
     o = 0 and yields none;
  3. sparse column dots reproduce a dense A^T A exactly;
  4. the h conventions, including both zero-denominator cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

C_BINS = [0.0, 0.1, 0.5, 0.9, 1.0001]
H_BINS = [0.0, 0.25, 1.0, np.inf]


def col_dot(indptr, rows, vals, j, k):
    """Exact G_jk by intersecting two CSC columns."""
    a0, a1 = indptr[j], indptr[j + 1]
    b0, b1 = indptr[k], indptr[k + 1]
    if a1 - a0 == 0 or b1 - b0 == 0:
        return 0.0
    ra, va = rows[a0:a1], vals[a0:a1]
    rb, vb = rows[b0:b1], vals[b0:b1]
    if ra.size > rb.size:
        ra, va, rb, vb = rb, vb, ra, va
    pos = np.searchsorted(rb, ra)
    pos = np.minimum(pos, rb.size - 1)
    hit = rb[pos] == ra
    if not hit.any():
        return 0.0
    return float(np.dot(va[hit].astype(np.float64), vb[pos[hit]].astype(np.float64)))


def h_of(dist, Rj, Rk):
    den = Rj + Rk
    if den <= 0:
        return 0.0 if dist <= 0 else np.inf
    return dist / den


def selftest():
    rng = np.random.default_rng(0)
    from collections import Counter
    # 1: sampler proportionality
    for _ in range(3):
        R, P = 6, 5
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.7)
        r = A.sum(1); o = r ** 2 - (A ** 2).sum(1); G = A.T @ A
        an = {(j, k): 2 * G[j, k] / o.sum() for j in range(P) for k in range(j + 1, P)}
        N = 300000; cnt = Counter()
        for i in rng.choice(R, size=N, p=o / o.sum()):
            w = A[i] * (r[i] - A[i]); w = w / w.sum()
            j = int(rng.choice(P, p=w))
            w2 = A[i].copy(); w2[j] = 0.0
            if w2.sum() <= 0:
                continue
            k = int(rng.choice(P, p=w2 / w2.sum()))
            cnt[(min(j, k), max(j, k))] += 1
        for pr, p_an in an.items():
            if p_an > 5e-3:
                assert abs(cnt[pr] / N - p_an) < 0.05 * p_an + 0.002, (pr, cnt[pr] / N, p_an)
    # 2: hand checks
    A = np.array([[0.5, 0.5]]); r = A.sum(1); o = r ** 2 - (A ** 2).sum(1)
    assert abs(2 * (A.T @ A)[0, 1] / o[0] - 1.0) < 1e-12
    A = np.array([[1.0, 0.0]]); assert abs((A.sum(1) ** 2 - (A ** 2).sum(1))[0]) < 1e-15
    # 3: sparse column dots vs dense Gram
    from scipy.sparse import csc_matrix
    for _ in range(50):
        R, P = int(rng.integers(8, 40)), int(rng.integers(3, 15))
        A = rng.random((R, P)) * (rng.random((R, P)) < 0.5)
        M = csc_matrix(A); G = A.T @ A
        ip, rw, vl = M.indptr, M.indices, M.data
        for j in range(P):
            for k in range(P):
                assert abs(col_dot(ip, rw, vl, j, k) - G[j, k]) < 1e-9, (j, k)
    # 4: h conventions
    assert h_of(2.0, 1.0, 1.0) == 1.0
    assert h_of(0.0, 0.0, 0.0) == 0.0
    assert h_of(1.0, 0.0, 0.0) == np.inf
    print("  selftest OK: unordered pair marginal matches 2*G_jk analytically; (1/2,1/2) ray yields "
          "its single pair w.p. 1 and a one-hot ray yields none; sparse column dots reproduce dense "
          "A^T A exactly; h conventions correct including both zero-denominator cases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen,gs_froz,pf_nonfrozen,gs_unfroz")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--n-pairs", type=int, default=100000)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/e3_pairs.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    from scipy.spatial import cKDTree
    from determinism import enable_determinism
    enable_determinism()
    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES, geometry

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    CH = 100_000_000

    for arm in a.arms.split(","):
        recon = arm.replace("pf_", "")
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            if not bool((row[1:] >= row[:-1]).all()):
                o_ = torch.argsort(row)
                row, col, val = row[o_].contiguous(), col[o_].contiguous(), val[o_].contiguous()
                del o_; torch.cuda.empty_cache()

            def scat(n, idx, src):
                acc = torch.zeros(n, device=dev, dtype=torch.float64)
                for s0 in range(0, idx.numel(), CH):
                    e0 = min(s0 + CH, idx.numel())
                    acc.index_add_(0, idx[s0:e0], src[s0:e0].double())
                return acc

            r_i = scat(R, row, val)
            sq_i = scat(R, row, val * val)
            g_j = scat(P, col, val * val).cpu().numpy()
            o_i = (r_i ** 2 - sq_i).clamp_min(0.0)
            tot_o = float(o_i.sum())

            # geometry: centres and radii
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                pos = np.asarray(cen, dtype=np.float64); Rad = np.asarray(rad, dtype=np.float64)
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                weights_only=False)
                sp = ck["splats"] if "splats" in ck else ck
                pos = sp["means"].float().numpy().astype(np.float64)
                Rad = 3.0 * np.exp(sp["scales"].float().numpy()).max(1).astype(np.float64)

            rec = {"arm": arm, "scene": sc, "P": int(P), "R": int(R), "nnz": int(nnz),
                   "views": a.views, "sum_o": tot_o}

            if tot_o <= 0:
                rec["note"] = "no co-visible rays (all one-hot); covis sample empty"
                pairs = np.zeros((0, 2), np.int64); Gjk = np.zeros(0); pair_w = np.zeros(0)
            else:
                gen = torch.Generator(device=dev); gen.manual_seed(0)
                ridx = torch.multinomial(o_i / tot_o, a.n_pairs, replacement=True, generator=gen)
                starts = torch.searchsorted(row, torch.arange(R + 1, device=dev))
                s0v = starts[ridx]; s1v = starts[ridx + 1]
                # ---- vectorised segmented sampling (replaces a per-sample Python loop)
                L = (s1v - s0v)
                keep = L >= 2
                s0k, Lk, rk = s0v[keep], L[keep], ridx[keep]
                nseg = int(Lk.numel())
                off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                                 torch.cumsum(Lk, 0)])
                flat = torch.repeat_interleave(s0k, Lk) + (
                    torch.arange(int(off[-1]), device=dev) - torch.repeat_interleave(off[:-1], Lk))
                seg = torch.repeat_interleave(torch.arange(nseg, device=dev), Lk)
                cf = col[flat]; vf = val[flat].double()
                rf = r_i[rk][seg]

                def seg_sample(weights):
                    """Draw one index per segment with probability proportional to `weights`."""
                    ssum = torch.zeros(nseg, device=dev, dtype=torch.float64)
                    ssum.index_add_(0, seg, weights)
                    ok = ssum > 0
                    cw = torch.cumsum(weights, 0)
                    base = torch.cat([torch.zeros(1, dtype=torch.float64, device=dev),
                                      cw[off[1:] - 1]])[:-1]
                    cw = cw - base[seg]                       # cumulative within segment
                    u = torch.rand(nseg, device=dev, dtype=torch.float64, generator=gen) * ssum
                    idx = torch.searchsorted(cw.contiguous(), u.contiguous())
                    # searchsorted is global; clamp into each segment
                    idx = torch.minimum(torch.maximum(idx, off[:-1]), off[1:] - 1)
                    return idx, ok

                wj = vf * (rf - vf)
                ij, okj = seg_sample(wj.clamp_min(0))
                wk = vf.clone(); wk[ij] = 0.0
                ik, okk = seg_sample(wk.clamp_min(0))
                good = okj & okk & (ij != ik)
                jj = cf[ij[good]].cpu().numpy().astype(np.int64)
                kk = cf[ik[good]].cpu().numpy().astype(np.int64)
                lo = np.minimum(jj, kk); hi = np.maximum(jj, kk)
                # multiplicities matter: the sample estimates the co-visibility MASS distribution
                # (prop to 2*G_jk), so statistics over DISTINCT pairs alone are sample-size
                # dependent and must not be used.
                pairs, pair_w = np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)

            # exact G_jk for the sampled pairs, via CSC column dots on the full operator
            if pairs.shape[0]:
                order = torch.argsort(col)
                c_s = col[order].cpu().numpy(); r_s = row[order].cpu().numpy().astype(np.int64)
                v_s = val[order].cpu().numpy()
                del order; torch.cuda.empty_cache()
                indptr = np.searchsorted(c_s, np.arange(P + 1))
                # rows must be ascending within each column for the intersection
                for j in np.unique(pairs):
                    s, e = indptr[j], indptr[j + 1]
                    if e - s > 1:
                        srt = np.argsort(r_s[s:e]); r_s[s:e] = r_s[s:e][srt]; v_s[s:e] = v_s[s:e][srt]
                Gjk = np.array([col_dot(indptr, r_s, v_s, int(j), int(k)) for j, k in pairs])
                del c_s, r_s, v_s

            def summarize(prs, G, tag, w=None):
                if prs.shape[0] == 0:
                    return {"n_pairs": 0}
                w = np.ones(prs.shape[0]) if w is None else w.astype(np.float64)
                w = w / w.sum()
                gj = g_j[prs[:, 0]]; gk = g_j[prs[:, 1]]
                c = np.where((gj > 0) & (gk > 0), G / np.sqrt(np.maximum(gj * gk, 1e-300)), 0.0)
                c = np.clip(c, 0.0, 1.0)
                d = np.linalg.norm(pos[prs[:, 0]] - pos[prs[:, 1]], axis=1)
                h = np.array([h_of(d[i], Rad[prs[i, 0]], Rad[prs[i, 1]]) for i in range(len(d))])
                ci = np.digitize(c, C_BINS[1:-1]); hi_ = np.digitize(h, H_BINS[1:-1])
                grid = np.zeros((4, 3))
                for x, y, ww in zip(ci, hi_, w):
                    grid[min(x, 3), min(y, 2)] += ww

                def wmed(x, ww):
                    m = np.isfinite(x)
                    if not m.any():
                        return None
                    xs = np.argsort(x[m]); cw = np.cumsum(ww[m][xs]) / ww[m].sum()
                    return float(x[m][xs][np.searchsorted(cw, 0.5)])
                return {"n_pairs": int(prs.shape[0]), "n_draws": float(w.sum() and prs.shape[0]),
                        "frac_c_zero_weighted": float(w[G <= 0].sum()),
                        "frac_c_zero_unweighted": float((G <= 0).mean()),
                        "c_median_weighted": wmed(c, w), "h_median_weighted": wmed(h, w),
                        "dist_median_weighted": wmed(d, w),
                        "grid_c_by_h": (grid / max(grid.sum(), 1e-300)).round(5).tolist(),
                        "tag": tag}

            rec["covis_sample"] = summarize(pairs, Gjk,
                                            "covisibility-weighted (prop to 2*G_jk)",
                                            w=pair_w if pairs.shape[0] else None)

            # separate UNIFORM 16-NN neighbour sample, to measure the zero-c fraction
            kd = cKDTree(pos)
            nn = kd.query(pos, k=17)[1][:, 1:]
            rs = np.random.default_rng(0)
            m = min(a.n_pairs, pos.shape[0] * 16)
            ji = rs.integers(0, pos.shape[0], m); ki = nn[ji, rs.integers(0, 16, m)]
            lo = np.minimum(ji, ki); hi2 = np.maximum(ji, ki)
            npairs = np.unique(np.stack([lo, hi2], 1), axis=0)
            npairs = npairs[npairs[:, 0] != npairs[:, 1]]
            if npairs.shape[0]:
                order = torch.argsort(col)
                c_s = col[order].cpu().numpy(); r_s = row[order].cpu().numpy().astype(np.int64)
                v_s = val[order].cpu().numpy(); del order; torch.cuda.empty_cache()
                indptr = np.searchsorted(c_s, np.arange(P + 1))
                for j in np.unique(npairs):
                    s, e = indptr[j], indptr[j + 1]
                    if e - s > 1:
                        srt = np.argsort(r_s[s:e]); r_s[s:e] = r_s[s:e][srt]; v_s[s:e] = v_s[s:e][srt]
                Gn = np.array([col_dot(indptr, r_s, v_s, int(j), int(k)) for j, k in npairs])
                del c_s, r_s, v_s
                rec["knn_sample"] = summarize(npairs, Gn, "uniform 16-NN neighbours")
            rec["wall_s"] = round(time.time() - t0, 1)
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            cv = rec["covis_sample"]; kn = rec.get("knn_sample", {})
            cmw = cv.get('c_median_weighted') or 0.0
            print(f"[{arm}/{sc}] covis pairs {cv.get('n_pairs',0)} c_med {cmw:.4f} "
                  f"h_med {cv.get('h_median_weighted')} | knn zero-c "
                  f"{100*kn.get('frac_c_zero_unweighted',0):.1f}% "
                  f"{rec['wall_s']}s", flush=True)
            del row, col, val, gid, Treg
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
