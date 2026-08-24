"""
test_davison_limited_angle.py

Empirical test of Davison 1983 (SIAM J. Appl. Math. 43(2):428-448),
"The ill-conditioned nature of the limited angle tomography problem",
against the operator we actually have in the CLIP-feature-lifting problem.

PART A. Davison's OWN operator, exactly.  Section 3, Gegenbauer lambda=1
        (p.439) gives the matrix of the finite-rank operator A_n in closed
        form:
            (mat A_n)_{k,l} = sin[2(k-l)Theta] / ((n+1)(k-l))
        with the diagonal 2*Theta/(n+1).  Its eigenvalues ARE the eigenvalues
        of PP* restricted to V_n, so the singular values of the limited-angle
        Radon transform are their square roots.  No discretization error.
        We reproduce his Fig. 1 (smallest eigenvalue vs Theta) and measure
        whether the decay in the index is polynomial or exponential.

PART B. A discretized 2D limited-angle X-ray transform with EXACT line/cell
        intersection lengths (Siddon), so that transparent and occluded
        operators can be compared on identical geometry.

PART C. The OCCLUDED analogue: rays TERMINATE at the first opaque cell,
        weights w_k = T_k * alpha_k with T_{k+1} = T_k(1-alpha_k), so rows
        are substochastic and the transmittance telescopes.  This is our A.

PART D. Where does the ill-conditioning come from?  Full cond vs cond after
        deleting never-touched columns.

CPU only, ~1 minute, no GPU, no shared-env writes.
"""

import json
import os
import numpy as np


# ==========================================================================
# PART A helpers -- Davison's exact prolate matrix
# ==========================================================================

def davison_matrix(n, Theta):
    """(mat A_n)_{k,l} = sin[2(k-l)Theta] / ((n+1)(k-l)), size (n+1)x(n+1).
    Davison 1983, p.439 (Gegenbauer case lambda=1).  Diagonal by l'Hopital
    is 2*Theta/(n+1).  Equals Slepian's p(n+1, W) / (n+1) with W = Theta/pi."""
    k = np.arange(n + 1)
    D = k[:, None] - k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.sin(2.0 * D * Theta) / ((n + 1) * D)
    np.fill_diagonal(M, 2.0 * Theta / (n + 1))
    return M


# ==========================================================================
# PART B/C helpers -- exact Siddon traversal on a uniform grid over [-1,1]^2
# ==========================================================================

def siddon(theta, s, N, half=1.8):
    """Exact intersection of the line {x cos th + y sin th = s} with an
    N x N uniform grid over [-1,1]^2.  Returns (cell_indices, lengths) in
    TRAVEL ORDER along direction d = (-sin th, cos th)."""
    c, sn = np.cos(theta), np.sin(theta)
    p0 = np.array([s * c, s * sn])
    d = np.array([-sn, c])

    h = 2.0 / N
    ts = [-half, half]
    for ax in (0, 1):
        if abs(d[ax]) > 1e-12:
            planes = -1.0 + h * np.arange(N + 1)
            ts.append((planes - p0[ax]) / d[ax])
    ts = np.concatenate([np.atleast_1d(t) for t in ts])
    ts = np.sort(ts[(ts >= -half) & (ts <= half)])
    if len(ts) < 2:
        return np.empty(0, np.int64), np.empty(0)

    lens = np.diff(ts)
    mids = 0.5 * (ts[:-1] + ts[1:])
    pts = p0[None, :] + mids[:, None] * d[None, :]
    ix = np.floor((pts[:, 0] + 1.0) / h).astype(np.int64)
    iy = np.floor((pts[:, 1] + 1.0) / h).astype(np.int64)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (lens > 1e-12)
    return (iy[ok] * N + ix[ok]), lens[ok]


def build_rays(theta_max_deg, n_angles, n_offsets, one_sided=False):
    Th = np.deg2rad(theta_max_deg)
    if one_sided:
        thetas = np.linspace(0.0, 2.0 * Th, n_angles)
    else:
        thetas = np.linspace(-Th, Th, n_angles)
    offs = np.linspace(-0.995, 0.995, n_offsets)
    return thetas, offs


def build_transparent(N, thetas, offs):
    rows = np.zeros((len(thetas) * len(offs), N * N))
    r = 0
    for th in thetas:
        for s in offs:
            idx, ln = siddon(th, s, N)
            if len(idx):
                np.add.at(rows[r], idx, ln)
            r += 1
    return rows


def make_opacity_field(N, alpha_solid, n_blobs=3):
    gx, gy = np.meshgrid((np.arange(N) + .5) / N * 2 - 1,
                         (np.arange(N) + .5) / N * 2 - 1, indexing="xy")
    alpha = np.zeros((N, N))
    for (cx, cy), r in zip([(-.35, -.20), (.30, .25), (.05, -.55)][:n_blobs],
                           [.42, .34, .24][:n_blobs]):
        alpha[((gx - cx) ** 2 + (gy - cy) ** 2) < r * r] = alpha_solid
    return alpha.reshape(-1)


def build_occluded(N, thetas, offs, alpha_flat, stop_T=1e-5):
    """w_k = T_k * alpha_k * len_k(normalized), T *= (1 - alpha_k)^len.
    Row-substochastic, telescoping transmittance, terminates at the surface."""
    rows = np.zeros((len(thetas) * len(offs), N * N))
    nnz = []
    h = 2.0 / N
    r = 0
    for th in thetas:
        for s in offs:
            idx, ln = siddon(th, s, N)
            T = 1.0
            cnt = 0
            for j, L in zip(idx, ln):
                a = alpha_flat[j]
                if a <= 0.0:
                    continue
                # per-cell absorption scaled by path length through the cell
                ae = 1.0 - (1.0 - a) ** (L / h)
                rows[r, j] += T * ae
                T *= (1.0 - ae)
                cnt += 1
                if T < stop_T:
                    break
            nnz.append(cnt)
            r += 1
    return rows, float(np.mean(nnz))


def lowpass_basis(N, K):
    """Orthonormal 2D-DCT basis restricted to the K lowest radial
    frequencies.  Stands in for Davison's smooth a priori class and removes
    the pixel-aliasing floor that otherwise masks the true decay."""
    x = (np.arange(N) + 0.5) / N
    freqs = []
    for u in range(N):
        for v in range(N):
            freqs.append((u * u + v * v, u, v))
    freqs.sort()
    cols = []
    for _, u, v in freqs[:K]:
        bu = np.cos(np.pi * u * x)
        bv = np.cos(np.pi * v * x)
        M = np.outer(bv, bu)
        cols.append((M / np.linalg.norm(M)).reshape(-1))
    B = np.array(cols).T
    Q, _ = np.linalg.qr(B)
    return Q


# ==========================================================================
# Spectrum analysis
# ==========================================================================

def fit_decay(sv, floor=1e-13):
    sv = sv / sv[0]
    k = int(np.sum(sv > floor))
    k = max(k, 5)
    i = np.arange(1, k + 1, dtype=float)
    y = np.log(sv[:k])
    Ae = np.vstack([np.ones_like(i), -i]).T
    ce, *_ = np.linalg.lstsq(Ae, y, rcond=None)
    r2e = 1 - np.sum((y - Ae @ ce) ** 2) / np.sum((y - y.mean()) ** 2)
    Ap = np.vstack([np.ones_like(i), -np.log(i)]).T
    cp, *_ = np.linalg.lstsq(Ap, y, rcond=None)
    r2p = 1 - np.sum((y - Ap @ cp) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(n_used=k, exp_rate=float(ce[1]), r2_exp=float(r2e),
                poly_p=float(cp[1]), r2_pol=float(r2p),
                verdict=("EXP" if r2e > r2p else "POLY"))


def analyse(A, label, B=None):
    if B is not None:
        A = A @ B
    m, n = A.shape
    cn = np.linalg.norm(A, axis=0)
    dead = int(np.sum(cn <= 1e-13 * max(cn.max(), 1e-300)))
    sv = np.linalg.svd(A, compute_uv=False)
    svn = sv / sv[0]
    f = fit_decay(sv)
    live = cn > 1e-13 * max(cn.max(), 1e-300)
    if 0 < live.sum() < n:
        svl = np.linalg.svd(A[:, live], compute_uv=False)
        svl = svl / svl[0]
        cond_live = float(svl[0] / max(svl[-1], 1e-300))
        f_live = fit_decay(svl)
    else:
        cond_live = float(svn[0] / max(svn[-1], 1e-300))
        f_live = f
    r = dict(label=label, m=m, n=n, dead=dead, dead_frac=dead / n,
             cond=float(svn[0] / max(svn[-1], 1e-300)), cond_live=cond_live,
             log10_sv=[float(np.log10(max(svn[min(n - 1, int(q * n))], 1e-300)))
                       for q in (0.1, 0.25, 0.5, 0.75, 0.95)],
             fit=f, fit_live=f_live)
    return r


def line(r, extra=""):
    f = r["fit_live"]
    return (f"  {r['label']:<30s} dead={r['dead_frac']*100:5.1f}%  "
            f"log10cond={np.log10(r['cond']):7.2f}  live={np.log10(r['cond_live']):6.2f}  "
            f"sv@[10,25,50,75,95]%={['%.2f' % v for v in r['log10_sv']]}\n"
            f"  {'':30s} live-spectrum fit: EXP rate={f['exp_rate']:.4f} R2={f['r2_exp']:.3f}"
            f" | POLY p={f['poly_p']:.2f} R2={f['r2_pol']:.3f}  -> {f['verdict']}{extra}")


# ==========================================================================
def main():
    res = {}

    # ---------------- PART A ----------------
    print("=" * 104)
    print("PART A -- DAVISON'S EXACT OPERATOR (his eq. for mat A_n, p.439, Gegenbauer lambda=1)")
    print("  eigenvalues of A_n = eigenvalues of PP* on V_n; singular values of P are their sqrt")
    print("=" * 104)
    res["davison_exact"] = []
    print(f"  {'n':>4s} {'Theta':>6s} | {'log10 lam_min':>13s} {'log10 lam_max':>13s} "
          f"| {'EXP fit rate':>12s} {'R2':>6s} | {'POLY p':>7s} {'R2':>6s} | verdict")
    for n in (15, 25, 40):
        for Wdeg in (90, 75, 60, 45, 30, 20, 12, 6):
            M = davison_matrix(n, np.deg2rad(Wdeg))
            ev = np.sort(np.linalg.eigvalsh(M))[::-1]
            ev = np.clip(ev, 1e-300, None)
            sv = np.sqrt(ev)                      # singular values of P on V_n
            f = fit_decay(sv, floor=1e-15)
            rec = dict(n=n, Theta_deg=Wdeg,
                       log10_lam_min=float(np.log10(ev[-1])),
                       log10_lam_max=float(np.log10(ev[0])), fit=f)
            res["davison_exact"].append(rec)
            print(f"  {n:4d} {Wdeg:5d}d | {rec['log10_lam_min']:13.3f} "
                  f"{rec['log10_lam_max']:13.3f} | {f['exp_rate']:12.4f} {f['r2_exp']:6.3f} "
                  f"| {f['poly_p']:7.2f} {f['r2_pol']:6.3f} | {f['verdict']}")
        print()

    # ---------------- PART B/C ----------------
    N = 28
    na, no = 40, 48
    wedges = [90, 60, 45, 30, 20, 12]
    K = 160
    B = lowpass_basis(N, K)
    alpha_hard = make_opacity_field(N, 1.0)
    alpha_soft = make_opacity_field(N, 0.06)

    print("=" * 104)
    print(f"PARTS B/C -- discretized operators, grid {N}x{N}={N*N} cells, "
          f"{na} angles x {no} offsets = {na*no} rays")
    print(f"          smooth-basis variant projects onto the {K} lowest 2D-DCT modes")
    print("=" * 104)

    for tag, use_B in (("PIXEL BASIS", False), (f"LOW-PASS BASIS (K={K} lowest DCT modes)", True)):
        Bb = B if use_B else None
        print(f"\n########## {tag} ##########")
        for key, kind in (("transparent", "T"), ("occ_hard", "OH"),
                          ("occ_soft", "OS"), ("occ_1sided", "O1")):
            res.setdefault(f"{key}_{'sm' if use_B else 'px'}", [])
            print(f"\n--- {key} ---")
            for W in wedges:
                th, off = build_rays(W, na, no, one_sided=(kind == "O1"))
                extra = ""
                if kind == "T":
                    A = build_transparent(N, th, off)
                else:
                    al = alpha_soft if kind == "OS" else alpha_hard
                    A, cpr = build_occluded(N, th, off, al)
                    extra = f"  [cells/ray={cpr:.1f}]"
                r = analyse(A, f"{key} Theta={W}d", Bb)
                res[f"{key}_{'sm' if use_B else 'px'}"].append(r)
                print(line(r, extra))

    # ---------------- PART D ----------------
    print("\n" + "=" * 104)
    print("PART D -- SOURCE OF THE ILL-CONDITIONING (pixel basis)")
    print("  drop = decades of cond() removed by simply DELETING never-touched columns.")
    print("  Large drop -> RANK DEFICIENCY (coverage). Small drop -> genuine ill-conditioning.")
    print("=" * 104)
    for key in ("transparent_px", "occ_hard_px", "occ_soft_px", "occ_1sided_px"):
        print(f"\n {key}")
        for r in res[key]:
            print(f"   {r['label']:<24s} dead={r['dead_frac']*100:5.1f}%  "
                  f"log10cond {np.log10(r['cond']):7.2f} -> {np.log10(r['cond_live']):6.2f}  "
                  f"(drop {np.log10(r['cond'])-np.log10(r['cond_live']):6.2f} decades)")

    # ---------------- PART E ----------------
    # Davison PROVES (p.440-441) that the limited-angle Radon transform has a
    # TRIVIAL KERNEL for every Theta > 0: if Pf = 0 on [-Theta,Theta] then each
    # trig polynomial sum_k pi f_{n,k} c_{n,k} e^{ik theta} vanishes on an
    # interval, hence identically, and since all c_{n,k} != 0, f = 0.
    # So his problem is UNIQUELY SOLVABLE and merely ill-conditioned.
    # Ours is not.  Quantify the difference, and check how it scales with
    # resolution (does the transparent collapse persist? is it high-frequency?)
    print("\n" + "=" * 104)
    print("PART E -- KERNEL / RANK DEFICIENCY vs RESOLUTION  (Theta = 30 deg, pixel basis)")
    print("  Davison p.440-441 PROVES ker(P) = {0} for every Theta > 0.")
    print("  'dead' = columns no ray ever touches (exact structural zeros).")
    print("  'numrank' = rank at tol = max(m,n)*eps*sigma_max.")
    print("=" * 104)
    res["resolution"] = []
    print(f"  {'N':>4s} {'cells':>6s} | {'operator':<14s} {'dead%':>7s} {'numrank%':>9s} "
          f"{'log10cond':>10s} {'log10cond_live':>15s}")
    for Nr in (16, 24, 32, 40):
        th, off = build_rays(30, 40, 48)
        for nm, kind in (("transparent", "T"), ("occluded", "OH")):
            if kind == "T":
                A = build_transparent(Nr, th, off)
            else:
                A, _ = build_occluded(Nr, th, off, make_opacity_field(Nr, 1.0))
            cn = np.linalg.norm(A, axis=0)
            dead = int(np.sum(cn <= 1e-13 * cn.max()))
            sv = np.linalg.svd(A, compute_uv=False)
            tol = max(A.shape) * np.finfo(float).eps * sv[0]
            nr = int(np.sum(sv > tol))
            live = cn > 1e-13 * cn.max()
            svl = np.linalg.svd(A[:, live], compute_uv=False)
            rec = dict(N=Nr, op=nm, dead_frac=dead / A.shape[1],
                       numrank_frac=nr / min(A.shape),
                       log10cond=float(np.log10(sv[0] / max(sv[-1], 1e-300))),
                       log10cond_live=float(np.log10(svl[0] / max(svl[-1], 1e-300))))
            res["resolution"].append(rec)
            print(f"  {Nr:4d} {Nr*Nr:6d} | {nm:<14s} {rec['dead_frac']*100:7.2f} "
                  f"{rec['numrank_frac']*100:9.2f} {rec['log10cond']:10.2f} "
                  f"{rec['log10cond_live']:15.2f}")

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_davison_limited_angle_results.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
