"""
test_bodin_rjmcmc_coverage.py

Test of the central claim of:
  Bodin & Sambridge (2009), "Seismic tomography with the reversible jump
  algorithm", Geophys. J. Int. 178(3), 1411-1436.

Claim under test (paper's own words):
  - Summary p.1411: "The size, position and shape of the cells defining the
    velocity model are directly determined by the data. ... The mobile position
    and number of cells means that global damping procedures, controlled by an
    optimal regularization parameter, are avoided."
  - Sec 2.1 p.1413: "this dynamic parametrization will adapt to the spatial
    variability in the information provided by the data."
  - Sec 4.1 p.1430: "There are no data in the ocean areas and according to
    Bayesian principles, we recover the mean and the variance of the prior
    probability density function."

Experiment: 2-D straight-ray (linear-in-slowness) traveltime tomography on the
unit square with DELIBERATELY UNEVEN ray coverage, including a region that no
ray touches at all (a "dead zone", the analogue of our 9.44% untouched cells).

We implement the paper's rj-MCMC exactly as specified:
  * model m = (c, v), n Voronoi nuclei, constant velocity per cell   [Sec 2.1]
  * forward t_j = sum_i L_ij / v_i                                    [eq. 3]
  * Gaussian likelihood on least-squares misfit                       [eq. 6,7]
  * uniform prior on n in [nmin,nmax], uniform on v in [Vmin,Vmax],
    uniform on nuclei positions                                       [eq. 9-14]
  * four proposals: velocity / birth / death / move                   [Sec 2.7]
  * acceptance terms                                                  [eq. 32,35,36]
Delayed rejection (Appendix A) is NOT implemented; the paper states it affects
only convergence rate, not the target distribution (Sec 2.7.2, p.1419-1420).

Baseline: damped+smoothed least squares in slowness on a fixed regular grid
(the paper's eq. 37 objective), with a sweep over the regularization weight.

Everything is CPU-only, numpy/scipy, a few minutes.
Run:  D:\conda\envs\powerfoam\python.exe D:\Downloads\powerfoam\test_bodin_rjmcmc_coverage.py
"""

import time
import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 1. True model, ray geometry with uneven coverage + a strictly dead zone
# ----------------------------------------------------------------------------
V_BG, V_FAST = 4.0, 5.0
DEAD_BOX = (0.62, 0.98, 0.62, 0.98)          # xmin,xmax,ymin,ymax : zero rays
COVERED_BOX = (0.05, 0.45, 0.05, 0.45)        # dense coverage region


def v_true(P):
    """True velocity field. One anomaly inside the well-covered region and one
    (identical) anomaly inside the dead zone, so we can measure recovery of
    each under identical structure but opposite data coverage."""
    x, y = P[:, 0], P[:, 1]
    v = np.full(len(P), V_BG)
    v[(x > 0.12) & (x < 0.38) & (y > 0.12) & (y < 0.38)] = V_FAST     # covered
    v[(x > 0.70) & (x < 0.92) & (y > 0.70) & (y < 0.92)] = V_FAST     # dead
    return v


def seg_hits_box(p0, p1, box, nchk=200):
    t = np.linspace(0, 1, nchk)[:, None]
    P = p0[None, :] * (1 - t) + p1[None, :] * t
    xm, xM, ym, yM = box
    return np.any((P[:, 0] > xm) & (P[:, 0] < xM) & (P[:, 1] > ym) & (P[:, 1] < yM))


def build_rays(n_target=140, ds=0.02):
    """Endpoints on the domain boundary, biased to the lower-left so coverage is
    strongly uneven; any ray entering DEAD_BOX is discarded."""
    rays = []
    tries = 0
    while len(rays) < n_target and tries < 20000:
        tries += 1
        # bias endpoints towards low coordinates -> dense lower-left coverage
        def endpoint():
            side = RNG.integers(0, 4)
            u = RNG.beta(1.3, 2.2)           # skewed towards 0
            if side == 0: return np.array([0.0, u])
            if side == 1: return np.array([u, 0.0])
            if side == 2: return np.array([1.0, u])
            return np.array([u, 1.0])
        p0, p1 = endpoint(), endpoint()
        if np.linalg.norm(p1 - p0) < 0.35:
            continue
        if seg_hits_box(p0, p1, DEAD_BOX):
            continue
        rays.append((p0, p1))
    # sample points along each ray (midpoint rule, as in the paper Fig.3)
    pts, dls, rid = [], [], []
    for j, (p0, p1) in enumerate(rays):
        L = np.linalg.norm(p1 - p0)
        m = max(int(np.ceil(L / ds)), 4)
        t = (np.arange(m) + 0.5) / m
        P = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
        pts.append(P)
        dls.append(np.full(m, L / m))
        rid.append(np.full(m, j, dtype=np.int64))
    return (len(rays), np.vstack(pts), np.concatenate(dls), np.concatenate(rid))


NRAY, PTS, DLS, RID = build_rays()
NPT = len(PTS)

SIG_D = 0.004                      # traveltime noise std (s)
t_true = np.bincount(RID, weights=DLS / v_true(PTS), minlength=NRAY)
d_obs = t_true + RNG.normal(0, SIG_D, NRAY)


def misfit(v_at_pts):
    """phi(m) = ||(g(m)-dobs)/sigma_d||^2   [eq. 6]"""
    t = np.bincount(RID, weights=DLS / v_at_pts, minlength=NRAY)
    r = (t - d_obs) / SIG_D
    return float(r @ r)


# ----------------------------------------------------------------------------
# 2. Coverage diagnostics on an evaluation grid
# ----------------------------------------------------------------------------
NG = 60
gx = (np.arange(NG) + 0.5) / NG
GX, GY = np.meshgrid(gx, gx, indexing='ij')
GRID = np.stack([GX.ravel(), GY.ravel()], 1)
VT = v_true(GRID)

# ray-length density per evaluation cell
gi = np.clip((PTS[:, 0] * NG).astype(int), 0, NG - 1)
gj = np.clip((PTS[:, 1] * NG).astype(int), 0, NG - 1)
cov = np.bincount(gi * NG + gj, weights=DLS, minlength=NG * NG)

in_box = lambda P, b: (P[:, 0] > b[0]) & (P[:, 0] < b[1]) & (P[:, 1] > b[2]) & (P[:, 1] < b[3])
M_DEAD = in_box(GRID, DEAD_BOX)
M_COV = in_box(GRID, COVERED_BOX)


# ----------------------------------------------------------------------------
# 3. rj-MCMC  (Bodin & Sambridge Sec 2.6-2.8)
# ----------------------------------------------------------------------------
VMIN, VMAX = 3.5, 5.5
DV = VMAX - VMIN
NMIN, NMAX = 2, 200
SIG1 = 0.10        # velocity proposal          (eq.19)
SIG2 = 0.30        # birth velocity proposal    (eq.26)
SIGC = 0.05        # nucleus move proposal      (eq.21)


def cells_of(pts, c):
    return cKDTree(c).query(pts, k=1)[1]


def run_chain(nsteps, burn, thin, seed, n_init=6, verbose=True):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 1, size=(n_init, 2))
    v = rng.uniform(VMIN, VMAX, size=n_init)
    idx = cells_of(PTS, c)
    phi = misfit(v[idx])

    acc = {k: [0, 0] for k in ('vel', 'birth', 'death', 'move')}
    ens_mean = np.zeros(NG * NG)
    ens_sq = np.zeros(NG * NG)
    nuc_hist = np.zeros(NG * NG)     # where nuclei sit, for density analysis
    n_list, phi_list, nsamp = [], [], 0

    for step in range(nsteps):
        n = len(v)
        if step % 2 == 0:
            kind = 'vel'
        else:
            kind = ('birth', 'death', 'move')[rng.integers(0, 3)]

        if kind == 'vel':
            i = rng.integers(0, n)
            vp = v.copy(); vp[i] += rng.normal(0, SIG1)
            if not (VMIN < vp[i] < VMAX):
                alpha, ok = -np.inf, False
            else:
                phi_p = misfit(vp[idx])
                alpha = -(phi_p - phi) / 2.0; ok = True          # eq.32
            acc['vel'][1] += 1
            if ok and np.log(rng.random()) < alpha:
                v, phi = vp, phi_p; acc['vel'][0] += 1

        elif kind == 'move':
            i = rng.integers(0, n)
            cp = c.copy(); cp[i] = c[i] + rng.normal(0, SIGC, 2)
            acc['move'][1] += 1
            if np.all((cp[i] > 0) & (cp[i] < 1)):
                idx_p = cells_of(PTS, cp)
                phi_p = misfit(v[idx_p])
                if np.log(rng.random()) < -(phi_p - phi) / 2.0:   # eq.32
                    c, idx, phi = cp, idx_p, phi_p; acc['move'][0] += 1

        elif kind == 'birth':
            acc['birth'][1] += 1
            if n + 1 <= NMAX:
                cn = rng.uniform(0, 1, 2)
                vi = v[cells_of(cn[None, :], c)[0]]      # velocity at birth site
                vn = vi + rng.normal(0, SIG2)
                if VMIN < vn < VMAX:
                    cp = np.vstack([c, cn]); vp = np.append(v, vn)
                    idx_p = cells_of(PTS, cp)
                    phi_p = misfit(vp[idx_p])
                    # eq.35
                    logA = (np.log(SIG2 * np.sqrt(2 * np.pi) / DV)
                            + (vn - vi) ** 2 / (2 * SIG2 ** 2)
                            - (phi_p - phi) / 2.0)
                    if np.log(rng.random()) < logA:
                        c, v, idx, phi = cp, vp, idx_p, phi_p; acc['birth'][0] += 1

        else:  # death
            acc['death'][1] += 1
            if n - 1 >= NMIN:
                i = rng.integers(0, n)
                cp = np.delete(c, i, 0); vp = np.delete(v, i)
                vj = vp[cells_of(c[i][None, :], cp)[0]]  # velocity now at that point
                idx_p = cells_of(PTS, cp)
                phi_p = misfit(vp[idx_p])
                # eq.36
                logA = (np.log(DV / (SIG2 * np.sqrt(2 * np.pi)))
                        - (vj - v[i]) ** 2 / (2 * SIG2 ** 2)
                        - (phi_p - phi) / 2.0)
                if np.log(rng.random()) < logA:
                    c, v, idx, phi = cp, vp, idx_p, phi_p; acc['death'][0] += 1

        if step >= burn:
            n_list.append(len(v)); phi_list.append(phi)
            if (step - burn) % thin == 0:
                vg = v[cells_of(GRID, c)]
                ens_mean += vg; ens_sq += vg * vg; nsamp += 1
                ci = np.clip((c[:, 0] * NG).astype(int), 0, NG - 1)
                cj = np.clip((c[:, 1] * NG).astype(int), 0, NG - 1)
                np.add.at(nuc_hist, ci * NG + cj, 1.0)
        if verbose and step % 20000 == 0:
            print(f"    step {step:7d}  n={len(v):3d}  chi2/N={phi/NRAY:8.3f}", flush=True)

    mean = ens_mean / nsamp
    std = np.sqrt(np.maximum(ens_sq / nsamp - mean ** 2, 0))
    return dict(mean=mean, std=std, nuc=nuc_hist / nsamp, nsamp=nsamp,
                n_list=np.array(n_list), phi=np.array(phi_list), acc=acc)


# ----------------------------------------------------------------------------
# 4. Baseline: damped + smoothed least squares on a fixed grid (eq. 37)
# ----------------------------------------------------------------------------
def lsq_baseline(NB=20):
    bi = np.clip((PTS[:, 0] * NB).astype(int), 0, NB - 1)
    bj = np.clip((PTS[:, 1] * NB).astype(int), 0, NB - 1)
    lin = bi * NB + bj
    A = np.zeros((NRAY, NB * NB))
    np.add.at(A, (RID, lin), DLS)                     # t = A s, s = slowness
    s0 = np.full(NB * NB, 1.0 / 4.5)
    # 2nd-difference smoothing operator
    D = []
    for i in range(NB):
        for j in range(NB):
            r = np.zeros(NB * NB); k = i * NB + j
            nb = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            nb = [(a, b) for a, b in nb if 0 <= a < NB and 0 <= b < NB]
            r[k] = -len(nb)
            for a, b in nb: r[a * NB + b] = 1.0
            D.append(r)
    D = np.array(D)
    Aw, dw = A / SIG_D, d_obs / SIG_D
    out = []
    for lam in np.logspace(-3, 3, 13):
        H = Aw.T @ Aw + lam * (D.T @ D) + lam * 1e-2 * np.eye(NB * NB)
        rhs = Aw.T @ dw + lam * 1e-2 * s0
        s = np.linalg.solve(H, rhs)
        vgrid = 1.0 / np.clip(s, 1e-6, None)
        gi_ = np.clip((GRID[:, 0] * NB).astype(int), 0, NB - 1)
        gj_ = np.clip((GRID[:, 1] * NB).astype(int), 0, NB - 1)
        vg = vgrid[gi_ * NB + gj_]
        chi2 = float(np.sum(((A @ s - d_obs) / SIG_D) ** 2)) / NRAY
        out.append((lam, vg, chi2))
    return out


def rmse(a, m=None):
    e = a - VT
    return float(np.sqrt(np.mean(e[m] ** 2))) if m is not None else float(np.sqrt(np.mean(e ** 2)))


# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Bodin & Sambridge (2009) rj-MCMC tomography -- coverage stress test")
    print("=" * 78)
    print(f"rays={NRAY}  sample points={NPT}  noise sigma_d={SIG_D}")
    print(f"eval grid {NG}x{NG}; cells with ZERO ray length: "
          f"{int((cov == 0).sum())}/{NG*NG} = {100*(cov==0).mean():.2f}%")
    print(f"mean ray length per cell  covered box {cov[M_COV].mean():.4f} | "
          f"dead box {cov[M_DEAD].mean():.4f}")

    t0 = time.time()
    NSTEP, BURN, THIN = 300000, 60000, 100
    chains = []
    for s in (1, 2):
        print(f"  chain seed {s} ...")
        chains.append(run_chain(NSTEP, BURN, THIN, s))
    print(f"  rj-MCMC wall time {time.time()-t0:.1f}s for {2*NSTEP} steps "
          f"({2*NSTEP/(time.time()-t0):.0f} steps/s)")

    for k, ch in enumerate(chains):
        a = ch['acc']
        print(f"  chain{k}: samples={ch['nsamp']}  n mean={ch['n_list'].mean():.2f} "
              f"sd={ch['n_list'].std():.2f} min={ch['n_list'].min()} max={ch['n_list'].max()}"
              f"  chi2/N={ch['phi'].mean()/NRAY:.3f}")
        print("          accept rates: " + "  ".join(
            f"{kk}={a[kk][0]/max(a[kk][1],1):.3f}" for kk in a))

    mean = np.mean([c['mean'] for c in chains], 0)
    std = np.mean([c['std'] for c in chains], 0)
    nuc = np.mean([c['nuc'] for c in chains], 0)

    print("\n--- CLAIM 1: does cell density adapt to data coverage? ---")
    dens_cov = nuc[M_COV].sum() / M_COV.sum()
    dens_dead = nuc[M_DEAD].sum() / M_DEAD.sum()
    dens_all = nuc.sum() / (NG * NG)
    print(f"  posterior mean nuclei per eval-cell:  covered={dens_cov:.4f}  "
          f"dead={dens_dead:.4f}  global={dens_all:.4f}")
    print(f"  covered/dead nuclei-density ratio = {dens_cov/max(dens_dead,1e-12):.2f}x")
    print(f"  covered/dead RAY-density ratio    = "
          f"{cov[M_COV].mean()/max(cov[M_DEAD].mean(),1e-12):.2f}x  (infinite = dead)")

    print("\n--- CLAIM 2: prior recovered where there are no data? ---")
    prior_mean, prior_sd = (VMIN + VMAX) / 2, DV / np.sqrt(12)
    print(f"  prior mean={prior_mean:.3f} prior sd={prior_sd:.3f}")
    print(f"  posterior in DEAD zone: mean={mean[M_DEAD].mean():.3f} "
          f"sd(of posterior std)={std[M_DEAD].mean():.3f}")
    print(f"  posterior in COVERED  : mean={mean[M_COV].mean():.3f} "
          f"post std={std[M_COV].mean():.3f}   (true mean {VT[M_COV].mean():.3f})")

    print("\n--- CLAIM 3: accuracy vs tuned-damping least squares ---")
    base = lsq_baseline()
    print(f"  {'lambda':>10} {'chi2/N':>8} {'RMSE all':>9} {'RMSE cov':>9} {'RMSE dead':>10}")
    best = None
    for lam, vg, chi2 in base:
        r = (rmse(vg), rmse(vg, M_COV), rmse(vg, M_DEAD))
        print(f"  {lam:10.3g} {chi2:8.3f} {r[0]:9.4f} {r[1]:9.4f} {r[2]:10.4f}")
        if best is None or r[0] < best[0]: best = (r[0], lam, r[1], r[2], chi2)
    print(f"  BEST-lambda LSQ (chosen using the TRUE model, i.e. cheating): "
          f"lambda={best[1]:.3g} RMSE all={best[0]:.4f} cov={best[2]:.4f} dead={best[3]:.4f}")
    print(f"  rj-MCMC (NO tuned parameter):  RMSE all={rmse(mean):.4f} "
          f"cov={rmse(mean, M_COV):.4f} dead={rmse(mean, M_DEAD):.4f}")

    print("\n--- CLAIM 4: is the posterior std a usable coverage diagnostic? ---")
    err = np.abs(mean - VT)
    ok = np.isfinite(std) & np.isfinite(err)
    cc = float(np.corrcoef(std[ok], err[ok])[0, 1])
    print(f"  corr(posterior std, |actual error|) over the whole grid = {cc:.3f}")
    print(f"  mean posterior std: covered={std[M_COV].mean():.3f}  "
          f"dead={std[M_DEAD].mean():.3f}  ratio={std[M_DEAD].mean()/max(std[M_COV].mean(),1e-9):.2f}x")
    zerocov = cov == 0
    if zerocov.any():
        print(f"  mean posterior std where ray length==0: {std[zerocov].mean():.3f} "
              f"vs where >0: {std[~zerocov].mean():.3f}")

    print("\n--- Cost scaling probe ---")
    for n in (10, 50, 200):
        cc_ = RNG.uniform(0, 1, (n, 2)); vv = RNG.uniform(VMIN, VMAX, n)
        t1 = time.time()
        for _ in range(200):
            misfit(vv[cells_of(PTS, cc_)])
        print(f"  n={n:4d} cells: {1000*(time.time()-t1)/200:.3f} ms per full "
              f"likelihood eval ({NPT} ray samples, {NRAY} rays)")


def knob_sweep():
    """The paper claims no tuning parameter controls model complexity.  But the
    birth/death acceptance terms (eq.35, eq.36) contain the explicit factor
    sigma2*sqrt(2*pi)/Delta_v.  Here we vary sigma2 (birth velocity proposal
    width) and Delta_v (prior width) and measure the posterior on n."""
    global SIG2, VMIN, VMAX, DV
    print("=" * 78)
    print("Hidden-knob sweep: does posterior complexity depend on sigma2/Delta_v?")
    print("=" * 78)
    base_vmin, base_vmax = 3.5, 5.5
    for s2, vmax in ((0.10, 5.5), (0.30, 5.5), (1.00, 5.5), (0.30, 7.5), (0.30, 4.5)):
        SIG2 = s2; VMIN, VMAX = base_vmin, vmax; DV = VMAX - VMIN
        ch = run_chain(120000, 30000, 100, seed=7, verbose=False)
        r = rmse(ch['mean'])
        print(f"  sigma2={s2:.2f} Delta_v={DV:.1f}  ratio sigma2*sqrt(2pi)/Dv="
              f"{s2*np.sqrt(2*np.pi)/DV:6.3f}  ->  n mean={ch['n_list'].mean():6.2f} "
              f"sd={ch['n_list'].std():5.2f} max={ch['n_list'].max():3d}  "
              f"chi2/N={ch['phi'].mean()/NRAY:6.3f}  RMSE={r:.4f} "
              f"birth_acc={ch['acc']['birth'][0]/max(ch['acc']['birth'][1],1):.3f}")
    SIG2, VMIN, VMAX, DV = 0.30, 3.5, 5.5, 2.0


if __name__ == '__main__':
    import sys
    if 'sweep' in sys.argv:
        knob_sweep()
    else:
        main()
