"""Sweep camera count for test_space_carving_photohull.py to check the
ordering (ray mass vs K&S-derived predictors) is stable, not a single-config
artifact.  CPU-only."""
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.linalg import lsqr
import test_space_carving_photohull as T

print("%5s %6s %7s %8s | %9s %9s %9s %9s" %
      ("ncam", "nrays", "dead%", "hull%", "rho_mass", "rho_seen", "rho_KSvis", "rho_hull"))
for ncam in [4, 6, 8, 12, 20, 32, 48]:
    T.RNG = np.random.default_rng(0)
    occ = T.build_scene(); f = T.build_values(occ)
    cams = T.cameras(n_cam=ncam)
    A, _ = T.build_A(cams, occ)
    rm = np.asarray(A.sum(axis=0)).ravel()
    sc = np.asarray((A > 0).sum(axis=0)).ravel().astype(float)
    b = A @ f.ravel()
    fhat = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=4000)[0]
    err = np.abs(fhat - f.ravel())
    imgs = T.render_hard(cams, occ, f)
    V, _, vis = T.space_carve(cams, imgs)
    hull = V.ravel().astype(float)
    ks = np.zeros(T.G * T.G)
    for j, o in vis.items():
        ks[j] = len(o)
    dead = rm <= T.TOL
    print("%5d %6d %6.2f%% %7.1f%% | %+9.4f %+9.4f %+9.4f %+9.4f" %
          (ncam, A.shape[0], 100 * dead.mean(), 100 * hull.mean(),
           spearmanr(rm, err)[0], spearmanr(sc, err)[0],
           spearmanr(ks, err)[0], spearmanr(hull, err)[0]))
