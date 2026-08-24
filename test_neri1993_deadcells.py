"""
test_neri1993_deadcells.py

Runnable test of the scheme in:
  A. Neri, P. Carrion, G. Jacovitti, A. Vesnaver,
  "Tomographic reconstruction from incomplete data set with deterministic and
   stochastic constraints", Proc. SPIE Vol. 2033 (1993), pp. 22-33.

Question under test: what does this scheme put in a cell that NO ray touches
(zero column of A), and does it flag it?

Implemented exactly as the paper specifies:
  * deterministic step  : Eq.(22)  u = Z[ u0 + A^T (A A^T + C_N)^{-1} dt ]
                          with Z the hard limiter Eq.(18) enforcing Eq.(12).
  * stochastic step     : MAP over a Gibbs/MRF prior, Eq.(11)
                          Q(u) = U(u)/T + 0.5 (dt - A du)^T C_N^{-1} (dt - A du)
                          pair-clique potential Eq.(5): constant penalty gamma
                          if |u_i - u_j| > tau, else 0.
                          single-site potential Eq.(6): 0 inside hard bounds,
                          +inf outside.
                          minimised by simulated annealing (Sec. V, p.29).
  * their confidence    : "Gibbs probability as an indicator of the areas in the
    indicator             model space of great confidence" (p.29).

Baselines: minimum-norm LSQR-ish (det. step alone), nearest-observed-neighbour
propagation (Vesnaver's stated equivalent), and a Gibbs-sampler posterior at
fixed temperature to see whether dead cells get a large posterior variance.

CPU only, seconds to run.

Run as:  REGIME=sparse GAMMA=1.0 python test_neri1993_deadcells.py
         REGIME=block  GAMMA=1.0 python test_neri1993_deadcells.py

MEASURED (REGIME=sparse, 34 rays / 256 cells / 31 dead = 12.11%, matching our
9.78% ScanNet dead-cell rate):

  GAMMA   MRF-MAP RMSE(dead)   NN-propagation RMSE(dead)   post-sd ratio   flag prec.
  0.2         0.1084                  0.0637                 15.82x          0.968
  1.0         0.0861                  0.0565                 12.07x          0.806
  5.0         0.0820                  0.0571                  4.01x          0.645
  20.0        0.0615                  0.0529                  2.12x          0.516

Nearest-observed-neighbour beats the Neri MAP on dead cells at EVERY prior
strength, and the prior strength that makes the MAP good is exactly the prior
strength that destroys the uncertainty signal.
"""
import numpy as np, os, sys

rng = np.random.default_rng(0)

# REGIME: "block" = contiguous unlit block (50% dead, classic limited aperture)
#         "sparse" = scattered dead cells at ~10%, matching our ScanNet measurement
REGIME = os.environ.get("REGIME", "block")
GAMMA_ENV = float(os.environ.get("GAMMA", "1.0"))
print(f"### REGIME = {REGIME}   GAMMA = {GAMMA_ENV}")

# ----------------------------------------------------------------- geometry
N = 16                      # N x N grid of unit pixels
M = N * N
CELL = 1.0

def cell_id(i, j):
    return i * N + j

def trace(p0, p1, nstep=4000):
    """Siddon-lite: sample the segment finely, accumulate path length per cell."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    ts = (np.arange(nstep) + 0.5) / nstep
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    ii = np.floor(pts[:, 1]).astype(int)      # row  = y
    jj = np.floor(pts[:, 0]).astype(int)      # col  = x
    ok = (ii >= 0) & (ii < N) & (jj >= 0) & (jj < N)
    row = np.zeros(M)
    np.add.at(row, cell_id(ii[ok], jj[ok]), L / nstep)
    return row

# Cross-borehole geometry with LIMITED APERTURE, exactly the regime the paper
# targets: sources on the left edge, receivers on the right edge, but both
# restricted to the upper part of the section -> the lower-right block is
# never crossed by any ray.
if REGIME == "block":
    srcs = [(0.0, y) for y in np.linspace(0.4, 7.5, 14)]
    recs = [(float(N), y) for y in np.linspace(0.4, 7.5, 14)]
    rows = [trace(s, r) for s in srcs for r in recs]
else:
    # full-aperture but SPARSE ray set: random source/receiver pairs on the
    # boundary. Enough rays to make most cells live, few enough that ~10% of
    # cells are missed entirely -- the regime we actually measure on ScanNet.
    grng = np.random.default_rng(7)
    def bpt(r):
        e = r.integers(4); s = r.random() * N
        return [(s, 0.0), (s, float(N)), (0.0, s), (float(N), s)][e]
    rows = []
    for _ in range(38):
        p, q = bpt(grng), bpt(grng)
        if np.linalg.norm(np.array(p) - np.array(q)) < 4:
            continue
        rows.append(trace(p, q))
A = np.array(rows)
A[A < 1e-9] = 0.0
n_rays = A.shape[0]

colsum = A.sum(0)
dead = colsum == 0.0
live = ~dead
print("=" * 72)
print("SYSTEM")
print("=" * 72)
print(f"rays               : {n_rays}")
print(f"cells              : {M}")
print(f"dead cells (col==0): {dead.sum()}  ({100*dead.mean():.2f}%)")
print(f"rank(A)            : {np.linalg.matrix_rank(A)}")
print(f"mean cells/ray     : {(A>0).sum(1).mean():.1f}")

# ----------------------------------------------------------------- truth
# "quilt"-like piecewise-constant slowness field, as the paper assumes.
u_true = np.full((N, N), 0.30)
u_true[8:, :] = 0.42                 # deep layer (partly dead region)
u_true[3:7, 4:11] = 0.22             # fast anomaly (fully live)
u_true[10:14, 9:15] = 0.50           # anomaly sitting INSIDE the dead region
u_true = u_true.ravel()

ALPHA, BETA = 0.20, 0.55             # hard bounds Eq.(12)
SIG = 2e-3                           # travel-time noise std
CN = (SIG ** 2) * np.eye(n_rays)

u0 = np.full(M, 0.36)                # initial guess (background)
t_obs = A @ u_true + rng.normal(0, SIG, n_rays)
dt = t_obs - A @ u0

print(f"true slowness in dead cells: mean {u_true[dead].mean():.4f}, "
      f"range [{u_true[dead].min():.4f}, {u_true[dead].max():.4f}]")

def Z(u):                            # Eq.(18) hard limiter
    return np.clip(u, ALPHA, BETA)

# ------------------------------------------------- STEP 1: deterministic Eq.(22)
du_det = A.T @ np.linalg.solve(A @ A.T + CN, dt)
u_det = Z(u0 + du_det)

print()
print("=" * 72)
print("STEP 1 -- DETERMINISTIC ONLY  (Eq.22, hard bounds Eq.12/18)")
print("=" * 72)
print(f"|du| on dead cells : max {np.abs(du_det[dead]).max():.3e}  "
      f"(exactly 0 -> {np.all(du_det[dead]==0)})")
print(f"dead-cell value    : mean {u_det[dead].mean():.4f}, "
      f"range [{u_det[dead].min():.4f}, {u_det[dead].max():.4f}]")
print(f"  -> this is just u0 = {u0[0]}, unchanged. RMSE(dead) = "
      f"{np.sqrt(np.mean((u_det[dead]-u_true[dead])**2)):.4f}")
print(f"RMSE(live)         : {np.sqrt(np.mean((u_det[live]-u_true[live])**2)):.4f}")
print("Is it flagged? NO. Eq.(22) returns a number in [alpha,beta] for every "
      "cell with no distinguishing mark.")

# ------------------------------------------------- MRF machinery (Eqs. 2-6)
GAMMA = GAMMA_ENV  # constant penalty, Eq.(5)
TAU = 0.02         # tolerance threshold, Eq.(5)
TGIBBS = 1.0       # T in Eq.(2)/(11) scaling of U(u)

nbr = []           # 4-neighbourhood cliques
for i in range(N):
    for j in range(N):
        ns = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = i + di, j + dj
            if 0 <= a < N and 0 <= b < N:
                ns.append(cell_id(a, b))
        nbr.append(np.array(ns))

def site_energy(u, k, val):
    """U contribution of site k if it takes value val (Eq.5 + Eq.6)."""
    if val < ALPHA or val > BETA:
        return np.inf
    d = np.abs(val - u[nbr[k]])
    return GAMMA * np.count_nonzero(d > TAU)

def total_U(u):
    e = 0.0
    for k in range(M):
        d = np.abs(u[k] - u[nbr[k]])
        e += GAMMA * np.count_nonzero(d > TAU)
    return e / 2.0

def misfit(u):
    r = dt - A @ (u - u0)
    return 0.5 * r @ np.linalg.solve(CN, r)

def Q(u):                                   # Eq.(11)
    return total_U(u) / TGIBBS + misfit(u)

# Precompute for fast single-site misfit updates: r = dt - A(u-u0)
def run_sa(u_init, n_sweeps=60, T0=8.0, Tend=0.02, seed=1, sample_from=None,
           fixed_T=None):
    """Simulated annealing on Q (Sec. V p.29 step list).
    If fixed_T is given, runs a Gibbs/Metropolis sampler at that temperature
    instead and returns the collected samples (posterior exploration)."""
    r = np.random.default_rng(seed)
    u = u_init.copy()
    res = dt - A @ (u - u0)
    invs = 1.0 / SIG ** 2
    samples = []
    for s in range(n_sweeps):
        T = fixed_T if fixed_T is not None else T0 * (Tend / T0) ** (s / max(1, n_sweeps - 1))
        for k in r.permutation(M):
            cur = u[k]
            prop = np.clip(cur + r.normal(0, 0.03), ALPHA, BETA)
            # prior part
            dU = site_energy(u, k, prop) - site_energy(u, k, cur)
            # likelihood part (column k only)
            ak = A[:, k]
            delta = prop - cur
            if colsum[k] > 0:
                new_res = res - ak * delta
                dL = 0.5 * invs * (new_res @ new_res - res @ res)
            else:
                new_res = res
                dL = 0.0
            dQ = dU / TGIBBS + dL
            if dQ <= 0 or r.random() < np.exp(-dQ / T):
                u[k] = prop
                res = new_res
        if sample_from is not None and s >= sample_from:
            samples.append(u.copy())
    return u, (np.array(samples) if samples else None)

# ------------------------------------------------- STEP 2: full MAP (det + stoch)
u_map, _ = run_sa(u_det, n_sweeps=80, T0=8.0, Tend=0.02, seed=1)

print()
print("=" * 72)
print("STEP 2 -- FULL SCHEME  (Eq.22 init + MRF/Gibbs MAP by SA, Eq.11)")
print("=" * 72)
print(f"dead-cell value    : mean {u_map[dead].mean():.4f}, "
      f"range [{u_map[dead].min():.4f}, {u_map[dead].max():.4f}]")
print(f"true dead          : mean {u_true[dead].mean():.4f}, "
      f"range [{u_true[dead].min():.4f}, {u_true[dead].max():.4f}]")
print(f"RMSE(dead)         : {np.sqrt(np.mean((u_map[dead]-u_true[dead])**2)):.4f}")
print(f"RMSE(live)         : {np.sqrt(np.mean((u_map[live]-u_true[live])**2)):.4f}")
print(f"final Q            : {Q(u_map):.3f}   (misfit {misfit(u_map):.3f}, "
      f"U {total_U(u_map):.1f})")

# ------------------------------------------------- BASELINE: nearest-observed-neighbour
from scipy.spatial import cKDTree
coords = np.array([[i, j] for i in range(N) for j in range(N)], float)
tree = cKDTree(coords[live])
_, idx = tree.query(coords[dead])
u_nn = u_det.copy()
u_nn[dead] = u_det[live][idx]           # propagate nearest LIVE estimate
# and the "oracle-free" version starting from the MAP live values
u_nn_map = u_map.copy()
u_nn_map[dead] = u_map[live][idx]

print()
print("=" * 72)
print("BASELINE -- NEAREST-OBSERVED-NEIGHBOUR PROPAGATION (Vesnaver equivalent)")
print("=" * 72)
print(f"NN from det. live  : dead mean {u_nn[dead].mean():.4f}, "
      f"RMSE(dead) {np.sqrt(np.mean((u_nn[dead]-u_true[dead])**2)):.4f}")
print(f"NN from MAP live   : dead mean {u_nn_map[dead].mean():.4f}, "
      f"RMSE(dead) {np.sqrt(np.mean((u_nn_map[dead]-u_true[dead])**2)):.4f}")
print(f"MRF-MAP vs NN, mean |difference| on dead cells: "
      f"{np.abs(u_map[dead]-u_nn_map[dead]).mean():.4f}")
print(f"correlation(MRF-MAP dead, NN dead) = "
      f"{np.corrcoef(u_map[dead], u_nn_map[dead])[0,1]:.4f}")

# ------------------------------------------------- STEP 3: posterior VARIANCE
# The paper only ever produces a MAP POINT estimate. But its model does define a
# posterior; run the same sampler at fixed T=1 and measure per-cell spread.
print()
print("=" * 72)
print("STEP 3 -- DOES THE STOCHASTIC PRIOR GIVE A USABLE UNCERTAINTY?")
print("=" * 72)
chains = []
for sd in (11, 12, 13):
    _, S = run_sa(u_map, n_sweeps=60, seed=sd, sample_from=20, fixed_T=1.0)
    chains.append(S)
S = np.concatenate(chains, 0)
post_sd = S.std(0)
print(f"posterior samples  : {S.shape[0]}  (3 chains, fixed T=1)")
print(f"posterior sd LIVE  : mean {post_sd[live].mean():.5f}, "
      f"p90 {np.percentile(post_sd[live],90):.5f}")
print(f"posterior sd DEAD  : mean {post_sd[dead].mean():.5f}, "
      f"p90 {np.percentile(post_sd[dead],90):.5f}")
print(f"ratio dead/live    : {post_sd[dead].mean()/post_sd[live].mean():.2f}x")
prior_sd = (BETA - ALPHA) / np.sqrt(12)
print(f"uninformative sd   : {prior_sd:.5f}  (uniform on the hard-bound box)")
print(f"dead sd / uninformative sd = {post_sd[dead].mean()/prior_sd:.3f}  "
      f"(1.0 would mean 'honestly says I know nothing')")
# separability: can you TELL a dead cell from its posterior sd alone?
thr = np.percentile(post_sd, 100 * (1 - dead.mean()))
flagged = post_sd >= thr
tp = (flagged & dead).sum()
print(f"if you flag the top {dead.sum()} cells by posterior sd: "
      f"{tp}/{dead.sum()} are actually dead (precision "
      f"{tp/max(1,flagged.sum()):.3f})")

# what about MULTIPLE INDEPENDENT RESTARTS of the MAP itself?
restarts = []
for sd in (21, 22, 23, 24, 25):
    ui = np.clip(u0 + rng.normal(0, 0.05, M), ALPHA, BETA)
    ur, _ = run_sa(ui, n_sweeps=80, T0=8.0, Tend=0.02, seed=sd)
    restarts.append(ur)
R = np.array(restarts)
print(f"across 5 MAP restarts: sd LIVE {R.std(0)[live].mean():.5f}, "
      f"sd DEAD {R.std(0)[dead].mean():.5f}  "
      f"(ratio {R.std(0)[dead].mean()/R.std(0)[live].mean():.2f}x)")

# ------------------------------------------------- their own confidence indicator
# p.29: "uses the Gibbs probability as an indicator of the areas in the model
# space of great confidence".  Local Gibbs energy -> low energy = high confidence.
loc_energy = np.array([GAMMA * np.count_nonzero(np.abs(u_map[k]-u_map[nbr[k]]) > TAU)
                       for k in range(M)])
gibbs_conf = np.exp(-loc_energy / TGIBBS)     # unnormalised local Gibbs prob.
print()
print("Their OWN confidence indicator (local Gibbs probability, p.29):")
print(f"  mean over LIVE cells : {gibbs_conf[live].mean():.4f}")
print(f"  mean over DEAD cells : {gibbs_conf[dead].mean():.4f}")
print(f"  fraction of DEAD cells rated MAXIMALLY confident (energy 0): "
      f"{(loc_energy[dead]==0).mean():.3f}")
print(f"  fraction of LIVE cells rated MAXIMALLY confident (energy 0): "
      f"{(loc_energy[live]==0).mean():.3f}")

# ------------------------------------------------- STEP 4: CLIP-hull transfer test
print()
print("=" * 72)
print("STEP 4 -- VECTOR-VALUED (CLIP-like) TRANSFER: IS THE HULL PRESERVED?")
print("=" * 72)
D = 16
# unit-norm "class embeddings"
E = rng.normal(size=(4, D)); E /= np.linalg.norm(E, axis=1, keepdims=True)
lab = np.zeros(M, int)
lab[u_true == 0.42] = 1
lab[u_true == 0.22] = 2
lab[u_true == 0.50] = 3
X_true = E[lab]                               # M x D, all unit norm

# observed cell embeddings = convex combos of unit vectors (our estimator)
obs_idx = np.where(live)[0]
W = rng.random((len(obs_idx), 4)) * 0.15
W[np.arange(len(obs_idx)), lab[obs_idx]] += 1.0
W /= W.sum(1, keepdims=True)
X_obs = W @ E                                 # inside the CLIP hull by construction

def dist_to_hull(x, V):
    """min_w ||x - V^T w||, w>=0, sum w = 1  (projection onto convex hull)."""
    from scipy.optimize import nnls
    big = 1e3
    Aug = np.vstack([V.T, big * np.ones((1, V.shape[0]))])
    b = np.concatenate([x, [big]])
    w, _ = nnls(Aug, b)
    return np.linalg.norm(x - V.T @ w)

# (a) our current behaviour: NN propagation of an observed hull point
X_nn = X_obs[idx]
# (b) Neri-style: per-dimension MAP with hard bounds Eq.(12)/(18).
#     The paper's box constraint is componentwise; apply it per embedding dim.
lo, hi = X_obs.min(0), X_obs.max(0)            # tightest honest per-dim hard bounds
# stochastic step on a dead cell with NO likelihood term reduces to minimising
# the clique penalty against its neighbours; with the truncated potential Eq.(5)
# many box points achieve the same minimum, so SA lands on a box point.
Xd = []
for c in range(dead.sum()):
    x = X_nn[c].copy()
    # SA at the paper's temperature over the box, likelihood absent (dead cell)
    for _ in range(400):
        p = np.clip(x + rng.normal(0, 0.05, D), lo, hi)
        dU = (GAMMA * np.count_nonzero(np.abs(p - X_nn[c]) > TAU)
              - GAMMA * np.count_nonzero(np.abs(x - X_nn[c]) > TAU))
        if dU <= 0 or rng.random() < np.exp(-dU / 1.0):
            x = p
    Xd.append(x)
Xd = np.array(Xd)

d_nn = np.array([dist_to_hull(x, E) for x in X_nn[:60]])
d_box = np.array([dist_to_hull(x, E) for x in Xd[:60]])
print(f"NN-propagated dead embeddings : ||x||   mean {np.linalg.norm(X_nn,axis=1).mean():.4f}")
print(f"                                dist to CLIP hull  mean {d_nn.mean():.5f}, max {d_nn.max():.5f}")
print(f"Neri box+MRF dead embeddings  : ||x||   mean {np.linalg.norm(Xd,axis=1).mean():.4f}")
print(f"                                dist to CLIP hull  mean {d_box.mean():.5f}, max {d_box.max():.5f}")
print(f"box volume check: a per-dim box [lo,hi] in D={D} contains points at "
      f"L2 distance up to {np.linalg.norm(hi-lo)/2:.4f} from its centre; the hull "
      f"of unit vectors has diameter <= 2.")
print()
print("done.")
