"""
test_visual_hull_laurentini.py

Does Laurentini's visual-hull characterisation (TPAMI 16(2):150-162, 1994)
transfer to our graded, transmittance-weighted, surface-terminating operator?

Construction:
  * synthetic 3-D scene with a genuine concavity (open-top box) plus a small
    cube hidden inside the concavity -- i.e. Laurentini's Fig. 2 configuration.
  * viewpoints strictly outside CH(S), non-planar (mimics handheld ScanNet).
  * SILHOUETTE side: per-view binary silhouettes -> voxel carving -> VH(S,R)
    exactly per Definition 1 / Proposition 2 (closest approximation obtainable
    by volume intersection with viewpoints in R).
  * OUR side: rays that terminate at the first opaque cell, per-cell weights
    w_rj = T_r(j) * alpha_j with telescoping transmittance. Row-substochastic.
    Ray mass of a cell = (A^T 1)_j.
  * recoverability ground truth: min-norm least squares recovery of a per-cell
    scalar feature from b = A f; per-cell absolute error.

Measured comparisons:
  1. set overlap between {dead cells: (A^T 1)_j = 0} and Laurentini's
     ambiguous set VH(S,R) \ S.
  2. correlation of per-cell recovery error with (a) ray mass, (b) VH indicator.
  3. AUC of each as a predictor of "dead".

CPU only, no GPU, no network.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- scene ----
N = 64                    # voxels per side
NVIEW = 16
IMW = 96                  # rays per view side
SIL_RES = 160             # silhouette raster resolution
NSAMP = 420               # samples per ray
ALPHA_SOLID = 0.99
ALPHA_EMPTY = 0.02        # thin ambient medium -> graded, not binary


def build_scene():
    """Open-top box (deep concavity) + small cube hidden inside the cavity."""
    lin = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

    def box(lo, hi):
        return ((X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
                & (Z >= lo[2]) & (Z <= hi[2]))

    outer = box((0.15, 0.15, 0.15), (0.85, 0.85, 0.85))
    cavity = box((0.30, 0.30, 0.32), (0.70, 0.70, 0.95))   # open at the top
    cube = box((0.42, 0.42, 0.36), (0.58, 0.58, 0.50))     # Laurentini Fig. 2
    solid = (outer & ~cavity) | cube
    return solid


def cameras():
    """Non-planar viewpoints on a sphere, all strictly outside CH(S)."""
    C = np.zeros((NVIEW, 3))
    for i in range(NVIEW):
        # jittered spiral over the sphere, biased to the upper hemisphere so
        # the concavity IS observable (worst case for our hypothesis)
        u = (i + 0.5) / NVIEW
        theta = np.arccos(1 - 1.55 * u)          # 0 .. ~114 deg from +z
        phi = 2.399963 * i + RNG.uniform(-0.2, 0.2)
        r = 2.2
        C[i] = 0.5 + r * np.array([np.sin(theta) * np.cos(phi),
                                   np.sin(theta) * np.sin(phi),
                                   np.cos(theta)])
    return C


def look_at(eye, target=np.array([0.5, 0.5, 0.5])):
    f = target - eye
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    if abs(f @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    rgt = np.cross(f, up); rgt /= np.linalg.norm(rgt)
    up2 = np.cross(rgt, f)
    return f, rgt, up2


FOV_TAN = 0.55   # half-tan; wide enough that the whole grid stays in frame


def ray_dirs(eye, w):
    f, rgt, up = look_at(eye)
    s = (np.arange(w) + 0.5) / w * 2 - 1
    sx, sy = np.meshgrid(s, s, indexing="xy")
    d = (f[None, None, :]
         + FOV_TAN * sx[..., None] * rgt[None, None, :]
         + FOV_TAN * sy[..., None] * up[None, None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


# ------------------------------------------------- transmittance operator --
def build_operator(solid, cams):
    """Rays terminate at the first opaque cell; per-cell telescoping weights."""
    alpha_vox = np.where(solid.ravel(), ALPHA_SOLID, ALPHA_EMPTY).astype(np.float64)
    rows_i, cols_i, vals_i = [], [], []
    row0 = 0
    tmin, tmax = 1.0, 3.4                       # bracket the unit cube
    ts = np.linspace(tmin, tmax, NSAMP)
    for eye in cams:
        d = ray_dirs(eye, IMW)                  # (R,3)
        R = d.shape[0]
        P = eye[None, None, :] + ts[None, :, None] * d[:, None, :]   # (R,S,3)
        idx = np.floor(P * N).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < N), axis=-1)
        idx = np.clip(idx, 0, N - 1)
        vid = (idx[..., 0] * N + idx[..., 1]) * N + idx[..., 2]
        vid = np.where(inside, vid, -1)
        # keep only the first sample in each contiguous run (no double count)
        first = np.ones_like(vid, dtype=bool)
        first[:, 1:] = vid[:, 1:] != vid[:, :-1]
        keep = first & (vid >= 0)
        a = np.where(keep, alpha_vox[np.clip(vid, 0, None)], 0.0)
        T = np.cumprod(np.concatenate([np.ones((R, 1)), 1.0 - a[:, :-1]], 1), 1)
        w = T * a
        rr, cc = np.nonzero(keep & (w > 1e-12))
        rows_i.append(rr + row0)
        cols_i.append(vid[rr, cc])
        vals_i.append(w[rr, cc])
        row0 += R
    A = sp.coo_matrix(
        (np.concatenate(vals_i), (np.concatenate(rows_i), np.concatenate(cols_i))),
        shape=(row0, N ** 3)).tocsr()
    A.sum_duplicates()
    return A


# ------------------------------------------------------------ visual hull --
def visual_hull(solid, cams):
    """Voxel carving == volume intersection (Laurentini Prop. 2)."""
    lin = (np.arange(N) + 0.5) / N
    G = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    solid_idx = np.nonzero(solid.ravel())[0]
    vh = np.ones(N ** 3, dtype=bool)
    half = 0.5 / N
    corners = np.array([[sx, sy, sz] for sx in (-1, 0, 1) for sy in (-1, 0, 1)
                        for sz in (-1, 0, 1)], dtype=float) * half
    for eye in cams:
        f, rgt, up = look_at(eye)
        M = np.stack([rgt, up, f])                # world -> cam rows

        def project(pts):
            c = (pts - eye) @ M.T                 # (n,3): x,y,depth
            u = c[:, 0] / (c[:, 2] * FOV_TAN)
            v = c[:, 1] / (c[:, 2] * FOV_TAN)
            pu = ((u + 1) * 0.5 * SIL_RES).astype(np.int64)
            pv = ((v + 1) * 0.5 * SIL_RES).astype(np.int64)
            return pu, pv, c[:, 2]

        # conservative silhouette: mark every pixel touched by a solid voxel
        sil = np.zeros((SIL_RES, SIL_RES), dtype=bool)
        for off in corners:
            pu, pv, dep = project(G[solid_idx] + off)
            ok = (pu >= 0) & (pu < SIL_RES) & (pv >= 0) & (pv < SIL_RES) & (dep > 0)
            sil[pu[ok], pv[ok]] = True
        # 1-px dilation: conservative (over-large) silhouette -> over-large VH,
        # which is the direction that FAVOURS the visual-hull hypothesis.
        d0 = sil.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sil |= np.roll(np.roll(d0, dx, 0), dy, 1)
        # carve: a voxel survives only if its centre projects inside the silhouette
        pu, pv, dep = project(G)
        inb = (pu >= 0) & (pu < SIL_RES) & (pv >= 0) & (pv < SIL_RES) & (dep > 0)
        hit = np.zeros(N ** 3, dtype=bool)
        hit[inb] = sil[pu[inb], pv[inb]]
        # points that fall outside the image are not constrained by this view
        vh &= (hit | ~inb)
    return vh


# ------------------------------------------------------------------ main --
def auc(score, label):
    """Rank AUC of `score` predicting `label` (bool)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    P = label.sum(); Nn = (~label).sum()
    if P == 0 or Nn == 0:
        return float("nan")
    return (ranks[label].sum() - P * (P + 1) / 2) / (P * Nn)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def spearman(a, b):
    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x)); r[o] = np.arange(len(x))
        return r
    return pearson(rk(a), rk(b))


def main():
    solid = build_scene()
    cams = cameras()
    ncell = N ** 3
    print(f"grid {N}^3 = {ncell} cells | solid {solid.sum()} "
          f"({100*solid.mean():.2f}%) | {NVIEW} views | {IMW*IMW*NVIEW} rays")

    A = build_operator(solid, cams)
    mass = np.asarray(A.sum(axis=0)).ravel()
    dead = mass <= 0.0
    per_ray = np.diff(A.indptr)
    print(f"operator: A {A.shape}, nnz {A.nnz}, cells/ray mean {per_ray.mean():.1f}")
    print(f"row sums: max {A.sum(axis=1).max():.4f} (row-substochastic check)")
    print(f"DEAD cells (A^T 1 = 0): {dead.sum()} ({100*dead.mean():.2f}%)")
    print(f"  of which solid: {(dead & solid.ravel()).sum()}, "
          f"empty: {(dead & ~solid.ravel()).sum()}")

    vh = visual_hull(solid, cams)
    S = solid.ravel()
    amb = vh & ~S                       # Laurentini's ambiguous set VH(S,R) \ S
    print(f"VISUAL HULL: |VH| {vh.sum()} ({100*vh.mean():.2f}%), "
          f"|S| {S.sum()}, |VH \\ S| = ambiguous {amb.sum()} ({100*amb.mean():.2f}%)")
    print(f"  VH contains S: {np.all(vh[S])}  (Def.1 sanity: S subset VH); "
          f"violations {int((~vh[S]).sum())} of {int(S.sum())}")
    # did the concavity actually survive carving? (Laurentini's claim)
    lin = (np.arange(N) + 0.5) / N
    Xg, Yg, Zg = np.meshgrid(lin, lin, lin, indexing="ij")
    cav = ((Xg >= 0.30) & (Xg <= 0.70) & (Yg >= 0.30) & (Yg <= 0.70)
           & (Zg >= 0.32) & (Zg <= 0.85)).ravel() & ~S
    print(f"  cavity voxels (empty, inside concavity): {cav.sum()}; "
          f"of those in VH (unrecoverable by carving): {(cav & vh).sum()} "
          f"({100*(cav & vh).sum()/max(cav.sum(),1):.1f}%)")

    # ---- overlap between the two "null" sets
    inter = (dead & amb).sum()
    union = (dead | amb).sum()
    print("\n--- SET COMPARISON: dead cells  vs  Laurentini ambiguous set ---")
    print(f"|dead| {dead.sum()}  |amb| {amb.sum()}  |dead & amb| {inter}  "
          f"Jaccard {inter/max(union,1):.4f}")
    print(f"P(dead | amb) = {inter/max(amb.sum(),1):.4f}   "
          f"P(amb | dead) = {inter/max(dead.sum(),1):.4f}")
    print(f"dead cells that are INSIDE VH: {(dead & vh).sum()} "
          f"({100*(dead & vh).sum()/dead.sum():.2f}% of dead)")
    print(f"cells carved away by VH (not in VH) that are ALIVE: "
          f"{((~vh) & ~dead).sum()} of {(~vh).sum()} carved "
          f"({100*((~vh)&~dead).sum()/max((~vh).sum(),1):.2f}%)")

    # ---- recoverability: min-norm LS on a smooth per-cell feature
    Xf = np.stack([Xg, Yg, Zg], -1).reshape(-1, 3)
    f_true = (np.sin(3.1 * Xf[:, 0]) + np.cos(2.7 * Xf[:, 1])
              + np.sin(4.3 * Xf[:, 2]) + 0.5 * np.sin(7.0 * Xf[:, 0] * Xf[:, 1]))
    b = A @ f_true
    sol = lsmr(A, b, atol=1e-10, btol=1e-10, maxiter=600)
    f_hat = sol[0]
    err = np.abs(f_hat - f_true)
    print(f"\nlsmr stop={sol[1]} iters={sol[2]} residual={sol[3]:.3e}")
    print(f"per-cell error: mean {err.mean():.4f} | live {err[~dead].mean():.4f} "
          f"| dead {err[dead].mean():.4f}")

    print("\n--- PREDICTOR COMPARISON (per-cell error) ---")
    r_mass = spearman(mass, err)
    r_vh = spearman(vh.astype(float), err)
    print(f"Spearman rho(ray mass,   err) = {r_mass:+.4f}   <-- baseline to beat")
    print(f"Spearman rho(VH indicator, err) = {r_vh:+.4f}")
    print(f"Pearson  r(ray mass,   err) = {pearson(mass, err):+.4f}")
    print(f"Pearson  r(VH indicator, err) = {pearson(vh.astype(float), err):+.4f}")
    print(f"AUC(-ray mass -> dead)   = {auc(-mass, dead):.4f}")
    print(f"AUC( VH  -> dead)        = {auc(vh.astype(float), dead):.4f}")
    print(f"AUC( amb -> dead)        = {auc(amb.astype(float), dead):.4f}")

    live = ~dead
    if live.sum() > 10:
        print("\n--- does VH explain anything AMONG LIVE CELLS? ---")
        print(f"live cells: {live.sum()}; of those in VH: {vh[live].sum()}")
        print(f"Spearman rho(VH, err | live)      = {spearman(vh[live].astype(float), err[live]):+.4f}")
        print(f"Spearman rho(ray mass, err | live)= {spearman(mass[live], err[live]):+.4f}")

    print("\n--- combined predictor (rank-average of the two) ---")
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    comb = rk(-mass) + rk(vh.astype(float))
    print(f"Spearman rho(comb, err) = {spearman(comb, err):+.4f}")


if __name__ == "__main__":
    main()
