"""
test_brankov_camm_node_density.py

Verifies the central quantitative claim of:
  J. G. Brankov, Y. Yang, M. N. Wernick, "Tomographic Image Reconstruction Based on a
  Content-Adaptive Mesh Model", IEEE TMI 23(2):202-212, Feb 2004.

Claim under test (Abstract, p.202): "both MDL and CHO suggested that the optimal number
of mesh nodes is roughly five to seven times smaller than the number of projection bins."

We reproduce the structure of their experiment on a small synthetic 2-D emission-tomography
problem, entirely on CPU:

  * pixel image  f_p  (32x32 = 1024 pixels)
  * content-adaptive mesh (Sec. IV.A): feature map = magnitude of 2nd directional
    derivatives (their eq. 21), Floyd-Steinberg error diffusion node placement,
    Delaunay triangulation, piecewise-linear (barycentric) interpolation.
  * pixel-domain interpolation matrix B (their eq. 6):  f_p = B f_m
  * mesh-domain system matrix (their eq. 9):            A_m = R B
  * reconstruct f_m from noisy data, map back to pixels, measure PSNR.

We sweep the number of mesh nodes N and measure, as a function of the ratio
(#measurements / #nodes):
  - reconstruction PSNR (noisy and noiseless)
  - representation ceiling PSNR (best possible mesh fit to the truth)
  - cond(A_m) and numerical rank
  - fraction of nodes with zero column mass in A_m ("dead" unknowns)  <-- coverage probe

Two data regimes:
  (a) full angular sampling  (many views)
  (b) view-starved / limited-angle (mimics our ScanNet scene0347_00 at 54 views)

Run:  D:\conda\envs\powerfoam\python.exe D:\Downloads\powerfoam\test_brankov_camm_node_density.py
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

RNG = np.random.default_rng(0)
NPIX = 32          # image side
NBIN = 32          # detector bins per view


# ---------------------------------------------------------------- phantom
def phantom(n=NPIX):
    """Small Shepp-Logan-like phantom: smooth background + sharp structures."""
    y, x = np.mgrid[0:n, 0:n]
    x = (x - n / 2 + 0.5) / (n / 2)
    y = (y - n / 2 + 0.5) / (n / 2)
    img = np.zeros((n, n))

    def ellipse(cx, cy, a, b, ang, val):
        c, s = np.cos(ang), np.sin(ang)
        xr = (x - cx) * c + (y - cy) * s
        yr = -(x - cx) * s + (y - cy) * c
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += val

    ellipse(0, 0, 0.80, 0.90, 0.0, 1.0)       # body
    ellipse(0, 0.10, 0.55, 0.60, 0.0, -0.4)   # cavity
    ellipse(-0.25, -0.20, 0.20, 0.28, 0.4, 0.9)   # "myocardium"-like blob
    ellipse(0.28, -0.15, 0.13, 0.13, 0.0, 0.7)
    ellipse(0.05, 0.35, 0.09, 0.09, 0.0, -0.5)    # cold defect
    return np.clip(img, 0, None)


# ---------------------------------------------------------------- forward operator
def radon_matrix(n=NPIX, nbin=NBIN, angles=None, nsamp=96):
    """Sparse parallel-beam line-integral matrix, bilinear pixel interpolation."""
    rows, cols, vals = [], [], []
    det = (np.arange(nbin) - nbin / 2 + 0.5) * (n / nbin)
    t = (np.linspace(-0.5, 0.5, nsamp)) * (n * 1.45)
    step = (t[1] - t[0])
    r = 0
    for th in angles:
        c, s = np.cos(th), np.sin(th)
        for d in det:
            # ray: base point d*(c,s) + t*(-s,c)
            px = d * c - t * s + n / 2 - 0.5
            py = d * s + t * c + n / 2 - 0.5
            x0 = np.floor(px).astype(int)
            y0 = np.floor(py).astype(int)
            fx = px - x0
            fy = py - y0
            for dx in (0, 1):
                for dy in (0, 1):
                    xi = x0 + dx
                    yi = y0 + dy
                    w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * step
                    ok = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n) & (w > 0)
                    rows.append(np.full(ok.sum(), r))
                    cols.append(yi[ok] * n + xi[ok])
                    vals.append(w[ok])
            r += 1
    R = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(r, n * n))
    R.sum_duplicates()
    return R


# ---------------------------------------------------------------- CAMM mesh generation
def feature_map(img, alpha=0.1):
    """Brankov eq.(21): approximation to largest magnitude of 2nd directional derivatives."""
    fxx = np.gradient(np.gradient(img, axis=1), axis=1)
    fyy = np.gradient(np.gradient(img, axis=0), axis=0)
    fxy = np.gradient(np.gradient(img, axis=1), axis=0)
    # largest |eigenvalue| of the Hessian
    tr = fxx + fyy
    dis = np.sqrt(np.maximum(((fxx - fyy) / 2) ** 2 + fxy ** 2, 0))
    lam = np.maximum(np.abs(tr / 2 + dis), np.abs(tr / 2 - dis))
    f = np.power(lam, 0.5)                     # sub-linear, as density ~ curvature^(1/2)
    f = f / (f.max() + 1e-12) + alpha
    return f


def floyd_steinberg_nodes(fmap, nnodes):
    """Error-diffusion halftoning to place nnodes with density proportional to fmap."""
    h, w = fmap.shape
    d = fmap * (nnodes / fmap.sum())
    d = d.copy()
    pts = []
    for i in range(h):
        for j in range(w):
            old = d[i, j]
            new = 1.0 if old > 0.5 else 0.0
            if new > 0:
                pts.append((j, i))
            err = old - new
            if j + 1 < w:
                d[i, j + 1] += err * 7 / 16
            if i + 1 < h:
                if j > 0:
                    d[i + 1, j - 1] += err * 3 / 16
                d[i + 1, j] += err * 5 / 16
                if j + 1 < w:
                    d[i + 1, j + 1] += err * 1 / 16
    pts = np.array(pts, float)
    # always pin the four corners so the mesh covers the whole domain
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    if len(pts):
        pts = np.vstack([pts, corners])
    else:
        pts = corners
    # dedup
    pts = np.unique(pts, axis=0)
    return pts


def uniform_nodes(nnodes, n=NPIX):
    k = int(np.ceil(np.sqrt(nnodes)))
    g = np.linspace(0, n - 1, k)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel()], 1)


def interp_matrix(nodes, n=NPIX):
    """B in f_pix = B f_nodes : barycentric interpolation on the Delaunay mesh."""
    tri = Delaunay(nodes)
    y, x = np.mgrid[0:n, 0:n]
    P = np.stack([x.ravel().astype(float), y.ravel().astype(float)], 1)
    simp = tri.find_simplex(P)
    rows, cols, vals = [], [], []
    inside = simp >= 0
    # for points outside the hull, snap to nearest node
    if (~inside).any():
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        _, nn = tree.query(P[~inside])
        idx = np.where(~inside)[0]
        rows.append(idx); cols.append(nn); vals.append(np.ones(len(idx)))
    idx = np.where(inside)[0]
    s = simp[inside]
    T = tri.transform[s]
    b = np.einsum('ijk,ik->ij', T[:, :2, :], P[inside] - T[:, 2, :])
    bary = np.hstack([b, 1 - b.sum(1, keepdims=True)])
    verts = tri.simplices[s]
    for k in range(3):
        rows.append(idx); cols.append(verts[:, k]); vals.append(bary[:, k])
    B = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n * n, len(nodes)))
    return B


# ---------------------------------------------------------------- metrics
def psnr(ref, est):
    mse = np.mean((ref - est) ** 2)
    return 10 * np.log10(ref.max() ** 2 / mse) if mse > 0 else np.inf


def mlem(Am, g, niter, f_true_img, B, n=NPIX):
    """Poisson EM in the mesh domain (Brankov eq.15). Returns best-over-iterations PSNR
    (early stopping, as the paper does: their best is at iteration 4-8) and the
    iteration index at which it occurs."""
    sens = np.asarray(Am.sum(0)).ravel()
    sens = np.where(sens > 0, sens, 1.0)
    f = np.ones(Am.shape[1])
    best, bi = -np.inf, 0
    for it in range(1, niter + 1):
        p = Am @ f
        r = np.divide(g, p, out=np.zeros_like(g), where=p > 1e-12)
        f = f * (Am.T @ r) / sens
        img = np.clip((B @ f).reshape(n, n), 0, None)
        v = psnr(f_true_img, img)
        if v > best:
            best, bi = v, it
    return best, bi


def mdl_curve(fref_img, counts, adaptive=True, n=NPIX):
    """Brankov eq.(25): MDL(N) = (N_pix/2) log(sigma_hat^2) + (3N/2) log(N_pix).
    Note: a mesh node costs 3 parameters (x, y, value). This objective depends ONLY on
    the reference IMAGE and its mesh approximation error -- the number of measurements
    never enters it."""
    fmap = feature_map(fref_img)
    ref = fref_img.ravel()
    Npix = ref.size
    out = []
    for N in counts:
        nodes = floyd_steinberg_nodes(fmap, N) if adaptive else uniform_nodes(N, n)
        B = interp_matrix(nodes, n)
        fm = np.linalg.lstsq(B.toarray(), ref, rcond=None)[0]
        err = ref - B @ fm
        s2 = max(np.mean(err ** 2), 1e-18)
        mdl = 0.5 * Npix * np.log(s2) + 1.5 * len(nodes) * np.log(Npix)
        out.append((len(nodes), mdl, 10 * np.log10(ref.max() ** 2 / s2)))
    return out


def run_regime(name, angles, counts, noise_counts, adaptive=True, verbose=True):
    n = NPIX
    img = phantom(n)
    f_true = img.ravel()
    R = radon_matrix(n, NBIN, angles)
    nmeas = R.shape[0]
    g_clean = R @ f_true
    # Poisson data at a fixed total count level
    scale = noise_counts / g_clean.sum()
    g_noisy = RNG.poisson(g_clean * scale) / scale

    # reference image for mesh generation (Brankov Sec.IV.A: smoothed FBP-like recon).
    # we use a smoothed least-squares pixel recon of the noisy data.
    fref = lsqr(R, g_noisy, damp=2.0, iter_lim=200)[0].reshape(n, n)
    from scipy.ndimage import gaussian_filter
    fref = gaussian_filter(np.clip(fref, 0, None), 1.0)
    fmap = feature_map(fref)

    rows = []
    for N in counts:
        nodes = floyd_steinberg_nodes(fmap, N) if adaptive else uniform_nodes(N, n)
        Nn = len(nodes)
        B = interp_matrix(nodes, n)
        Am = (R @ B)
        Ad = Am.toarray()
        # ---- conditioning / rank / dead unknowns
        colmass = np.asarray(Am.sum(0)).ravel()
        dead = float((colmass <= 1e-12).mean())
        sv = np.linalg.svd(Ad, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        rank = int((sv > sv[0] * 1e-10).sum())
        # ---- representation ceiling: best mesh fit to the true image
        fm_best = np.linalg.lstsq(B.toarray(), f_true, rcond=None)[0]
        ceil_psnr = psnr(img, (B @ fm_best).reshape(n, n))
        # ---- reconstruction from data (mild Tikhonov, same damp for all N)
        fm = lsqr(Am, g_noisy, damp=1e-2, iter_lim=800)[0]
        rec = np.clip((B @ fm).reshape(n, n), 0, None)
        p_noisy = psnr(img, rec)
        fm_c = lsqr(Am, g_clean, damp=1e-2, iter_lim=800)[0]
        rec_c = np.clip((B @ fm_c).reshape(n, n), 0, None)
        p_clean = psnr(img, rec_c)
        # ---- Poisson EM in the mesh domain, best over iterations (the paper's method)
        em_psnr, em_it = mlem(Am, np.maximum(g_noisy, 0), 60, img, B)
        ratio = nmeas / Nn
        rows.append((Nn, ratio, ceil_psnr, p_clean, p_noisy, cond, rank / Nn, dead,
                     em_psnr, em_it))
        if verbose:
            print(f"  N={Nn:5d}  meas/nodes={ratio:6.2f}  ceil={ceil_psnr:6.2f}dB  "
                  f"LSclean={p_clean:6.2f}dB  LSnoisy={p_noisy:7.2f}dB  "
                  f"EMbest={em_psnr:6.2f}dB@it{em_it:<3d}  "
                  f"cond={cond:9.3e}  rank/N={rank/Nn:5.3f}  dead={dead*100:5.2f}%")
    return nmeas, rows


def radon_matrix_occluded(occ, n=NPIX, nbin=NBIN, angles=None, depth=3, nsamp=96):
    """Rays are emitted from one side only and TERMINATE `depth` samples after the first
    hit on the occluder `occ` (a boolean map). This is the analogue of our transmittance
    model: everything behind the first surface is never observed. Produces a genuine
    geometric coverage hole (unlike detector truncation, which does not)."""
    det = (np.arange(nbin) - nbin / 2 + 0.5) * (n / nbin)
    t = np.linspace(-0.5, 0.5, nsamp) * (n * 1.45)
    step = t[1] - t[0]
    rows, cols, vals = [], [], []
    r = 0
    for th in angles:
        c, s = np.cos(th), np.sin(th)
        for d in det:
            px = d * c - t * s + n / 2 - 0.5
            py = d * s + t * c + n / 2 - 0.5
            xi0 = np.clip(np.round(px).astype(int), 0, n - 1)
            yi0 = np.clip(np.round(py).astype(int), 0, n - 1)
            inb = (px >= -0.5) & (px < n - 0.5) & (py >= -0.5) & (py < n - 0.5)
            hit = occ[yi0, xi0] & inb
            stop = nsamp
            if hit.any():
                stop = min(nsamp, int(np.argmax(hit)) + depth)
            m = np.zeros(nsamp, bool)
            m[:stop] = True
            x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
            fx = px - x0; fy = py - y0
            for dx in (0, 1):
                for dy in (0, 1):
                    xi = x0 + dx; yi = y0 + dy
                    w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * step
                    ok = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n) & (w > 0) & m
                    rows.append(np.full(ok.sum(), r)); cols.append(yi[ok] * n + xi[ok])
                    vals.append(w[ok])
            r += 1
    R = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(r, n * n))
    R.sum_duplicates()
    return R


def coverage_hole_probe(counts, n=NPIX):
    img = phantom(n)
    from scipy.ndimage import gaussian_filter
    ang = np.linspace(0, np.pi, 32, endpoint=False)
    occ = img > 0.05
    R = radon_matrix_occluded(occ, n, NBIN, ang, depth=3)
    nmeas = R.shape[0]
    pixmass = np.asarray(R.sum(0)).ravel()
    print(f"  measurements = {nmeas}; pixels never crossed by any ray = "
          f"{(pixmass <= 1e-12).mean()*100:.1f}% of {n*n}")
    seen = (pixmass > 1e-12)
    f_true = img.ravel()
    g = R @ f_true
    scale = 20000.0 / g.sum()
    g = RNG.poisson(g * scale) / scale
    fmap = feature_map(gaussian_filter(img, 1.0))
    for N in counts:
        nodes = floyd_steinberg_nodes(fmap, N)
        B = interp_matrix(nodes, n)
        Am = R @ B
        colmass = np.asarray(Am.sum(0)).ravel()
        dead = float((colmass <= 1e-12).mean())
        sv = np.linalg.svd(Am.toarray(), compute_uv=False)
        rank = int((sv > sv[0] * 1e-10).sum())
        nz = sv[sv > sv[0] * 1e-10]
        cond_live = nz[0] / nz[-1]
        cond_all = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        # EM recon; report accuracy separately on OBSERVED and HIDDEN pixels
        sens = np.asarray(Am.sum(0)).ravel(); sens = np.where(sens > 0, sens, 1.0)
        f = np.ones(Am.shape[1])
        for _ in range(30):
            p = Am @ f
            rr = np.divide(g, p, out=np.zeros_like(g), where=p > 1e-12)
            f = f * (Am.T @ rr) / sens
        rec = np.clip(np.asarray(B @ f).ravel(), 0, None)
        e_s = np.mean((f_true[seen] - rec[seen]) ** 2)
        e_h = np.mean((f_true[~seen] - rec[~seen]) ** 2)
        ps = 10 * np.log10(img.max() ** 2 / e_s)
        ph = 10 * np.log10(img.max() ** 2 / e_h)
        print(f"  N={len(nodes):5d}  meas/nodes={nmeas/len(nodes):6.2f}  "
              f"dead nodes={dead*100:6.2f}%  rank/N={rank/len(nodes):5.3f}  "
              f"cond(all)={cond_all:9.3e}  cond(live)={cond_live:9.3e}  "
              f"PSNR seen={ps:6.2f}dB  hidden={ph:6.2f}dB")


def main():
    print("=" * 100)
    print("Brankov et al. 2004 CAMM node-density claim -- synthetic replication")
    print(f"image {NPIX}x{NPIX} = {NPIX*NPIX} pixels, {NBIN} detector bins/view")
    print("=" * 100)

    counts = [32, 64, 96, 128, 171, 205, 256, 341, 512, 683, 1024]

    print("\n[A] FULL ANGULAR SAMPLING: 32 views over 180 deg -> 1024 measurements "
          "(#meas == #pixels, as in the paper: 64x64 bins == 64x64 pixels)")
    ang = np.linspace(0, np.pi, 32, endpoint=False)
    nm, rowsA = run_regime("full", ang, counts, noise_counts=2400.0, adaptive=True)

    print("\n[A-uniform] same but NON-adaptive (uniform grid) node placement")
    _, rowsAu = run_regime("full-unif", ang, counts, noise_counts=2400.0, adaptive=False)

    print("\n[B] VIEW-STARVED: 8 views over 180 deg -> 256 measurements "
          "(mimics our 54-view ScanNet scene)")
    ang2 = np.linspace(0, np.pi, 8, endpoint=False)
    nmB, rowsB = run_regime("starved", ang2, counts, noise_counts=2400.0, adaptive=True)

    print("\n[C] LIMITED ANGLE: 32 views over 60 deg -> 1024 measurements "
          "(coverage hole in ANGLE, not in count)")
    ang3 = np.linspace(0, np.pi / 3, 32, endpoint=False)
    nmC, rowsC = run_regime("limited", ang3, counts, noise_counts=2400.0, adaptive=True)

    def best(rows, col, label):
        i = int(np.argmax([r[col] for r in rows]))
        print(f"  {label}: best at N={rows[i][0]} -> meas/nodes = {rows[i][1]:.2f} "
              f"(PSNR {rows[i][col]:.2f} dB)")

    print("\n" + "=" * 100)
    print("SUMMARY -- where is the optimum in (#measurements / #nodes)?")
    print("Brankov's claim: optimum at ratio ~5-7")
    print("=" * 100)
    best(rowsA, 8, "[A] full sampling, EM (paper's method)")
    best(rowsA, 4, "[A] full sampling, damped LS")
    best(rowsA, 3, "[A] full sampling, noiseless LS")
    best(rowsAu, 8, "[A-uniform] non-adaptive, EM")
    best(rowsB, 8, "[B] view-starved, EM")
    best(rowsC, 8, "[C] limited-angle, EM")

    # ---------------- MDL: the paper's own model-selection criterion ----------------
    print("\n" + "=" * 100)
    print("MDL CRITERION (their eq.25) -- note it uses ONLY the reference image, never the data")
    print("=" * 100)
    img = phantom(NPIX)
    from scipy.ndimage import gaussian_filter
    for blur, tag in ((0.0, "no blur"), (1.0, "sigma=1 blur"), (2.0, "sigma=2 blur")):
        ref = gaussian_filter(img, blur) if blur > 0 else img
        mc = mdl_curve(ref, counts)
        i = int(np.argmin([m[1] for m in mc]))
        print(f"  {tag:14s}: MDL minimum at N={mc[i][0]:4d}  "
              f"(N_pixels/N = {NPIX*NPIX/mc[i][0]:5.2f}, "
              f"approx PSNR at that N = {mc[i][2]:.2f} dB)")
        print("      " + "  ".join(f"{m[0]}:{m[1]:.0f}" for m in mc))
    print("  -> the MDL optimum is a property of the IMAGE, identical for any number of views.")

    # -------- regime D: genuine geometric coverage hole (truncated field of view) -----
    print("\n" + "=" * 100)
    print("[D] GEOMETRIC COVERAGE HOLE: 32 views x 32 bins, but rays TERMINATE 3 samples")
    print("    after the first surface hit (occlusion / transmittance model), so the")
    print("    interior of the object is NEVER touched by any ray.")
    print("    This is the analogue of our occluded / never-observed ScanNet cells.")
    print("=" * 100)
    coverage_hole_probe(counts)

    print("\nCOVERAGE PROBE (regimes A-C) -- does shrinking the model kill the "
          "dead-unknown fraction?")
    for label, rows in (("[B] view-starved", rowsB), ("[C] limited-angle", rowsC),
                        ("[A] full", rowsA)):
        print(f"  {label}: " + "  ".join(
            f"N={r[0]}:dead={r[7]*100:.1f}%,rank/N={r[6]:.3f}" for r in rows[::3]))


if __name__ == "__main__":
    main()
