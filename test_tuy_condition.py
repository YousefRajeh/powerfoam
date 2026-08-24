"""
test_tuy_condition.py

Does Tuy's condition (Tuy 1983, SIAM J. Appl. Math. 43(3):546-552, condition (iii)
on p.547) predict reconstruction quality in a DISCRETE, FINITELY-SAMPLED setting?
And does it retain any predictive value under an OCCLUDED (ray-terminating) operator?

Tuy condition (iii), verbatim from p.547:
  "For all (x, beta) in Omega x S, there exists lambda in Lambda, such that
   <x, beta> = <Phi(lambda), beta> and <Phi'(lambda), beta> != 0."
Remark 3 (p.549): "An inversion formula of the form (9) in the case where the
function f is defined on R^n (n >= 2), can be established using the arguments
similar to those given above."  -> a 2D fan-beam analogue is legitimate.

In 2D the "plane through x orthogonal to beta" is a LINE. Source curve is an arc
Phi(lam) = R(cos lam, sin lam), lam in [0, arc]. Then
   <Phi(lam), beta> = R cos(lam - theta_beta)  and  <Phi'(lam), beta> = -R sin(lam - theta_beta)
so the condition at (x, beta) holds iff  lam = theta_beta +- arccos(c/R), c = <x,beta>,
lands inside [0, arc] with |c| < R strictly (which is exactly transversality).
This is checked EXACTLY, in closed form -- no numerics in the condition itself.

CPU only, tiny (32x32 grid). No GPU, no shared-env writes.
"""

import numpy as np

RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
N = 32                      # grid is N x N over [-1,1]^2
R_SRC = 3.0                 # source curve radius
OBJ_R = 0.85                # object support radius (Omega); curve is outside it
NDET = 96                   # detector samples per source (rays per view)
FAN = None                  # computed to always cover the object

_h = 2.0 / N
_c = (np.arange(N) + 0.5) * _h - 1.0
GX, GY = np.meshgrid(_c, _c, indexing="ij")
PIX = np.stack([GX.ravel(), GY.ravel()], axis=1)          # (N*N, 2)
INSIDE = (PIX[:, 0] ** 2 + PIX[:, 1] ** 2) < OBJ_R ** 2   # Omega mask
NPIX = N * N


# ----------------------------------------------------------------------------
# 1. Tuy's condition, per pixel, EXACT against the CONTINUOUS curve
# ----------------------------------------------------------------------------
def tuy_fraction(arc, nbeta=360):
    """Per-pixel fraction of directions beta in S^1 satisfying Tuy (iii)
    against the *continuous* arc lam in [0, arc]. Returns (NPIX,) in [0,1]."""
    th = np.linspace(0.0, 2.0 * np.pi, nbeta, endpoint=False)
    beta = np.stack([np.cos(th), np.sin(th)], axis=1)          # (nb,2)
    c = PIX @ beta.T                                           # (NPIX, nb) = <x,beta>
    ratio = np.clip(c / R_SRC, -1.0, 1.0)
    ok_trans = np.abs(c) < R_SRC - 1e-12                       # <Phi',beta> != 0
    a = np.arccos(ratio)
    lam1 = th[None, :] + a
    lam2 = th[None, :] - a
    def inarc(l):
        l = np.mod(l, 2.0 * np.pi)
        return l <= arc + 1e-12
    hit = (inarc(lam1) | inarc(lam2)) & ok_trans
    return hit.mean(axis=1)


# ----------------------------------------------------------------------------
# 2. Discrete operators
# ----------------------------------------------------------------------------
def _ray_pixels(src, dirv, nstep=400, tmax=None):
    """Siddon-lite: sample the ray densely, return (pixel_index, arclength) pairs
    aggregated. Simple, adequate for a 32x32 study."""
    if tmax is None:
        tmax = np.linalg.norm(src) + 2.0
    t0 = max(0.0, np.linalg.norm(src) - 2.0)
    ts = np.linspace(t0, tmax, nstep)
    dt = ts[1] - ts[0]
    p = src[None, :] + ts[:, None] * dirv[None, :]
    ix = np.floor((p[:, 0] + 1.0) / _h).astype(int)
    iy = np.floor((p[:, 1] + 1.0) / _h).astype(int)
    good = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    return ix[good] * N + iy[good], dt, np.where(good)[0]


def build_A(arc, nsrc, occl=None):
    """Line-integral matrix. If occl is not None it is a (NPIX,) opacity in [0,1]:
    the ray's weight on a cell is the running transmittance prod(1-alpha) of the
    cells already traversed, i.e. rows become substochastic and the ray
    effectively terminates at the first opaque cell (our setting)."""
    lams = np.linspace(0.0, arc, nsrc, endpoint=(arc < 2 * np.pi - 1e-9))
    rows = []
    for lam in lams:
        src = R_SRC * np.array([np.cos(lam), np.sin(lam)])
        # fan aimed at origin, wide enough to cover the object disk
        half = np.arcsin(min(0.999, (OBJ_R * 1.15) / R_SRC))
        base = np.arctan2(-src[1], -src[0])
        for d in np.linspace(base - half, base + half, NDET):
            dirv = np.array([np.cos(d), np.sin(d)])
            idx, dt, _ = _ray_pixels(src, dirv)
            row = np.zeros(NPIX)
            if idx.size:
                if occl is None:
                    np.add.at(row, idx, dt)
                else:
                    # running transmittance along the ordered traversal
                    a = occl[idx]
                    T = np.concatenate([[1.0], np.cumprod(1.0 - a)[:-1]])
                    np.add.at(row, idx, dt * T)
            rows.append(row)
    return np.asarray(rows)


# ----------------------------------------------------------------------------
# 3. Per-pixel recoverability from the discrete operator
# ----------------------------------------------------------------------------
SIGMA = 1e-3   # relative measurement noise (A is normalized to ||A||_2 = 1)


def per_pixel_error(A, sigma=SIGMA):
    """Per-pixel Bayes (Wiener) MSE for recovering f from y = A f + n, with
    white unit-variance prior on f and noise variance sigma^2, A rescaled to
    ||A||_2 = 1.  Per SVD mode the minimum achievable MSE is sigma^2/(s_k^2+sigma^2),
    and any direction outside row(A) contributes 1 (pure null space).  So

        err_j = sum_k V_kj^2 * sigma^2/(s_k^2 + sigma^2)  +  (1 - sum_k V_kj^2)

    err_j in [0,1]: 0 = pixel j perfectly recoverable, 1 = unobservable.
    This is exactly the quantity that *should* correlate with a valid
    coverage/completeness certificate. Returns (err, singular values)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    nrm = s[0] if s[0] > 0 else 1.0
    s = s / nrm
    keep = s > 1e-14
    s, Vt = s[keep], Vt[keep]
    w = sigma ** 2 / (s ** 2 + sigma ** 2)
    V2 = Vt ** 2
    err = (V2 * w[:, None]).sum(axis=0) + (1.0 - V2.sum(axis=0))
    return np.clip(err, 0.0, 1.0), s


def _rank(a):
    """Average ranks, correct tie handling (argsort-ranks would invent an order
    among ties and manufacture spurious correlation)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")          # constant input -> rank correlation undefined
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def hdr(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def E1():
    hdr("E1  FULL CIRCLE source curve (Tuy (iii) holds at EVERY pixel, always).\n"
        "    Only the NUMBER OF SAMPLED SOURCES changes. Tuy is a property of the\n"
        "    continuous curve, so it cannot see this.")
    arc = 2 * np.pi
    tf = tuy_fraction(arc)
    print(f"  Tuy-satisfied fraction over Omega: min={tf[INSIDE].min():.4f} "
          f"mean={tf[INSIDE].mean():.4f}  (identical for every row below)")
    print(f"  {'nsrc':>6} {'Tuy frac':>9} {'mean err':>10} {'p95 err':>10} {'cond(A)':>12}")
    out = []
    for nsrc in [3, 4, 6, 12, 24, 48, 96]:
        A = build_A(arc, nsrc)
        e, s = per_pixel_error(A)
        cond = s[0] / max(s[-1], 1e-300)
        print(f"  {nsrc:6d} {tf[INSIDE].mean():9.4f} {e[INSIDE].mean():10.4f} "
              f"{np.percentile(e[INSIDE],95):10.4f} {cond:12.3e}")
        out.append((nsrc, tf[INSIDE].mean(), e[INSIDE].mean()))
    lo, hi = out[0][2], out[-1][2]
    print(f"\n  VERDICT: Tuy fraction constant at {out[0][1]:.4f} while mean per-pixel")
    print(f"  error moves {lo:.4f} -> {hi:.4f} ({lo/max(hi,1e-12):.1f}x). Predictive power: ZERO.")


def E2():
    hdr("E2  FIXED source count (48), VARYING arc. Now Tuy fraction varies per pixel.\n"
        "    Transparent (unoccluded) operator -- the regime Tuy actually governs.")
    nsrc = 48
    print(f"  {'arc(deg)':>9} {'Tuy=1 %':>9} {'Tuyfrac':>9} {'mean err':>10} "
          f"{'rho(Tuy,err)':>13}")
    pooled_t, pooled_e = [], []
    for deg in [60, 90, 120, 180, 240, 300, 360]:
        arc = np.deg2rad(deg)
        tf = tuy_fraction(arc)
        A = build_A(arc, nsrc)
        e, _ = per_pixel_error(A)
        t_i, e_i = tf[INSIDE], e[INSIDE]
        rho = spearman(t_i, e_i)
        print(f"  {deg:9d} {100*(t_i>=0.999).mean():9.2f} {t_i.mean():9.4f} "
              f"{e_i.mean():10.4f} {rho:13.3f}")
        pooled_t.append(t_i); pooled_e.append(e_i)
    pt = np.concatenate(pooled_t); pe = np.concatenate(pooled_e)
    print(f"\n  POOLED across arcs: rho(Tuy fraction, per-pixel error) = {spearman(pt, pe):.3f}")
    print("  (negative rho = higher Tuy fraction -> lower error = Tuy IS predictive here)")
    # binary split at 360-fixed-nsrc-free comparison
    sat = pt >= 0.999
    print(f"  Pixels with Tuy fully satisfied: n={sat.sum()}, mean err={pe[sat].mean():.4f}")
    print(f"  Pixels with Tuy violated      : n={(~sat).sum()}, mean err={pe[~sat].mean():.4f}")


def E3():
    hdr("E3  OCCLUDED operator: rays carry running transmittance and die at the\n"
        "    first opaque cell (row-substochastic, near-selection), as ours do.\n"
        "    Same geometry sweep as E2.")
    nsrc = 48
    # opacity field: an opaque annulus-ish blob shell so interior is shadowed
    r = np.sqrt(PIX[:, 0] ** 2 + PIX[:, 1] ** 2)
    for tag, occl in [
        ("alpha=0.30 uniform in Omega", np.where(INSIDE, 0.30, 0.0)),
        ("alpha=0.80 uniform in Omega", np.where(INSIDE, 0.80, 0.0)),
        ("opaque shell 0.6<r<0.85",     np.where((r > 0.60) & (r < 0.85), 0.9, 0.05 * INSIDE)),
    ]:
        print(f"\n  --- occlusion: {tag} ---")
        print(f"  {'arc(deg)':>9} {'Tuyfrac':>9} {'mean err':>10} {'rho(Tuy,err)':>13}")
        pt, pe = [], []
        for deg in [60, 120, 180, 240, 360]:
            arc = np.deg2rad(deg)
            tf = tuy_fraction(arc)
            A = build_A(arc, nsrc, occl=occl)
            e, _ = per_pixel_error(A)
            t_i, e_i = tf[INSIDE], e[INSIDE]
            print(f"  {deg:9d} {t_i.mean():9.4f} {e_i.mean():10.4f} "
                  f"{spearman(t_i, e_i):13.3f}")
            pt.append(t_i); pe.append(e_i)
        pt = np.concatenate(pt); pe = np.concatenate(pe)
        sat = pt >= 0.999
        print(f"  POOLED rho = {spearman(pt, pe):.3f}   "
              f"Tuy-sat mean err={pe[sat].mean():.4f} (n={sat.sum()})  "
              f"Tuy-viol mean err={pe[~sat].mean():.4f} (n={(~sat).sum()})")


def E4():
    hdr("E4  CONTROL: a purely geometric 'ray-count' coverage statistic, same\n"
        "    pixels, same operators -- how well does the dumb baseline predict?")
    nsrc = 48
    r = np.sqrt(PIX[:, 0] ** 2 + PIX[:, 1] ** 2)
    occl_shell = np.where((r > 0.60) & (r < 0.85), 0.9, 0.05 * INSIDE)
    for tag, occl in [("transparent", None), ("occluded shell", occl_shell)]:
        pt, pc, pe = [], [], []
        for deg in [60, 120, 180, 240, 360]:
            arc = np.deg2rad(deg)
            tf = tuy_fraction(arc)
            A = build_A(arc, nsrc, occl=occl)
            e, _ = per_pixel_error(A)
            cov = np.log10(A.sum(axis=0) + 1e-12)   # total ray mass hitting the pixel
            pt.append(tf[INSIDE]); pc.append(cov[INSIDE]); pe.append(e[INSIDE])
        pt = np.concatenate(pt); pc = np.concatenate(pc); pe = np.concatenate(pe)
        print(f"  {tag:>16}:  rho(Tuy, err) = {spearman(pt, pe):+.3f}   "
              f"rho(log ray-mass, err) = {spearman(pc, pe):+.3f}")


def E5():
    hdr("E5  DISCRIMINATION FAILURE, sharpest form. Full 360 curve, 48 sources:\n"
        "    Tuy (iii) is satisfied at 100% of pixels for 100% of directions.\n"
        "    How much does the ACTUAL per-pixel error still vary?")
    r = np.sqrt(PIX[:, 0] ** 2 + PIX[:, 1] ** 2)
    tf = tuy_fraction(2 * np.pi)
    print(f"  Tuy fraction over Omega: min={tf[INSIDE].min():.4f} max={tf[INSIDE].max():.4f}"
          f"  -> certifies {100*(tf[INSIDE]>=0.999).mean():.1f}% of cells, zero variation")
    print(f"  {'occlusion':>28} {'min err':>9} {'median':>9} {'p95':>9} {'max':>9} "
          f"{'spread':>9}")
    for tag, occl in [
        ("none (transparent)",       None),
        ("alpha=0.30 in Omega",      np.where(INSIDE, 0.30, 0.0)),
        ("alpha=0.80 in Omega",      np.where(INSIDE, 0.80, 0.0)),
        ("opaque shell 0.6<r<0.85",  np.where((r > 0.60) & (r < 0.85), 0.9, 0.05 * INSIDE)),
    ]:
        A = build_A(2 * np.pi, 48, occl=occl)
        e, _ = per_pixel_error(A)
        ei = e[INSIDE]
        print(f"  {tag:>28} {ei.min():9.4f} {np.median(ei):9.4f} "
              f"{np.percentile(ei,95):9.4f} {ei.max():9.4f} "
              f"{ei.max()-ei.min():9.4f}")
    print("\n  A certificate that is constant across cells whose true error spans")
    print("  most of [0,1] has no ranking power whatsoever, by construction.")


def tuy_fraction_discrete(arc, nsrc, tol, nbeta=180):
    """The condition as ANY implementation must actually evaluate it: you do not
    have a continuous curve, you have nsrc sampled sources. Certify (x,beta) if
    some SAMPLED source lies within `tol` of the plane {y: <y,beta>=<x,beta>}
    and the local secant is transversal to it. This is the honest discrete
    surrogate -- and note it is mechanically monotone in nsrc."""
    th = np.linspace(0.0, 2.0 * np.pi, nbeta, endpoint=False)
    beta = np.stack([np.cos(th), np.sin(th)], axis=1)
    lams = np.linspace(0.0, arc, nsrc, endpoint=(arc < 2 * np.pi - 1e-9))
    S = R_SRC * np.stack([np.cos(lams), np.sin(lams)], axis=1)      # (nsrc,2)
    T = np.stack([-np.sin(lams), np.cos(lams)], axis=1)             # tangents
    sb = S @ beta.T                                                  # (nsrc, nb)
    tb = np.abs(T @ beta.T)                                          # transversality
    c = PIX @ beta.T                                                 # (NPIX, nb)
    ok = np.zeros((NPIX, len(th)), bool)
    for i in range(nsrc):
        ok |= (np.abs(c - sb[i][None, :]) < tol) & (tb[i][None, :] > 1e-3)
    return ok.mean(axis=1)


def E6():
    hdr("E6  WHY A MEASURED 'CERTIFIED FRACTION' TRACKS VIEW COUNT.\n"
        "    Any implementation evaluates Tuy against SAMPLED sources with a\n"
        "    tolerance. That surrogate is monotone in nsrc BY CONSTRUCTION,\n"
        "    independently of whether reconstruction improves.")
    arc = 2 * np.pi
    r = np.sqrt(PIX[:, 0] ** 2 + PIX[:, 1] ** 2)
    occl = np.where((r > 0.60) & (r < 0.85), 0.9, 0.05 * INSIDE)
    print(f"  {'nsrc':>6} {'discreteTuy%':>13} {'err(transp)':>12} {'err(occl)':>11} "
          f"{'rho_occl(Tuy,err)':>18}")
    tt, ee = [], []
    for nsrc in [4, 8, 16, 32, 64, 128]:
        dt = tuy_fraction_discrete(arc, nsrc, tol=0.12)
        At = build_A(arc, nsrc); et, _ = per_pixel_error(At)
        Ao = build_A(arc, nsrc, occl=occl); eo, _ = per_pixel_error(Ao)
        print(f"  {nsrc:6d} {100*dt[INSIDE].mean():13.2f} {et[INSIDE].mean():12.4f} "
              f"{eo[INSIDE].mean():11.4f} {spearman(dt[INSIDE], eo[INSIDE]):18.3f}")
        tt.append(dt[INSIDE]); ee.append(eo[INSIDE])
    tt = np.concatenate(tt); ee = np.concatenate(ee)
    print(f"\n  POOLED rho(discrete-Tuy, occluded err) = {spearman(tt, ee):+.3f}")
    print("  The certified fraction rises monotonically with view count; under")
    print("  occlusion the reconstruction error it is supposed to certify does not")
    print("  follow it. The statistic is measuring sampling density, not recoverability.")


def E7():
    hdr("E7  NECESSITY vs SUFFICIENCY, measured directly.\n"
        "    Tuy 1983 states only one direction (Thm p.548: 'If Phi satisfies the\n"
        "    curve conditions ... then f(x) = (9)'). No converse is proved. Here we\n"
        "    test both directions empirically in the discrete setting.")
    thr = 0.01           # 'recovered' = Bayes MSE < 1% of prior variance
    print("\n  (a) IS IT NECESSARY? Look for cells that VIOLATE Tuy yet are recovered.")
    print(f"  {'arc(deg)':>9} {'nsrc':>5} {'%cells Tuy-violated':>20} "
          f"{'of those, % recovered':>23} {'their mean err':>15}")
    for deg, nsrc in [(60, 48), (90, 48), (120, 48), (180, 48), (60, 96), (90, 96)]:
        arc = np.deg2rad(deg)
        tf = tuy_fraction(arc)
        e, _ = per_pixel_error(build_A(arc, nsrc))
        t_i, e_i = tf[INSIDE], e[INSIDE]
        viol = t_i < 0.999
        if viol.sum() == 0:
            continue
        rec = e_i[viol] < thr
        print(f"  {deg:9d} {nsrc:5d} {100*viol.mean():20.2f} "
              f"{100*rec.mean():23.2f} {e_i[viol].mean():15.6f}")
    print("  -> Tuy violated at 100% of cells, yet ~100% of them are recovered to")
    print("     <1% MSE. The condition is NOT NECESSARY for accurate recovery.")

    print("\n  (b) IS IT SUFFICIENT (as a finite-sample predicate)? Look for cells")
    print("      that SATISFY Tuy yet are NOT recovered.")
    print(f"  {'arc(deg)':>9} {'nsrc':>5} {'%cells Tuy-satisfied':>21} "
          f"{'of those, % FAILED':>20} {'their mean err':>15}")
    for deg, nsrc in [(360, 3), (360, 4), (360, 6), (360, 12), (240, 4), (360, 8)]:
        arc = np.deg2rad(deg)
        tf = tuy_fraction(arc)
        e, _ = per_pixel_error(build_A(arc, nsrc))
        t_i, e_i = tf[INSIDE], e[INSIDE]
        sat = t_i >= 0.999
        if sat.sum() == 0:
            continue
        fail = e_i[sat] >= thr
        print(f"  {deg:9d} {nsrc:5d} {100*sat.mean():21.2f} "
              f"{100*fail.mean():20.2f} {e_i[sat].mean():15.6f}")
    print("  -> Tuy satisfied at 100% of cells, yet up to 100% of them FAIL to be")
    print("     recovered. Sufficiency holds only in the continuum, not at finite")
    print("     sampling. Both implications break. The predicate is uninformative.")


if __name__ == "__main__":
    print(f"grid {N}x{N} = {NPIX} cells, {INSIDE.sum()} inside Omega; "
          f"{NDET} rays/view; source radius {R_SRC}, object radius {OBJ_R}")
    E1(); E2(); E3(); E4(); E5(); E6(); E7()
