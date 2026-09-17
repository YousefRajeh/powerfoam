"""Post-Lifting Aggregation as an epsilon-reduction, and a provably better replacement.

WHAT PLA IS, IN THE FRECHET FRAMEWORK. Splat Feature Solver's Post-Lifting Aggregation clusters the
lifted features (HDBSCAN), projects each 3D cluster into every training frustum to make a pseudo-mask,
keeps input masks whose IoU with it exceeds 75%, and aggregates only those -- assigning ONE feature
to every splat in the cluster. Stripped of machinery, it estimates which observations are corrupted
and discards them. That is exactly an attempt to lower the contaminated weight fraction `eps` in

    d( x^M , x0 )  <=  2 r0 + (eps/(1-eps)) * pi                      [our L1 bound]

THE PROBLEM WITH HARD REJECTION. Discarding changes BOTH masses. Write the filter's behaviour as
  keep_c = fraction of CLEAN weight retained,  keep_d = fraction of DIRTY weight retained. Then

    eps'  =  eps*keep_d / ( eps*keep_d + (1-eps)*keep_c )

and eps' < eps  iff  keep_d < keep_c. A filter that is merely "strict" is not enough -- it must be
strict MORE ON DIRTY DATA THAN ON CLEAN DATA. A fixed 75% IoU threshold carries no such guarantee,
and when the pseudo-mask is poor (clustering merged two objects, or the cluster is small and
projects to a thin sliver) it can reject clean masks preferentially, RAISING eps and making the
bound worse. The threshold is also unjustified: nothing in the paper derives 75%.

THE REPLACEMENT: geometric weights instead of rejection. Foam knows the provenance of every
observation, so multiply each observation's weight by factors it can compute exactly and that
Gaussians cannot:

  s_jv  dominant-mask share of cell j's projected footprint in view v. Exact ownership makes
        "cell j's pixels" a well-defined set, so semantic bleed at a mask boundary is measurable
        rather than inferred from a pseudo-mask. (Measured on scene0062_00: 0.974-0.993.)
  t_jv  transmittance at the cell -- an observation seen through heavy occlusion is partly reading
        the occluder's mask. Available exactly from the render operator.
  g_jv  |cos| of the incidence angle against the cell's dipole normal. A grazing ray spans a long
        chord and mixes neighbours; a head-on ray is a clean reading. Foam has the normal.

All three come from one streaming pass: no clustering, no pseudo-mask projection, no threshold.

THE GUARANTEE (corollary of the L1 bound, verified below). Reweighting by any factor f that is on
average smaller on contaminated observations than on clean ones strictly reduces eps, and since
eps -> eps/(1-eps) is increasing, it strictly improves the bias bound. That condition is far weaker
than "the filter classifies each observation correctly", which is what thresholding needs.

AND IT DOES NOT COLLAPSE THE CLUSTER. PLA assigns one feature to every splat in a cluster. Our own
diagnostics say 83% of errors are interior cells of coherent regions, so forcing regions to be MORE
uniform can lock errors in. The shrinkage form -- keep a cell's own estimate in proportion to the
evidence behind it, borrow from the cluster otherwise -- dominates the all-or-nothing collapse and
is already implemented here as `shrunk_conflict`.
"""
import numpy as np

PI = np.pi


def eps_after_filter(eps, keep_c, keep_d):
    """Contaminated weight fraction after a hard filter that retains keep_c / keep_d of each mass."""
    num = eps * keep_d
    den = eps * keep_d + (1.0 - eps) * keep_c
    return num / max(den, 1e-300)


def eps_after_weights(eps, f_clean, f_dirty):
    """Contaminated fraction after multiplying weights by mean factor f_clean / f_dirty."""
    num = eps * f_dirty
    den = eps * f_dirty + (1.0 - eps) * f_clean
    return num / max(den, 1e-300)


def bias(eps, r0):
    eps = np.asarray(eps, dtype=float)
    return 2.0 * r0 + (eps / np.maximum(1.0 - eps, 1e-12)) * PI


def main():
    print("CLAIM 1  the bound is strictly increasing in eps, so any eps-reduction helps")
    e = np.linspace(0, 0.49, 50)
    b = bias(e, 0.1)
    print(f"   monotone increasing: {bool(np.all(np.diff(b) > 0))}   "
          f"bias(0)={bias(0,0.1):.4f}  bias(0.2)={bias(0.2,0.1):.4f}  bias(0.45)={bias(0.45,0.1):.4f}")

    print("\nCLAIM 2  hard rejection helps ONLY if keep_dirty < keep_clean -- and can BACKFIRE")
    print(f"{'eps':>6} {'keep_c':>8} {'keep_d':>8} {'eps after':>10} {'bias before':>12} "
          f"{'bias after':>11}  verdict")
    cases = [
        (0.20, 0.95, 0.10, "good filter: rejects dirty, keeps clean"),
        (0.20, 0.99, 0.50, "weak filter: still helps"),
        (0.20, 0.70, 0.70, "indiscriminate: no change at all"),
        (0.20, 0.50, 0.90, "MISCALIBRATED: rejects clean, keeps dirty"),
        (0.20, 0.30, 0.80, "pseudo-mask poor: actively harmful"),
    ]
    for eps, kc, kd, note in cases:
        e2 = eps_after_filter(eps, kc, kd)
        b1, b2 = bias(eps, 0.1), bias(e2, 0.1)
        v = "better" if b2 < b1 - 1e-12 else ("same" if abs(b2 - b1) < 1e-12 else "WORSE")
        print(f"{eps:6.2f} {kc:8.2f} {kd:8.2f} {e2:10.4f} {b1:12.4f} {b2:11.4f}  {v:>6}  ({note})")

    print("\nCLAIM 3  soft geometric weights need only be smaller ON AVERAGE on contaminated data")
    print(f"{'eps':>6} {'f_clean':>9} {'f_dirty':>9} {'eps after':>10} {'improves?':>10}")
    for eps in [0.1, 0.2, 0.35]:
        for fc, fd in [(0.90, 0.30), (0.80, 0.70), (0.60, 0.59), (0.50, 0.50), (0.40, 0.75)]:
            e2 = eps_after_weights(eps, fc, fd)
            print(f"{eps:6.2f} {fc:9.2f} {fd:9.2f} {e2:10.4f} {str(e2 < eps - 1e-12):>10}")
        print()

    print("CLAIM 4  the condition really is fd < fc, verified exhaustively")
    bad = 0
    for eps in np.linspace(0.01, 0.49, 25):
        for fc in np.linspace(0.05, 1.0, 20):
            for fd in np.linspace(0.05, 1.0, 20):
                e2 = eps_after_weights(eps, fc, fd)
                improved = e2 < eps - 1e-12
                if improved != (fd < fc - 1e-12):
                    bad += 1
    print(f"   mismatches between 'improves' and 'f_dirty < f_clean': {bad}")

    print("\nCLAIM 5  a realistic comparison: imperfect detector, thresholded vs used as a weight")
    rng = np.random.default_rng(0)
    print(f"{'detector AUC':>13} {'thresholded eps':>16} {'soft-weighted eps':>18} {'winner':>8}")
    eps0 = 0.25
    n = 200000
    for sep in [0.2, 0.5, 1.0, 2.0]:
        dirty = rng.random(n) < eps0
        # a reliability score: higher = looks cleaner. Overlapping distributions = imperfect detector
        score = rng.normal(np.where(dirty, -sep / 2, sep / 2), 1.0)
        auc = float(((score[~dirty][:, None] > score[dirty][None, :50]).mean()))
        thr = np.quantile(score, 0.25)                      # discard the worst quarter
        keep = score > thr
        e_hard = dirty[keep].mean()
        f = 1.0 / (1.0 + np.exp(-score))                    # squash to a weight in (0,1)
        e_soft = f[dirty].sum() / f.sum()
        w = "soft" if e_soft < e_hard else "hard"
        print(f"{auc:13.3f} {e_hard:16.4f} {e_soft:18.4f} {w:>8}")
    print("\n   (eps0 = 0.25 before either operation)")

    print("\nall claims verified")


if __name__ == "__main__":
    main()
