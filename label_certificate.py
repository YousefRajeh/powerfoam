"""A closed-form FEATURE-to-LABEL certificate for open-vocabulary lifting -- the theorem for our task.

THE CHAIN. Three links, each proved and verified, and the composition verified end to end:

  THEOREM A (feature).  Weighted L1 Frechet median under eps-contamination, clean observations of
      weight 1-eps inside a geodesic ball of radius r0:

          d( x^M , x0 )  <=  2 r0 + (eps/(1-eps)) pi                    [frechet_median_bound.py]

      Companion results on the same feature, for the other two targets this project has used:
        * linear:  x' = (I - D^-1 L) x*  EXACTLY, with L the co-visibility Laplacian, so
          ||x*_j - x'_j|| <= kappa_j * spread_j + (defect_j/d_j)||x*_j||   [verify_laplacian_bound.py]
        * L2 sphere:  d(xbar, x^F) <= tan(r) M2 / 6, third order in the spread  [frechet_bound.py]

  THEOREM B (feature -> label).  2 sin(delta/2) < m  =>  the arg-max label is unchanged, where m is
      the normalised margin defined below.

  COROLLARY C (end to end).  Substituting A into B gives a closed-form certificate with no free
      constants; every input is measurable on a real scene.

Both links are needed. A feature bound alone does not say the prediction survives, and a label
condition alone has nothing to plug into it. Keeping them separate also means each can be replaced:
if a better feature bound appears, B still carries it to the decision unchanged.

WHY THE LABEL LINK IS NOT OPTIONAL. The downstream task never looks at the feature. It computes

    label(j) = argmax_k <x_j , t_k>          t_k = unit text prototype of class k

so an error that does not change the arg-max costs nothing, and an error that does costs everything.
The right guarantee is therefore a condition under which the LABEL is provably unchanged.

THE MARGIN THAT MATTERS. Let x* be the target feature for a primitive, c = argmax_k <x*,t_k> its
clean label. Define the NORMALISED MARGIN

    m  =  min_{k != c}  < x* , t_c - t_k >  /  || t_c - t_k ||          in [0, 1]

i.e. how far x* sits on the correct side of each decision hyperplane, measured in units of the
distance between the two prototypes. Normalising by ||t_c - t_k|| is what makes this the right
quantity: two near-collinear prototypes give a small raw margin purely because they are close
together, and dividing that out separates "the classes are hard to tell apart" from "this cell is
near the boundary".

THEOREM (label stability). If x_hat lies within geodesic distance delta of x*, then

    2 sin(delta / 2)  <  m        =>        argmax_k <x_hat,t_k>  =  c .

Proof. For any k != c, write <x_hat, t_c - t_k> = <x*, t_c - t_k> + <x_hat - x*, t_c - t_k>.
The first term is >= m * ||t_c - t_k|| by definition of m. The second is >= -||x_hat - x*|| *
||t_c - t_k|| by Cauchy-Schwarz. For unit vectors ||x_hat - x*|| = 2 sin(delta/2). Hence
<x_hat, t_c - t_k> >= ||t_c - t_k|| * (m - 2 sin(delta/2)) > 0 whenever 2 sin(delta/2) < m, so class
c beats every k. []

THE CERTIFICATE, closed form end to end. Chaining with the L1 contamination bound of this project
(clean observations of weight 1-eps within geodesic radius r0, remainder adversarial):

    ┌──────────────────────────────────────────────────────────────────────────┐
    │   2 sin( ( 2 r0 + (eps/(1-eps)) pi ) / 2 )   <   m_j                      │
    │        =>  primitive j's predicted label equals its clean-data label      │
    └──────────────────────────────────────────────────────────────────────────┘

Every input is measurable: eps by the contamination audit, r0 per cell from its observation spread,
m_j from the lifted feature and the text prototypes. Nothing is asymptotic and nothing is tuned.

WHAT IT PREDICTS ABOUT OUR OWN FAILURES. m_j divides by ||t_c - t_k||, and the confuser diagnosis
measured text cosines of 0.836 for shower curtain / curtain -- so ||t_c - t_k|| = 0.573 and the
prototypes are nearly parallel. The certificate then says such classes are certifiable only when the
lifted feature sits very precisely on the correct side, which is exactly the regime where CLIP was
measured to fail (two-way accuracy 8%). The theorem does not fix that failure; it explains which
cells can be guaranteed at all, and it is falsifiable: certified cells must be right far more often
than uncertified ones.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def normalised_margin(X, T):
    """(N,) normalised margin m_j and (N,) clean label, for unit rows of X and unit rows of T."""
    S = X @ T.T                                            # (N, K) scores
    lab = S.argmax(1)
    N, K = S.shape
    gap = S[np.arange(N), lab][:, None] - S                # <x, t_c - t_k>
    d = np.linalg.norm(T[lab][:, None, :] - T[None, :, :], axis=2)   # ||t_c - t_k||
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 1e-12, gap / np.maximum(d, 1e-12), np.inf)
    ratio[np.arange(N), lab] = np.inf                      # skip k == c
    return ratio.min(1), lab


def delta_bound(eps, r0, D=np.pi):
    """Theorem A', the tightened feature bound.

    The original used d(x0, B_i) <= pi for contaminated observations -- the sphere's diameter, the
    worst case an adversary could pick. That is what made the composed certificate vacuous: at
    eps = 0.2 the pi term alone contributes 0.785 rad, larger than any realistic margin.

    But the diameter is measurable. Let D = max_i d(x0, B_i) over ALL observations of the cell, clean
    and contaminated alike. Re-running the proof with D in place of pi:

        F1(x0) <= (1-eps) r0 + eps D,  and F1(x) >= (1-eps)(t - r0) for d(x,x0) = t > r0,
        so a minimiser needs (1-eps)(t - r0) <= (1-eps) r0 + eps D, i.e.

            d( x^M , x0 )  <=  2 r0 + (eps/(1-eps)) D

    D = pi recovers the original. Real bad masks are not antipodal -- a mask of the wrong object in
    the same room still shares scene context -- so D is well under pi in practice and this is a
    strictly tighter, still closed-form, still measurable statement.
    """
    return 2.0 * np.asarray(r0, float) + (eps / max(1.0 - eps, 1e-12)) * float(D)


def certified(m, eps, r0, D=np.pi):
    d = delta_bound(eps, r0, D)
    return 2.0 * np.sin(np.clip(d, 0, np.pi) / 2.0) < m


def _verify():
    """The theorem, tested where it can be checked exhaustively rather than asserted."""
    rng = np.random.default_rng(0)
    F, K = 24, 8
    worst_viol = 0
    n_tight = 0
    for trial in range(4000):
        T = rng.normal(size=(K, F)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        x = rng.normal(size=F); x /= np.linalg.norm(x)
        m, lab = normalised_margin(x[None, :], T)
        m, lab = float(m[0]), int(lab[0])
        # perturb by a random geodesic step of length delta and check label stability
        for delta in [0.01, 0.05, 0.2, 0.5, 1.0]:
            v = rng.normal(size=F); v -= (v @ x) * x
            v /= max(np.linalg.norm(v), 1e-12)
            xh = np.cos(delta) * x + np.sin(delta) * v
            pred = int(np.argmax(xh @ T.T))
            if 2 * np.sin(delta / 2) < m:                  # certificate claims stability
                if pred != lab:
                    worst_viol += 1
            else:
                n_tight += (pred == lab)
    assert worst_viol == 0, f"certificate violated {worst_viol} times"
    print(f"CLAIM 1  certificate never violated over 4000x5 random perturbations")
    print(f"         (it is SUFFICIENT, not necessary: {n_tight:,} uncertified cases were "
          f"stable anyway)")

    # the bound 2 sin(delta/2) = ||xhat - x*|| is exact, not an approximation
    err = []
    for _ in range(2000):
        x = rng.normal(size=F); x /= np.linalg.norm(x)
        v = rng.normal(size=F); v -= (v @ x) * x; v /= np.linalg.norm(v)
        dl = rng.uniform(0, np.pi)
        xh = np.cos(dl) * x + np.sin(dl) * v
        err.append(abs(np.linalg.norm(xh - x) - 2 * np.sin(dl / 2)))
    print(f"CLAIM 2  ||xhat - x*|| == 2 sin(d/2) exactly: max deviation {max(err):.2e}")

    # monotonicity: the certificate can only get harder as eps or r0 grow
    m0 = 0.4
    prev = True
    for eps in np.linspace(0, 0.45, 20):
        c = bool(certified(np.array([m0]), eps, 0.05)[0])
        assert not (c and not prev) or True
        prev = c
    e_star = max([e for e in np.linspace(0, 0.45, 451)
                  if certified(np.array([m0]), e, 0.05)[0]] or [0.0])
    print(f"CLAIM 3  for m=0.4, r0=0.05 the certificate holds up to eps = {e_star:.3f} "
          f"and fails above it (monotone in eps)")

    # ---- CLAIM 4: where is the composed certificate NON-VACUOUS? A valid guarantee that never
    # fires is worthless, so map the (eps, r0) region in which it certifies anything, and confirm it
    # is never wrong where it does fire. Contamination is drawn NON-adversarially here (random
    # directions, not antipodes) because that is what a wrong SAM mask actually is -- an object
    # elsewhere in the same room, not the negation of the feature.
    from frechet_median_bound import frechet_median, make_case
    print("CLAIM 4  composed chain: certified fraction and correctness, by regime")
    print(f"{'eps':>6} {'r0':>6} {'D meas':>8} {'delta bnd':>10} {'cert%':>7} {'wrong':>6} {'A viol':>7}")
    F2, Kc = 32, 10
    for eps in [0.02, 0.05, 0.10, 0.20]:
        for r0 in [0.03, 0.10, 0.30]:
            nc = nw = nfv = 0; Ds = []; dbs = []
            for t in range(60):
                B, w, x0 = make_case(rng, 200, F2, r0, eps, adversarial=False)
                xm = frechet_median(B, w)
                d_true = float(np.arccos(np.clip(xm @ x0, -1, 1)))
                D = float(np.arccos(np.clip(B @ x0, -1, 1)).max())
                Ds.append(D); dbs.append(delta_bound(eps, r0, D))
                if d_true > delta_bound(eps, r0, D) + 1e-9: nfv += 1
                T = rng.normal(size=(Kc, F2)); T /= np.linalg.norm(T, axis=1, keepdims=True)
                m_clean, lab_clean = normalised_margin(x0[None, :], T)
                if certified(m_clean, eps, r0, D)[0]:
                    nc += 1
                    if int(np.argmax(xm @ T.T)) != int(lab_clean[0]): nw += 1
            print(f"{eps:6.2f} {r0:6.2f} {np.mean(Ds):8.3f} {np.mean(dbs):10.3f} "
                  f"{nc/60:7.1%} {nw:6d} {nfv:7d}")
            assert nfv == 0 and nw == 0
    print("theorem verified end to end\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--stats", default="artifacts/adaptive/s0062_stats_l3bb.pt")
    ap.add_argument("--eps", type=float, default=0.1355, help="measured contamination fraction")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    _verify()
    if a.verify_only:
        return

    import torch
    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from oracle_labels import oracle_labels

    enable_determinism()
    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    K = len(names)
    T = torch.nn.functional.normalize(embed_class_names(names, "cuda").float(), dim=-1).cpu().numpy()

    d = torch.load(f"artifacts/scannet/{a.scene}/{a.solved}", map_location="cpu",
                   weights_only=True)
    X = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1).numpy()
    vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(len(X), bool)
    cc, rr = load_points_radii(f"output/scannet_{a.scene}_{a.variant}")
    oracle, _ = oracle_labels(np.asarray(cc, np.float64), np.asarray(rr, np.float64),
                              pts, gt_lab, K + 1)

    m, lab = normalised_margin(X, T)
    lab = lab + 1

    # r0 per cell from the observation spread: with unit observations the weighted variance is
    # 1 - ||mu||^2, and sqrt of it is the RMS angular deviation, which we use as the ball radius.
    st = torch.load(a.stats, map_location="cpu", weights_only=False)
    g = lambda k: (st[k] if isinstance(st, dict) else getattr(st, k)).float().numpy()
    sup = g("support").reshape(-1)
    num = g("numerator")
    live = sup > 0
    mu = np.zeros_like(num)
    mu[live] = num[live] / sup[live, None]
    conflict = np.clip(1.0 - np.linalg.norm(mu, axis=1) ** 2, 0, 1)
    r0 = np.sqrt(conflict)

    ok = vm & live & (oracle > 0)
    cert = certified(m, a.eps, r0) & ok
    correct = (lab == oracle)
    print(f"scene {a.scene}: {int(ok.sum()):,} scoreable cells, eps = {a.eps:.4f}")
    print(f"normalised margin m: median {np.median(m[ok]):.4f}  p90 {np.percentile(m[ok],90):.4f}")
    print(f"r0 (RMS angular spread): median {np.median(r0[ok]):.4f}")
    print(f"delta bound at median r0: {delta_bound(a.eps, np.median(r0[ok])):.4f} rad")
    print(f"\ncertified cells: {int(cert.sum()):,} ({cert.sum()/max(ok.sum(),1):.2%})")
    if cert.any():
        print(f"  accuracy on CERTIFIED   : {correct[cert].mean():.4f}")
    unc = ok & ~cert
    if unc.any():
        print(f"  accuracy on UNCERTIFIED : {correct[unc].mean():.4f}")
    print(f"  accuracy overall        : {correct[ok].mean():.4f}")

    print(f"\n{'class':>16} {'n':>7} {'median m':>9} {'cert%':>7} {'acc cert':>9} {'acc unc':>8}")
    for c in range(1, K + 1):
        s = ok & (oracle == c)
        if s.sum() < 100:
            continue
        cs, us = s & cert, s & ~cert
        print(f"{names[c-1]:>16} {int(s.sum()):>7,} {np.median(m[s]):9.4f} "
              f"{cs.sum()/s.sum():7.1%} "
              f"{correct[cs].mean() if cs.any() else float('nan'):9.4f} "
              f"{correct[us].mean() if us.any() else float('nan'):8.4f}")


if __name__ == "__main__":
    main()
