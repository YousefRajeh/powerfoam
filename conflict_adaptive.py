"""Per-cell conflict from disjoint ownership, and a conflict-adaptive lift.

THE IDEA. Splat Feature Solver regularises with a SINGLE global Tikhonov lambda (Module 8). That is
all it can do: with overlapping Gaussians there is no well-defined set of observations belonging to
one primitive. A power diagram partitions space, so every cell has an exact evidence set -- the rays
that deposit weight in it -- and we can measure how much those observations DISAGREE, per cell,
before choosing how much to trust that cell.

THE CONFLICT MEASURE IS FREE. SAM+CLIP mask embeddings are unit norm, so for cell j with weights
w_i = A_ij summing to W_j and mean mu_j = (1/W_j) sum_i w_i B_i:

    (1/W_j) sum_i w_i ||B_i - mu_j||^2
        = (1/W_j) sum_i w_i (||B_i||^2 - 2 B_i.mu_j + ||mu_j||^2)
        = 1 - 2||mu_j||^2 + ||mu_j||^2
        = 1 - ||mu_j||^2                                     <-- conflict, in [0, 1]

So the weighted variance of a cell's observations is just the shrinkage of its mean vector's LENGTH.
This is the resultant length from directional statistics. ||mu|| = 1 means every view agreed
exactly; ||mu|| -> 0 means the views pointed all over the sphere. It needs no second pass and no
extra storage -- but it IS destroyed by the solver, which returns unit vectors, which is why this
reads the accumulator's `numerator`/`support` rather than a solved .pt.

WHY THIS IS THE RIGHT TERM TO ATTACK. Measured on scene0062_00 (measure_participation.py), a foam
ray depends on 1.01 primitives at the median, so co-visibility -- the solver's own error term -- is
already near zero on foam. What remains is observation conflict, exactly the ramen-bowl failure the
paper describes. Note also what this is NOT: a smoothing prior over the facet graph. That family has
been tested three times here (Potts, linear, bilateral) and does not pay, because 83% of errors are
interior cells of coherent regions rather than boundary cells.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")


def conflict_from_stats(numerator, support, eps=1e-12):
    """(P,) conflict in [0,1] and the mean vectors, from the accumulator's sufficient statistics.

    conflict_j = 1 - ||mu_j||^2 with mu_j = numerator_j / support_j. Exact when the observations
    are unit norm; an upper bound on the normalised variance otherwise, since ||B_i|| <= 1 only
    lowers the first term.
    """
    num = np.asarray(numerator, dtype=np.float64)
    sup = np.asarray(support, dtype=np.float64)
    live = sup > eps
    mu = np.zeros_like(num)
    mu[live] = num[live] / sup[live, None]
    r = np.linalg.norm(mu, axis=1)
    return np.clip(1.0 - r ** 2, 0.0, 1.0), mu, live


def effective_views(support, sum_view_weight_sq, eps=1e-12):
    """Kish effective number of VIEWS backing each cell: (sum_v W)^2 / sum_v W^2."""
    sup = np.asarray(support, dtype=np.float64).reshape(-1)
    svw = np.asarray(sum_view_weight_sq, dtype=np.float64).reshape(-1)
    out = np.zeros_like(sup)
    m = svw > eps
    out[m] = sup[m] ** 2 / svw[m]
    return out


def shrunk_conflict(conflict, n_eff, prior_strength=2.0, live=None):
    """Conflict, corrected for how little evidence the cell actually has.

    A cell seen by ONE view has ||mu|| = 1 and therefore conflict exactly 0 -- not because its
    observations agree but because it has nothing to disagree with. Measured on scene0062_00 the
    lowest-conflict decile averages 2.45 effective views against ~6 in the middle, and it is the
    ONLY decile that breaks the accuracy trend. So the raw statistic conflates "unanimous" with
    "unconfirmed".

    The fix is the standard shrinkage estimator: pull each cell's conflict toward the population
    mean with weight equal to the evidence behind it,

        chat_j = (n_eff_j * c_j + k * cbar) / (n_eff_j + k)

    so a 1-view cell is treated as average-until-proven-otherwise while a 20-view cell keeps its
    measured value almost untouched. `k` is in units of views.
    """
    c = np.asarray(conflict, dtype=np.float64)
    n = np.asarray(n_eff, dtype=np.float64)
    sel = np.ones(len(c), bool) if live is None else np.asarray(live)
    cbar = float(c[sel].mean()) if sel.any() else 0.0
    return (n * c + prior_strength * cbar) / np.maximum(n + prior_strength, 1e-12)


def _self_test():
    """The identity above, on inputs where the answer is known by hand."""
    rng = np.random.default_rng(0)
    F = 8
    # 1. all observations identical  -> conflict 0
    b = rng.normal(size=F); b /= np.linalg.norm(b)
    w = np.array([0.3, 0.5, 0.2])
    num = (w[:, None] * np.tile(b, (3, 1))).sum(0)[None, :]
    c, _, _ = conflict_from_stats(num, np.array([w.sum()]))
    assert abs(c[0]) < 1e-12, c
    # 2. two antipodal observations, equal weight -> mean 0 -> conflict 1
    num2 = (0.5 * b - 0.5 * b)[None, :]
    c2, _, _ = conflict_from_stats(num2, np.array([1.0]))
    assert abs(c2[0] - 1.0) < 1e-12, c2
    # 3. brute force against the definition on random unit vectors
    B = rng.normal(size=(40, F)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    w3 = rng.random(40)
    mu = (w3[:, None] * B).sum(0) / w3.sum()
    brute = (w3 * ((B - mu) ** 2).sum(1)).sum() / w3.sum()
    c3, _, _ = conflict_from_stats((w3[:, None] * B).sum(0)[None, :], np.array([w3.sum()]))
    assert abs(brute - c3[0]) < 1e-10, (brute, c3[0])
    # 4. a dead cell (no support) is reported as live=False, not as conflict 0
    _, _, live = conflict_from_stats(np.zeros((1, F)), np.array([0.0]))
    assert not live[0]
    print("conflict_from_stats self-test passed (identity matches brute force to 1e-10)")



def evaluate(a, mu, conflict, live, names, gt_pts, gt_lab, centers, radii, text):
    """Point-level mIoU/mAcc under the published protocol, with and without conflict adaptation.

    The lift itself is unchanged -- the same mean features, the same arg-max. What conflict buys is
    an ABSTENTION: a cell whose own observations disagree declares no class rather than guessing.
    In IoU terms that trades recall on the abstaining cell's true class against the false positives
    it would otherwise scatter across other classes, so the direction of the effect is genuinely an
    empirical question, not something to assert.
    """
    import numpy as np
    import torch
    from evaluate_point_cloud_miou import classify_primitives
    from point_cloud_query import assign_points_to_power_cells

    owner = np.asarray(assign_points_to_power_cells(gt_pts, centers, radii, valid=None, k=8))
    base = classify_primitives(torch.from_numpy(mu).float().cuda(), text).cpu().numpy() + 1
    base[~live] = 0
    nc = len(names) + 1

    def score(cell_lab):
        pred = np.where(owner >= 0, cell_lab[np.clip(owner, 0, len(cell_lab) - 1)], 0)
        ious, accs = [], []
        for k in range(1, nc):
            g = gt_lab == k
            if not g.any():
                continue
            p = pred == k
            inter = float((g & p).sum()); union = float((g | p).sum())
            ious.append(inter / union if union else 0.0)
            accs.append(inter / float(g.sum()))
        return float(np.mean(ious)), float(np.mean(accs))

    rows = []
    m0, a0 = score(base)
    rows.append(("baseline (no abstention)", 1.01, 1.0, m0, a0))
    for tau in [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]:
        lab = base.copy()
        drop = live & (conflict > tau)
        lab[drop] = 0
        kept = 1.0 - drop.sum() / max(live.sum(), 1)
        m, ac = score(lab)
        rows.append((f"abstain if conflict > {tau}", tau, kept, m, ac))

    print(f"\n{'variant':32s} {'kept':>7s} {'mIoU':>8s} {'mAcc':>8s} {'d mIoU':>8s}")
    for name, tau, kept, m, ac in rows:
        d = m - m0
        print(f"{name:32s} {kept:6.1%} {m:8.4f} {ac:8.4f} {d:+8.4f}")
    best = max(rows[1:], key=lambda r: r[3])
    print(f"\nbest abstention: {best[0]}  mIoU {best[3]:.4f} "
          f"({best[3]-m0:+.4f} vs baseline)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--stats", default="artifacts/adaptive/s0062_stats_l3bb.pt")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--prior-strength", type=float, default=2.0,
                    help="views of prior pulling an under-observed cell toward the mean")
    ap.add_argument("--raw-conflict", action="store_true",
                    help="skip the shrinkage correction (shows the single-view confound)")
    ap.add_argument("--eval", action="store_true", help="score point mIoU with abstention")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from oracle_labels import oracle_labels

    enable_determinism()
    _self_test()

    st = torch.load(a.stats, map_location="cpu", weights_only=False)
    get = (lambda k: getattr(st, k)) if hasattr(st, "support") else (lambda k: st[k])
    num = get("numerator").float().numpy()
    sup = get("support").float().numpy().reshape(-1)
    conflict, mu, live = conflict_from_stats(num, sup)
    n_eff = effective_views(sup, get("sum_view_weight_sq").float().numpy())
    conflict_raw = conflict.copy()
    if not a.raw_conflict:
        conflict = shrunk_conflict(conflict_raw, n_eff, a.prior_strength, live)
    print(f"\n{len(sup):,} primitives, {int(live.sum()):,} with support "
          f"({live.mean():.1%})")
    print(f"conflict: min {conflict[live].min():.4f}  median {np.median(conflict[live]):.4f}  "
          f"mean {conflict[live].mean():.4f}  max {conflict[live].max():.4f}")

    # ---- the gate: does conflict predict whether the cell is classified correctly? ------------
    pts, raw, all_names = load_scannet_pointcept_gt(
        os.path.join(POINTCEPT, SPLIT[a.scene], a.scene), "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    names = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
    cc, rr = load_points_radii(f"output/scannet_{a.scene}_{a.variant}")
    oracle, ost = oracle_labels(np.asarray(cc, float), np.asarray(rr, float), pts, gt_lab,
                                len(names) + 1)

    text = embed_class_names(names, "cuda")
    mu_t = torch.from_numpy(mu).float().cuda()
    pred = classify_primitives(mu_t, text).cpu().numpy() + 1

    scoreable = live & (oracle > 0)
    correct = (pred == oracle) & scoreable
    print(f"\nscoreable cells (have support AND own GT): {int(scoreable.sum()):,}")
    print(f"overall per-cell accuracy: {correct[scoreable].mean():.4f}")

    q = np.quantile(conflict[scoreable], np.linspace(0, 1, a.deciles + 1))
    q[-1] += 1e-9
    print(f"\n{'decile':>7} {'conflict range':>22} {'n':>8} {'accuracy':>9}")
    accs = []
    for d in range(a.deciles):
        m = scoreable & (conflict >= q[d]) & (conflict < q[d + 1])
        if not m.any():
            continue
        acc = correct[m].mean()
        accs.append(acc)
        print(f"{d + 1:>7} {q[d]:>10.4f}-{q[d + 1]:<10.4f} {int(m.sum()):>8,} {acc:>9.4f}")
    if len(accs) >= 2:
        spread = accs[0] - accs[-1]
        rho = np.corrcoef(conflict[scoreable], correct[scoreable].astype(float))[0, 1]
        print(f"\nlowest-conflict decile {accs[0]:.4f}  ->  highest-conflict decile {accs[-1]:.4f}"
              f"   (spread {spread:+.4f})")
        print(f"point-biserial corr(conflict, correct) = {rho:+.4f}")
        print("\nGATE: " + ("PASS - conflict is a usable reliability signal"
                            if spread > 0.10 else
                            "FAIL - conflict does not separate correct from incorrect cells; "
                            "an adaptive lambda built on it cannot help"))
        if a.eval:
            evaluate(a, mu, conflict, live, names, pts, gt_lab,
                     np.asarray(cc, dtype=np.float64), np.asarray(rr, dtype=np.float64), text)


if __name__ == "__main__":
    main()
