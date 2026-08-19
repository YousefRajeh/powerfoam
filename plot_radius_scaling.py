"""Does semantic accuracy scale with a cell's SURFACE (r^2) or its VOLUME (r^3)?

Hypothesis worth a paper claim: in the Laguerre partition a cell with power radius r wins
a region whose projected cross-section grows like r^2, so the number of rays (and hence
view observations) it receives should grow like r^2 too. If so, the power radius is a
geometric proxy for observability, and semantic reliability inherits that scaling. This
would be a Feature-Foam-only statement -- Gaussians have scales but no partition weight
deciding how much space (and therefore how much evidence) a primitive owns.

The claim is tested where it is actually sharp -- on SUPPORT vs radius via a log-log fit,
whose slope estimates the exponent directly (2 = surface, 3 = volume) -- rather than on
accuracy vs radius, which is bounded in [0,1] and must saturate regardless of the
underlying law. Accuracy is then shown against both radius and support to complete the
chain radius -> evidence -> accuracy.
"""
import argparse
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import two_level_position_aware, K_FLAT, SCENES


def r2_of(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--classes", default="opengaussian19")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--out", default="artifacts/scannet/radius_scaling.png")
    args = p.parse_args()

    device = "cuda"
    scene = args.scene
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{args.gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{args.variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{args.variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit = F.normalize(feats[vi], dim=-1)
    positions = torch.from_numpy(centers).to(device).float()
    leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")

    stats = AccumulatedFeatureStats.load(
        f"artifacts/scannet/{scene}/train_stats_sam_{args.variant}_l3.pt")
    support = stats.support.cpu().numpy()          # summed rendering weight over all views
    nviews = stats.support_iv.cpu().numpy() if hasattr(stats, "support_iv") else None
    del stats

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    P = centers.shape[0]
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[args.classes] if n2i[n] in present]
    tids = [i for i, _ in kept]
    tnames = [n for _, n in kept]
    K = len(tids)
    gt = remap_gt_labels(raw_labels, tids)

    text = embed_class_names(tnames, device)
    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
    pooled.index_add_(0, leaf, unit)
    pooled = F.normalize(pooled, dim=-1)
    sim = pooled @ text.T
    cell_cls = (sim - args.lam * sim.mean(0, keepdim=True)).argmax(-1)
    pc = np.zeros(P, dtype=np.int64)
    pc[vi.cpu().numpy()] = cell_cls[leaf].cpu().numpy() + 1

    sel = gt[owned] > 0
    vote = np.zeros((P, K + 1), dtype=np.int64)
    np.add.at(vote, (assigned[owned][sel], gt[owned][sel]), 1)
    npts = vote[:, 1:].sum(1)
    maj = vote.argmax(1)
    owner = maj > 0
    correct = (pc == maj) & owner

    r = radii[owner].astype(np.float64)
    c = correct[owner].astype(np.float64)
    w = npts[owner].astype(np.float64)          # points per cell = metric weight
    sup = support[owner].astype(np.float64)

    # equal-count bins in radius
    qs = np.quantile(r, np.linspace(0, 1, args.bins + 1))
    br, bacc, bsup, bn = [], [], [], []
    for i in range(args.bins):
        m = (r >= qs[i]) & (r <= qs[i + 1])
        if m.sum() < 30:
            continue
        br.append(r[m].mean())
        bacc.append((c[m] * w[m]).sum() / w[m].sum())   # point-weighted accuracy
        bsup.append(sup[m].mean())
        bn.append(int(m.sum()))
    br, bacc, bsup = np.array(br), np.array(bacc), np.array(bsup)

    # --- the sharp test: support vs radius, log-log slope = scaling exponent ---
    ok = (br > 0) & (bsup > 0)
    A = np.vstack([np.log(br[ok]), np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(bsup[ok]), rcond=None)
    expo = float(coef[0])
    pred_log = A @ coef
    r2_pow = r2_of(np.log(bsup[ok]), pred_log)

    # per-cell (unbinned) exponent as a robustness check
    okc = (r > 0) & (sup > 0)
    Ac = np.vstack([np.log(r[okc]), np.ones(okc.sum())]).T
    coefc, *_ = np.linalg.lstsq(Ac, np.log(sup[okc]), rcond=None)

    print(f"=== {scene} ({args.classes}), {len(r)} owner cells ===")
    print(f"SUPPORT ~ radius^k :  k = {expo:.2f} (binned, R^2={r2_pow:.3f})   "
          f"k = {float(coefc[0]):.2f} (per-cell)")
    print(f"   interpretation: k~2 => evidence scales with SURFACE/projected area; "
          f"k~3 => with VOLUME")

    # --- accuracy vs radius: is it linear in r, r^2, or log r? ---
    fits = {}
    for name, x in [("linear in r", br), ("quadratic in r (r^2)", br ** 2),
                    ("cubic in r (r^3)", br ** 3), ("log r", np.log(br))]:
        X = np.vstack([x, np.ones(len(x))]).T
        cf, *_ = np.linalg.lstsq(X, bacc, rcond=None)
        fits[name] = (r2_of(bacc, X @ cf), cf)
        print(f"accuracy ~ {name:<22}: R^2 = {fits[name][0]:.3f}")
    # explicit quadratic (with curvature term) to see if curvature is real
    X2 = np.vstack([br ** 2, br, np.ones(len(br))]).T
    cf2, *_ = np.linalg.lstsq(X2, bacc, rcond=None)
    print(f"accuracy ~ a*r^2 + b*r + c : R^2 = {r2_of(bacc, X2 @ cf2):.3f}  "
          f"(a={cf2[0]:+.2f}, b={cf2[1]:+.2f}) -> a<0 means concave/saturating")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(br, bacc, "o-", color="#2b6cb0", label="point-weighted accuracy")
    xs = np.linspace(br.min(), br.max(), 200)
    ax[0].plot(xs, np.vstack([xs, np.ones_like(xs)]).T @ fits["linear in r"][1], "--",
               color="#a0aec0", label=f"linear (R²={fits['linear in r'][0]:.2f})")
    ax[0].plot(xs, np.vstack([xs ** 2, xs, np.ones_like(xs)]).T @ cf2, "-",
               color="#e53e3e", label=f"quadratic (R²={r2_of(bacc, X2 @ cf2):.2f})")
    ax[0].set_xlabel("power radius r (m)")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("accuracy vs power radius")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].loglog(br, bsup, "o", color="#2f855a")
    ax[1].loglog(br[ok], np.exp(pred_log), "-", color="#e53e3e",
                 label=f"slope k = {expo:.2f} (R²={r2_pow:.2f})")
    # reference power laws anchored at the first bin (axline cannot be used on log scales)
    xr = br[ok]
    for k_ref, style, col, lbl in [(2.0, ":", "#718096", "slope 2 (surface)"),
                                   (3.0, "-.", "#a0aec0", "slope 3 (volume)")]:
        ax[1].loglog(xr, bsup[ok][0] * (xr / xr[0]) ** k_ref, style, color=col, label=lbl)
    ax[1].set_xlabel("power radius r (m)")
    ax[1].set_ylabel("mean support (summed rendering weight)")
    ax[1].set_title("evidence vs radius (log-log)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    ax[2].semilogx(bsup, bacc, "o-", color="#805ad5")
    ax[2].set_xlabel("mean support (evidence)")
    ax[2].set_ylabel("accuracy")
    ax[2].set_title("accuracy vs evidence")
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle(f"{scene}: does semantic accuracy scale with cell surface (r²) or volume (r³)?")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
