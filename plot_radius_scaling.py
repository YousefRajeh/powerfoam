"""Does semantic accuracy scale with a cell's SURFACE (r^2) or its VOLUME (r^3)?

In the Laguerre partition a cell with power radius r wins a region whose projected
cross-section grows like r^2, so the rays it receives -- and therefore its accumulated
semantic evidence -- might be expected to grow like r^2 too. If so the power radius is a
geometric proxy for observability, a Feature-Foam-only statement (Gaussians have scales
but no partition weight deciding how much space, and hence how much evidence, a primitive
owns).

The claim is tested where an exponent is actually identifiable -- SUPPORT vs radius via a
log-log fit whose slope estimates the exponent directly (2 = surface, 3 = volume) --
rather than on accuracy vs radius, which is bounded in [0,1] and must saturate regardless
of the underlying law. Accuracy is then shown against both radius and support to complete
the chain radius -> evidence -> accuracy.

Multi-scene: bins are pooled across scenes for the headline fit, and a per-scene exponent
is reported so the spread is visible rather than hidden by pooling.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, "/home/rajehyl/feature-foam-lifting/src")
sys.path.insert(0, "/home/rajehyl/powerfoam")

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


def collect_scene(scene, variant, gt_root, classes, lam, device="cuda"):
    """Per-owner-cell arrays (radius, correct, n_points, support) for one scene."""
    split = SCENES[scene]
    gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
        f"{gt_root}/{split}/{scene}", "segment20")
    centers, radii = load_foam(f"output/scannet_{scene}_{variant}", device)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_{variant}_l3.pt",
                        map_location=device, weights_only=True)
    feats = solved["primitive_features"].to(device).float()
    vm = solved["valid_mask"].cpu().numpy()
    vm_t = torch.from_numpy(vm).to(device)
    vi = torch.where(vm_t)[0]
    unit = F.normalize(feats[vi], dim=-1)
    positions = torch.from_numpy(centers).to(device).float()
    leaf = two_level_position_aware(positions[vi], unit, seed=0, leaf_init="fps")

    stats = AccumulatedFeatureStats.load(
        f"artifacts/scannet/{scene}/train_stats_sam_{variant}_l3.pt")
    support = stats.support.cpu().numpy()
    del stats

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
    owned = assigned >= 0
    P = centers.shape[0]
    n2i = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[classes] if n2i[n] in present]
    tids = [i for i, _ in kept]
    tnames = [n for _, n in kept]
    K = len(tids)
    gt = remap_gt_labels(raw_labels, tids)

    text = embed_class_names(tnames, device)
    pooled = torch.zeros(K_FLAT, unit.shape[1], device=device)
    pooled.index_add_(0, leaf, unit)
    pooled = F.normalize(pooled, dim=-1)
    sim = pooled @ text.T
    cell_cls = (sim - lam * sim.mean(0, keepdim=True)).argmax(-1)
    pc = np.zeros(P, dtype=np.int64)
    pc[vi.cpu().numpy()] = cell_cls[leaf].cpu().numpy() + 1

    sel = gt[owned] > 0
    vote = np.zeros((P, K + 1), dtype=np.int64)
    np.add.at(vote, (assigned[owned][sel], gt[owned][sel]), 1)
    npts = vote[:, 1:].sum(1)
    maj = vote.argmax(1)
    owner = maj > 0
    correct = (pc == maj) & owner

    del feats, unit, solved, pooled
    torch.cuda.empty_cache()
    return (radii[owner].astype(np.float64),
            correct[owner].astype(np.float64),
            npts[owner].astype(np.float64),
            support[owner].astype(np.float64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--classes", default="opengaussian19")
    p.add_argument("--lam", type=float, default=0.4)
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--out", default="artifacts/scannet/radius_scaling.png")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    scene_list = args.scenes.split(",")
    R, C, W, S = [], [], [], []
    per_scene_k = {}
    for scene in scene_list:
        r, c, w, s = collect_scene(scene, args.variant, args.gt_root, args.classes, args.lam)
        R.append(r); C.append(c); W.append(w); S.append(s)
        ok = (r > 0) & (s > 0)
        A = np.vstack([np.log(r[ok]), np.ones(ok.sum())]).T
        cf, *_ = np.linalg.lstsq(A, np.log(s[ok]), rcond=None)
        per_scene_k[scene] = float(cf[0])
        print(f"  [{scene}] owner cells={len(r)} per-cell k={cf[0]:.2f} "
              f"acc={(c * w).sum() / w.sum():.3f} r-range=[{r.min():.3f},{r.max():.3f}]",
              flush=True)

    r = np.concatenate(R); c = np.concatenate(C)
    w = np.concatenate(W); sup = np.concatenate(S)

    qs = np.quantile(r, np.linspace(0, 1, args.bins + 1))
    br, bacc, bsup = [], [], []
    for i in range(args.bins):
        m = (r >= qs[i]) & (r <= qs[i + 1])
        if m.sum() < 30:
            continue
        br.append(r[m].mean())
        bacc.append((c[m] * w[m]).sum() / w[m].sum())
        bsup.append(sup[m].mean())
    br, bacc, bsup = np.array(br), np.array(bacc), np.array(bsup)

    ok = (br > 0) & (bsup > 0)
    A = np.vstack([np.log(br[ok]), np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(bsup[ok]), rcond=None)
    expo = float(coef[0])
    pred_log = A @ coef
    r2_pow = r2_of(np.log(bsup[ok]), pred_log)
    okc = (r > 0) & (sup > 0)
    Ac = np.vstack([np.log(r[okc]), np.ones(okc.sum())]).T
    coefc, *_ = np.linalg.lstsq(Ac, np.log(sup[okc]), rcond=None)

    ks = np.array(list(per_scene_k.values()))
    print(f"\n=== {len(scene_list)} scene(s), {args.classes}, {len(r)} owner cells pooled ===")
    print(f"per-scene exponent k: min={ks.min():.2f} median={np.median(ks):.2f} "
          f"max={ks.max():.2f} mean={ks.mean():.2f} sd={ks.std():.2f}")
    print(f"POOLED  support ~ radius^k :  k = {expo:.2f} (binned, R^2={r2_pow:.3f})   "
          f"k = {float(coefc[0]):.2f} (per-cell)")
    print("   k~2 => evidence scales with SURFACE/projected area; k~3 => with VOLUME")

    fits = {}
    for name, x in [("linear in r", br), ("quadratic in r (r^2)", br ** 2),
                    ("cubic in r (r^3)", br ** 3), ("log r", np.log(br))]:
        X = np.vstack([x, np.ones(len(x))]).T
        cf, *_ = np.linalg.lstsq(X, bacc, rcond=None)
        fits[name] = (r2_of(bacc, X @ cf), cf)
        print(f"accuracy ~ {name:<22}: R^2 = {fits[name][0]:.3f}")
    X2 = np.vstack([br ** 2, br, np.ones(len(br))]).T
    cf2, *_ = np.linalg.lstsq(X2, bacc, rcond=None)
    r2_quad = r2_of(bacc, X2 @ cf2)
    print(f"accuracy ~ a*r^2 + b*r + c : R^2 = {r2_quad:.3f}  (a={cf2[0]:+.2f}, b={cf2[1]:+.2f}) "
          f"-> a<0 means concave/saturating")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(br, bacc, "o-", color="#2b6cb0", label="point-weighted accuracy")
    xs = np.linspace(br.min(), br.max(), 200)
    ax[0].plot(xs, np.vstack([xs, np.ones_like(xs)]).T @ fits["linear in r"][1], "--",
               color="#a0aec0", label=f"linear (R²={fits['linear in r'][0]:.2f})")
    ax[0].plot(xs, np.vstack([xs ** 2, xs, np.ones_like(xs)]).T @ cf2, "-",
               color="#e53e3e", label=f"quadratic (R²={r2_quad:.2f})")
    ax[0].set_xlabel("power radius r (m)")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("accuracy vs power radius")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].loglog(br, bsup, "o", color="#2f855a")
    ax[1].loglog(br[ok], np.exp(pred_log), "-", color="#e53e3e",
                 label=f"fit slope k = {expo:.2f} (R²={r2_pow:.2f})")
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

    fig.suptitle(f"{len(scene_list)} ScanNet scene(s): accuracy vs cell surface (r²) or "
                 f"volume (r³)?  pooled k={expo:.2f}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"per_scene_k": per_scene_k, "pooled_k_binned": expo,
                       "pooled_k_percell": float(coefc[0]), "r2_powerlaw": r2_pow,
                       "acc_fit_r2": {k: v[0] for k, v in fits.items()},
                       "acc_quad_a": float(cf2[0]), "acc_quad_r2": r2_quad,
                       "bins": {"r": br.tolist(), "acc": bacc.tolist(), "sup": bsup.tolist()}},
                      f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
