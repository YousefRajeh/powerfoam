"""Put the cross-primitive coupling BACK, the way the theorem says it went missing.

The exact lift solves  G Xhat = A^T B  with G = A^T A; the cheap one solves  D X' = A^T B,
i.e. it throws away every off-diagonal of G -- all of the coupling between primitives. The
theorem says that is harmless exactly when rays are disjoint, and on foam they nearly are, so
the lift is essentially UNBIASED. But unbiasedness is not the whole story: with G diagonal each
primitive is estimated from its own rays alone, so its VARIANCE is set by how many views it
actually got, and nothing in the estimator can borrow strength from a neighbour. 3DGS never
faces this -- its splats overlap, so its G is far from diagonal and neighbouring primitives are
averaged together whether anyone asked for it or not.

So we restore a coupling explicitly, using geometry where rays no longer provide one:

    Xs = argmin_X  sum_j w_j ||x_j - xp_j||^2  +  lam * sum_{(j,k) in E} ||x_j - x_k||^2
       = (W + lam * L)^{-1} W Xp

with E the Delaunay dual (the foam's own adjacency, free) and L its graph Laplacian. The trust
w_j is where the theory earns its keep: the diagnostic says accuracy is governed jointly by
n_eff (effective views) and R (whether those views agreed), spanning 0.348 -> 0.773 across
their joint terciles. A primitive seen often and consistently should barely move; one seen once
should be filled in by its neighbours. Setting w_j = 1 uniformly is the ablation that isolates
whether the trust weighting, and not merely the smoothing, is doing the work.

Nothing here looks at a class, a text query, or the other primitives' labels: it acts on the
FEATURE field before any query exists, so a single-query deployment is unaffected.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics,
                                       apply_gt_opacity_mask)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_holes import geometry, SCENES, GT_ROOT


def load_graph(scene, recon):
    tag = "tfroz" if recon == "truefrozen" else "nonfroz"
    g = torch.load(f"artifacts/ablation_cache/{scene}_pf_{tag}_delaunay.pt",
                   map_location="cpu", weights_only=False)
    return g["offsets"].numpy().astype(np.int64), g["adjacent"].numpy().astype(np.int64)


def smooth(X, indptr, indices, w, lam, iters=30, dev="cuda", live=None):
    """(W + lam L) Xs = W Xp by Jacobi-preconditioned CG on the sparse graph.

    L x = deg*x - sum_neighbours x, so the operator needs only a scatter-add: no matrix is
    ever formed. deg is the unweighted degree of the Delaunay dual (edge weights are 1).

    EDGES TOUCHING AN UNOBSERVED CELL ARE DROPPED. A primitive with D_jj = 0 carries a zero
    feature and, being unconstrained by any data term, would both drag its neighbours toward
    zero and act as a conduit: two surfaces on opposite sides of an empty region are Delaunay
    neighbours of the same interior cells, so leaving them in lets a wall average with whatever
    is behind it. Restricting the graph to observed cells is what makes the smoother a local
    operator on the SURFACE rather than through the volume.
    """
    X = X.to(dev).float()
    src_np = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    dst_np = indices
    if live is not None:
        keep = live[src_np] & live[dst_np]
        src_np, dst_np = src_np[keep], dst_np[keep]
    src = torch.from_numpy(np.ascontiguousarray(src_np)).to(dev)
    dst = torch.from_numpy(np.ascontiguousarray(dst_np)).to(dev)
    deg = torch.zeros(X.shape[0], device=dev).index_add_(
        0, src, torch.ones(src.shape[0], device=dev))
    w = torch.as_tensor(w, device=dev).float().unsqueeze(-1)

    def Aop(V):
        out = w * V + lam * deg.unsqueeze(-1) * V
        out.index_add_(0, src, -lam * V[dst])
        return out

    b = w * X
    diag = (w.squeeze(-1) + lam * deg).clamp_min(1e-12).unsqueeze(-1)
    Xs = X.clone()
    r = b - Aop(Xs)
    z = r / diag
    p = z.clone()
    rz = (r * z).sum()
    for _ in range(iters):
        Ap = Aop(p)
        alpha = rz / (p * Ap).sum().clamp_min(1e-30)
        Xs += alpha * p
        r -= alpha * Ap
        z = r / diag
        rz_new = (r * z).sum()
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
    return Xs


def scene_arms(scene, recon, class_set, lams, opacity_threshold, gt_opacity_mask, dev="cuda"):
    ap = f"artifacts/scannet/{scene}"
    st = torch.load(f"{ap}/stats_{recon}_ogl3.pt", map_location="cpu", weights_only=False)
    D = st["support"].numpy().astype(np.float64)
    svw = st["sum_view_weight_sq"].numpy().astype(np.float64)
    n_eff = np.where(svw > 0, D ** 2 / np.maximum(svw, 1e-12), 0.0)
    R = (st["numerator"].norm(dim=-1).numpy().astype(np.float64) /
         np.maximum(st["intra_sum"].numpy().astype(np.float64), 1e-12))
    sol = torch.load(f"{ap}/solved_geometric_median_{recon}_ogl3.pt", map_location="cpu",
                     weights_only=True)
    X = sol["primitive_features"].float()
    valid = sol["valid_mask"].numpy()

    centers, radii, density = geometry(scene, recon)
    cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    text = embed_class_names(kept, dev)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)

    if gt_opacity_mask:
        alpha = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
        gt_lab, _ = apply_gt_opacity_mask(gt_lab, assigned, alpha, opacity_threshold, scene)

    indptr, indices = load_graph(scene, recon)
    trust = {"neffR": np.clip(n_eff, 0, None) * np.clip(R, 0, 1),
             "neff": np.clip(n_eff, 0, None),
             "unit": np.ones_like(D)}

    def score(feats):
        pred_cls = (F.normalize(feats.to(dev), dim=-1) @ text.T).argmax(1).cpu().numpy()
        pl = np.zeros(len(gt_pts), dtype=np.int64)
        own = assigned >= 0
        pl[own] = pred_cls[assigned[own]] + 1
        ious, miou, _, macc = calculate_metrics(torch.from_numpy(gt_lab).long(),
                                                torch.from_numpy(pl).long(), len(kept) + 1)
        return float(miou), float(macc)

    out = {"plain": score(X)}
    for tname, w in trust.items():
        ww = w / max(np.median(w[w > 0]), 1e-12)   # scale-free, so lam is comparable across arms
        for lam in lams:
            out[f"{tname}_lam{lam}"] = score(
                smooth(X, indptr, indices, ww, lam, live=valid.astype(bool)))
    return out, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-sets", default="opengaussian19,opengaussian15,opengaussian10")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--lams", default="0.25,1,4")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--no-gt-opacity-mask", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/smooth_results.json")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]
    res = {}
    for cs in a.class_sets.split(","):
        per = {}
        for sc in a.scenes.split(","):
            try:
                arms, kept = scene_arms(sc, a.recon, cs, lams, a.opacity_threshold,
                                        not a.no_gt_opacity_mask)
            except Exception as e:
                print(f"[{cs}/{sc}] SKIP {type(e).__name__}: {e}")
                continue
            per[sc] = {k: {"miou": v[0], "macc": float(v[1])} for k, v in arms.items()}
            print(f"[{cs}/{sc}] " + "  ".join(f"{k} {v[0]*100:.2f}" for k, v in arms.items()))
        res[cs] = per
        if per:
            names = list(next(iter(per.values())).keys())
            print(f"\n--- {cs}: mean over {len(per)} scenes ---")
            base = np.mean([per[s]["plain"]["miou"] for s in per]) * 100
            for n in names:
                mi = np.mean([per[s][n]["miou"] for s in per]) * 100
                ma = np.mean([per[s][n]["macc"] for s in per]) * 100
                wins = sum(per[s][n]["miou"] > per[s]["plain"]["miou"] for s in per)
                print(f"  {n:<16} mIoU {mi:6.2f} ({mi-base:+5.2f})  mAcc {ma:6.2f}  "
                      f"wins {wins}/{len(per)}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
