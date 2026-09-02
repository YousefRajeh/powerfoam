"""Is mass conservation a virtue or a trap? And fix the TPFA flow's missing data term.

ARM E CAME BACK -0.59/-0.35/+1.04. Two candidate causes, tested here separately.

BUG 1 -- HEAT DEATH. Arm E was PURE conservative diffusion, p <- p + dt*div/V, with nothing
anchoring it to p0. On a connected graph the steady state of that flow is p_i = const for all
i, namely (sum_j V_j p0_j)/(sum_j V_j): every cell ends up with the SAME posterior and all
spatial information is destroyed. Plain diffusion avoids this through its (1-alpha) p0 anchor.
Fixed below by arm E', the gradient flow of a convex energy with a fidelity term:

    E(p) = 1/2 sum_ij T_ij ||p_i - p_j||^2  +  mu/2 sum_i V_i ||p_i - p0_i||^2,  p_i in simplex

Both terms remain foam-exact: T_ij = A_ij/d_ij needs facet areas and the power-diagram
orthogonality that makes TPFA consistent; the V_i-weighted fidelity says a large cell is
anchored more strongly because it speaks for more space, which needs exact volumes.

BUG 2 -- CONSERVATION MAY BE A TRAP. The flow conserves sum_i V_i p_ic, i.e. the INITIAL
volume-weighted class mass. That is only desirable if the initial mass is approximately
correct. Our measured pathology says it is not: `floor` has the highest mean cosine to
everything, `picture` wins 22.47% of cells, and chair/sofa/table win 0.00%. If the initial
mass distribution is far from the GT distribution, conservation preserves the error instead of
correcting it -- and the entire finite-volume framing is wrong for this problem, not merely
mis-tuned.

The diagnostic below compares, per class:
    M_c = sum_i V_i p0_ic / sum_i V_i     (initial volume-weighted mass)
    G_c = N_c / N                          (GT mass)
and reports total variation distance. It also reports what plain diffusion does to M_c -- if
diffusion MOVES the mass toward G, then not conserving is exactly why it wins, which would
settle the question.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)

SP = (r"C:\Users\rajehyl\AppData\Local\Temp\claude"
      r"\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad")
SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train"}


def project_simplex(v):
    n, C = v.shape
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, C + 1, device=v.device, dtype=v.dtype)
    rho = (u - css / ind > 0).float().cumsum(-1).argmax(-1)
    theta = css.gather(1, rho[:, None]) / (rho[:, None].to(v.dtype) + 1)
    return (v - theta).clamp_min(0)


def damped_tpfa(p0, i, j, T, V, mu=1.0, iters=300, cfl=0.4):
    """Arm E': gradient flow of the convex energy above. Fidelity kills the heat death."""
    n = p0.shape[0]
    outflux = torch.zeros(n, device=p0.device).index_add_(0, i, T).index_add_(0, j, T)
    dt = cfl / ((outflux / V.clamp_min(1e-12)).max() + mu)
    p = p0.clone()
    for _ in range(iters):
        flux = T[:, None] * (p[j] - p[i])
        div = torch.zeros_like(p).index_add_(0, i, flux).index_add_(0, j, -flux)
        p = p + dt * (div / V[:, None].clamp_min(1e-12) - mu * (p - p0))
        p = project_simplex(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00")
    ap.add_argument("--mus", default="0.3,1.0,3.0")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        m = torch.load(f"output/scannet_{scene}_nonfrozen/model.pt",
                       map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = P.shape[0]
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
        keep = src < adjacent
        ei, ej = src[keep], adjacent[keep]

        A = np.load(os.path.join(SP, f"area_{scene}_pf_nonfroz.npz"))
        bnd = ~A["unbounded"].astype(bool)
        key = {(int(x), int(y)): float(w) for x, y, w in
               zip(A["i"][bnd], A["j"][bnd], A["area"][bnd])}
        eiv, ejv = ei.cpu().numpy(), ej.cpu().numpy()
        area = torch.from_numpy(np.array([key.get((int(x), int(y)), 0.0)
                                          for x, y in zip(eiv, ejv)])).float().to(dev)
        dist = (P[ei] - P[ej]).norm(dim=-1)
        T = area / dist.clamp_min(1e-12)
        V = torch.from_numpy(np.load(os.path.join(SP, f"cellgeom_{scene}_pf_nonfroz.npz"))
                             ["V"].astype(np.float32)).to(dev)

        d = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                       map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy")
        owned = assign >= 0

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())
        cs = "opengaussian19"
        names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        gt = remap_gt_labels(raw, [n2i[n] for n in names])
        nc = len(names) + 1
        text = embed_class_names(names, dev)
        p0 = torch.softmax(1000.0 * (unit @ text.T), dim=-1)
        p0[~vt] = 0.0

        # ---------- BUG 2 diagnostic: is the initial mass anywhere near GT?
        Vn = V.clone()
        Vn[~vt] = 0.0
        M0 = (Vn[:, None] * p0).sum(0)
        M0 = (M0 / M0.sum()).cpu().numpy()
        G = np.array([(gt == c + 1).sum() for c in range(len(names))], dtype=np.float64)
        G = G / G.sum()
        pdif = diffuse = p0.clone()
        for _ in range(60):
            agg = torch.zeros_like(pdif).index_add_(0, src, pdif[adjacent])
            pdif = 0.1 * p0 + 0.9 * (agg / deg[:, None])
        Md = (Vn[:, None] * pdif).sum(0)
        Md = (Md / Md.sum()).cpu().numpy()
        tv0 = 0.5 * np.abs(M0 - G).sum()
        tvd = 0.5 * np.abs(Md - G).sum()
        print(f"\n[{scene}] volume-weighted class mass vs GT ({cs[11:]} classes)")
        print(f"  {'class':<14}{'GT':>8}{'initial':>9}{'diffused':>10}")
        order = np.argsort(-G)
        for k in order[:8]:
            print(f"  {names[k]:<14}{100*G[k]:7.2f}%{100*M0[k]:8.2f}%{100*Md[k]:9.2f}%")
        print(f"  TV distance to GT:  initial {tv0:.4f}   after diffusion {tvd:.4f}"
              f"   ({'diffusion MOVES mass toward GT' if tvd < tv0 else 'diffusion moves it AWAY'})")
        print(f"  => conservation would LOCK IN the initial distribution"
              f" (TV {tv0:.4f} from GT)")

        # ---------- BUG 1 fix: damped TPFA, swept over the fidelity weight
        def score(pr_field):
            cls = pr_field.argmax(-1).cpu().numpy() + 1
            live = (pr_field.sum(-1) > 0).cpu().numpy()
            sc = owned.copy()
            sc[owned] = live[assign[owned]]
            pred = np.zeros(len(gt), dtype=np.int64)
            pred[sc] = cls[assign[sc]]
            _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                            torch.from_numpy(pred).long(), nc)
            return float(mi) * 100

        print(f"\n  {'arm':<28}{'mIoU19':>8}")
        print(f"  {'plain diffusion':<28}{score(pdif):8.2f}")
        for mu in [float(x) for x in a.mus.split(",")]:
            pe = damped_tpfa(p0, ei, ej, T, V, mu=mu)
            print(f"  {'E-prime damped TPFA mu=' + str(mu):<28}{score(pe):8.2f}")


if __name__ == "__main__":
    main()
