"""Item 3: signed decision-plane margins -- the empirical semantic bridge FINDINGS4 asks for.

A60 killed the L2 route: a bound of the form "small ||x - t_y|| => correct argmax" needs
cos > 0.9611 while the CLIP modality gap caps the achievable cos at 0.3127. The failure is
STRUCTURAL, so FINDINGS4 recommends reporting signed decision-plane errors instead of a certificate.
This measures them.

For a primitive with lifted feature X_j, reference class y (its majority GT class) and unit text
embeddings T:

    x = X_j / ||X_j||,   s_c = <x, t_c>,   c* = argmax_{c != y} s_c
    m_j = s_y - s_{c*}                          SIGNED margin; m_j < 0 <=> misclassified
    b_j = m_j / ||t_y - t_{c*}||                EXACT min-norm perturbation of x that flips c*
    e_j = ||x - t_y||                           the actual L2 error to the reference field

`b_j` is exact, not a bound: the flip condition is <t_y - t_c*, x + delta> < 0, i.e.
<t_y - t_c*, delta> < -m_j, whose minimum-norm solution has norm m_j / ||t_y - t_c*||.

THE TIE TO A60. Evaluating `b` at the reference itself, x = t_y, gives s_y = 1, s_c = <t_y,t_c>,
so m = 1 - cos and ||t_y - t_c|| = sqrt(2 - 2 cos), hence b = sqrt((1 - cos)/2) -- EXACTLY the A60
threshold d_y = sqrt(Delta_min / 2). The margin route therefore strictly generalises the L2 route,
and the gap between them is the quantity of interest: primitives that are CORRECT despite e_j > b_j
are ones a worst-case L2 certificate must give up on while the actual readout succeeds. That
fraction measures, empirically, how much the L2 route throws away.

--selftest verifies:
  1. sign(m) agrees with argmax correctness on random configurations;
  2. b is the exact min-norm flip: the constructed perturbation at (1+eps)b flips and at (1-eps)b
     does not;
  3. at x = t_y, b reduces to A60's sqrt(Delta_min/2);
  4. the L2 certificate is SOUND -- e < d_y always implies correct -- so "correct but uncertified"
     is looseness, never a bug.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np


def margins(x, T, y):
    """x (n,d) unit rows, T (C,d) unit rows, y (n,). Returns m, b, cstar."""
    s = x @ T.T
    n = np.arange(len(y))
    sy = s[n, y]
    s2 = s.copy(); s2[n, y] = -np.inf
    cstar = s2.argmax(1); sc = s2[n, cstar]
    m = sy - sc
    diff = np.linalg.norm(T[y] - T[cstar], axis=1)
    return m, m / np.maximum(diff, 1e-30), cstar


def selftest():
    rng = np.random.default_rng(0)
    for _ in range(300):
        C, d, n = int(rng.integers(3, 9)), 16, 40
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        x = rng.normal(size=(n, d)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = rng.integers(0, C, n)
        m, b, cs = margins(x, T, y)
        # 1: sign of the margin is exactly argmax correctness
        assert np.all(((x @ T.T).argmax(1) == y) == (m > 0)), "margin sign != correctness"
        # 2: b is the EXACT min-norm flip distance
        # ...for m>0 it is the distance to LOSE correctness; for m<0 the distance to REGAIN it.
        for i in range(0, n, 7):
            u = T[y[i]] - T[cs[i]]; nu = float(u @ u)
            for scale in (1.0 + 1e-6, 1.0 - 1e-6):
                delta = -(m[i] * scale / nu) * u
                assert abs(np.linalg.norm(delta) - abs(b[i]) * scale) < 1e-9
                wrong_after = float(u @ (x[i] + delta)) < 0
                # crossing happens exactly at scale = 1, in whichever direction m points
                assert wrong_after == (m[i] * (1.0 - scale) < 0), (m[i], scale, wrong_after)
            # and nothing shorter than |b| can cross, in any direction
            for _ in range(5):
                g = rng.normal(size=d); g *= (abs(b[i]) * 0.999) / np.linalg.norm(g)
                assert (float(u @ (x[i] + g)) < 0) == (m[i] < 0), "sub-budget perturbation crossed"
        # 4: the L2 certificate is sound
        G = T @ T.T
        dmin = np.array([min(1.0 - G[c, j] for j in range(C) if j != c) for c in range(C)])
        dy = np.sqrt(dmin / 2.0)
        e = np.linalg.norm(x - T[y], axis=1)
        assert np.all(~(e < dy[y]) | (m > 0)), "L2 certificate unsound"
    # 3: at x = t_y, b collapses to A60's threshold
    for _ in range(100):
        C, d = int(rng.integers(3, 9)), 16
        T = rng.normal(size=(C, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        y = np.arange(C)
        m, b, _ = margins(T.copy(), T, y)
        G = T @ T.T
        dmin = np.array([min(1.0 - G[c, j] for j in range(C) if j != c) for c in range(C)])
        assert np.allclose(b, np.sqrt(dmin / 2.0), atol=1e-9), (b, np.sqrt(dmin / 2))
    print("  selftest OK: sign(m) == argmax correctness; b is the exact min-norm flip distance "
          "((1+eps)b flips, (1-eps)b does not); at x=t_y it reduces to A60's sqrt(Dmin/2); the L2 "
          "certificate is sound so 'correct but uncertified' is pure looseness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None); ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12); ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--cat-on-cpu", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/bridge_margins.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    import torch
    from determinism import enable_determinism
    enable_determinism()
    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names, remap_gt_labels
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}
    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True); continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                         cat_on_cpu=a.cat_on_cpu)
            nnz = val.numel()
            acc = torch.zeros(P, device=dev, dtype=torch.float64)
            for s0 in range(0, nnz, 100_000_000):
                e0 = min(s0 + 100_000_000, nnz)
                acc.index_add_(0, col[s0:e0], val[s0:e0].double())
            colsum = acc.float(); live = colsum > 0; d = Treg.shape[1]

            dd = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(dd, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            Cc = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=None, k=64)
            else:
                ck = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                weights_only=False)
                spp = ck["splats"] if "splats" in ck else ck
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(), valid=None)
            gtv = gl[m]; okm = own >= 0
            ow = torch.from_numpy(own[okm]).to(dev); gv = torch.from_numpy(gtv[okm]).to(dev)
            cnt = torch.zeros((P, Cc + 1), device=dev)
            cnt.index_put_((ow, gv), torch.ones_like(gv, dtype=torch.float32), accumulate=True)
            w = cnt.sum(1); maj = cnt.argmax(1)

            rhs = torch.zeros((P, d), device=dev)
            CH = max(1, int(4e8 // max(d, 1)))
            for s0 in range(0, nnz, CH):
                e0 = min(s0 + CH, nnz)
                rhs.index_add_(0, col[s0:e0], val[s0:e0, None] * Treg[gid[row[s0:e0]]])
            X = rhs / colsum.clamp_min(torch.finfo(val.dtype).eps)[:, None]

            sel = live & (w > 0) & (maj > 0)
            xs = torch.nn.functional.normalize(X[sel], dim=-1).cpu().numpy().astype(np.float64)
            Tn = T.cpu().numpy().astype(np.float64)
            ys = (maj[sel] - 1).cpu().numpy(); ws = w[sel].cpu().numpy().astype(np.float64)
            ws = ws / ws.sum()
            mg, bg, _ = margins(xs, Tn, ys)
            e = np.linalg.norm(xs - Tn[ys], axis=1)
            G = Tn @ Tn.T
            dmin = np.array([min(1.0 - G[c, j] for j in range(Cc) if j != c) for c in range(Cc)])
            dy = np.sqrt(dmin / 2.0)[ys]
            correct = mg > 0; certified = e < dy
            rec = {"arm": arm, "scene": sc, "C": Cc, "n_prim_scored": int(sel.sum().item()),
                   "w_correct": float(ws[correct].sum()),
                   "w_certified": float(ws[certified].sum()),
                   "w_correct_uncertified": float(ws[correct & ~certified].sum()),
                   "margin_median": float(np.median(mg)),
                   "margin_p10": float(np.percentile(mg, 10)),
                   "margin_p90": float(np.percentile(mg, 90)),
                   "budget_median": float(np.median(bg)), "e_median": float(np.median(e)),
                   "dy_median": float(np.median(dy)),
                   "e_over_budget_median": float(np.median(e / np.maximum(np.abs(bg), 1e-30))),
                   "wall_s": round(time.time() - t0, 1)}
            out.append(rec); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] C={Cc} | correct {100*rec['w_correct']:.1f}% certified "
                  f"{100*rec['w_certified']:.1f}% -> correct-but-uncertified "
                  f"{100*rec['w_correct_uncertified']:.1f}% | margin med {rec['margin_median']:+.4f} "
                  f"budget med {rec['budget_median']:+.4f} vs e med {rec['e_median']:.4f} "
                  f"(e/b {rec['e_over_budget_median']:.1f}x)  {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
