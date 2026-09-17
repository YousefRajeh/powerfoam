"""Is the ladder's bottom-rung residual the hard-vote quantisation, or bookkeeping?

The `sam_clip` rung quantises each ray to a hard class BEFORE lifting (`oracle_projected.py` builds
`AtS` over C class bins), so it is a PLURALITY VOTE over rays. The real pipeline lifts the 512-d
features and takes the argmax at the end, i.e. it AVERAGES SOFT SCORES. In class space both are the
same operator on different observations:

    hard : D^-1 A^T . onehot(argmax(B T^T))      assign each ray, then average
    soft : D^-1 A^T . (B T^T)                    average the scores, then assign

These are not equal -- a weighted mean is a sum of score MARGINS, not a majority vote. The ladder
residual (sam_clip - real) was -0.60 / +0.11, but it bundles this quantisation together with
scoring-set and solver differences. This isolates the quantisation by computing every arm inside ONE
codepath, on one operator, one point set, one ownership rule, so the only thing that varies is how
a ray's class evidence is formed.

Three arms:
  hard        onehot(argmax) per region, mass-weighted  -- reproduces the rung
  soft_raw    raw region features dotted with the text head -- what the real pipeline does
  soft_unit   region features L2-normalised first -- isolates region-norm weighting, which is a
              per-ray positive scalar that leaves each region's OWN argmax alone but changes how
              loudly it votes

--selftest checks, before any GPU time:
  1. the soft path equals D^-1 A^T (B T^T) computed directly;
  2. the hard path equals an explicit plurality vote;
  3. the two provably disagree on the known counterexample (two rays predict A, the mean predicts B);
  4. normalising B per region never changes that region's own argmax (so arm 3 differs from arm 2
     only through vote weighting, not through any region changing its mind).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch


def _hard(A, S, C):
    """Plurality vote: each ray casts its argmax class, weighted by its mass."""
    lab = S.argmax(1)
    out = np.zeros((A.shape[1], C))
    for i in range(A.shape[0]):
        out[:, lab[i]] += A[i]
    d = A.sum(0)[:, None]
    return np.divide(out, np.maximum(d, 1e-30))


def _soft(A, S):
    """Score averaging: D^-1 A^T S."""
    d = A.sum(0)[:, None]
    return np.divide(A.T @ S, np.maximum(d, 1e-30))


def selftest():
    rng = np.random.default_rng(0)
    # 1 & 2: both paths against explicit definitions
    for _ in range(50):
        R, P, C = 30, 7, 4
        A = np.abs(rng.normal(size=(R, P))) * (rng.random((R, P)) < 0.5)
        A[:, A.sum(0) == 0] = 1.0
        B = rng.normal(size=(R, 9)); T = rng.normal(size=(9, C))
        S = B @ T
        assert np.allclose(_soft(A, S), np.divide(A.T @ S, A.sum(0)[:, None]))
        oh = np.zeros_like(S); oh[np.arange(R), S.argmax(1)] = 1.0
        assert np.allclose(_hard(A, S, C), np.divide(A.T @ oh, A.sum(0)[:, None]))

    # 3: they disagree, on the known counterexample -- 2 of 3 rays predict A, the mean predicts B
    tA, tB = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    T = np.stack([tA, tB], 1)
    z = np.array([[0.1, np.sqrt(0.99)], [0.1, np.sqrt(0.99)], [-1.0, 0.0]])
    A = np.ones((3, 1))
    S = z @ T
    h = _hard(A, S, 2).argmax(1)[0]
    s = _soft(A, S).argmax(1)[0]
    assert h == 0 and s == 1, (h, s)

    # 4: per-row normalisation never changes that row's own argmax
    for _ in range(200):
        B = rng.normal(size=(20, 9)); T = rng.normal(size=(9, 5))
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        assert ((B @ T).argmax(1) == (Bn @ T).argmax(1)).all()
    print("selftest OK  (soft == D^-1 A^T S; hard == explicit plurality vote; they disagree on the "
          "known counterexample; per-region normalisation preserves each region's own argmax)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--arms", default="pf_truefrozen")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--feat-dir", default="openclip_features_sam_l3",
                    help="region-feature dir; openclip_features_sam_l3_nonorm has real norms")
    ap.add_argument("--raw-norms", action="store_true",
                    help="skip the loader's L2 normalisation (needs a *_nonorm feature dir)")
    ap.add_argument("--out", default="artifacts/scannet/vote_vs_score.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    from determinism import enable_determinism
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    if a.selftest:
        selftest()
        if a.scenes is None:
            return

    dev = "cuda"
    sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    sys.path.insert(0, r"D:\Downloads\powerfoam")
    import measure_xball2 as XB
    from diagnose_holes import SCENES, GT_ROOT, geometry
    from diagnose_scannet_miou import load_scannet_pointcept_gt
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           remap_gt_labels, calculate_metrics)
    from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

    scenes = (a.scenes or ",".join(SCENES)).split(",")
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["scene"]) for r in out}

    for arm in a.arms.split(","):
        for sc in scenes:
            if (arm, sc) in done:
                print(f"[{arm}/{sc}] cached", flush=True)
                continue
            t0 = time.time()
            row, col, val, gid, Treg, P, R, _ = XB.build(sc, arm, a.views, a.cap, dev,
                                                          feat_dirname=a.feat_dir,
                                                          normalize_features=not a.raw_norms)
            colsum = torch.zeros(P, device=dev).index_add_(0, col, val)
            live = colsum > 0

            d = [q for q in glob.glob(os.path.join(GT_ROOT, "*", sc)) if os.path.isdir(q)][0]
            pts, raw, names = load_scannet_pointcept_gt(d, "segment20")
            n2i = {n: i for i, n in enumerate(names)}
            pres = set(np.unique(raw).tolist())
            kept = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in pres]
            C = len(kept)
            T = embed_class_names(kept, dev); T = T / T.norm(dim=-1, keepdim=True)
            gl = remap_gt_labels(raw, [n2i[n] for n in kept]).astype(np.int64)
            vis = np.load(os.path.join("artifacts", "scannet", sc, "gt_visible.npy"))
            m = (gl > 0) & vis
            livenp = live.cpu().numpy(); recon = arm.replace("pf_", "")
            if recon in XB.FOAM:
                cen, rad, _ = geometry(sc, recon)
                own = assign_points_to_power_cells(pts[m], cen, rad, valid=livenp, k=64)
            else:
                ckp = torch.load(f"recon_remote/{arm}/{sc}/ckpt.pt", map_location="cpu",
                                 weights_only=False)
                spp = ckp["splats"] if "splats" in ckp else ckp
                own = assign_points_to_nearest_center(pts[m], spp["means"].float().numpy(),
                                                      valid=livenp)
            gtv = gl[m]; okm = own >= 0

            def score(W):
                lab = torch.zeros(P, dtype=torch.long, device=dev)
                lab[live] = W[live].argmax(1) + 1
                pr = np.zeros(gtv.shape[0], np.int64); pr[okm] = lab.cpu().numpy()[own[okm]]
                _, mi, ac, _ = calculate_metrics(torch.from_numpy(gtv), torch.from_numpy(pr), C + 1)
                return float(mi) * 100, float(ac) * 100

            Tn = Treg / Treg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            S_raw = Treg @ T.T                      # (M, C) region -> class scores, raw features
            S_unit = Tn @ T.T                       # unit-normalised region features
            k_reg = S_unit.argmax(1)                # region hard label (same for raw or unit)
            assert bool((S_raw.argmax(1) == k_reg).all()), "normalisation changed a region's argmax"
            _nrm = Treg.norm(dim=-1)
            print(f"   region-feature norms: {_nrm.min():.4f}..{_nrm.max():.4f} "
                  f"(spread {float(_nrm.max()/_nrm.min().clamp_min(1e-30)):.3f}x)", flush=True)

            res = {"arm": arm, "scene": sc, "P": int(P), "C": C, "M": int(Treg.shape[0]),
                   "feat_dir": a.feat_dir,
                   "norm_min": float(Treg.norm(dim=-1).min()),
                   "norm_max": float(Treg.norm(dim=-1).max())}
            for nm, src in (("soft_raw", S_raw), ("soft_unit", S_unit)):
                W = torch.zeros(P, C, device=dev)
                for s in range(0, val.numel(), 8_000_000):
                    e = min(s + 8_000_000, val.numel())
                    W.index_add_(0, col[s:e], val[s:e, None] * src[gid[row[s:e]]])
                W[live] /= colsum[live].unsqueeze(-1)
                res[nm + "_miou"], res[nm + "_acc"] = score(W)
                del W
            W = torch.zeros(P, C, device=dev)
            for s in range(0, val.numel(), 8_000_000):
                e = min(s + 8_000_000, val.numel())
                W.index_put_((col[s:e], k_reg[gid[row[s:e]]]), val[s:e], accumulate=True)
            W[live] /= colsum[live].unsqueeze(-1)
            res["hard_miou"], res["hard_acc"] = score(W)
            del W

            res["wall_s"] = round(time.time() - t0, 1)
            out.append(res); json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{arm}/{sc}] hard {res['hard_miou']:.2f} | soft_raw {res['soft_raw_miou']:.2f} "
                  f"({res['soft_raw_miou']-res['hard_miou']:+.2f}) | soft_unit "
                  f"{res['soft_unit_miou']:.2f} ({res['soft_unit_miou']-res['hard_miou']:+.2f})  "
                  f"{res['wall_s']}s", flush=True)
            del row, col, val, gid, Treg
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
