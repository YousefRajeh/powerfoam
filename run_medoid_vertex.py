"""Spherical MEDOID on the Delaunay, then barycentric on the resulting posteriors.

THE IDEA (user's). Keep the copy discipline UPSTREAM and average only DOWNSTREAM.

  step 1 (on-manifold, discrete)  each vertex adopts the "common direction" of its Delaunay
                                  neighbourhood -- the spherical MEDOID, i.e. the actually
                                  observed direction maximising total cosine agreement:

      f_i* = argmax_{f in {f_i} u {f_j : j in N(i)}}  sum_{k in N(i) u {i}} cos(f, f_k)

  step 2 (convex, safe)           softmax to posteriors, then barycentric interpolation at
                                  each GT point over its Delaunay tetrahedron.

WHY THE SPLIT MATTERS. simplex-vs-sphere-extension.md Thm 1 gives an explicit witness where
averaging two CLIP directions yields a class NEITHER supported. NormLift avoids this by copying
a neighbour outright. Barycentric interpolation IS averaging -- it is only admissible because
we apply it to POSTERIORS, after the text-side argmax, where simplex closure holds. So this
pipeline never averages on the sphere and always averages on the simplex.

HOW THE MEDOID DIFFERS FROM MODE-VOTING. NormLift copies the single best-supported neighbour
under a reliability-weighted score with a margin guard. The medoid copies the most CENTRAL
observed direction -- the L1 analogue of the Frechet mean restricted to observed points, so it
is a robust consensus rather than a single pick, and it is insensitive to one outlying
neighbour in a way an argmax-of-support is not.

CONNECTION WORTH RECORDING. The class embeddings t_c induce a spherical Voronoi partition of
S^{F-1}, and `argmax_c <f, t_c>` IS the cell-membership query on it. The pipeline is therefore
a map between TWO Voronoi structures: the foam partitions R^3, the text embeddings partition
the CLIP sphere. Our measured pathology reads naturally in that language -- a median top1-top2
margin of 0.0136 means nearly every cell sits close to a face of the spherical diagram.

ARMS (all scored with the same assignment and scorer)
    percell                     bare argmax
    medoid                      step 1 only, hard assignment
    medoid+diff                 step 1 then posterior diffusion
    bary+medoid                 steps 1 and 2
    bary+medoid+diff            steps 1, diffusion, then 2
    modevote / modevote+diff    NormLift's operator, for the paired comparison

FALSIFIER: medoid must beat modevote by >= +0.3 mIoU at 19cls on the pilot, since it is the
same class of operation (an on-manifold copy) and has to justify replacing a published one.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_surface import GTSurfaceIndex, semantic_surface_metrics
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, calculate_metrics,
                                       embed_class_names, load_scannet_pointcept_gt,
                                       remap_gt_labels)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from run_normlift_refine_eval import mode_vote_refine

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def diffuse(p0, src, dst, deg, alpha=0.9, iters=60):
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p).index_add_(0, src, p[dst])
        p = (1 - alpha) * p0 + alpha * (agg / deg[:, None])
    return p


def spherical_medoid_linear(unit, adjacent, offsets, vt):
    """Same medoid, computed in O(P*D*C) instead of O(P*D^2*C) -- EXACTLY equivalent.

    The (D,D) Gram matrix the first implementation built is unnecessary, because the objective
    is LINEAR in f: every feature is unit, so

        score(f) = sum_k cos(f, f_k) = sum_k <f, f_k> = <f, sum_k f_k>.

    So one scatter-add gives the closed-neighbourhood sum S_i, and each candidate is scored by
    a single dot product against it. Not an approximation -- identical arithmetic, D times less
    of it (D = max degree ~ 20-80 here; measured 11.2 min -> seconds on scene0140).

    Ties are broken toward the lower primitive index so the result stays deterministic.

    NOTE, corrected: an earlier version of this docstring claimed exact equivalence with the
    Gram form. The ALGEBRA is equivalent, but the reference implementation also wrote a
    neighbour's feature into cells with no valid feature of their own (via `anyok`), which is a
    COVERAGE gain rather than a tie-break -- that is why its changed-fraction was 81.5% against
    75.1% here. This version reproduces that behaviour explicitly.
    """
    dev = unit.device
    P = unit.shape[0]
    deg = offsets[1:] - offsets[:-1]
    src = torch.repeat_interleave(torch.arange(P, device=dev), deg)
    uv = unit * vt[:, None].float()                      # invalid cells contribute nothing

    # CHUNKED over edges: uv[adjacent] is (E, C) and E*C*4 is 37 GiB on scene0140.
    S = uv.clone()
    E = adjacent.shape[0]
    CH = max(1, int(2e8 // unit.shape[1]))
    for b in range(0, E, CH):
        sl = slice(b, min(b + CH, E))
        S.index_add_(0, src[sl], uv[adjacent[sl]])

    self_score = (unit * S).sum(-1)
    # a cell with no feature of its own has self_score 0 and MUST be allowed to adopt a
    # neighbour -- the reference implementation did this via `anyok`, and it is a coverage
    # gain, not a tie-break. Matching it here explicitly rather than by accident.
    self_score = torch.where(vt, self_score, torch.full_like(self_score, -float("inf")))

    best = torch.full((P,), -float("inf"), device=dev)
    pick = torch.arange(P, device=dev)
    for b in range(0, E, CH):
        sl = slice(b, min(b + CH, E))
        sc = (unit[adjacent[sl]] * S[src[sl]]).sum(-1)
        sc = torch.where(vt[adjacent[sl]], sc, torch.full_like(sc, -float("inf")))
        best = best.scatter_reduce(0, src[sl], sc, reduce="amax", include_self=True)
    for b in range(0, E, CH):
        sl = slice(b, min(b + CH, E))
        sc = (unit[adjacent[sl]] * S[src[sl]]).sum(-1)
        sc = torch.where(vt[adjacent[sl]], sc, torch.full_like(sc, -float("inf")))
        win = sc >= (best[src[sl]] - 1e-9)
        idx = torch.full((P,), P, dtype=torch.long, device=dev).scatter_reduce(
            0, src[sl][win], adjacent[sl][win], reduce="amin", include_self=True)
        pick = torch.where((best > self_score) & (idx < P) & (pick == torch.arange(P, device=dev)),
                           idx, pick)
    out = unit[pick]
    return out


def spherical_medoid(unit, adjacent, offsets, valid, chunk=2048):
    """REFERENCE implementation (Gram form). Kept for equivalence checking; the linear form
    above is what is used, and is exactly equivalent -- see its docstring.

    Each cell adopts the most central OBSERVED direction in its closed neighbourhood.

    Score(f) = sum_k cos(f, f_k) over the neighbourhood; the winner is copied verbatim, so the
    output is always an observed CLIP direction -- never a synthetic mixture.

    Implemented against the padded neighbour table rather than a dense Gram matrix: degrees are
    ~20-40 here, so the per-cell cost is (deg+1)^2 dot products.
    """
    dev = unit.device
    P = unit.shape[0]
    deg = (offsets[1:] - offsets[:-1])
    D = int(deg.max().item()) + 1
    out = unit.clone()
    ar = torch.arange(D, device=dev)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        B = e - s
        rows = torch.arange(s, e, device=dev)
        cand = torch.full((B, D), -1, dtype=torch.long, device=dev)
        cand[:, 0] = rows
        dslice = deg[s:e]
        mask = ar[None, :] < (dslice[:, None] + 1)
        for k in range(D - 1):
            sel = k < dslice
            if not sel.any():
                break
            idx = torch.nonzero(sel).squeeze(1)
            cand[idx, k + 1] = adjacent[offsets[rows[idx]] + k]
        cand_c = cand.clamp_min(0)
        Fv = unit[cand_c]                                  # (B, D, C)
        ok = mask & (cand >= 0) & torch.from_numpy(valid).to(dev)[cand_c]
        Fv = Fv * ok[..., None].float()
        G = torch.bmm(Fv, Fv.transpose(1, 2))              # (B, D, D) pairwise cosines
        score = (G * ok[:, None, :].float()).sum(-1)
        score = score.masked_fill(~ok, -1e9)
        best = score.argmax(-1)
        pick = cand_c.gather(1, best[:, None]).squeeze(1)
        anyok = ok.any(-1)
        out[rows[anyok]] = unit[pick[anyok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0347_00")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    res = {}

    for scene in [s for s in a.scenes.split(",") if s in SPLIT]:
        t0 = time.time()
        mp = f"output/scannet_{scene}_nonfrozen/model.pt"
        fp = f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt"
        stp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
        bz = f"artifacts/ablation_cache/{scene}_bary.npz"
        apth = f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign_validmask.npy"
        if not all(os.path.exists(p) for p in (mp, fp, stp, bz, apth)):
            print(f"[skip] {scene}", flush=True)
            continue

        m = torch.load(mp, map_location="cpu", weights_only=False)
        P = m["points"].float().to(dev)
        adjacent = m["adjacency"].long().to(dev)
        offsets = m["adjacency_offsets"].long().to(dev)
        n_prim = P.shape[0]
        src = torch.repeat_interleave(torch.arange(n_prim, device=dev),
                                      offsets[1:] - offsets[:-1])
        deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()

        d = torch.load(fp, map_location=dev, weights_only=True)
        feats = d["primitive_features"].to(dev).float()
        valid = d["valid_mask"].cpu().numpy()
        vt = torch.from_numpy(valid).to(dev)
        unit = torch.zeros_like(feats)
        unit[vt] = F.normalize(feats[vt], dim=-1)
        assign = np.load(apth)
        owned = assign >= 0

        med = spherical_medoid_linear(unit, adjacent, offsets, vt)
        torch.cuda.empty_cache()          # release the (B,D,D) Gram buffers before scoring
        changed = float(((med - unit).abs().sum(-1) > 1e-6).float().mean())
        R = AccumulatedFeatureStats.load(stp).reliability()["reliability"].to(dev).float() * vt
        mv = mode_vote_refine(unit, R, P, adjacent, offsets)
        print(f"  [{scene}] medoid changed {100*changed:.1f}% of cells "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

        z = np.load(bz)
        bverts = torch.from_numpy(z["verts"]).to(dev)
        blam = torch.from_numpy(z["lam"]).float().to(dev)
        bins = torch.from_numpy(z["inside"]).to(dev)

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SPLIT[scene]}\{scene}", "segment20")
        pts = np.asarray(gt_pts, dtype=np.float64)
        n2i = {n: q for q, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            gt = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            index = GTSurfaceIndex(pts, gt, nc)

            def emit(tag, pred):
                sm = semantic_surface_metrics(index, pred)
                _, mi, _, _ = calculate_metrics(torch.from_numpy(gt).long(),
                                                torch.from_numpy(pred).long(), nc)
                res.setdefault((tag, cs), []).append(
                    (float(mi) * 100, sm.get("scd", np.nan) * 100,
                     sm.get("boundary_f1", np.nan)))

            def hard(tag, Pf):
                cls = Pf.argmax(-1).cpu().numpy() + 1
                live = (Pf.sum(-1) > 0).cpu().numpy()
                sc = owned.copy()
                sc[owned] = live[assign[owned]]
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[sc] = cls[assign[sc]]
                emit(tag, pr)

            def bary(tag, Pf):
                # chunked: Pf[bverts] is (N_gt, 4, C) and OOM'd on scene0140 when the medoid's
                # (B, D, D) Gram intermediates were still resident
                parts = []
                CH = 200_000
                for b0 in range(0, bverts.shape[0], CH):
                    sl = slice(b0, min(b0 + CH, bverts.shape[0]))
                    parts.append((Pf[bverts[sl]] * blam[sl][..., None]).sum(1))
                mix = torch.cat(parts)
                cls = mix.argmax(-1).cpu().numpy() + 1
                live = ((mix.sum(-1) > 1e-8) & bins).cpu().numpy()
                pr = np.zeros(len(gt), dtype=np.int64)
                pr[live] = cls[live]
                hv = (Pf.sum(-1) > 0).cpu().numpy()
                hc = Pf.argmax(-1).cpu().numpy() + 1
                fall = (~live) & owned & hv[assign.clip(0)]
                pr[fall] = hc[assign[fall]]
                emit(tag, pr)

            for tag, u in (("percell", unit), ("medoid", med), ("modevote", mv)):
                p0 = torch.softmax(1000.0 * (u @ text.T), dim=-1)
                p0[~vt] = 0.0
                pd = diffuse(p0, src, adjacent, deg)
                hard(tag, p0)
                hard(tag + "+diff", pd)
                bary("bary+" + tag, p0)
                bary("bary+" + tag + "+diff", pd)
        print(f"[{scene}] scored {(time.time()-t0)/60:.1f} min", flush=True)

    order = ["percell", "medoid", "modevote", "percell+diff", "medoid+diff", "modevote+diff",
             "bary+percell+diff", "bary+medoid+diff", "bary+modevote+diff"]
    n = len(res.get(("percell", CLASS_SETS[0]), []))
    print(f"\n=== {n} scenes ===")
    print(f"{'arm':<22}" + "".join(f"{c[11:]:>9}" for c in CLASS_SETS) + f"{'scd cm':>9}{'bF1':>7}")
    for tag in order:
        if (tag, CLASS_SETS[0]) not in res:
            continue
        row = "".join(f"{np.mean([r[0] for r in res[(tag,c)]]):9.2f}" for c in CLASS_SETS)
        scd = np.mean([r[1] for r in res[(tag, CLASS_SETS[0])]])
        bf1 = np.mean([r[2] for r in res[(tag, CLASS_SETS[0])]])
        print(f"{tag:<22}{row}{scd:9.2f}{bf1:7.3f}")
    print("\n=== medoid vs modevote (same class of operation) ===")
    for suf in ("", "+diff", ""):
        pass
    for suf in ("", "+diff"):
        A, B = "medoid" + suf, "modevote" + suf
        if (A, CLASS_SETS[0]) not in res:
            continue
        dl = " ".join(f"{np.mean([r[0] for r in res[(A,c)]]) - np.mean([r[0] for r in res[(B,c)]]):+.2f}"
                      for c in CLASS_SETS)
        print(f"  medoid{suf:<6} - modevote{suf:<6}: {dl}")


if __name__ == "__main__":
    main()
