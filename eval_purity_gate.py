"""IDEA 1: is mask purity `p*_j` a usable signal? Judged on AUC **and** IoU **and** COHERENCE.

WHY THREE METRICS, NOT ONE.
  * AUC alone has misled this project repeatedly: the single-query gate showed +5.5 mAP that
    collapsed to +0.7 deployable IoU, a 6x overstatement. AUC rewards reordering anywhere in the
    ranking; a deployed threshold only sees the top. So AUC is necessary, not sufficient.
  * IoU alone is blind to STRUCTURE. Two masks with identical IoU look completely different if one
    is a compact object and the other is speckle scattered across the room. IoU counts points, not
    connectivity.

So this also reports SPATIAL COHERENCE on the exact facet graph:
    lcc_frac  -- share of a predicted mask's primitives lying in its LARGEST connected component
    n_comp    -- how many components the predicted mask fragments into
Both are computed on the power diagram's Delaunay dual, which for this representation IS the true
cell adjacency (verified jaccard 1.0000 against radfoam's own CUDA Delaunay). A 3DGS arm cannot
compute this: its alpha graph is degenerate, mean degree 0.05 even at gsplat's 3-sigma bound.

WHAT IS BEING TESTED. The validated angular bound says the lifting error is controlled by
    tan theta_j <= (1 - p*_j) Omega / (1 - (1 - p*_j) Omega)
with `p*_j` the dominant SAM mask's share of primitive j's ray weight. Measured p* is poor
(truefrozen scene0062: mean 0.394, median 0.293, 6.17 masks per primitive), so the bound says there
is a lot of contamination. This asks whether that quantity is USABLE:
  (a) does p*_j predict per-primitive correctness at all?
  (b) does gating on p* beat gating on the agreement signal `||x_weighted||` that already works?
  (c) does either produce more COHERENT masks, or just better point counts?

If p* has AUC ~ 0.5 the bound's controlling quantity does not bind in practice -- itself worth
knowing, since it would close the operator-side line.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from graphcut import binary_graphcut
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import (assign_points_to_power_cells, load_foam,
                                   load_scannet_pointcept_gt)

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
GT_ROOT = r"D:\Downloads\scannet_pointcept"
QUANTS = (0.90, 0.95)
THRESHES = (0.20, 0.21, 0.22)


# ----------------------------------------------------------------- metrics
def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def iou_at_score(scores, y, t):
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    sel = scores >= t
    inter = int((sel & (y > 0)).sum())
    union = int(sel.sum()) + n_pos - inter
    return float(inter / union) if union else 0.0


def iou_at_quantile(scores, y, q):
    fin = scores[np.isfinite(scores)]
    if fin.size == 0 or int(y.sum()) == 0:
        return float("nan")
    return iou_at_score(scores, y, float(np.quantile(fin, q)))


def components(sel_idx, indptr, indices):
    """(n_components, largest_component_size) for the induced subgraph on `sel_idx`.

    Iterative BFS on the CSR facet graph -- recursion would blow the stack at 10^5 primitives.
    `sel_idx` is a sorted array of selected primitive ids; membership is tested with a dense
    boolean over P, which is far cheaper than searchsorted per edge.
    """
    if sel_idx.size == 0:
        return 0, 0
    P = indptr.size - 1
    inset = np.zeros(P, dtype=bool)
    inset[sel_idx] = True
    seen = np.zeros(P, dtype=bool)
    ncomp, largest = 0, 0
    stack = np.empty(sel_idx.size, dtype=np.int64)
    for s in sel_idx:
        if seen[s]:
            continue
        ncomp += 1
        top = 0
        stack[top] = s
        top += 1
        seen[s] = True
        size = 0
        while top:
            top -= 1
            v = stack[top]
            size += 1
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                if inset[u] and not seen[u]:
                    seen[u] = True
                    stack[top] = u
                    top += 1
        largest = max(largest, size)
    return ncomp, largest


def neighbour_frac(keep, indptr, indices):
    """For every primitive, the fraction of its facet neighbours that are currently admitted.

    Vectorised over the CSR arrays: `keep[indices]` is the admitted-flag of every directed edge,
    and a segment sum over `indptr` gives each node's admitted-neighbour count. Isolated nodes
    (degree 0) return 0.0 and are therefore never repaired, which is deliberate -- a primitive with
    no facet neighbours carries no connectivity evidence either way.
    """
    # CUMSUM, not np.add.reduceat: reduceat raises IndexError when a ZERO-DEGREE node's offset
    # equals len(indices), which happens whenever the last primitive has no facet neighbours --
    # and ~6 % of primitives here are unobserved. The prefix-sum difference is total and needs no
    # special-casing.
    deg = np.diff(indptr)
    cum = np.concatenate(([0], np.cumsum(keep[indices].astype(np.int64))))
    adm = cum[indptr[1:]] - cum[indptr[:-1]]
    return np.divide(adm, np.maximum(deg, 1), dtype=np.float64)


def repair(keep, indptr, indices, tau=0.8, iters=1):
    """CONNECTIVITY REPAIR (graph morphological closing).

    Re-admit an EXCLUDED primitive whose facet neighbours are (nearly) all admitted. The gate is a
    precision filter, and A16.2 measured that it fragments masks by 48 % -- it preferentially drops
    the boundary/interface cells that hold a region together. Repair restores those without
    re-admitting the isolated low-confidence cells the gate exists to remove.

    Only computable on a representation with an EXACT adjacency graph. 3DGS cannot: its alpha graph
    has mean degree 0.05 at gsplat's own 3-sigma bound.

    `iters > 1` lets a repair cascade one hop further per pass; each pass uses the PREVIOUS pass's
    admitted set, so a chain of excluded cells cannot bootstrap itself in a single sweep.
    """
    out = keep.copy()
    has_nb = np.diff(indptr) > 0          # a node with no facet neighbours carries no evidence
    for _ in range(iters):
        frac = neighbour_frac(out, indptr, indices)
        add = (~out) & has_nb & (frac >= tau)
        if not add.any():
            break
        out = out | add
    return out


def prune(keep, indptr, indices, tau=0.5):
    """CONNECTIVITY PRUNE (graph morphological opening): drop an ADMITTED primitive whose facet
    neighbourhood is mostly excluded -- i.e. an isolated survivor, which is exactly the speckle a
    viewer sees. Applied after repair it removes stragglers the gate left behind."""
    frac = neighbour_frac(keep, indptr, indices)
    return keep & (frac >= tau)


# ----------------------------------------------------------------- main
def main():
    enable_determinism()
    device = "cuda"
    only = [s for s in os.environ.get("ONLY_SCENES", "").split(",") if s]
    scenes = [s for s in SCENES if (s in only if only else True)]
    cs_name = os.environ.get("CLASS_SET", "opengaussian19")
    feat_file = os.environ.get("FEAT_FILE", "solved_geometric_median_truefrozen_ogl3")
    agree_file = os.environ.get("AGREE_FILE", "solved_weighted_truefrozen_ogl3")
    recon = os.environ.get("RECON", "truefrozen")
    KEEP = float(os.environ.get("KEEP", "0.75"))
    print(f"[config] recon={recon} feat={feat_file} agree={agree_file} keep={KEEP} "
          f"scenes={len(scenes)}", flush=True)

    ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a] or [
        "plain", "gate_agree", "gate_pstar", "gate_both", "rand"]
    res = {a: {} for a in ARMS}
    aucs = []

    for scene in scenes:
        ps_path = f"artifacts/scannet/{scene}/pstar_{recon}.npz"
        if not os.path.exists(ps_path):
            print(f"[skip] {scene}: no {os.path.basename(ps_path)} "
                  f"-- run measure_mask_purity.py --variant {recon}", flush=True)
            continue
        cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
        centers, radii = load_foam(f"output/scannet_{scene}_{recon}", device)
        d = torch.load(f"artifacts/scannet/{scene}/{feat_file}.pt", map_location=device,
                       weights_only=True)
        feats = d["primitive_features"].to(device).float()
        valid = d["valid_mask"].cpu().numpy()
        ag = torch.load(f"artifacts/scannet/{scene}/{agree_file}.pt", map_location=device,
                        weights_only=True)["primitive_features"].to(device).float().norm(dim=-1)
        z = np.load(ps_path)
        pstar = torch.from_numpy(z["pstar"]).to(device).float()
        assert pstar.numel() == feats.shape[0], (pstar.numel(), feats.shape[0])

        g = torch.load(f"artifacts/ablation_cache/{scene}_pf_{'tfroz' if recon=='truefrozen' else 'nonfroz'}_delaunay.pt",
                       map_location="cpu", weights_only=False)
        indptr = g["offsets"].numpy().astype(np.int64)
        indices = g["adjacent"].numpy().astype(np.int64)

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid, k=64)
        owned = assigned >= 0
        vidx = np.where(valid)[0]
        vt = torch.from_numpy(vidx).to(device)
        unit = F.normalize(feats[vt], dim=-1)
        conf_a = (ag[vt] / ag[vt].median().clamp_min(1e-12)).cpu().numpy()
        conf_p = pstar[vt].cpu().numpy()
        row_of = np.full(centers.shape[0], -1, dtype=np.int64)
        row_of[vidx] = np.arange(vidx.size)
        prow = np.full(raw_labels.shape[0], -1, dtype=np.int64)
        prow[owned] = row_of[assigned[owned]]
        scored = prow >= 0

        name_to_id = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        cls_list = [(name_to_id[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs_name]
                    if name_to_id[n] in present]
        text = embed_class_names([n for _, n in cls_list], device)
        sim0 = (unit @ text.T).cpu().numpy()
        print(f"\n===== {scene} (valid={vidx.size:,}, classes={len(cls_list)}) =====", flush=True)

        # ---- (a) do the two signals predict per-primitive correctness? ----
        votes = np.zeros((centers.shape[0], len(cls_list)), dtype=np.int64)
        lab, prim = raw_labels[owned], assigned[owned]
        for k, (tid, _) in enumerate(cls_list):
            m = lab == tid
            if m.any():
                np.add.at(votes[:, k], prim[m], 1)
        has_gt = votes.sum(1) > 0
        gt_slot = votes.argmax(1)
        selp = has_gt[vidx]
        correct = (sim0.argmax(1)[selp] == gt_slot[vidx][selp]).astype(np.int64)
        a_ag = auc(conf_a[selp], correct)
        a_ps = auc(conf_p[selp], correct)
        aucs.append((scene, a_ag, a_ps))
        print(f"  AUC vs correctness:  agreement {a_ag:.4f}   p* {a_ps:.4f}", flush=True)

        # ---- masks for each arm ----
        def mask_for(kind):
            if kind == "plain":
                return np.ones(vidx.size, dtype=bool)
            if kind == "gate_agree":
                return conf_a >= np.quantile(conf_a, 1 - KEEP)
            if kind == "gate_pstar":
                return conf_p >= np.quantile(conf_p, 1 - KEEP)
            if kind == "gate_both":
                return (conf_a >= np.quantile(conf_a, 1 - np.sqrt(KEEP))) & \
                       (conf_p >= np.quantile(conf_p, 1 - np.sqrt(KEEP)))
            if kind.startswith("repair") or kind.startswith("rp"):
                # gate on agreement, then re-admit excluded cells surrounded by admitted ones
                base = conf_a >= np.quantile(conf_a, 1 - KEEP)
                tau = float(kind.split("@")[1]) if "@" in kind else 0.8
                it = int(kind.split("x")[1].split("@")[0]) if "x" in kind else 1
                full = np.zeros(indptr.size - 1, dtype=bool)
                full[vidx] = base
                full = repair(full, indptr, indices, tau=tau, iters=it)
                if kind.startswith("rp"):                      # repair THEN prune
                    full = prune(full, indptr, indices, tau=0.5)
                return full[vidx]
            if kind.startswith("prune"):
                base = conf_a >= np.quantile(conf_a, 1 - KEEP)
                tau = float(kind.split("@")[1]) if "@" in kind else 0.5
                full = np.zeros(indptr.size - 1, dtype=bool)
                full[vidx] = base
                return prune(full, indptr, indices, tau=tau)[vidx]
            rng = np.random.default_rng(abs(hash(scene)) % (2 ** 31))
            return rng.random(vidx.size) < KEEP

        for arm in ARMS:
            # `cut@lam[:gate]` runs the exact binary graph cut per class at threshold 0.21,
            # optionally restricted to the confidence-gated subset. It is per-CLASS, so unlike the
            # mask arms it cannot be expressed as a single keep-vector.
            is_cut = arm.startswith("cut")
            if is_cut:
                lam = float(arm.split("@")[1].split(":")[0])
                gated = arm.endswith(":gate")
                base = (conf_a >= np.quantile(conf_a, 1 - KEEP)) if gated else None
                sub_full = None
                if base is not None:
                    sub_full = np.zeros(indptr.size - 1, dtype=bool)
                    sub_full[vidx] = base
                keep = base if base is not None else np.ones(vidx.size, dtype=bool)
                sim = np.where(keep[:, None], sim0, -np.inf)
            else:
                keep = mask_for(arm)
                sim = np.where(keep[:, None], sim0, -np.inf)
            per_cls = {}
            for k, (tid, nm) in enumerate(cls_list):
                y = (raw_labels[scored] == tid).astype(np.int64)
                sc = sim[prow[scored], k]
                fin = np.isfinite(sc)
                if fin.sum() < 10 or y[fin].sum() == 0:
                    continue
                if is_cut:
                    # a cut yields a LABELLING, not a score, so every column is that labelling's IoU
                    sfull_i = np.full(indptr.size - 1, -1.0)
                    sfull_i[vidx] = sim0[:, k]
                    lab_i = binary_graphcut(sfull_i, 0.21, indptr, indices, lam=lam,
                                            subset=sub_full)
                    selpt = lab_i[assigned[scored]][fin]
                    inter = int((selpt & (y[fin] > 0)).sum())
                    union = int(selpt.sum()) + int(y[fin].sum()) - inter
                    v_ = float(inter / union) if union else 0.0
                    rec = {f"iou_q{q}": v_ for q in QUANTS}
                    rec.update({f"iou_t{t}": v_ for t in THRESHES})
                else:
                    rec = {f"iou_q{q}": iou_at_quantile(sc[fin], y[fin], q) for q in QUANTS}
                    rec.update({f"iou_t{t}": iou_at_score(sc[fin], y[fin], t) for t in THRESHES})
                # ---- (c) spatial coherence of the PREDICTED mask at a fixed cut ----
                if is_cut:
                    sfull = np.full(indptr.size - 1, -1.0)
                    sfull[vidx] = sim0[:, k]
                    lab = binary_graphcut(sfull, 0.21, indptr, indices, lam=lam,
                                          subset=sub_full)
                    pred_prim = np.where(lab[vidx])[0]
                else:
                    pred_prim = np.where(keep & (sim0[:, k] >= 0.21))[0]
                nc, lcc = components(vidx[np.sort(pred_prim)], indptr, indices)
                rec["n_comp"] = float(nc)
                rec["lcc_frac"] = float(lcc / max(pred_prim.size, 1))
                per_cls[nm] = rec
            res[arm][scene] = per_cls
            ious = [v["iou_t0.21"] for v in per_cls.values() if v["iou_t0.21"] == v["iou_t0.21"]]
            lccs = [v["lcc_frac"] for v in per_cls.values()]
            print(f"  {arm:<12} IoU@0.21 {st.mean(ious)*100:6.2f}   "
                  f"lcc_frac {st.mean(lccs):.3f}   n_comp {st.mean([v['n_comp'] for v in per_cls.values()]):7.1f}",
                  flush=True)

    # ---------------------------------------------------------------- report
    print("\n\n=== IDEA 1: mask purity as a gate signal ===")
    if aucs:
        print(f"\nAUC vs per-primitive correctness ({len(aucs)} scenes):")
        print(f"  agreement ||x_w||  mean {st.mean([a for _, a, _ in aucs]):.4f}")
        print(f"  purity    p*_j     mean {st.mean([p for _, _, p in aucs]):.4f}")

    keys = [f"iou_q{q}" for q in QUANTS] + [f"iou_t{t}" for t in THRESHES]

    def agg(a, k):
        v = [st.mean([c[k] for c in res[a][s].values() if c[k] == c[k]])
             for s in res[a] if res[a][s]]
        return st.mean(v) if v else float("nan")

    print(f"\n{'arm':<12}" + "".join(f"{k[4:]:>9}" for k in keys) +
          f"{'lcc_frac':>10}{'n_comp':>9}")
    for a in ARMS:
        if not any(res[a].values()):
            continue
        lcc = agg(a, "lcc_frac")
        nc = agg(a, "n_comp")
        print(f"{a:<12}" + "".join(f"{agg(a, k)*100:>9.2f}" for k in keys) +
              f"{lcc:>10.3f}{nc:>9.1f}")
    print(f"\n{'arm':<12}" + "".join(f"{k[4:]:>9}" for k in keys) +
          f"{'d lcc':>10}{'d ncomp':>9}   (delta vs plain)")
    for a in ARMS:
        if a == "plain" or not any(res[a].values()):
            continue
        print(f"{a:<12}" + "".join(f"{(agg(a,k)-agg('plain',k))*100:>+9.2f}" for k in keys) +
              f"{agg(a,'lcc_frac')-agg('plain','lcc_frac'):>+10.3f}"
              f"{agg(a,'n_comp')-agg('plain','n_comp'):>+9.1f}")
    out = f"artifacts/scannet/purity_gate_{len(scenes)}scene.json"
    with open(out, "w") as f:
        json.dump({"auc": aucs, "res": res}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
