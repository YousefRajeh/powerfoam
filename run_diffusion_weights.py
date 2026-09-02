"""P2 + P4 + P5: does WEIGHTING the diffusion graph beat uniform row-normalisation?

All three proposals reweight the same posterior diffusion (p <- (1-a) p0 + a S p) and are tested
against the same uniform-S baseline on the same scenes, so the deltas are paired and attributable.

  P2  TPFA / transmissibility   w_ij = A_ij / d_ij
      The finite-volume two-point flux approximation. A power diagram is exactly the mesh class on
      which TPFA is consistent, because the site-to-site segment is orthogonal to the shared facet
      by construction. So the foam admits an EXACT discrete Laplacian where a Gaussian kNN graph
      admits only a heuristic one. Facet area is the literal measure of shared boundary and a
      Gaussian pair has no shared boundary at all.
  P2a AREA only                 w_ij = A_ij
      Isolates whether the gain (if any) comes from area or from the 1/d factor, which a Gaussian
      graph could also compute. If area-only ~= dist-only, the "impossible for splats" claim is
      weaker than it looks.
  P5  VOLUME-directed           w_ij = A_ij / d_ij, then row-normalise by V_i
      A large cell's evidence should outvote a small one; uniform row-normalisation gives every
      neighbour equal say regardless of how much space it speaks for.
  P4  FEATURE-GATED (bilateral) w_ij *= sigmoid((cos(f_i,f_j) - tau)/gamma)
      Down-weight edges crossing a semantic boundary. This is the SOFT variant of the dead Potts
      move -- a sigmoid gate, never a threshold -- using the facet boundary cue re-measured at
      AUC 0.67-0.71 on the true facet graph (the 0.65 in the vault was a Cech number).

WHAT WOULD MAKE THIS INTERESTING RATHER THAN A TUNING EXERCISE. Raw area is unusable as a weight:
measured, 52 facets (0.0033%) carry 50% of total area, max facet 602 m^2. So every variant here is
row-normalised, and P2 divides by distance, which is what makes it a flux rather than a mass.

FALSIFIER, stated before running: < +0.2 mean mIoU over uniform. And a specific warning from the
surface agent's measurement -- largest-facet neighbour was INDISTINGUISHABLE from nearest-centre
neighbour as a top-1 vote selector (0.9757 vs 0.9741). Area saturates under hard selection. The
open question this script answers is whether it survives as a SOFT reweighting, where it never had
a fair test.
"""
import argparse
import glob
import json
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
SCENES = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train"}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def build_csr(i, j, w, n):
    """Symmetric CSR from an undirected weighted edge list."""
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    ww = np.concatenate([w, w])
    order = np.argsort(src, kind="stable")
    src, dst, ww = src[order], dst[order], ww[order]
    counts = np.bincount(src, minlength=n)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return dst.astype(np.int64), offsets.astype(np.int64), ww.astype(np.float64)


def diffuse(p0, dst, off, w, n, alpha, iters, dev, gate=None):
    src = torch.repeat_interleave(torch.arange(n, device=dev),
                                  torch.as_tensor(np.diff(off), device=dev))
    dst_t = torch.as_tensor(dst, device=dev)
    w_t = torch.as_tensor(w, device=dev, dtype=torch.float32)
    if gate is not None:
        w_t = w_t * gate
    rowsum = torch.zeros(n, device=dev).index_add_(0, src, w_t).clamp_min(1e-12)
    p = p0.clone()
    for _ in range(iters):
        agg = torch.zeros_like(p)
        agg.index_add_(0, src, p[dst_t] * w_t[:, None])
        agg /= rowsum[:, None]
        p = (1 - alpha) * p0 + alpha * agg
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--scale", type=float, default=1000.0)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--out", default="artifacts/scannet/diffusion_weights.json")
    a = ap.parse_args()
    enable_determinism()
    dev = "cuda"
    out = {}

    for scene in [s for s in a.scenes.split(",") if s in SCENES]:
        af = os.path.join(SP, f"area_{scene}_pf_nonfroz.npz")
        if not os.path.exists(af):
            print(f"[skip] {scene}: no facet areas", flush=True)
            continue
        A = np.load(af)
        i, j, area = A["i"].astype(np.int64), A["j"].astype(np.int64), A["area"].astype(np.float64)
        bounded = ~A["unbounded"].astype(bool)
        i, j, area = i[bounded], j[bounded], area[bounded]
        n = int(A["n_primitives"])

        sol = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                         map_location=dev, weights_only=True)
        feats = sol["primitive_features"].to(dev).float()
        valid = sol["valid_mask"].cpu().numpy()
        if feats.shape[0] != n:
            print(f"[skip] {scene}: {feats.shape[0]} feats vs {n} sites", flush=True)
            continue
        unit = F.normalize(feats, dim=-1)
        assigned = np.load(f"artifacts/ablation_cache/{scene}_pf_nonfroz_assign.npy")
        owned = assigned >= 0

        # geometry for the weight variants
        prim = torch.load(f"output/scannet_{scene}_nonfrozen/model.pt",
                          map_location="cpu", weights_only=False)
        pts = prim["points"].float().numpy().astype(np.float64)
        d_ij = np.linalg.norm(pts[i] - pts[j], axis=1).clip(1e-9)
        cg = os.path.join(SP, f"cellgeom_{scene}_pf_nonfroz.npz")
        V = np.load(cg)["V"].astype(np.float64) if os.path.exists(cg) else None

        variants = {
            "uniform":  np.ones_like(area),
            "area":     area,
            "tpfa":     area / d_ij,
            "invdist":  1.0 / d_ij,
        }
        if V is not None:
            variants["tpfa_vol"] = area / d_ij      # volume enters via row scaling below

        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\Downloads\scannet_pointcept\{SCENES[scene]}\{scene}", "segment20")
        n2i = {nm: k for k, nm in enumerate(names_all)}
        present = set(np.unique(raw).tolist())

        for cs in CLASS_SETS:
            names = [nm for nm in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[nm] in present]
            gt = remap_gt_labels(raw, [n2i[nm] for nm in names])
            nc = len(names) + 1
            text = embed_class_names(names, dev)
            p0 = torch.softmax(a.scale * (unit @ text.T), dim=-1)
            p0[~torch.from_numpy(valid).to(dev)] = 0.0

            # P4 gate, computed once per scene/classset
            cosij = (unit[torch.as_tensor(i, device=dev)] *
                     unit[torch.as_tensor(j, device=dev)]).sum(-1)
            gate = torch.sigmoid((torch.cat([cosij, cosij]) - a.tau) / a.gamma)

            for name, w in variants.items():
                dst, off, ww = build_csr(i, j, w, n)
                if name == "tpfa_vol":
                    src = np.repeat(np.arange(n), np.diff(off))
                    ww = ww * V[dst] / V[dst].mean()      # neighbour's volume = its say
                pd = diffuse(p0, dst, off, ww, n, a.alpha, a.iters, dev)
                for tag, prob in ((name, pd),
                                  (name + "+gate", diffuse(p0, dst, off, ww, n, a.alpha,
                                                           a.iters, dev, gate=gate))):
                    if tag.endswith("+gate") and name not in ("uniform", "tpfa"):
                        continue
                    cls = prob.argmax(-1).cpu().numpy()
                    sc_ = owned.copy()
                    sc_[owned] = (prob.sum(-1) > 0).cpu().numpy()[assigned[owned]]
                    pred = np.zeros(len(gt), dtype=np.int64)
                    pred[sc_] = cls[assigned[sc_]] + 1
                    _, miou, _, macc = calculate_metrics(
                        torch.from_numpy(gt).long(), torch.from_numpy(pred).long(), nc)
                    out.setdefault(f"{tag}|{cs}", {})[scene] = {"mIoU": float(miou),
                                                                "mAcc": float(macc)}
                    print(f"  {scene} {cs[11:]:>3} {tag:<14} mIoU={miou*100:6.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n=== mean over scenes (vs uniform) ===")
    for cs in CLASS_SETS:
        base = out.get(f"uniform|{cs}", {})
        if not base:
            continue
        b = np.mean([v["mIoU"] for v in base.values()]) * 100
        print(f"--- {cs[11:]} classes (uniform = {b:.2f}) ---")
        for k in sorted(out):
            tag, c = k.split("|")
            if c != cs or tag == "uniform":
                continue
            common = set(out[k]) & set(base)
            if not common:
                continue
            m = np.mean([out[k][s]["mIoU"] for s in common]) * 100
            bb = np.mean([base[s]["mIoU"] for s in common]) * 100
            print(f"    {tag:<16}{m:7.2f}{m-bb:+8.2f}")


if __name__ == "__main__":
    main()
