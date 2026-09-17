"""Write covis-diffused features to disk as a `solved_*.pt` artifact.

WHY THIS EXISTS. covis diffusion was only ever applied IN MEMORY inside
`run_cluster_classify_eval.py`, so the +0.0053 it measured is entangled with that script's
protocol: 320-leaf cluster-then-classify, no opacity culling, weighted-mean features. That is not
the configuration the paper reports. To ask what diffusion buys under the ACTUAL protocol
(per-primitive cosine argmax + OpenGaussian's low-opacity GT masking + the geometric-median
solver) the diffused features have to exist as a file that `evaluate_point_cloud_miou.py` can load.

WHAT TRANSFERS AND WHAT DOES NOT. Diffusion is a per-primitive feature transform applied before
any clustering, so it transfers to per-primitive argmax unchanged. The threshold gate does NOT: it
is a POOLING weight, and per-primitive argmax has no pooling step to weight. Of the two positive
score-side ideas, only this one is testable in that configuration.

GRAPH PROVENANCE. `covis_graph` reads the gram cache, so G is the co-visibility of the ray set
that cache was built from (sam_level=3, background rays dropped). G depends on GEOMETRY and the
ray set, NOT on the features, so it is a legitimate graph for any feature file on the same
checkpoint -- but it is the same graph the +0.0053 was measured with, which keeps the comparison
consistent rather than introducing a second change.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F


def diffuse_features(feats, valid_mask, src, dst, w, P, alpha, iters, sphere):
    """Diffuse only over valid-valid edges; invalid rows are returned untouched.

    Restricting to valid-valid matters: an unobserved primitive holds a zero row, and letting it
    into the graph would pull its neighbours toward zero, which is a data-absence artefact rather
    than smoothing.
    """
    from covis_graph import ppr_diffuse

    device = feats.device
    vmask = valid_mask.to(device)
    keep = vmask[src] & vmask[dst]
    src, dst, w = src[keep], dst[keep], w[keep]

    unit = torch.zeros_like(feats)
    unit[vmask] = F.normalize(feats[vmask], dim=-1)
    out = ppr_diffuse(unit, src, dst, P, alpha=alpha, iters=iters, weights=w, sphere=sphere)

    res = feats.clone()
    # Preserve each row's ORIGINAL norm: ||x_j|| is the multi-view agreement signal the gate uses,
    # and diffusion is meant to rotate the direction, not to overwrite that magnitude.
    nrm = feats[vmask].norm(dim=-1, keepdim=True)
    res[vmask] = F.normalize(out[vmask], dim=-1) * nrm
    moved = float((F.normalize(out[vmask], dim=-1) * unit[vmask]).sum(-1).mean())
    return res, int(keep.sum()), moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--in-file", required=True, help="basename under artifacts/scannet/<scene>/")
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--sphere", action="store_true")
    ap.add_argument("--graph-file", default=None,
                    help="a build_covis_graph.py output. Required for arms with no gram cache "
                         "(the caches on disk are nonfrozen only, so truefrozen needs this). "
                         "Verified to reproduce the gram cache exactly at --max-hits 64: "
                         "mine-only=0, weight ratio 1.00000 at p10/median/p90, mass ratio 1.0000.")
    a = ap.parse_args()

    device = "cuda"
    base = f"artifacts/scannet/{a.scene}"
    src_path = os.path.join(base, a.in_file if a.in_file.endswith(".pt") else a.in_file + ".pt")
    d = torch.load(src_path, map_location=device)
    feats = d["primitive_features"].to(device).float()
    valid = d["valid_mask"].to(device)
    P = feats.shape[0]

    if a.graph_file:
        from covis_graph import top_k_edges
        g = torch.load(a.graph_file, map_location="cpu", weights_only=True)
        assert int(g["P"]) == P, f"graph P={int(g['P'])} != model P={P} (wrong arm?)"
        keys = g["S_keys"].to(device)
        gj, gl = keys // P, keys % P
        gsrc, gdst, gw = top_k_edges(gj, gl, g["S_vals"].to(device).float(), P, a.k)
        print(f"  graph {os.path.basename(a.graph_file)}: {gsrc.numel():,} directed edges "
              f"after top-{a.k}", flush=True)
    else:
        from covis_graph import covis_graph
        gsrc, gdst, gw = covis_graph(a.scene, P, K=a.k, device=device)

    out, n_edges, moved = diffuse_features(
        feats, valid, gsrc, gdst, gw, P, a.alpha, a.iters, a.sphere)

    dst_path = os.path.join(base, a.out_file if a.out_file.endswith(".pt") else a.out_file + ".pt")
    torch.save({"primitive_features": out.cpu(), "valid_mask": valid.cpu()}, dst_path)
    print(f"  {a.scene}: P={P:,} valid={int(valid.sum()):,} edges={n_edges:,} "
          f"alpha={a.alpha} iters={a.iters} sphere={a.sphere} "
          f"mean cos to undiffused {moved:.4f} -> {os.path.basename(dst_path)}", flush=True)


if __name__ == "__main__":
    main()
