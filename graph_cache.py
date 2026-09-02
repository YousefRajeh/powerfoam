"""Disk cache for built neighbourhood graphs, keyed by (scene, builder, params).

WHY THIS IS REQUIRED, not an optimisation. Measured build costs at the assumption-optimum settings,
extrapolated to the largest scene (P=2,252,236) with the fitted exponents:

    knn_pos  290s   knn_feat  598s   knn_maha ~1300s   delaunay 59s   kmeans/codebook ~2s

Across 12 scenes that is ~190 minutes of pure graph construction EXCLUDING knn_maha. The sweep
evaluates ~26 configs per (scene, solver), and several of those configs share a graph, so rebuilding
per config would multiply a one-time 3-hour cost by an order of magnitude and dominate the entire
sweep. Built once per (scene, builder, K), it is paid once.

Stores only the edge list (src, dst) as int32 -- the degree vector and CSR form are cheap to
recompute and would otherwise triple the file size. Edge counts reach 57.9M, so int32 rather than
int64 halves the footprint; primitive counts are ~2.25M, far inside int32 range (checked on save).

The cache key includes every parameter that changes the graph. A builder whose parameters are not
all captured here would silently return a stale graph, so `params` is required, not optional.
"""
import hashlib
import json
import os

import torch

CACHE_DIR = os.path.join("artifacts", "graph_cache")


def key(scene, builder, params):
    p = json.dumps(params, sort_keys=True)
    h = hashlib.sha1(f"{scene}|{builder}|{p}".encode()).hexdigest()[:12]
    return f"{scene}_{builder}_{h}"


def path(scene, builder, params):
    return os.path.join(CACHE_DIR, key(scene, builder, params) + ".pt")


def load(scene, builder, params, device="cuda"):
    f = path(scene, builder, params)
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location=device, weights_only=True)
    return d["src"].long(), d["dst"].long()


def save(scene, builder, params, src, dst, P):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if P > 2_147_483_647:
        raise ValueError(f"P={P} exceeds int32; widen the cache dtype before storing this scene")
    f = path(scene, builder, params)
    tmp = f + ".tmp"
    # write-then-rename: a cache file truncated by an interrupted run would load as a VALID but
    # incomplete graph, which is worse than a miss because nothing would flag it.
    torch.save({"src": src.to(torch.int32).cpu(), "dst": dst.to(torch.int32).cpu(),
                "P": int(P), "scene": scene, "builder": builder, "params": params}, tmp)
    os.replace(tmp, f)
    return f


def get_or_build(scene, builder, params, build_fn, P, device="cuda"):
    """Return (src, dst), building and caching on a miss. `build_fn()` returns (src, dst)."""
    hit = load(scene, builder, params, device)
    if hit is not None:
        return hit[0], hit[1], True
    src, dst = build_fn()
    save(scene, builder, params, src, dst, P)
    return src.long(), dst.long(), False
