"""Adapter: a trained OFFICIAL-radfoam scene -> the tensors our ScanNet mIoU pipeline eats.

The point of this file is that radfoam and PowerFoam numbers come out of the SAME scoring
code (run_cluster_classify_eval.py / evaluate_point_cloud_miou.py / run_normlift_refine_eval.py),
so a delta between them is attributable to the reconstruction and the lifting, not to two
different evaluators.

WHAT RADFOAM ALREADY GIVES US (do not re-derive any of this)
------------------------------------------------------------
radfoam maintains a full GPU Delaunay triangulation as part of its own formulation and
persists the useful parts straight into its checkpoint.  `RadFoamScene.save_pt`
(radfoam_model/scene.py) writes:

    xyz               (N,3) float32  primal points == cell centres
    density           (N,1) float32  RAW density; activation is
                                     args.activation_scale * softplus(density, beta=10)
                                     (RadFoamScene.get_primal_density)
    color_dc          (N,3)          SH DC term
    color_sh          (N, 3*((d+1)^2-1))
    adjacency         (E,) int64     CSR neighbour list  <-- the TRUE Delaunay adjacency
    adjacency_offsets (N+1,) int64   CSR row offsets

`adjacency` / `adjacency_offsets` come from `Triangulation.point_adjacency()` /
`point_adjacency_offsets()` (torch_bindings/triangulation_bindings.cpp), produced by
`find_adjacency` in src/delaunay/delaunay.cu.  Reading that kernel: it takes the 6 edges of
every Delaunay tet, sorts+uniques them, emits BOTH orientations, sorts by source and builds
the offsets -- i.e. exactly the symmetric CSR of the unique Delaunay edge set, the same
object our build_true_facet_graph.build_csr produces.  So for radfoam there is NO 4D lift,
NO ConvexHull, and NO scipy: adjacency is a checkpoint read.  Use --validate to confirm the
native graph and our lifted builder agree on a subsample.

TWO THINGS TO KNOW WHEN COMPARING TO POWERFOAM
----------------------------------------------
1. radii.  radfoam has none.  Our eval helpers take a radii array, so this adapter hands
   them ZEROS, which is not a fudge: with all radii equal, the power distance
   ||x-c||^2 - r^2 differs from squared Euclidean distance by a CONSTANT, so its argmin is
   the Euclidean argmin.  assign_points_to_power_cells(radii=0) is therefore bit-identical
   to assign_points_to_nearest_center and both are EXACT membership for radfoam's cells
   (see the header of this project's test_unweighted_delaunay_adjacency.py, and the
   verification of radfoam's own cell definition recorded in the notes below).
2. ORDER.  radfoam PERMUTES its points whenever the triangulation is rebuilt
   (RadFoamScene.update_triangulation -> permute_points).  The permutation is applied to
   every per-point parameter before saving, so index i in `xyz` matches index i in
   `adjacency`/`density`/`color_*` WITHIN one checkpoint -- but indices are NOT stable
   across checkpoints.  Any per-primitive feature file must therefore be produced against
   the SAME model.pt it will be evaluated with, never against an earlier one.

USAGE
-----
  # convert a trained radfoam scene (pure torch.load; no radfoam/CUDA import needed)
  python radfoam_adapter.py --ckpt /path/to/radfoam_out/model.pt \\
      --out-dir artifacts/radfoam/scene0000_00 --validate --validate-points 60000

  # then, from any of our eval scripts:
  from radfoam_adapter import load_radfoam_foam
  centers, radii = load_radfoam_foam("/path/to/radfoam_out/model.pt")   # radii == zeros
"""
import argparse
import json
import os
import time

import numpy as np
import torch

REQUIRED_KEYS = ("xyz", "adjacency", "adjacency_offsets")


def load_radfoam_checkpoint(pt_path, map_location="cpu"):
    """Raw dict from RadFoamScene.save_pt.  Pure torch.load -- no radfoam import, no GPU."""
    sd = torch.load(pt_path, map_location=map_location, weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in sd]
    if missing:
        raise KeyError(
            f"{pt_path} is not a radfoam save_pt checkpoint (missing {missing}); "
            f"keys present: {sorted(sd.keys())}")
    return sd


def load_radfoam_foam(pt_path, device=None):
    """Drop-in replacement for diagnose_scannet_miou.load_foam, for radfoam scenes.

    Returns (centers (N,3) float64 ndarray, radii (N,) float64 ZEROS).  Zeros are exact,
    not a placeholder -- see note 1 in the module docstring.
    """
    sd = load_radfoam_checkpoint(pt_path)
    centers = sd["xyz"].float().cpu().numpy().astype(np.float64)
    return centers, np.zeros(centers.shape[0], dtype=np.float64)


def radfoam_density(pt_path, activation_scale=1.0):
    """Activated volumetric density, mirroring RadFoamScene.get_primal_density().

    NOTE this is an unbounded volumetric density (softplus), NOT a 0-1 opacity -- the same
    caveat evaluate_point_cloud_miou's --gt-opacity-mask carries for PowerFoam.  Pass the
    run's args.activation_scale if you need the true scale.
    """
    sd = load_radfoam_checkpoint(pt_path)
    raw = sd["density"].float().reshape(-1)
    return (activation_scale * torch.nn.functional.softplus(raw, beta=10)).numpy()


def native_csr(sd, centers=None, with_dist=True):
    """radfoam's own CSR -> our adjacency_*.pt schema.

    Ours (build_true_facet_graph.build_csr): int32 `adjacent`, int32 `offsets`,
    int `num_primitives`, optional float32 `dist`.  radfoam stores int64 (cast up from the
    uint32 the kernels use), already symmetric and already sorted by source, so this is a
    dtype change plus an edge-length computation -- no re-derivation of the graph.
    """
    adjacent = sd["adjacency"].cpu().long()
    offsets = sd["adjacency_offsets"].cpu().long()
    n = offsets.numel() - 1
    if int(adjacent.numel()) > 2 ** 31 - 1 or int(offsets[-1]) > 2 ** 31 - 1:
        raise OverflowError(
            f"CSR has {int(adjacent.numel())} directed entries, which does not fit the "
            f"int32 offsets our adjacency schema uses; widen the schema before proceeding")
    out = {"adjacent": adjacent.to(torch.int32),
           "offsets": offsets.to(torch.int32),
           "num_primitives": int(n)}
    if with_dist and centers is not None:
        c = torch.as_tensor(centers, dtype=torch.float64)
        deg = (offsets[1:] - offsets[:-1])
        src = torch.repeat_interleave(torch.arange(n, dtype=torch.long), deg)
        out["dist"] = (c[src] - c[adjacent]).norm(dim=-1).to(torch.float32)
    return out


def csr_to_undirected_edges(adjacent, offsets):
    n = offsets.numel() - 1
    deg = (offsets[1:] - offsets[:-1]).long()
    src = np.repeat(np.arange(n), deg.numpy())
    dst = adjacent.numpy().astype(np.int64)
    e = np.sort(np.stack([src, dst], axis=1), axis=1)
    e = np.unique(e, axis=0)
    return e[e[:, 0] != e[:, 1]]


def check_csr_wellformed(adjacent, offsets, n):
    """Structural checks that do not need a second triangulation: monotone offsets, in-range
    indices, no self-loops, and exact symmetry (j in N(i) <=> i in N(j))."""
    off = offsets.numpy().astype(np.int64)
    adj = adjacent.numpy().astype(np.int64)
    rep = {"monotone_offsets": bool(np.all(np.diff(off) >= 0)),
           "offsets_cover_adjacency": bool(off[0] == 0 and off[-1] == adj.size),
           "indices_in_range": bool(adj.min() >= 0 and adj.max() < n)}
    deg = np.diff(off)
    src = np.repeat(np.arange(n), deg)
    rep["no_self_loops"] = bool(not np.any(src == adj))
    fwd = set(map(tuple, np.stack([src, adj], axis=1).tolist()))
    rep["symmetric"] = all((b, a) in fwd for a, b in list(fwd)[:200000])
    rep["mean_degree"] = float(deg.mean())
    rep["max_degree"] = int(deg.max())
    rep["isolated"] = int((deg == 0).sum())
    rep["num_undirected_edges"] = int(adj.size // 2)
    return rep


def validate_against_lifted(centers, adjacent, offsets, max_points=60000, seed=0):
    """Cross-check radfoam's native GPU Delaunay adjacency against our 4D-lift builder.

    A SUBSET of a Delaunay triangulation is not the Delaunay triangulation of the subset, so
    comparing on a random subsample of a big scene would be meaningless.  Instead the check
    is run on the FULL point set when it is small enough, and on a spatially CONTIGUOUS
    block otherwise, where only the block's boundary can disagree -- so the reported overlap
    is a lower bound and boundary-attributable.  Interpret <100% only via the interior
    figure, which excludes sites that touch the block boundary.
    """
    from build_true_facet_graph import regular_triangulation_edges

    n = centers.shape[0]
    native = csr_to_undirected_edges(adjacent, offsets)
    if n <= max_points:
        sub = np.arange(n)
        mode = "full"
    else:
        # contiguous block: take the max_points sites nearest a random seed point
        rng = np.random.default_rng(seed)
        anchor = centers[rng.integers(n)]
        sub = np.argsort(((centers - anchor) ** 2).sum(1))[:max_points]
        sub = np.sort(sub)
        mode = "contiguous-block"

    inset = np.zeros(n, dtype=bool)
    inset[sub] = True
    remap = -np.ones(n, dtype=np.int64)
    remap[sub] = np.arange(sub.size)

    keep = inset[native[:, 0]] & inset[native[:, 1]]
    nat_sub = set(map(tuple, remap[native[keep]].tolist()))
    lifted, _, _ = regular_triangulation_edges(centers[sub].astype(np.float64), None)
    lif = set(map(tuple, lifted.tolist()))

    # interior = sites of the block not on its boundary: those whose native neighbours are
    # ALL inside the block.  Their neighbourhoods cannot be truncated by the crop.
    deg_in = np.zeros(n, dtype=np.int64)
    np.add.at(deg_in, native[keep].ravel(), 1)
    deg_all = np.zeros(n, dtype=np.int64)
    np.add.at(deg_all, native.ravel(), 1)
    interior = inset & (deg_in == deg_all)
    ii = set(remap[np.where(interior)[0]].tolist())

    def restrict(s):
        return {e for e in s if e[0] in ii and e[1] in ii}

    ni, li = restrict(nat_sub), restrict(lif)
    rep = {
        "mode": mode, "n_total": int(n), "n_compared": int(sub.size),
        "native_edges_in_block": len(nat_sub), "lifted_edges_in_block": len(lif),
        "block_jaccard": len(nat_sub & lif) / max(len(nat_sub | lif), 1),
        "block_overlap_frac": len(nat_sub & lif) / max(len(nat_sub), 1),
        "n_interior": len(ii),
        "interior_native_edges": len(ni), "interior_lifted_edges": len(li),
        "interior_identical": ni == li,
        "interior_overlap_frac": len(ni & li) / max(len(ni), 1),
        "interior_only_native": len(ni - li), "interior_only_lifted": len(li - ni),
    }
    return rep


def save_solved_from_accumulator(numerator, denominator, out_path,
                                 min_weight=1e-6, normalize=False):
    """Agent C's FeatureAccumulator state -> the `solved_*.pt` schema our eval reads.

    run_cluster_classify_eval.py / run_normlift_refine_eval.py / evaluate_point_cloud_miou.py
    all consume exactly two keys:
        primitive_features  (P, C) float   per-primitive feature, ROW ORDER == centers order
        valid_mask          (P,)   bool    primitives with enough support to be trusted
    radfoam_model/feature_operator.py's FeatureAccumulator holds `numerator` = A^T b and
    `denominator` = A^T 1, whose training-free ratio is the back-projection estimate
    f_j = (A^T b)_j / (A^T 1)_j.  A primitive no ray ever weighted has denominator 0 and is
    marked invalid, which is the radfoam analogue of PowerFoam's `support > 0`.

    Normalisation is left OFF by default because the consumers normalise themselves
    (`F.normalize(feats[valid])`), and double-normalising would silently discard the
    magnitude that a later weighting scheme might want.
    """
    num = torch.as_tensor(numerator).float().cpu()
    den = torch.as_tensor(denominator).float().cpu().reshape(-1)
    valid = den > min_weight
    feats = torch.zeros_like(num)
    feats[valid] = num[valid] / den[valid, None]
    if normalize:
        feats[valid] = torch.nn.functional.normalize(feats[valid], dim=-1)
    obj = {"primitive_features": feats, "valid_mask": valid}
    if out_path:
        torch.save(obj, out_path)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="radfoam model.pt (from save_pt)")
    p.add_argument("--out-dir", default=None,
                   help="write centers.npz + adjacency_radfoam.pt + manifest.json here")
    p.add_argument("--activation-scale", type=float, default=1.0)
    p.add_argument("--validate", action="store_true",
                   help="cross-check the native adjacency against the 4D-lift builder")
    p.add_argument("--validate-points", type=int, default=60000)
    p.add_argument("--no-dist", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    sd = load_radfoam_checkpoint(a.ckpt)
    centers = sd["xyz"].float().cpu().numpy().astype(np.float64)
    n = centers.shape[0]
    print(f"[adapter] {a.ckpt}: N={n} primitives "
          f"sh_coeffs={tuple(sd['color_sh'].shape) if 'color_sh' in sd else None} "
          f"({time.time()-t0:.1f}s)", flush=True)

    csr = native_csr(sd, centers, with_dist=not a.no_dist)
    manifest = {"ckpt": os.path.abspath(a.ckpt), "num_primitives": n,
                "radii": "zeros (radfoam is an unweighted foam; exact, not a placeholder)",
                "adjacency_source": "radfoam Triangulation.point_adjacency (native GPU "
                                    "Delaunay, read from the checkpoint)"}
    manifest["csr"] = check_csr_wellformed(csr["adjacent"], csr["offsets"], n)
    print("[adapter] native CSR: " + json.dumps(manifest["csr"]), flush=True)

    if a.validate:
        rep = validate_against_lifted(centers, csr["adjacent"], csr["offsets"],
                                      max_points=a.validate_points)
        manifest["validation_vs_lifted"] = rep
        print("[adapter] native-vs-lifted: " + json.dumps(rep, indent=2), flush=True)

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(a.out_dir, "centers.npz"),
                            points=centers.astype(np.float32),
                            radii=np.zeros(n, dtype=np.float32),
                            density=radfoam_density(a.ckpt, a.activation_scale))
        torch.save(csr, os.path.join(a.out_dir, "adjacency_radfoam.pt"))
        with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[adapter] wrote {a.out_dir} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
