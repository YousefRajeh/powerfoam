"""End-to-end test of radfoam_adapter.py against a checkpoint in radfoam's EXACT save_pt
format, synthesised here with scipy so the adapter can be validated before Agent A's CUDA
build lands.

The synthetic CSR is produced the way src/delaunay/delaunay.cu::find_adjacency produces it
(unique Delaunay tet edges -> both orientations -> sorted by source -> offsets), so this
exercises every code path in the adapter.  It does NOT prove that radfoam's GPU kernel
agrees with scipy; that check is radfoam_adapter.py --validate against a REAL model.pt.

Run:  python test_radfoam_adapter.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
from scipy.spatial import Delaunay

sys.path.insert(0, r"D:\Downloads\powerfoam")

from radfoam_adapter import (load_radfoam_foam, native_csr, check_csr_wellformed,
                             validate_against_lifted, load_radfoam_checkpoint,
                             radfoam_density, csr_to_undirected_edges)
from point_cloud_query import (assign_points_to_power_cells,
                               assign_points_to_nearest_center)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def make_fake_radfoam_ckpt(path, n=4000, seed=1, sh_degree=3):
    """Mirror src/delaunay/delaunay.cu::find_adjacency exactly."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    tri = Delaunay(pts.astype(np.float64))
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    e = np.concatenate([tri.simplices[:, list(p)] for p in pairs], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)          # unique undirected edges
    src = np.concatenate([e[:, 0], e[:, 1]])           # both orientations
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.lexsort((dst, src))                     # sorted by source
    src, dst = src[order], dst[order]
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=off[1:])
    sh = 3 * ((1 + sh_degree) ** 2 - 1)
    torch.save({"xyz": torch.from_numpy(pts),
                "density": torch.from_numpy(rng.normal(size=(n, 1)).astype(np.float32)),
                "color_dc": torch.zeros(n, 3),
                "color_sh": torch.zeros(n, sh),
                "adjacency": torch.from_numpy(dst.astype(np.int64)),
                "adjacency_offsets": torch.from_numpy(off)}, path)
    return pts, e


def main():
    tmp = tempfile.mkdtemp()
    ckpt = os.path.join(tmp, "model.pt")
    n = 4000
    pts, true_edges = make_fake_radfoam_ckpt(ckpt, n=n)
    print(f"synthetic radfoam checkpoint: N={n}, {true_edges.shape[0]} undirected edges")

    print("A  loaders")
    centers, radii = load_radfoam_foam(ckpt)
    check("centers shape/dtype", centers.shape == (n, 3) and centers.dtype == np.float64)
    check("radii are zeros (exact for an unweighted foam)",
          radii.shape == (n,) and not radii.any())
    check("centers round-trip float32 -> float64 exactly",
          np.array_equal(centers.astype(np.float32), pts))
    d = radfoam_density(ckpt, activation_scale=2.0)
    check("density is activated softplus(beta=10)*scale and strictly positive",
          d.shape == (n,) and bool((d > 0).all()))
    try:
        load_radfoam_checkpoint.__wrapped__ if False else None
        torch.save({"foo": 1}, os.path.join(tmp, "bad.pt"))
        load_radfoam_checkpoint(os.path.join(tmp, "bad.pt"))
        check("non-radfoam checkpoint is rejected", False)
    except KeyError:
        check("non-radfoam checkpoint is rejected", True)

    print("B  CSR conversion + well-formedness")
    sd = load_radfoam_checkpoint(ckpt)
    csr = native_csr(sd, centers)
    check("schema matches ours (adjacent/offsets int32 + num_primitives + dist)",
          csr["adjacent"].dtype == torch.int32 and csr["offsets"].dtype == torch.int32
          and csr["num_primitives"] == n and csr["dist"].dtype == torch.float32)
    rep = check_csr_wellformed(csr["adjacent"], csr["offsets"], n)
    check("well-formed (monotone, in-range, no self-loops, symmetric)",
          all(rep[k] for k in ("monotone_offsets", "offsets_cover_adjacency",
                               "indices_in_range", "no_self_loops", "symmetric")),
          str(rep))
    check("undirected edge count preserved",
          rep["num_undirected_edges"] == true_edges.shape[0])
    got = csr_to_undirected_edges(csr["adjacent"], csr["offsets"])
    check("recovered edge set == Delaunay edge set", np.array_equal(got, true_edges))
    deg = (csr["offsets"][1:].long() - csr["offsets"][:-1].long()).numpy()
    src = np.repeat(np.arange(n), deg)
    ref = np.linalg.norm(centers[src] - centers[csr["adjacent"].numpy()], axis=1)
    check("dist == Euclidean edge length",
          bool(np.allclose(csr["dist"].numpy(), ref, atol=1e-5)))

    print("C  native-vs-lifted validator (the check that will be run on real radfoam)")
    v = validate_against_lifted(centers, csr["adjacent"], csr["offsets"], max_points=n)
    check("full-set mode, identical to the 4D lift",
          v["mode"] == "full" and v["interior_identical"]
          and abs(v["block_jaccard"] - 1.0) < 1e-12, str(v))
    vb = validate_against_lifted(centers, csr["adjacent"], csr["offsets"], max_points=1500)
    check("contiguous-block mode: interior agrees exactly",
          vb["mode"] == "contiguous-block" and vb["interior_identical"]
          and vb["n_interior"] > 0,
          f"n_interior={vb['n_interior']} interior_overlap={vb['interior_overlap_frac']:.4f} "
          f"block_overlap={vb['block_overlap_frac']:.4f}")

    print("D  membership is exact via our existing helpers, unchanged")
    rng = np.random.default_rng(3)
    q = rng.uniform(-2.2, 2.2, size=(20000, 3))
    a_pow = assign_points_to_power_cells(q, centers, radii, k=64)
    a_nn = assign_points_to_nearest_center(q, centers)
    check("assign_points_to_power_cells(radii=0) == assign_points_to_nearest_center",
          bool(np.array_equal(a_pow, a_nn)))
    # brute-force truth on a subsample (full (20000,4000) is fine, but keep it cheap)
    qs = q[:2000]
    bf = ((qs[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    check("== brute-force nearest centre (i.e. exact Voronoi membership)",
          bool(np.array_equal(a_pow[:2000], bf)))
    # and the cell a point lands in must be consistent with the traversal rule radfoam uses:
    # for the owning cell i, every neighbour j must satisfy (x - (p_i+p_j)/2).(p_j-p_i) <= 0
    own = a_pow[:2000]
    bad = 0
    off = csr["offsets"].numpy().astype(np.int64)
    adj = csr["adjacent"].numpy().astype(np.int64)
    for t in range(2000):
        i = own[t]
        nb = adj[off[i]:off[i + 1]]
        nrm = centers[nb] - centers[i]
        if np.any(((qs[t] - (centers[i] + centers[nb]) / 2.0) * nrm).sum(1) > 1e-9):
            bad += 1
    check("owning cell satisfies radfoam's perpendicular-bisector half-space test "
          "for every Delaunay neighbour", bad == 0, f"{bad}/2000 violations")

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
