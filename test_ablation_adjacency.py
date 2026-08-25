"""Do the three complexes satisfy the relations they must, on REAL reconstructions?

Structural invariants, each of which would catch a different real bug:

  I1  alpha SUBSET delaunay. The alpha complex is the Delaunay edge set filtered by
      |xi-xj| < ri+rj, so every alpha edge must be a Delaunay edge. A violation means the
      filter is being applied to the wrong edge list.

  I2  cech vs delaunay OVERLAP (powerfoam), reported not asserted. The intuition that cech
      (AABB overlap) is a superset of the facet graph is FALSE, and measuring it is the point:
      a power cell extends far beyond its own radius, so two cells can share a facet while
      their bounded volumes never meet. Measured on scene0062_00 the two have near-identical
      edge counts yet differ on ~half their edges, which is why the ablation carries them as
      separate arms rather than treating cech as a cheap approximation of the truth.

  I2b RADFOAM'S OWN ADJACENCY as an oracle. radfoam checkpoints ship `adjacency`/
      `adjacency_offsets` built by radfoam's own CUDA Delaunay (Shewchuk predicates). Our
      Delaunay of the same sites must reproduce it. This is the strongest available check on
      the delaunay path: an independent implementation, on real scene geometry, at full scale.

  I3  Symmetry and self-loop freedom of every CSR: j in adj(i) <=> i in adj(j), and i not in
      adj(i). The renderer's cech CSR and our edge-list construction are built by completely
      different code paths, so this is a genuine check on both.

  I4  radfoam's weighted and unweighted builds agree. radfoam is unweighted, so passing its
      zero radii must give exactly the same graph as passing None. If it does not, the
      degenerate-lift shortcut in build_delaunay is wrong.

Run:  D:\\conda\\envs\\powerfoam\\python.exe test_ablation_adjacency.py [scene]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
from ablation_adjacency import (build_alpha, build_delaunay, load_cech_powerfoam,
                                gaussian_reach_radii, _csr_from_edges)
from ablation_assign import load_primitives


def edge_set(e):
    e = np.asarray(e, dtype=np.int64)
    if len(e) == 0:
        return set()
    lo = np.minimum(e[:, 0], e[:, 1]); hi = np.maximum(e[:, 0], e[:, 1])
    return set(zip(lo.tolist(), hi.tolist()))


def csr_edge_set(adjacent, offsets):
    adj = adjacent.numpy().astype(np.int64)
    off = offsets.numpy().astype(np.int64)
    src = np.repeat(np.arange(len(off) - 1), np.diff(off))
    lo = np.minimum(src, adj); hi = np.maximum(src, adj)
    return set(zip(lo.tolist(), hi.tolist())), src, adj


def check_csr(adjacent, offsets, label):
    es, src, adj = csr_edge_set(adjacent, offsets)
    loops = int((src == adj).sum())
    # symmetry: the directed multiset must equal its own transpose
    d = set(zip(src.tolist(), adj.tolist()))
    asym = sum(1 for (i, j) in d if (j, i) not in d)
    print(f"    {label:<12} undirected={len(es):>9,}  self_loops={loops}  asymmetric={asym}")
    return loops == 0 and asym == 0


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0062_00"
    ok = True

    # ---- powerfoam: delaunay vs alpha vs cech -------------------------------------------
    for recon in ("pf_nonfroz",):
        try:
            prim = load_primitives(recon, scene, device="cpu")
        except FileNotFoundError as e:
            print(f"{recon}: missing {e}"); continue
        c = prim["centers"].numpy().astype(np.float64)
        r = prim["radii"].numpy().astype(np.float64)
        print(f"\n=== {recon} {scene}: N={c.shape[0]:,}")
        de = build_delaunay(c, r)
        al = build_alpha(c, r, edges=de)
        sd, sa = edge_set(de), edge_set(al)
        print(f"    delaunay={len(sd):,}  alpha={len(sa):,} "
              f"({100*len(sa)/max(len(sd),1):.1f}% of delaunay)")
        i1 = sa.issubset(sd)
        print(f"    I1 alpha subset delaunay: {i1}")
        ok &= i1
        a_csr, o_csr = _csr_from_edges(de, c.shape[0])
        ok &= check_csr(a_csr, o_csr, "delaunay")

        ce = load_cech_powerfoam(recon, scene)
        if ce is None:
            print("    cech: no cached graph")
        else:
            adjacent, offsets, st = ce
            ok &= check_csr(adjacent, offsets, "cech")
            sc, _, _ = csr_edge_set(adjacent, offsets)
            if len(offsets) - 1 != c.shape[0]:
                print(f"    [SKIP] I2: cech has {len(offsets)-1:,} nodes but checkpoint has "
                      f"{c.shape[0]:,} -- different reconstruction")
            else:
                inter = len(sd & sc)
                print(f"    I2 cech vs delaunay: |cech|={len(sc):,} |delaunay|={len(sd):,} "
                      f"shared={inter:,} ({100*inter/max(len(sd),1):.1f}% of delaunay); "
                      f"{len(sd-sc):,} facet edges absent from cech, "
                      f"{len(sc-sd):,} cech edges are not facets")

    # ---- radfoam: weighted(zeros) must equal unweighted ----------------------------------
    for recon in ("rf_unfroz",):
        try:
            prim = load_primitives(recon, scene, device="cpu")
        except FileNotFoundError:
            print(f"\n{recon}: not downloaded yet"); continue
        c = prim["centers"].numpy().astype(np.float64)
        n = min(c.shape[0], 40000)          # hull cost grows fast; a subset settles the identity
        sub = c[:n]
        print(f"\n=== {recon} {scene}: N={c.shape[0]:,} (testing on {n:,})")
        a = edge_set(build_delaunay(sub, np.zeros(n)))
        b = edge_set(build_delaunay(sub, None))
        i4 = a == b
        print(f"    I4 zero-radii == unweighted: {i4}  (|a|={len(a):,} |b|={len(b):,})")
        ok &= i4

        # I2b: radfoam ships its own Delaunay adjacency -- an independent oracle at full scale
        import torch as _t
        sd_ck = _t.load(f"recon_remote/{recon}/{scene}/model.pt", map_location="cpu",
                        weights_only=False)
        if "adjacency" in sd_ck and "adjacency_offsets" in sd_ck:
            native, _, _ = csr_edge_set(sd_ck["adjacency"].cpu(),
                                        sd_ck["adjacency_offsets"].cpu())
            ours = edge_set(build_delaunay(c, None))
            inter = len(native & ours)
            jac = inter / max(len(native | ours), 1)
            print(f"    I2b vs radfoam's own Delaunay: |native|={len(native):,} "
                  f"|ours|={len(ours):,} shared={inter:,} jaccard={jac:.4f}")
            i2b = jac > 0.99
            print(f"        agrees: {i2b}")
            ok &= i2b
        else:
            print("    I2b: checkpoint carries no native adjacency")

    # ---- gaussian: delaunay of means, alpha by max-scale reach ---------------------------
    for recon in ("gs_froz",):
        try:
            prim = load_primitives(recon, scene, device="cpu")
        except FileNotFoundError:
            print(f"\n{recon}: not downloaded yet"); continue
        c = prim["centers"].numpy().astype(np.float64)
        n = min(c.shape[0], 40000)
        sub = c[:n]
        rr = gaussian_reach_radii(prim["scales"].numpy())[:n]
        print(f"\n=== {recon} {scene}: N={c.shape[0]:,} (testing on {n:,})")
        de = build_delaunay(sub, None)
        al = build_alpha(sub, rr, edges=de)
        sd, sa = edge_set(de), edge_set(al)
        print(f"    delaunay={len(sd):,}  alpha={len(sa):,} "
              f"({100*len(sa)/max(len(sd),1):.1f}% of delaunay)")
        i1 = sa.issubset(sd)
        print(f"    I1 alpha subset delaunay: {i1}")
        ok &= i1
        a_csr, o_csr = _csr_from_edges(de, n)
        ok &= check_csr(a_csr, o_csr, "gs delaunay")

    print("\nVERDICT:", "ALL INVARIANTS HOLD" if ok else "VIOLATION -- do not run the ablation")


if __name__ == "__main__":
    main()
