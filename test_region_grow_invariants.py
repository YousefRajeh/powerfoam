"""Does the region grower produce what region growing is FOR?

The grower exists because k-means regions are spatially incoherent: only 6.6% of the 320
k-means regions are a single connected piece on the facet graph, the median region is
scattered over 15.5 disjoint fragments, and one pooled feature is then broadcast across all
of it -- the "desk predicted on a distant wall" artifact. Growing is supposed to fix that BY
CONSTRUCTION, because a region is built by walking edges.

So the invariants that actually matter are structural, not numerical:

  G1  CONNECTIVITY. Every grown region must be a connected subgraph of the adjacency graph.
      This is the entire point. If regions come out fragmented, the grower is not doing the
      one thing it exists to do, and its numbers mean nothing.

  G2  PARTITION. Every valid primitive gets exactly one label; no invalid primitive gets one.
      Silent double-assignment or dropped primitives would corrupt the pooled features.

  G3  THE FEATURE GATE IS REAL. With a high threshold, growth must be more restrictive:
      region count rises monotonically with the threshold, reaching one-region-per-primitive
      in the limit. A gate that is not actually applied would leave the count flat.

  G4  DETERMINISM. Same inputs, same labels. The batched grower resolves same-level conflicts
      by highest similarity; if that tie-break is not deterministic, ablation rows are not
      reproducible.

Run:  D:\\conda\\envs\\powerfoam\\python.exe test_region_grow_invariants.py [scene]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from ablation_assign import load_primitives
from determinism import enable_determinism
from run_region_grow_eval import batched_region_grow


def components_per_label(labels, adjacent, offsets, n):
    """Number of connected components each label spans, via label-restricted BFS."""
    lab = labels.cpu().numpy()
    adj = adjacent.cpu().numpy().astype(np.int64)
    off = offsets.cpu().numpy().astype(np.int64)
    seen = np.zeros(n, dtype=bool)
    comps = {}
    for start in range(n):
        if seen[start] or lab[start] < 0:
            continue
        L = lab[start]
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for k in range(off[u], off[u + 1]):
                v = adj[k]
                if not seen[v] and lab[v] == L:
                    seen[v] = True
                    stack.append(v)
        comps[L] = comps.get(L, 0) + 1
    return comps


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0062_00"
    enable_determinism()   # the ablation enables it; the grower must be tested as used
    dev = "cuda"
    prim = load_primitives("pf_nonfroz", scene, device=dev)
    sol = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                     map_location=dev, weights_only=True)
    feats = sol["primitive_features"].to(dev).float()
    valid = sol["valid_mask"].to(dev)
    unit = torch.nn.functional.normalize(feats, dim=-1)

    d = torch.load(f"artifacts/ablation_cache/{scene}_pf_nonfroz_delaunay.pt",
                   map_location=dev, weights_only=True)
    adjacent, offsets = d["adjacent"].to(dev).long(), d["offsets"].to(dev).long()
    P = unit.shape[0]
    print(f"{scene} pf_nonfroz: P={P:,} valid={int(valid.sum()):,} "
          f"edges={adjacent.numel()//2:,}\n")

    ok = True
    prev_regions = None
    for thr in (0.80, 0.90, 0.95):
        labels, nreg = batched_region_grow(adjacent, offsets, unit, valid, thr)

        # G2 partition
        lab_valid = labels[valid]
        unlabeled = int((lab_valid < 0).sum())
        invalid_labeled = int((labels[~valid] >= 0).sum())
        g2 = unlabeled == 0 and invalid_labeled == 0
        ok &= g2

        # G1 connectivity -- the reason the grower exists
        comps = components_per_label(labels, adjacent, offsets, P)
        frag = sum(1 for c in comps.values() if c > 1)
        worst = max(comps.values()) if comps else 0
        g1 = frag == 0
        ok &= g1

        print(f"thr={thr:.2f}  regions={nreg:,}")
        print(f"   G1 connectivity : {'ALL CONNECTED' if g1 else f'{frag:,} FRAGMENTED'} "
              f"(worst spans {worst} components)")
        g2msg = "ok" if g2 else f"FAIL unlabeled={unlabeled} invalid_labeled={invalid_labeled}"
        print(f"   G2 partition    : {g2msg}")
        if prev_regions is not None:
            g3 = nreg >= prev_regions
            print(f"   G3 gate is real : regions {prev_regions:,} -> {nreg:,} "
                  f"{'ok (monotone)' if g3 else 'FAIL (not monotone)'}")
            ok &= g3
        prev_regions = nreg

        lab2, nreg2 = batched_region_grow(adjacent, offsets, unit, valid, thr)
        g4 = bool(torch.equal(labels, lab2)) and nreg == nreg2
        print(f"   G4 determinism  : {'ok' if g4 else 'FAIL -- labels differ across runs'}")
        ok &= g4

    print("\nVERDICT:", "GROWER IS SOUND" if ok else "INVARIANT VIOLATED -- do not use in the ablation")


if __name__ == "__main__":
    main()
