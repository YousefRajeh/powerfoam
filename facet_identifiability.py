"""Per-facet identifiability: is the DIFFERENCE between two adjacent cells determined by the data?

WHY THIS REPLACES THE PER-CELL QUINTO CERTIFICATE.

The first attempt (`quinto_certificate.py`) tested, per cell, whether the camera trajectory crosses
the plane through the cell centre conormal to the GT surface normal. It was measured on 10 scenes
and RETIRED, for two reasons:

  1. It reads only centres, GT normals and cameras -- so it returns the SAME answer for any
     partition sharing those seed points. It is blind to cell shape, size and connectivity, i.e.
     to the representation itself.
  2. Measured, it tracked view count (8.2% at 37 views, 44.2% at 279) and ANTI-correlated with
     mIoU: scene0062_00 certified worst (8.23%) and scores best (48.33 og19).

Curtis & Snieder (Geophysics 62(4):1524-1532, 1997) name exactly this failure. Their Figure 1
shows two cell geometries with IDENTICAL raypath coverage and completely different identifiability,
and they conclude: "raypath density, although necessary, is not nearly a sufficient criterion to
design the model space structure." A criterion built from coverage and camera geometry alone cannot
distinguish their (a) from their (b). Ours could not either.

THE RIGHT OBJECT. A feature field on a disjoint partition is piecewise constant per cell, so every
discontinuity lives on a FACET, and the question "is this field determined?" decomposes into, for
each facet (i,j), "is f_i - f_j determined?". In Curtis & Snieder's Figure 1(a) the two left cells
are always traversed TOGETHER by the same rays, so only v1 + v2 is constrained and v1 - v2 is a
null direction. That is the zero eigenvalue in their spectrum, localised to a facet.

THE MEASURE.

    d_ij = sum_r (A_ri - A_rj)^2  /  ( sum_r A_ri^2 + sum_r A_rj^2 )

d_ij = 0 exactly when every ray weights the two cells identically -- the difference is invisible and
the pair is only determined in sum. d_ij = 1 when no ray sees both. Expanding the numerator,

    sum_r (A_ri - A_rj)^2 = S_ii + S_jj - 2 S_ij

with S = A^T A, so this is computable from the Gram entries we ALREADY form (`gram_blocks.py`),
and the normaliser is just S_ii + S_jj. No SVD, no microlocal analysis, nothing idealised away:
cell shape, connectivity, occlusion and camera geometry all enter through A itself.

WHAT IT PREDICTS, so it can be falsified. Low-d_ij facets are where the data genuinely cannot
separate neighbours, so a graph prior is supplying information rather than smoothing away real
signal. Region growing and mode-vote refinement should therefore help MOST on low-d_ij facets and
be neutral-to-harmful on high-d_ij ones. If refinement gain is flat in d_ij, this measure is not
capturing what we claim and should be dropped like the last one.
"""
import numpy as np


def facet_identifiability(S_ii, S_jj, S_ij, eps=1e-30):
    """d_ij from Gram entries. Vectorised over facets.

    d = (S_ii + S_jj - 2 S_ij) / (S_ii + S_jj), clamped to [0, 1]. The numerator is
    sum_r (A_ri - A_rj)^2 >= 0 exactly; any negative value is floating-point noise on a facet where
    the two columns are nearly identical, which is precisely the d ~ 0 case we care about, so it is
    clamped rather than reported as a defect.
    """
    num = S_ii + S_jj - 2.0 * S_ij
    den = S_ii + S_jj
    d = np.where(den > eps, num / np.maximum(den, eps), 0.0)
    return np.clip(d, 0.0, 1.0)


def gram_entries_from_dense_A(A, pairs):
    """Reference path for tests: form S = A^T A densely and read the needed entries."""
    S = A.T @ A
    i, j = pairs[:, 0], pairs[:, 1]
    return S[i, i], S[j, j], S[i, j]


# ----------------------------------------------------------------------------------------------
# Validation against Curtis & Snieder Figure 1 -- the published example this measure must reproduce
# ----------------------------------------------------------------------------------------------

def curtis_snieder_figure1a():
    """(a) Cells BISECT each path. 2x2 grid; sources on top, receivers below.

    Two left-hand paths each run vertically through cell 1 (top-left) then cell 2 (bottom-left),
    contributing equal path length to both. Two right-hand paths do the same for cells 3 and 4.

    Their result: v1 + v2 determined exactly, v1 - v2 "remains entirely unresolved by the data",
    and likewise for 3,4 -> TWO ZERO EIGENVALUES.
    """
    A = np.zeros((4, 4))
    for r in (0, 1):          # two left paths
        A[r, 0] = A[r, 1] = 0.5
    for r in (2, 3):          # two right paths
        A[r, 2] = A[r, 3] = 0.5
    return A


def curtis_snieder_figure1b():
    """(b) Four vertical cells, each containing EXACTLY ONE path over its full length.

    Their result: eigenvectors are the identity, eigenvalues all equal, "velocities v1 to v4 are
    determined completely" -> NO zero eigenvalues.

    Note both geometries have "exactly the same homogeneous path coverage within each cell" (their
    caption), so any coverage-based criterion scores them identically. That is the whole point.
    """
    return np.eye(4)


def _report(name, A, pairs):
    S_ii, S_jj, S_ij = gram_entries_from_dense_A(A, pairs)
    d = facet_identifiability(S_ii, S_jj, S_ij)
    ev = np.linalg.eigvalsh(A.T @ A)
    ev = np.sort(ev)[::-1]
    ev_n = ev / ev.max()
    print(f"\n{name}")
    print(f"  column sums (raypath coverage per cell): {A.sum(axis=0)}")
    print(f"  normalised eigenvalues: {np.array2string(ev_n, precision=4)}")
    print(f"  zero eigenvalues: {int((ev_n < 1e-9).sum())}")
    for (i, j), dd in zip(pairs, d):
        print(f"  d_({i},{j}) = {dd:.6f}")
    return d, ev_n


def main():
    pairs = np.array([[0, 1], [2, 3], [0, 2]])

    dA, evA = _report("Curtis & Snieder Fig 1(a) -- cells bisect each path", curtis_snieder_figure1a(), pairs)
    dB, evB = _report("Curtis & Snieder Fig 1(b) -- one path per cell", curtis_snieder_figure1b(), pairs)

    print("\n--- checks ---")
    # Their stated result for (a): v1-v2 and v3-v4 unresolved => d = 0 on those facets.
    assert dA[0] < 1e-12, dA[0]
    assert dA[1] < 1e-12, dA[1]
    print("PASS (a): d_(0,1) = d_(2,3) = 0 -- the two unresolved differences C&S identify")
    # ...and exactly two zero eigenvalues, matching their Figure 1(a) spectrum.
    assert int((evA < 1e-9).sum()) == 2
    print("PASS (a): spectrum has exactly 2 zero eigenvalues, as published")
    # (0,2) spans left/right: no ray sees both, so the difference IS determined.
    assert dA[2] > 0.99, dA[2]
    print(f"PASS (a): d_(0,2) = {dA[2]:.4f} -- disjoint ray sets, difference determined")

    # (b): everything determined, no zero eigenvalues, every facet fully identifiable.
    assert int((evB < 1e-9).sum()) == 0
    assert np.all(dB > 0.99), dB
    print("PASS (b): no zero eigenvalues and every d = 1 -- 'determined completely'")

    # THE DISCRIMINATION TEST: coverage is identical, d is not.
    cova = curtis_snieder_figure1a().sum(axis=0)
    covb = curtis_snieder_figure1b().sum(axis=0)
    assert np.allclose(cova, covb), (cova, covb)
    print(f"PASS: identical coverage {cova} in BOTH geometries, but d_(0,1) is "
          f"{dA[0]:.4f} vs {dB[0]:.4f}")
    print("      -> the measure sees the structural difference that coverage cannot. This is the")
    print("         exact failure mode that retired the per-cell Quinto certificate.")

    print("\nALL FACET-IDENTIFIABILITY CHECKS PASS")


if __name__ == "__main__":
    main()
