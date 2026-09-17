"""Verify that `foam_solid` reproduces the render kernels' geometry.

The claim under test: what a PowerFoam primitive renders is the convex solid

    R_i = Ball(p_i, r_i)  n  (radical half-spaces)  n  {n_i.(x - p_i) <= h}

and the kernels' `t_far - t_near` is the length of the ray's chord through it. Every earlier surface
extraction assumed instead that the primitive renders its dipole FACE, so this is the assumption
that has to be pinned down before any extraction is rebuilt on it.

The central test is deliberately independent: `ray_segment` is a transcription of the kernel's
clipping arithmetic, while `contains` is a ray-free membership predicate. Densely sampling a ray and
comparing the sampled-inside interval against the analytic segment cross-checks one against the
other, so a mistake in either shows up as a disagreement.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foam_solid import cell_halfspaces, contains, ray_segment  # noqa: E402


def _scene(seed=0, n=40, spread=0.35, rmin=0.10, rmax=0.22):
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, spread, (n, 3))
    r = rng.uniform(rmin, rmax, n)
    q = rng.normal(0.0, 1.0, (n, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return c, r, q, rng


def _rays(rng, m=250, radius=2.0):
    o = rng.normal(0.0, 1.0, (m, 3))
    o *= radius / np.linalg.norm(o, axis=1, keepdims=True)
    d = -o + rng.normal(0.0, 0.35, (m, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


def test_segment_matches_membership_sampling():
    """The analytic chord must equal the interval a dense point-membership scan finds."""
    c, r, nrm, rng = _scene(0)
    nb = [np.array([j for j in range(len(c)) if j != i]) for i in range(len(c))]
    o, d = _rays(rng, 600)
    checked = 0
    for i in rng.choice(len(c), 16, replace=False):
        for k in range(len(o)):
            hit, t0, t1 = ray_segment(int(i), o[k], d[k], c, r, nrm, nb[int(i)],
                                      use_dipole=True)
            ts = np.linspace(0.0, 4.0, 4001)
            pts = o[k][None, :] + ts[:, None] * d[k][None, :]
            ins = contains(int(i), pts, c, r, nrm, nb[int(i)], height=0.0)
            if not ins.any():
                # sampling found nothing: the analytic chord must be empty or thinner than the
                # sample spacing (1 mm here)
                assert (not hit) or (t1 - t0) < 1.5e-3, (i, k, hit, t1 - t0)
                continue
            assert hit, (i, k, "sampling found interior points but the chord was empty")
            lo, hi = ts[ins].min(), ts[ins].max()
            assert abs(lo - t0) < 2e-3 and abs(hi - t1) < 2e-3, (i, k, lo, t0, hi, t1)
            checked += 1
    assert checked > 100, f"too few non-empty cases exercised ({checked})"


def test_dipole_keeps_the_negative_half_space():
    """The kernel's dp-conditional clipping keeps {n.(x - p) <= h}, from BOTH ray directions."""
    c = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])
    n = np.array([[0.0, 0.0, 1.0]])
    nb = [np.zeros(0, dtype=int)]
    # ray travelling along +n, and the same ray reversed: both must keep the z <= 0 half
    for o, d in (([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]),
                 ([0.0, 0.0, 2.0], [0.0, 0.0, -1.0])):
        hit, t0, t1 = ray_segment(0, np.array(o), np.array(d), c, r, n, nb[0])
        assert hit
        mid = np.array(o) + 0.5 * (t0 + t1) * np.array(d)
        assert mid[2] <= 1e-9, (o, d, mid)
        assert abs((t1 - t0) - 1.0) < 1e-9, (o, d, t1 - t0)   # half of a unit-diameter sphere


def test_without_neighbours_the_solid_is_a_half_ball():
    """No neighbours -> Ball n half-space. Chord length must match the analytic half-ball chord."""
    c = np.array([[0.0, 0.0, 0.0]])
    r = np.array([0.7])
    n = np.array([[0.0, 0.0, 1.0]])
    rng = np.random.default_rng(3)
    for _ in range(200):
        o = rng.normal(0, 1, 3)
        o *= 3.0 / np.linalg.norm(o)
        d = -o + rng.normal(0, 0.3, 3)
        d /= np.linalg.norm(d)
        hit, t0, t1 = ray_segment(0, o, d, c, r, n, np.zeros(0, dtype=int))
        if not hit:
            continue
        mids = o[None, :] + np.linspace(t0, t1, 32)[:, None] * d[None, :]
        assert (np.linalg.norm(mids, axis=1) <= r[0] + 1e-9).all()
        assert (mids[:, 2] <= 1e-9).all()


def test_radical_plane_is_the_power_bisector():
    """A point on the extracted face must be power-equidistant from the two sites."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        c = rng.normal(0, 1, (2, 3))
        r = rng.uniform(0.1, 1.0, 2)
        A, b = cell_halfspaces(0, c, r, np.array([1]))
        nrm = A[0]
        x = c[0] + nrm * ((b[0] - c[0] @ nrm) / (nrm @ nrm))     # project site 0 onto the face
        p0 = np.sum((x - c[0]) ** 2) - r[0] ** 2
        p1 = np.sum((x - c[1]) ** 2) - r[1] ** 2
        assert abs(p0 - p1) < 1e-9, (p0, p1)


def test_adjacent_solids_share_their_face_and_do_not_overlap():
    """THE CONNECTIVITY CLAIM. Two overlapping spheres share a radical face: points just inside
    it belong to one solid and points just outside to the other, and no point belongs to both."""
    c = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])       # spheres overlap (0.25 < 0.2 + 0.2)
    r = np.array([0.2, 0.2])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    A0, b0 = cell_halfspaces(0, c, r, np.array([1]))
    nrm = A0[0] / np.linalg.norm(A0[0])
    x_face = c[0] + nrm * ((b0[0] - c[0] @ A0[0]) / (A0[0] @ nrm))
    rng = np.random.default_rng(11)
    both = inside0 = inside1 = 0
    for _ in range(2000):
        # jitter within the face plane, then step to either side
        t = rng.normal(0, 0.05, 3)
        t -= (t @ nrm) * nrm
        for eps in (-1e-3, 1e-3):
            x = (x_face + t + eps * nrm)[None, :]
            i0 = contains(0, x, c, r, n, np.array([1]))[0]
            i1 = contains(1, x, c, r, n, np.array([0]))[0]
            both += int(i0 and i1)
            inside0 += int(i0)
            inside1 += int(i1)
    assert both == 0, f"{both} points belong to BOTH solids -- they must be disjoint"
    assert inside0 > 50 and inside1 > 50, (inside0, inside1)


def test_non_overlapping_spheres_leave_a_gap():
    """The other half of the connectivity story: when spheres do NOT overlap, the solids are
    separated by empty space that belongs to neither. Disconnection there is CORRECT, not a bug."""
    c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    r = np.array([0.2, 0.2])                                   # 0.4 < 1.0 -> no overlap
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mid = np.array([[0.5, 0.0, -0.05]])
    assert not contains(0, mid, c, r, n, np.array([1]))[0]
    assert not contains(1, mid, c, r, n, np.array([0]))[0]


def test_chord_length_drives_alpha_monotonically():
    """Sanity on the rendering relation itself: alpha = 1 - exp(-sigma * chord) is increasing in
    both sigma and chord, so a longer path through a cell can only make it more opaque."""
    c, r, nrm, rng = _scene(5, n=12)
    nb = [np.array([j for j in range(len(c)) if j != i]) for i in range(len(c))]
    o, d = _rays(rng, 60)
    lens = []
    for k in range(len(o)):
        hit, t0, t1 = ray_segment(0, o[k], d[k], c, r, nrm, nb[0])
        if hit:
            lens.append(t1 - t0)
    lens = np.array(lens)
    assert (lens >= -1e-12).all(), "negative chord length"
    for sigma in (1.0, 10.0, 100.0):
        a = 1.0 - np.exp(-sigma * lens)
        assert ((a >= 0) & (a <= 1)).all()
    a1 = 1.0 - np.exp(-1.0 * lens)
    a2 = 1.0 - np.exp(-10.0 * lens)
    assert (a2 >= a1 - 1e-12).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
