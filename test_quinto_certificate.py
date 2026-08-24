"""Analytic tests for the Quinto Theorem 4.1 certificate.

Every case here has an answer derivable by hand, so a failure means the implementation is wrong
rather than the geometry being surprising. The certificate is a plane-versus-curve transversality
test, and the two ways to get it wrong are (a) treating a trajectory that LIES IN the plane as a
crossing, and (b) treating a trajectory that never reaches the plane as a crossing. Both are
covered.
"""
import numpy as np
import torch

from quinto_certificate import certify, signed_plane_distances, trajectory_planarity

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(centers, normals, cams, min_margin=0.0):
    c = torch.tensor(centers, dtype=torch.float32)
    n = torch.tensor(normals, dtype=torch.float32)
    n = n / n.norm(dim=-1, keepdim=True)
    k = torch.tensor(cams, dtype=torch.float32)
    seg = (k[1:] - k[:-1]).norm(dim=-1).to(DEV)
    s = signed_plane_distances(c, n, k, device=DEV)
    return certify(s, seg, min_margin)


def test_perpendicular_crossing_is_certified():
    """Cell at origin, normal +z. Plane pi = {z = 0}. Trajectory runs along z from -1 to +1, so it
    stabs the plane at right angles: certified, and the crossing margin |cos| must be exactly 1."""
    cert, ncross, margin = run([[0, 0, 0]], [[0, 0, 1]], [[0, 0, -1], [0, 0, 1]])
    assert bool(cert[0]), "a perpendicular stab through the plane must certify"
    assert int(ncross[0]) == 1
    assert abs(float(margin[0]) - 1.0) < 1e-5, f"margin should be 1.0, got {float(margin[0])}"
    print("PASS perpendicular crossing -> certified, margin 1.0")


def test_trajectory_lying_in_the_plane_is_NOT_certified():
    """THE CASE THAT MATTERS. Cell at origin with normal +x, so pi = {x = 0}, the yz-plane. The
    trajectory runs along z AT x = 0 -- entirely inside the plane. Every signed distance is 0.

    This is a tangential (non-transversal) intersection, which Theorem 4.1 explicitly excludes,
    and it is Quinto's own failure mode: sources on a curve that does not cross the conormal plane
    transversally leave that singularity undetected. A naive implementation using a non-strict
    test (s_a * s_b <= 0) would call this a crossing and certify a cell that is provably
    unrecoverable."""
    cert, ncross, _ = run([[0, 0, 0]], [[1, 0, 0]], [[0, 0, -1], [0, 0, 0], [0, 0, 1]])
    assert not bool(cert[0]), "a trajectory lying IN the plane is tangential, not transversal"
    assert int(ncross[0]) == 0
    print("PASS trajectory inside the plane -> NOT certified")


def test_trajectory_parallel_and_offset_is_NOT_certified():
    """Same normal +x (pi = {x = 0}), trajectory along z but offset to x = 1. Signed distance is a
    constant +1: the curve never reaches the plane, so no crossing and no certificate."""
    cert, ncross, _ = run([[0, 0, 0]], [[1, 0, 0]], [[1, 0, -1], [1, 0, 1]])
    assert not bool(cert[0])
    assert int(ncross[0]) == 0
    print("PASS parallel offset trajectory -> NOT certified")


def test_oblique_crossing_has_margin_cos_of_angle():
    """Cell at origin, normal +z, pi = {z = 0}. Step from (-1,0,-1) to (1,0,1): it does cross, and
    the component along the plane normal is 2 out of a step length of 2*sqrt(2), so the margin is
    1/sqrt(2) = 0.7071. This checks the margin is a genuine |cos| and not just a flag."""
    cert, _, margin = run([[0, 0, 0]], [[0, 0, 1]], [[-1, 0, -1], [1, 0, 1]])
    assert bool(cert[0])
    assert abs(float(margin[0]) - 1 / np.sqrt(2)) < 1e-5, float(margin[0])
    print(f"PASS oblique crossing -> margin {float(margin[0]):.4f} = 1/sqrt(2)")


def test_min_margin_rejects_grazing_crossings():
    """A crossing that is nearly tangential should be rejected once we demand a clean stab.
    Step from (-1,0,-0.01) to (1,0,0.01): crosses z=0, but the normal component is 0.02 over a
    step of ~2, giving margin ~0.01. Certified at min_margin=0, rejected at 0.05."""
    cert0, _, m = run([[0, 0, 0]], [[0, 0, 1]], [[-1, 0, -0.01], [1, 0, 0.01]], min_margin=0.0)
    cert1, _, _ = run([[0, 0, 0]], [[0, 0, 1]], [[-1, 0, -0.01], [1, 0, 0.01]], min_margin=0.05)
    assert bool(cert0[0]) and not bool(cert1[0]), (bool(cert0[0]), bool(cert1[0]))
    print(f"PASS grazing crossing margin {float(m[0]):.4f}: certified at 0, rejected at 0.05")


def test_sign_convention_is_symmetric():
    """Flipping the normal flips every signed distance but must not change the certificate --
    the conormal direction xi and -xi describe the same plane."""
    cams = [[0, 0, -1], [0, 0, 1]]
    a, _, ma = run([[0, 0, 0]], [[0, 0, 1]], cams)
    b, _, mb = run([[0, 0, 0]], [[0, 0, -1]], cams)
    assert bool(a[0]) == bool(b[0]) and abs(float(ma[0]) - float(mb[0])) < 1e-6
    print("PASS certificate is invariant to the sign of the conormal")


def test_planarity_detects_a_circular_trajectory():
    """Quinto's bad case is sources on a single circle. A planar circle must report planarity ~0;
    a helix must report clearly above 0. This is the diagnostic behind the per-scene prediction."""
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    helix = np.stack([np.cos(t), np.sin(t), t / 3.0], axis=1)
    pc, _ = trajectory_planarity(torch.tensor(circle, dtype=torch.float32))
    ph, _ = trajectory_planarity(torch.tensor(helix, dtype=torch.float32))
    assert pc < 1e-6, pc
    assert ph > 0.1, ph
    print(f"PASS planarity: circle {pc:.2e} (degenerate), helix {ph:.4f}")


def test_many_cells_batched_matches_one_at_a_time():
    """The chunked (P,V) path must agree with evaluating each cell alone -- guards the einsum
    contraction and the chunk boundary."""
    rng = np.random.default_rng(0)
    C = rng.normal(size=(500, 3)); N = rng.normal(size=(500, 3))
    K = np.cumsum(rng.normal(size=(40, 3)) * 0.3, axis=0)
    cert_all, _, m_all = run(C, N, K)
    for i in (0, 1, 250, 499):
        ci, _, mi = run(C[i:i + 1], N[i:i + 1], K)
        assert bool(ci[0]) == bool(cert_all[i]), i
        assert abs(float(mi[0]) - float(m_all[i])) < 1e-5, i
    print("PASS batched == per-cell on 500 random cells")


if __name__ == "__main__":
    for f in list(globals().values()):
        if callable(f) and getattr(f, "__name__", "").startswith("test_"):
            f()
    print("\nALL CERTIFICATE TESTS PASS")
