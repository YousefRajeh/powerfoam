"""Verify the whitening / ABTT / Procrustes math before spending a sweep on it.

The decisive test is `test_wccn_recovers_separation_on_synthetic_data`: build a cloud with a known
large NUISANCE direction and a small CLASS-SEPARATING direction -- the exact geometry WCCN exists to
fix -- and check that whitening by the within-class scatter flips which one dominates. If that fails,
the method is not doing what the docstring claims and no scene result would be interpretable.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import torch
import torch.nn.functional as F

from run_whitening_eval import within_object_scatter, inv_power, all_but_the_top, procrustes_align


def test_within_object_scatter_matches_definition():
    torch.manual_seed(0)
    f = torch.randn(6, 4)
    src = torch.tensor([0, 1, 2]); dst = torch.tensor([3, 4, 5])
    got = within_object_scatter(f, src, dst)
    d = f[src] - f[dst]
    assert torch.allclose(got, (d.T @ d) / (2 * 3), atol=1e-6)


def test_within_object_scatter_is_symmetric_psd():
    torch.manual_seed(1)
    f = torch.randn(50, 8)
    src = torch.randint(0, 50, (300,)); dst = torch.randint(0, 50, (300,))
    S = within_object_scatter(f, src, dst)
    assert torch.allclose(S, S.T, atol=1e-6)
    assert float(torch.linalg.eigvalsh(S).min()) > -1e-6


def test_inv_power_alpha_zero_is_identity():
    torch.manual_seed(2)
    A = torch.randn(6, 6); S = A @ A.T + torch.eye(6)
    assert torch.allclose(inv_power(S, 0.0), torch.eye(6), atol=1e-4)


def test_inv_power_half_whitens():
    """S^{-1/2} S S^{-1/2} == I is the defining property of whitening."""
    torch.manual_seed(3)
    A = torch.randn(6, 6); S = A @ A.T + 0.5 * torch.eye(6)
    M = inv_power(S, 0.5)
    assert torch.allclose(M @ S @ M.T, torch.eye(6), atol=1e-3), (M @ S @ M.T)


def test_wccn_recovers_separation_on_synthetic_data():
    """THE test. e0 is a huge within-class nuisance direction, e1 a small between-class one.
    Before whitening the nuisance dominates the cosine; after, the class direction should."""
    torch.manual_seed(4)
    D, n = 16, 400
    nuis, cls = torch.zeros(D), torch.zeros(D)
    nuis[0] = 1.0; cls[1] = 1.0
    # two classes separated by 0.6 along cls, each smeared by 6.0 along nuis
    a = 0.6 * cls + 6.0 * torch.randn(n, 1) * nuis + 0.05 * torch.randn(n, D)
    b = -0.6 * cls + 6.0 * torch.randn(n, 1) * nuis + 0.05 * torch.randn(n, D)
    feats = torch.cat([a, b])
    # "adjacent pairs" = same-class pairs, which is what the facet graph supplies
    src = torch.cat([torch.arange(0, n, 2), torch.arange(n, 2 * n, 2)])
    dst = src + 1
    S = within_object_scatter(feats, src, dst)
    M = inv_power(S, 0.5)

    def sep(X):
        X = F.normalize(X, dim=-1)
        ca, cb = X[:n].mean(0), X[n:].mean(0)
        within = (X[:n] - ca).norm(dim=1).mean() + (X[n:] - cb).norm(dim=1).mean()
        return float((ca - cb).norm() / (within + 1e-9))

    before, after = sep(feats), sep(feats @ M.T)
    assert after > before * 2, f"whitening did not improve separation: {before:.4f} -> {after:.4f}"


def test_all_but_the_top_removes_the_top_directions():
    torch.manual_seed(5)
    D, n = 12, 5000
    top = F.normalize(torch.randn(1, D), dim=-1)
    X = 10.0 * torch.randn(n, 1) * top + torch.randn(n, D) * 0.1
    Pk = all_but_the_top(X, 1)
    Y = X @ Pk.T
    assert float((Y @ top.T).abs().mean()) < 0.05 * float((X @ top.T).abs().mean())


def test_all_but_the_top_is_an_idempotent_projector():
    torch.manual_seed(6)
    X = torch.randn(3000, 10)
    Pk = all_but_the_top(X, 3)
    assert torch.allclose(Pk @ Pk, Pk, atol=1e-4)
    assert torch.allclose(Pk, Pk.T, atol=1e-4)


def test_projector_applied_to_both_sides_preserves_a_matched_pair():
    """Sanity on the usage pattern: if a text vector IS a cell vector, projecting both keeps their
    cosine at 1. Guards against applying the map to one side only."""
    torch.manual_seed(7)
    X = torch.randn(2000, 16)
    Pk = all_but_the_top(X, 2)
    v = X[0:1]
    a = F.normalize(v @ Pk.T, dim=-1); b = F.normalize(v @ Pk.T, dim=-1)
    assert abs(float((a * b).sum()) - 1.0) < 1e-5


def _rot(D, ang):
    Q = torch.eye(D)
    c, s = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
    Q[0, 0] = c; Q[0, 1] = -s; Q[1, 0] = s; Q[1, 1] = c
    return Q


def test_procrustes_recovers_a_moderate_rotation():
    """Valid regime: the two spaces are already roughly aligned, which is OUR case (cell features
    and text embeddings are both CLIP vectors, cosine ~0.25). Recovery is exact up to ~30 degrees."""
    torch.manual_seed(8)
    D, C = 8, 60
    cells = F.normalize(torch.randn(4000, D), dim=-1)
    for ang in (0.05, 0.17, 0.52):                     # 3, 10, 30 degrees
        txt = F.normalize(cells[:C] @ _rot(D, ang).T, dim=-1)
        W = procrustes_align(cells, txt, iters=5)
        cos = float((F.normalize(txt @ W.T, dim=-1) * cells[:C]).sum(-1).mean())
        assert cos > 0.95, f"failed at {ang} rad: cos {cos:.4f}"


def test_procrustes_CANNOT_bootstrap_from_an_arbitrary_rotation():
    """Documented limitation, not a bug. Under a random rotation the initial mutual-NN dictionary is
    0% correct, so there is nothing for Procrustes to refine. This is exactly why Conneau et al. run
    ADVERSARIAL TRAINING (section 2.1) to initialise W before the Procrustes refinement (2.2); we
    implement only 2.2. Pinned so that a future failure of the B arm is read as this limitation
    rather than a coding error."""
    torch.manual_seed(8)
    D, C = 8, 60
    cells = F.normalize(torch.randn(4000, D), dim=-1)
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    txt = F.normalize(cells[:C] @ Q.T, dim=-1)
    W = procrustes_align(cells, txt, iters=5)
    cos = float((F.normalize(txt @ W.T, dim=-1) * cells[:C]).sum(-1).mean())
    assert cos < 0.5, f"unexpectedly recovered a random rotation ({cos:.4f}) -- revisit the docs"


def test_procrustes_returns_an_orthogonal_map():
    torch.manual_seed(9)
    cells = F.normalize(torch.randn(2000, 10), dim=-1)
    txt = F.normalize(torch.randn(40, 10), dim=-1)
    W = procrustes_align(cells, txt)
    assert torch.allclose(W @ W.T, torch.eye(10), atol=1e-3)
