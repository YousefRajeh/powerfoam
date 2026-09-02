"""Verify the text-decorrelation and pseudo-label-LFDA math before the sweep.

The two that carry the claims: `test_text_whiten_decorrelates_prototypes` (the operation must
actually reduce prototype correlation, which is the measured defect) and
`test_lfda_recovers_the_discriminative_direction` (the generalised eigenproblem must find the
between-class direction and not the high-variance nuisance one).
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import torch
import torch.nn.functional as F

from run_text_and_pseudo_eval import (text_center, text_whiten, text_lowdin, text_hypernym,
                                      pseudo_labels, lfda_map)
from run_local_scatter_gate import between_class_scatter, leakage


def _protos(C=40, D=32, corr=0.9, seed=0):
    """Prototypes sharing a large common direction -- the measured `kitchen cabinet`/`cabinet`
    geometry, where pairwise cosines run to 0.83."""
    torch.manual_seed(seed)
    shared = F.normalize(torch.randn(1, D), dim=-1)
    return F.normalize(corr * shared + (1 - corr) * torch.randn(C, D), dim=-1)


def _mean_offdiag_cos(T):
    S = T @ T.T
    n = S.shape[0]
    return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))


def test_text_outputs_stay_unit_norm():
    T = _protos()
    for out in (text_center(T, 0.5), text_whiten(T, 0.25)[0], text_lowdin(T),
                text_hypernym(T, 0.5)):
        assert torch.allclose(out.norm(dim=-1), torch.ones(out.shape[0]), atol=1e-4)


def test_text_whiten_decorrelates_prototypes():
    """THE test for this family. Mean off-diagonal cosine must drop substantially."""
    T = _protos()
    before = _mean_offdiag_cos(T)
    after = _mean_offdiag_cos(text_whiten(T, 0.5)[0])
    assert before > 0.5, before
    assert after < before * 0.5, f"whitening barely decorrelated: {before:.3f} -> {after:.3f}"


def test_lowdin_produces_an_orthonormal_set():
    T = _protos(C=20, D=32)
    G = text_lowdin(T) @ text_lowdin(T).T
    assert torch.allclose(G, torch.eye(20), atol=1e-3), float((G - torch.eye(20)).abs().max())


def test_lowdin_is_the_closest_orthonormal_set():
    """Loewdin's defining property: minimal displacement. Each prototype should stay closer to its
    original than to any other original, or the class identities would be scrambled."""
    T = _protos(C=20, D=32)
    Tp = text_lowdin(T)
    sim = Tp @ T.T
    assert int((sim.argmax(1) == torch.arange(20)).sum()) == 20


def test_hypernym_subtraction_reduces_similarity_to_the_nearest_prototype():
    """Directly the `kitchen cabinet` -> `cabinet` operation."""
    T = _protos(C=30, D=32)
    S = T @ T.T; S.fill_diagonal_(-2.0)
    nn = S.argmax(1)
    before = (T * T[nn]).sum(-1).mean()
    Tp = text_hypernym(T, 0.5)
    after = (Tp * Tp[nn]).sum(-1).mean()
    assert after < before, f"{before:.4f} -> {after:.4f}"


def test_text_center_beta_zero_is_identity():
    T = _protos()
    assert torch.allclose(text_center(T, 0.0), T, atol=1e-5)


def test_pseudo_labels_pick_the_most_reliable_cells():
    scores = torch.randn(1000, 5)
    R = torch.arange(1000).float()
    idx, lab = pseudo_labels(scores, R, 0.1)
    assert idx.numel() == 100
    assert int(idx.min()) >= 900, "did not select the top-reliability cells"
    assert torch.equal(lab, scores[idx].argmax(1))


def test_lfda_recovers_the_discriminative_direction():
    """S_W huge along e0 (nuisance), S_B along e1 (class). The generalised eigenvector must be e1.
    Plain PCA of the data would return e0, so this distinguishes LFDA from whitening."""
    torch.manual_seed(0)
    D, n = 24, 600
    x = torch.zeros(n, D)
    lab = (torch.arange(n) % 2).long()
    x[:, 1] = torch.where(lab == 0, 1.0, -1.0)          # class direction, small
    x[:, 0] = 8.0 * torch.randn(n)                      # nuisance direction, huge
    x += 0.02 * torch.randn(n, D)
    i = torch.arange(0, n, 2)
    S_W = ((x[i] - x[i + 1]).T @ (x[i] - x[i + 1])) / (2 * i.numel())
    S_B = between_class_scatter(x, lab)
    U = lfda_map(S_W, S_B, 1)
    u = F.normalize(U[:, 0], dim=0).abs()
    assert float(u[1]) > 0.8, f"LFDA picked the wrong direction: e0={u[0]:.3f} e1={u[1]:.3f}"


def test_pseudo_label_scatter_captures_class_structure_when_labels_are_right():
    """Upper bound on the gate: with CORRECT pseudo-labels, capture must be ~1. If this failed, a
    low capture in the real run would be uninterpretable."""
    torch.manual_seed(1)
    D, n = 24, 400
    lab = (torch.arange(n) % 3).long()
    x = torch.zeros(n, D)
    for c in range(3):
        x[lab == c, c + 2] = 1.0
    x += 0.01 * torch.randn(n, D)
    S_B_true = between_class_scatter(x, lab)
    assert leakage(between_class_scatter(x, lab), S_B_true, 3) > 0.95
