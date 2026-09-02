"""Verify the hierarchy and Mutual-Proximity math before either sweep runs.

The MP test is the important one, and it needed rewriting: on iid random scores MP changes NOTHING
(0% of argmaxes), which looks like a no-op but is correct behaviour -- random data contains no hubs,
so a hubness correction has nothing to correct. The valid test INJECTS a hub and checks that MP
demotes it. Same lesson as the 4x3 toy in test_csls_paper_ideas.py: a degenerate fixture makes a
working method look broken.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import torch

from run_hierarchy_eval import build_hierarchy, pool_nodes
from run_hubness_mp_eval import mp_empirical, mp_gaussian
from run_derived_stack_eval import rank_encode


def _hubbed(N=2000, C=30, hub=7, lift=0.05, seed=0):
    """CLIP-like scores (mean .25, sd .02) with one class uniformly elevated."""
    torch.manual_seed(seed)
    cv = 0.25 + 0.02 * torch.randn(N, C)
    truth = cv.clone()
    cv[:, hub] += lift
    return cv, truth, hub


def test_hierarchy_levels_get_finer_and_ids_are_valid():
    torch.manual_seed(0)
    pos = torch.randn(2000, 3)
    lv = build_hierarchy(pos, torch.ones(2000), branch=4, min_size=20, max_levels=4)
    ns = [n for _, n in lv]
    assert all(b >= a for a, b in zip(ns, ns[1:])), ns
    for asg, n in lv:
        assert int(asg.min()) >= 0 and int(asg.max()) < n
        assert asg.numel() == pos.shape[0]


def test_hierarchy_separates_well_separated_blobs_at_the_coarsest_level():
    """Level 0 must not mix two clouds 50 units apart, or the spatial scale is meaningless."""
    torch.manual_seed(1)
    pos = torch.cat([torch.randn(500, 3), torch.randn(500, 3) + 50.0])
    asg, _ = build_hierarchy(pos, torch.ones(1000), branch=2, min_size=100, max_levels=1)[0]
    assert len(set(asg[:500].tolist()) & set(asg[500:].tolist())) == 0


def test_pool_nodes_returns_the_member_vector_when_members_agree():
    f = torch.zeros(10, 4); f[:, 0] = 1.0
    got = pool_nodes(f, torch.ones(10), torch.zeros(10, dtype=torch.long), 1)[0]
    assert torch.allclose(got, f[0], atol=1e-6)


def test_pool_nodes_respects_support_weights():
    f = torch.zeros(2, 4); f[0, 0] = 1.0; f[1, 1] = 1.0
    p = pool_nodes(f, torch.tensor([9.0, 1.0]), torch.zeros(2, dtype=torch.long), 1)[0]
    assert p[0] > 5 * p[1]


def test_mp_is_bounded_in_zero_one():
    cv, _, _ = _hubbed()
    for fn in (mp_empirical, mp_gaussian):
        m = fn(cv)
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0001


def test_mp_demotes_an_injected_hub_better_than_csls():
    """THE test. A class uniformly lifted by 0.05 steals 66.9% of argmaxes; MP must give them back."""
    cv, truth, hub = _hubbed()
    base_ok = (cv.argmax(1) == truth.argmax(1)).float().mean()
    csls = cv - 0.5 * cv.topk(50, dim=0).values.mean(0)[None, :]
    csls_ok = (csls.argmax(1) == truth.argmax(1)).float().mean()
    for fn in (mp_empirical, mp_gaussian):
        ok = (fn(cv).argmax(1) == truth.argmax(1)).float().mean()
        share = (fn(cv).argmax(1) == hub).float().mean()
        assert ok > csls_ok, f"{fn.__name__} {ok:.3f} did not beat CSLS {csls_ok:.3f}"
        assert share < 0.15, f"{fn.__name__} left the hub at {share:.1%}"
    assert csls_ok > base_ok      # sanity: CSLS itself helps, so the fixture is meaningful


def test_mp_is_a_no_op_when_there_are_no_hubs():
    """Complement, pinned deliberately: on iid scores MP changes nothing. This is CORRECT -- it also
    means a null result on real data would be ambiguous without the hub-injection test above."""
    torch.manual_seed(0)
    cv = torch.randn(500, 30)
    assert (cv.argmax(1) == mp_empirical(cv).argmax(1)).float().mean() > 0.99


def test_mp_survives_rank_encode():
    """Per-cell transforms are erased by rank_encode; MP must not be one."""
    cv, _, _ = _hubbed()
    assert not torch.equal(rank_encode(cv, 8.0, "cpu"),
                           rank_encode(mp_empirical(cv), 8.0, "cpu"))
