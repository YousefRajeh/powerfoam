"""Reject cells that belong to no queried class, instead of forcing an arg-max onto every one.

WHY THIS EXISTS. `classify_primitives` is an unconditional arg-max: every primitive with a feature
gets one of the K classes, because arg-max always returns something. For a point-mIoU protocol that
is correct and is what OpenGaussian/NormLift do -- every GT point has a label, so every prediction
is scored. For SURFACE extraction it is wrong: a foam is a space-filling tessellation, so forcing a
label onto every cell makes the "class-c surface" include every interior cell that happened to lean
towards c, and the extracted surface becomes the whole room volume (measured: 1926 m^2 of claimed
surface inside a 17.8 m^3 room). The useful volume is the semantically-claimed volume, which is a
strict subset -- the same distinction RadFoam draws between the geometry it fills and the geometry
it publishes.

THE RULE (LERF's relevancy score, Kerr et al. 2023, as used by LangSplat and LEGaussians). Embed a
canonical bank of negative phrases alongside the class names. The relevancy of class c against
negative m is the pairwise softmax

    rel(c, m) = exp(s_c) / (exp(s_c) + exp(n_m))  =  1 / (1 + exp(n_m - s_c)),

and a class is claimed only if it beats the WORST negative, i.e. min_m rel(c, m) > 1/2. That
inequality is strictly equivalent to s_c > max_m n_m, with no temperature to tune -- the softmax is
monotone in (s_c - n_m), so the threshold at 1/2 is exactly the sign test. We therefore implement
the sign test directly and expose `margin` for a stricter cut (s_c > max_m n_m + margin), which is
the only free parameter and defaults to 0.

Cells that fail the test get label 0, the same "unlabelled" code that the opacity mask and the
zero-feature guard already use, so downstream per-class code needs no change.
"""
import numpy as np
import torch

# LERF's canonical negatives (lerf/encoders/openclip_encoder.py: "object", "things", "stuff",
# "texture"). Kept verbatim rather than tuned per scene -- they are deliberately generic, and
# picking them per dataset would make the rejection threshold a fitted parameter.
CANONICAL_NEGATIVES = ["object", "things", "stuff", "texture"]


def classify_with_rejection(primitive_features, text_feats, neg_feats, margin=0.0):
    """(P,C) raw features x (K,C) class embeddings x (M,C) negative embeddings -> (P,) labels.

    Returns 0 where the cell claims no class, and 1..K otherwise (already in GT label space, so
    callers must NOT add 1 as they do for `classify_primitives`).
    """
    f = torch.nn.functional.normalize(primitive_features, dim=-1)
    t = torch.nn.functional.normalize(text_feats, dim=-1)
    n = torch.nn.functional.normalize(neg_feats, dim=-1)
    sims = f @ t.T                                     # (P, K)
    negs = (f @ n.T).max(dim=-1).values                # (P,) the strongest negative
    best = sims.max(dim=-1)
    lab = best.indices + 1
    lab[best.values <= negs + margin] = 0
    lab[primitive_features.norm(dim=-1) == 0] = 0      # zero features claim nothing
    return lab


def _self_test():
    """The equivalence the docstring claims, on hand-checkable inputs."""
    torch.manual_seed(0)
    # 3 classes, 2 negatives, orthonormal so cosines are exact and readable.
    t = torch.zeros(3, 5)
    t[0, 0] = t[1, 1] = t[2, 2] = 1.0
    n = torch.zeros(2, 5)
    n[0, 3] = 1.0
    n[1, 4] = 1.0
    # a: purely class 1 -> s=1 beats both negatives (0) -> claimed as 1
    # b: purely negative 0 -> every s=0, strongest negative 1 -> rejected
    # c: mixed, class 2 at 0.6 vs negative at 0.8 -> rejected
    f = torch.zeros(3, 5)
    f[0, 0] = 1.0
    f[1, 3] = 1.0
    f[2, 1], f[2, 3] = 0.6, 0.8
    lab = classify_with_rejection(f, t, n)
    assert lab.tolist() == [1, 0, 0], lab.tolist()

    # the pairwise-softmax form and the sign test must agree on random data
    f = torch.randn(500, 16)
    t = torch.randn(7, 16)
    n = torch.randn(4, 16)
    fn = torch.nn.functional.normalize(f, dim=-1)
    s = fn @ torch.nn.functional.normalize(t, dim=-1).T
    g = fn @ torch.nn.functional.normalize(n, dim=-1).T
    rel = 1.0 / (1.0 + torch.exp(g[:, None, :] - s[:, :, None]))   # (P,K,M)
    claimed_soft = (rel.min(dim=-1).values.max(dim=-1).values > 0.5)
    claimed_sign = classify_with_rejection(f, t, n) > 0
    assert torch.equal(claimed_soft, claimed_sign), "softmax form != sign test"

    # margin only ever removes claims
    strict = classify_with_rejection(f, t, n, margin=0.05)
    assert bool((((strict > 0).int() - (claimed_sign).int()) <= 0).all())
    # a zero feature never claims a class
    f[0] = 0
    assert int(classify_with_rejection(f, t, n)[0]) == 0
    print("semantic_reject self-test passed")


if __name__ == "__main__":
    _self_test()
