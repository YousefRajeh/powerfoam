"""Negative-query relevancy readout (LERF / Splat Feature Solver), and what it can and cannot do.

Instead of scoring a primitive by the raw cosine to the query embedding, score it against a
small fixed set of CANONICAL NEGATIVES ("object", "things", "stuff", "texture"):

    r_c(phi) = min_i  exp(phi . t_c) / ( exp(phi . t_c) + exp(phi . n_i) )
             = min_i  sigmoid( phi . t_c  -  phi . n_i )
             = sigmoid( phi . t_c  -  max_i phi . n_i )                      (*)

The negatives are fixed and class-independent, so this is single-query safe: querying one class
needs only that class embedding and the same four negatives, with no reference to any other
class in the scene.

(*) IS THE IMPORTANT LINE, and it is worth stating before measuring anything. `max_i phi . n_i`
depends on the primitive but NOT on the class, so it is a per-primitive constant and sigmoid is
strictly increasing. Therefore

    argmax_c r_c(phi)  ==  argmax_c  phi . t_c     exactly,

and the readout cannot change a single label in the closed-set argmax protocol -- the same
per-primitive inertness that made every confidence-weighting experiment a no-op. `assert_inert`
checks this numerically rather than asking anyone to trust the algebra.

What the readout DOES change is the absolute scale. A raw cosine lives in a narrow,
scene-dependent band (roughly 0.13-0.23 here), which is why a fixed cosine threshold
over-selected every class in A19 with the per-class selections summing to 410%. The relevancy
is a probability against a fixed reference, so 0.5 is a meaningful operating point that means
"closer to the query than to any generic negative" -- which is what a single-query mask needs.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

# LERF's canonical negatives, reused by LangSplat and Splat Feature Solver.
CANONICAL_NEGATIVES = ["object", "things", "stuff", "texture"]


def embed_text(phrases, device, model=None, tokenizer=None, overrides=None):
    """Plain CLIP text embeddings, one per phrase, L2-normalised. No templates."""
    import open_clip
    from evaluate_point_cloud_miou import (CLIP_MODEL, CLIP_PRETRAINED,
                                           OPENGAUSSIAN_NAME_OVERRIDES)
    if overrides is None:
        overrides = OPENGAUSSIAN_NAME_OVERRIDES
    if model is None:
        model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL,
                                                            pretrained=CLIP_PRETRAINED)
        model.eval().to(device)
    if tokenizer is None:
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    names = [overrides.get(p, p) for p in phrases]
    with torch.no_grad():
        e = model.encode_text(tokenizer(names).to(device)).float()
    return F.normalize(e, dim=-1), model, tokenizer


def relevancy(feats, T_pos, T_neg, normalize_feats=True):
    """(P,F) features x (C,F) queries x (N,F) negatives -> (P,C) relevancy in (0,1).

    Uses the pairwise-softmax form directly rather than the algebraic shortcut, so the
    implementation matches the published definition and `assert_inert` is a real check on it.
    """
    phi = F.normalize(feats, dim=-1) if normalize_feats else feats
    sp = phi @ T_pos.T                                   # (P, C)
    sn = phi @ T_neg.T                                   # (P, N)
    # softmax over {positive, negative_i} for each (class, negative) pair, then min over i
    pair = torch.stack([torch.sigmoid(sp - sn[:, i:i + 1]) for i in range(sn.shape[1])], -1)
    return pair.min(-1).values


def assert_inert(feats, T_pos, T_neg, atol=0):
    """The readout must not change any closed-set argmax. Returns the disagreement count."""
    phi = F.normalize(feats, dim=-1)
    a = (phi @ T_pos.T).argmax(1)
    b = relevancy(feats, T_pos, T_neg).argmax(1)
    return int((a != b).sum())


_NEG_CACHE = {}


def embed_negatives(device, phrases=None):
    """Canonical negatives, embedded once and cached -- they never change between scenes."""
    key = (str(device), tuple(phrases or CANONICAL_NEGATIVES))
    if key not in _NEG_CACHE:
        e, m, t = embed_text(list(key[1]), device, overrides={})
        _NEG_CACHE[key] = (e, m, t)
    return _NEG_CACHE[key]
