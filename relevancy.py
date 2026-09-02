"""Splat Feature Solver's / LERF's contrastive relevancy readout, vectorised over classes.

WHY. Their `Runner.quantize()` docstring states the recommended 3D-segmentation configuration
explicitly: "we recommend to use the original features plus our contrastive comparison to segment
the splats". Our baseline instead took a plain cosine argmax over class names, which is THEIR
FEATURES WITH OUR READOUT -- so it does not reproduce their method end to end.

Their implementation (splat-distiller/pre_processing.py::get_relevancy) scores ONE positive at a
time against four canonical negatives:

    negatives = ("object", "things", "stuff", "texture")
    sims    = stack((positive_repeated, negative_vals), -1)     # (rays, N_neg, 2)
    softmax = softmax(10 * sims, -1)
    best_id = softmax[..., 0].argmin(1)                          # the WORST-CASE negative
    return  gather(softmax, 1, best_id...)[:, 0, :]

Two details that a casual reimplementation gets wrong, so they are called out:
  - the temperature is 10, applied to the RAW cosine pair, not to a full class distribution;
  - `argmin` selects the negative that makes the positive look WEAKEST, i.e. relevancy is a
    worst-case-over-negatives quantity, not an average.

For multi-class segmentation we evaluate this per class and take the argmax over classes. That
extension is ours -- their function is single-positive -- and is stated as such.

Verified against a literal transcription of their loop in test_relevancy.py.
"""
import torch

CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture")
TEMPERATURE = 10.0


def relevancy_scores(feats, pos_embeds, neg_embeds, temperature=TEMPERATURE, chunk=200_000):
    """(P, D) features x (C, D) class prototypes -> (P, C) relevancy, their formula per class.

    Chunked over primitives: the intermediate is (chunk, C, N_neg, 2), which at P=2.25M, C=100,
    N_neg=4 would be 1.8e9 floats if materialised whole.
    """
    P = feats.shape[0]
    C = pos_embeds.shape[0]
    out = torch.zeros((P, C), device=feats.device, dtype=feats.dtype)
    neg = neg_embeds.to(feats.dtype)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        f = feats[s:e]
        pos_val = f @ pos_embeds.to(f.dtype).T            # (n, C)
        neg_val = f @ neg.T                               # (n, N_neg)
        n_neg = neg.shape[0]
        # (n, C, N_neg, 2): positive repeated against each negative
        sims = torch.stack([pos_val[:, :, None].expand(-1, -1, n_neg),
                            neg_val[:, None, :].expand(-1, C, -1)], dim=-1)
        sm = torch.softmax(temperature * sims, dim=-1)    # (n, C, N_neg, 2)
        # worst-case negative: the one MINIMISING the positive's softmax share
        out[s:e] = sm[..., 0].min(dim=2).values
        del sims, sm, pos_val, neg_val
    return out


def embed_negatives(device, negatives=CANONICAL_NEGATIVES):
    """Embed the canonical negatives with the SAME encoder used for class names, so the two sets
    live in one space (embed_class_names applies no prompt template, matching this project)."""
    from evaluate_point_cloud_miou import embed_class_names
    return embed_class_names(list(negatives), device)
