"""Consensus-vote solver: pick the direction the most views AGREE on, not the average direction.

THE PROBLEM IT TARGETS. A cell accumulates one CLIP direction per view. Our estimators so far
(weighted mean, geometric median) all return a CENTRAL direction. If the evidence is bimodal --
most views say "chair", a few say "floor" because the ray clipped a neighbour or an occluder
leaked -- a central estimate lands BETWEEN the modes and belongs to neither class. A minority
class that genuinely owns the cell can also be averaged out of existence.

Measured on scene0347_00, this is not hypothetical: within-cell pairwise cosine among the
accumulated observations has median MIN of 0.688, and

    53.18% of cells contain at least one pair below cos 0.7
    27.15% contain a pair below cos 0.5

so a majority of cells carry genuinely conflicting evidence.

WHY THIS IS NOT THE vMF EM THAT ALREADY FAILED. The vMF solver reweighted observations by
agreement with the CURRENT estimate, initialised from back-projection -- i.e. from the blend. Its
own post-mortem: "the iteration is pure confirmation ... no reweighting keyed on
agreement-with-the-majority can reach [cells where the majority is wrong]". It was mode-seeking
from the mean, so it could only sharpen whatever the mean already believed, and it degraded
monotonically in kappa (55.00% -> 51.00%).

This does NOT start from the blend. Every observation is a candidate mode; each one's support is
counted independently; the winner is the one with the largest weighted support set, and the
dissenters are DISCARDED rather than down-weighted. A minority cluster can win outright if it is
the most coherent, which is exactly the case vMF could not reach.

Implementation is a one-step flat-kernel mode seek (weighted medoid over a cosine ball), which on
&lt;=K observations per cell is exact and costs one KxK Gram per cell.
"""
import argparse

import torch
import torch.nn.functional as F


def consensus(U, w, tau, min_support=1):
    """U: (P,K,D) per-view observations. w: (P,K) weights. Returns (P,D) unit features.

    For every candidate observation k, its support is the total weight of observations lying
    within cosine `tau` of it (including itself). The candidate with the greatest support wins,
    and the feature is the weighted mean of that candidate's supporters only.
    """
    n = F.normalize(U, dim=-1)
    live = w > 0
    G = torch.einsum("pkd,pld->pkl", n, n)                      # (P,K,K) cosine
    agree = (G >= tau) & live.unsqueeze(1) & live.unsqueeze(2)  # who supports whom
    support = (agree.float() * w.unsqueeze(1)).sum(-1)          # (P,K) weighted support
    support = support.masked_fill(~live, -1.0)
    best = support.argmax(dim=1)                                # (P,) winning candidate
    P = U.shape[0]
    members = agree[torch.arange(P), best]                      # (P,K) its supporters
    wm = w * members.float()
    # A cell whose winner has too little support falls back to the plain weighted mean rather
    # than to an arbitrary single view.
    thin = members.sum(1) < min_support
    wm = torch.where(thin.unsqueeze(1), w, wm)
    f = (wm.unsqueeze(-1) * n).sum(1)
    return F.normalize(f, dim=-1), members.sum(1), thin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    c = torch.load(a.cache, map_location="cpu", weights_only=False)
    U = c["U"].float()
    w = c["top_w"].float()
    f, nmem, thin = consensus(U, w, a.tau, a.min_support)
    valid = w.sum(1) > 0
    print(f"[consensus] tau={a.tau} min_support={a.min_support}")
    print(f"  cells: {U.shape[0]}  valid: {int(valid.sum())}")
    print(f"  median supporters in winning group: {int(nmem[valid].median())}")
    print(f"  fell back to plain mean (thin support): {float(thin[valid].float().mean())*100:.2f}%")
    torch.save({"primitive_features": f, "valid_mask": valid}, a.output)
    print(f"  wrote {a.output}")


if __name__ == "__main__":
    main()
