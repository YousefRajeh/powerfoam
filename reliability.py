"""Single source of truth for the per-primitive reliability `R` used by feature consensus.

WHY THIS MODULE EXISTS (OPEN_ISSUES section K). Two orderings of the same two lines were in use
across the eval scripts and they compute DIFFERENT quantities:

    raw[vm] = F.normalize(feats[vm], dim=-1)
    R = raw.norm(dim=-1)   * vm      # AFTER normalising  -> R == 1 EXACTLY. Weighting is INERT.
    R = feats.norm(dim=-1) * vm      # BEFORE normalising -> live, but only a PROXY.

Five scripts used the first, four the second, and the difference is invisible at the call site: both
run without error and both produce plausible numbers. Worse, `run_spp_gs_eval.py` -- the source of
the ScanNet++ 3DGS component ladder -- used the inert form, which undercut the stated mechanism for
Issue F confound #1 (that reliability is live on 3DGS and constant on foam; it was constant on both).

WHAT THE CORRECT QUANTITY IS. Neither norm is NormLift's reliability. That is the decomposition
`reliability = c_intra * c_inter` computed by `AccumulatedFeatureStats.reliability()` from
`support`, `intra_sum`, `sum_view_weight_sq` and `numerator`. The `||f||` proxy was a workaround for
not having accumulator stats on 3DGS; since task #53 those stats exist for all 12 ScanNet++ scenes,
so the workaround is no longer needed.

EVERY CALLER GETS A SOURCE TAG BACK. The tag is meant to be recorded next to the result, because a
number produced with inert weighting is not comparable to one produced with live weighting, and that
was exactly the failure this module prevents.
"""
import os

import torch


def from_stats(stats_path, device="cuda"):
    """NormLift's reliability from accumulator stats. The correct quantity when stats exist."""
    import sys
    if r"D:\Downloads\feature-foam-lifting\src" not in sys.path:
        sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
    from feature_foam_lifting.operator import AccumulatedFeatureStats

    st = AccumulatedFeatureStats.load(stats_path, device=device)
    r = st.reliability()["reliability"].to(device).float()
    del st
    return r


def from_norm(feats_unnormalised, valid_mask):
    """Fallback proxy: ||f|| of the SOLVED (un-normalised) feature.

    Only meaningful for solvers whose output norm varies. The geometric-median solve renormalises
    every update, so ||f|| == 1 by construction and this proxy is CONSTANT there -- which is a
    property of the solver, not a bug, but it does mean consensus weighting is inert on that arm and
    must be reported as such.
    """
    return feats_unnormalised.norm(dim=-1) * valid_mask


def get(feats_unnormalised, valid_mask, stats_path=None, device="cuda", prefer_stats=True):
    """Return (R, source_tag). Prefers real stats; falls back to the norm proxy with a clear tag."""
    if prefer_stats and stats_path and os.path.exists(stats_path):
        r = from_stats(stats_path, device)
        if r.numel() != valid_mask.numel():
            raise ValueError(f"reliability from stats has {r.numel()} entries but valid_mask has "
                             f"{valid_mask.numel()} -- mismatched scene/artifact pair")
        return r * valid_mask, "stats.reliability"
    r = from_norm(feats_unnormalised, valid_mask)
    const = bool(torch.allclose(r[valid_mask], torch.ones_like(r[valid_mask]))) \
        if int(valid_mask.sum()) else False
    return r, "norm_proxy_constant" if const else "norm_proxy_live"


def describe(r, valid_mask):
    """One-line summary for logs, so an inert weighting is visible in the output rather than silent."""
    v = r[valid_mask]
    if v.numel() == 0:
        return "R: no valid primitives"
    return (f"R: min={float(v.min()):.4f} med={float(v.median()):.4f} max={float(v.max()):.4f} "
            f"std={float(v.std()):.4f}"
            + ("  [CONSTANT -> consensus weighting is INERT]" if float(v.std()) < 1e-6 else ""))
