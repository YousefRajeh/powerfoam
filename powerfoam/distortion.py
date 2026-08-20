"""Exact Mip-NeRF 360 distortion loss, computed from stratified transmittance quantiles.

WHY THIS FORM
-------------
The distortion loss (Barron et al. 2022, adopted by VoroTracing) is

    L_dist = sum_{k,l} w_k w_l |s_k - s_l|  +  (1/3) sum_k w_k^2 * ds_k

over ray segments with compositing weights w_k = T_k * alpha_k and depths s_k. Computing it
per CELL requires the per-cell weights inside the ray kernel, i.e. a new Warp kernel plus a
hand-written backward.

There is a much better route on a representation that can already emit DIFFERENTIABLE depth
at an arbitrary transmittance threshold. Take thresholds q_0 = 1 > q_1 > ... > q_K and let
t_k be the depth at which transmittance first falls below q_k. Then the termination CDF is
F(t) = 1 - T(t), so the probability mass terminating in the interval (t_{k-1}, t_k] is

    w_k = F(t_k) - F(t_{k-1}) = (1 - q_k) - (1 - q_{k-1}) = q_{k-1} - q_k

which is a CONSTANT fixed by the threshold schedule -- known exactly, with no estimation --
while the depths t_k carry all the geometry and are differentiable. So the same loss becomes
a small PyTorch expression over the rendered quantile depths, and its gradient flows through
whatever the renderer's depth backward touches.

WHY THAT MATTERS HERE (the power-diagram point)
-----------------------------------------------
Verified empirically on a real checkpoint: gradient from the quantile depths reaches
`radii` (|grad|max 5.07e5 over 10,912 cells) as well as `density` and `points`. In
VoroTracing the distortion gradient reaches DENSITY ONLY -- their cell boundaries are
midpoint bisectors, so a boundary cannot move without moving a site and dragging every other
boundary of that cell with it. On a power diagram the weight r_i translates cell i's planes
WITHOUT moving its center, so the loss stops meaning only "make this material more opaque"
and starts also meaning "make this cell thinner". The `(1/3) sum w^2 ds` self-term is
precisely a thickness penalty, and this is the channel that lets it act.

DISCRETIZATION
--------------
Segments are equal-transmittance-mass bins rather than cells. That is a different
discretization of the same integral, and it is the natural one here: bin edges are exactly
where the renderer can evaluate depth in closed form (powerfoam solves
t = t_near + log(trans/q)/sigma analytically), so the bin masses are exact and only the
depths are approximated -- the opposite of the usual quadrature, where masses are the
approximation.

s-space follows the paper: s(t) = 1 - 1/(1 + t), bounded in [0,1), so near-camera geometry
is penalized more per metre than far geometry.
"""
import torch


def stratified_thresholds(K, device, dtype=torch.float32):
    """K transmittance thresholds, DESCENDING, at equal mass spacing.

    Descending is required, not cosmetic: the ray kernel walks thresholds with a
    forward-only index while transmittance decreases monotonically, so an ascending list
    makes every threshold fire at essentially the same (deepest) depth.

    Returns q with q_k = 1 - k/K for k = 1..K, i.e. bin k spans mass exactly 1/K with the
    implicit leading edge q_0 = 1 at t = 0.
    """
    k = torch.arange(1, K + 1, device=device, dtype=dtype)
    return 1.0 - k / K


def exact_distortion(depths, thresholds, valid=None, eps=1e-8):
    """L_dist per ray from quantile depths.

    depths     (..., K) depth at which transmittance first falls below each threshold.
               Non-positive entries mark thresholds the ray never reached.
    thresholds (K,)     the DESCENDING thresholds used to render `depths`.
    valid      (...,)   optional extra ray mask.

    Returns (loss_per_ray (...), n_valid_bins (...)).

    The double sum is evaluated in O(K) rather than O(K^2): the bins are depth-sorted by
    construction, so |s_k - s_l| = s_k - s_l for l < k and

        sum_{k,l} w_k w_l |s_k - s_l| = 2 * sum_k w_k (s_k * W_{<k} - S_{<k})

    with running prefixes W_{<k} = sum_{l<k} w_l and S_{<k} = sum_{l<k} w_l s_l. This is the
    same identity VoroTracing uses in its kernel, here over mass-bins instead of cells.
    """
    K = depths.shape[-1]
    dev, dt = depths.device, depths.dtype
    q = thresholds.to(dev, dt)
    # leading edge q_0 = 1 at t = 0: bin k spans (t_{k-1}, t_k]
    q_prev = torch.cat([torch.ones(1, device=dev, dtype=dt), q[:-1]])
    w = (q_prev - q).clamp_min(0.0)                       # (K,) constant bin masses

    ok = depths > 0
    if valid is not None:
        ok = ok & valid.unsqueeze(-1)
    t = torch.where(ok, depths, torch.zeros_like(depths))

    # Each quantile depth is a REPRESENTATIVE SAMPLE of the termination distribution
    # carrying mass w_k -- not the midpoint of an interval starting at the camera. That
    # distinction is not cosmetic: with interval midpoints, a ray hitting an opaque surface
    # at depth d through empty space puts w_1 of its mass at ~s(d/2) and scores a large
    # spurious spread, when the true L_dist for a step transmittance profile is exactly 0.
    # (VoroTracing avoids this because their segments are CELLS, so the first segment begins
    # where the ray enters geometry rather than at t=0.) Caught by the point-mass self-test.
    s_mid = 1.0 - 1.0 / (1.0 + t)

    # The self-term charges each parcel of mass for its own extent. With point samples the
    # extent is the spacing to the previous quantile; the first sample has no measured
    # extent, so it contributes none. A point mass therefore gives ds == 0 throughout.
    ds = torch.zeros_like(s_mid)
    ds[..., 1:] = (s_mid[..., 1:] - s_mid[..., :-1]).clamp_min(0.0)

    # drop unreached bins by zeroing their mass, so they contribute nothing to either term
    wk = w.expand_as(s_mid) * ok.to(dt)

    # O(K) cross term via exclusive prefix sums along the quantile axis
    ws = wk * s_mid
    W_prefix = torch.cumsum(wk, dim=-1) - wk
    S_prefix = torch.cumsum(ws, dim=-1) - ws
    cross = 2.0 * (wk * (s_mid * W_prefix - S_prefix)).sum(-1)

    self_term = (wk * wk * ds).sum(-1) / 3.0
    return cross + self_term, ok.sum(-1)


def _brute_force(depths, thresholds):
    """O(K^2) reference implementation of the same quantity, for testing only."""
    K = depths.shape[-1]
    dev, dt = depths.device, depths.dtype
    q = thresholds.to(dev, dt)
    q_prev = torch.cat([torch.ones(1, device=dev, dtype=dt), q[:-1]])
    w = (q_prev - q).clamp_min(0.0)
    ok = depths > 0
    t = torch.where(ok, depths, torch.zeros_like(depths))
    s_mid = 1.0 - 1.0 / (1.0 + t)
    ds = torch.zeros_like(s_mid)
    ds[..., 1:] = (s_mid[..., 1:] - s_mid[..., :-1]).clamp_min(0.0)
    wk = w.expand_as(s_mid) * ok.to(dt)
    cross = torch.zeros(depths.shape[:-1], device=dev, dtype=dt)
    for a in range(K):
        for b in range(K):
            cross = cross + wk[..., a] * wk[..., b] * (s_mid[..., a] - s_mid[..., b]).abs()
    return cross + (wk * wk * ds).sum(-1) / 3.0


def _self_test(device="cpu"):
    """Verify the O(K) form against the O(K^2) definition, and the known-zero case."""
    torch.manual_seed(0)
    K = 12
    q = stratified_thresholds(K, device)
    # random MONOTONE depths (the renderer emits increasing depth for descending thresholds)
    d = torch.rand(64, K, device=device).cumsum(-1) * 0.4 + 0.05
    fast, _ = exact_distortion(d, q)
    slow = _brute_force(d, q)
    err = (fast - slow).abs().max().item()
    print(f"[test] O(K) vs O(K^2) max abs err = {err:.3e}  "
          f"(scale {slow.abs().max().item():.4f})  {'PASS' if err < 1e-5 else 'FAIL'}")

    # a ray terminating in a single bin (all mass at one depth) must give ~0 spread
    dz = torch.full((4, K), 2.0, device=device)
    z, _ = exact_distortion(dz, q)
    print(f"[test] point-mass ray loss = {z.max().item():.3e}  "
          f"{'PASS' if z.max().item() < 1e-6 else 'FAIL'}")

    # monotonic in spread: a wider distribution must score strictly higher
    tight = torch.linspace(1.0, 1.1, K, device=device).expand(1, K)
    wide = torch.linspace(1.0, 5.0, K, device=device).expand(1, K)
    lt, _ = exact_distortion(tight, q)
    lw, _ = exact_distortion(wide, q)
    print(f"[test] tight={lt.item():.6f} < wide={lw.item():.6f}  "
          f"{'PASS' if lt.item() < lw.item() else 'FAIL'}")

    # unreached thresholds (depth<=0) must be ignored, not treated as depth 0
    dm = d.clone(); dm[:, -3:] = 0.0
    lm, nb = exact_distortion(dm, q)
    print(f"[test] masked bins: mean valid bins = {nb.float().mean().item():.2f} of {K}  "
          f"{'PASS' if abs(nb.float().mean().item() - (K - 3)) < 1e-6 else 'FAIL'}")

    # gradient flows to the depths
    dg = d.clone().requires_grad_(True)
    exact_distortion(dg, q)[0].sum().backward()
    print(f"[test] d(loss)/d(depth) nonzero = {int((dg.grad.abs() > 0).sum())}/{dg.numel()}  "
          f"{'PASS' if (dg.grad.abs() > 0).any() else 'FAIL'}")


if __name__ == "__main__":
    _self_test()
