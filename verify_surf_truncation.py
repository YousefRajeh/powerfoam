"""Verify the 'surf<tau>' weight transform before spending GPU time on the ablation.

Two things have to be true for `T_k = 1 - exclusive_cumsum(w)` to reconstruct the
transmittance the export kernel actually held:

  (A) the algebra: w_k = alpha_k * T_k and T_{k+1} = T_k - w_k, so T telescopes.
      Checked against a directly-simulated forward pass with random alphas.

  (B) the ordering: each pixel's exported slots are in front-to-back traversal order.
      This is an assumption about the CUDA kernel, not algebra, so it is checked on a
      REAL exported view: reconstructed transmittance must be monotone non-increasing
      down each pixel's slots, must start at 1, and must bottom out near the kernel's
      own transmittance_threshold rather than somewhere arbitrary.

If (B) failed -- e.g. if atomics interleaved slots out of order -- the cumsum would
reconstruct garbage and the ablation would silently measure nothing. That is exactly
the failure mode that made the earlier distortion loss a no-op for a whole run, so it
gets checked directly rather than assumed.
"""
import argparse
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch


def check_algebra(n_trials=200, n_slots=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n_trials):
        alpha = torch.rand(n_slots, generator=g)
        # direct forward simulation, exactly as the kernel does it
        w = torch.zeros(n_slots)
        T_true = torch.zeros(n_slots)
        T = 1.0
        for k in range(n_slots):
            T_true[k] = T
            w[k] = alpha[k] * T
            T = T * (1.0 - alpha[k])
        T_rec = 1.0 - (w.cumsum(0) - w)
        worst = max(worst, float((T_rec - T_true).abs().max()))
    return worst


def check_truncation_semantics():
    """tau=0.5 must keep a prefix of the slots and zero the rest -- never a hole."""
    w = torch.tensor([[0.2, 0.25, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0]])
    T = 1.0 - (w.cumsum(1) - w)
    keep = (T >= 0.5)
    k = keep.long().flatten().tolist()
    # must be a contiguous prefix (T is non-increasing, so the mask is a prefix)
    first_zero = k.index(0) if 0 in k else len(k)
    assert all(v == 1 for v in k[:first_zero]) and all(v == 0 for v in k[first_zero:]), k
    return T.flatten().tolist(), k


def check_real_view(scene, variant, tau):
    """Export one real view and validate the ordering assumption on it."""
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    ckpt = f"output/scannet_{scene}_{variant}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])

    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ckpt}/model.pt")

    cam = dh.cameras[0]
    MAX_HITS = 64
    THRESH = 1e-3
    out_col, out_val, slot_counter, overflow, _ = model.export_feature_operator(
        cam, transmittance_threshold=THRESH, max_intersections=1024,
        max_hits_per_pixel=MAX_HITS)

    npix = cam.height * cam.width
    vmat = out_val.reshape(npix, MAX_HITS)
    used = slot_counter.clamp(max=MAX_HITS)
    T = 1.0 - (vmat.cumsum(1) - vmat)

    live = used > 0
    print(f"  view 0: {int(live.sum())}/{npix} pixels hit something, "
          f"overflow={int(overflow.item())}, median slots/pixel={float(used[live].float().median()):.1f}")

    # (B1) T starts at exactly 1 for every pixel that hit anything
    print(f"  T[:,0] == 1 for all live pixels: {bool((T[live, 0] == 1.0).all())}")

    # (B2) monotone non-increasing across USED slots only
    ar = torch.arange(MAX_HITS, device=vmat.device)
    valid = ar[None, :] < used[:, None]
    d = T[:, 1:] - T[:, :-1]
    both = valid[:, 1:] & valid[:, :-1]
    max_increase = float(d[both].max()) if both.any() else 0.0
    print(f"  max transmittance INCREASE across used slots: {max_increase:.3e} "
          f"(must be <= 0 -- any positive value means slots are NOT depth-ordered)")

    # (B3) weights are non-negative and sum to <= 1 per pixel (they are alpha*trans)
    print(f"  min weight: {float(vmat.min()):.3e}   max per-pixel weight sum: "
          f"{float(vmat.sum(1).max()):.6f} (must be <= 1)")

    # (B4) the last used slot's transmittance should sit near the kernel's own cutoff
    #      for pixels that terminated by absorption (i.e. used all/most slots)
    last_idx = (used - 1).clamp(min=0)
    T_last = T[torch.arange(npix, device=T.device), last_idx]
    deep = live & (used >= 8)
    if deep.any():
        print(f"  median final transmittance on pixels with >=8 hits: "
              f"{float(T_last[deep].median()):.5f}  (kernel cutoff {THRESH})")

    # what the truncation actually costs
    keep = T >= tau
    kept_w = (vmat * keep).sum()
    tot_w = vmat.sum()
    kept_slots = (keep & valid).sum()
    tot_slots = valid.sum()
    print(f"\n  tau={tau}: keeps {float(kept_w/tot_w)*100:.2f}% of total ray WEIGHT "
          f"but only {float(kept_slots)/float(tot_slots)*100:.2f}% of (pixel,cell) PAIRS")
    print(f"  median cells/pixel: {float(used[live].float().median()):.1f} -> "
          f"{float((keep & valid).sum(1)[live].float().median()):.1f}")
    # distinct cells receiving any feature at all
    cols = out_col.reshape(npix, MAX_HITS)
    before = torch.unique(cols[valid]).numel()
    after = torch.unique(cols[valid & keep]).numel()
    print(f"  distinct cells touched by this view: {before} -> {after} "
          f"({after/before*100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0347_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--skip-real", action="store_true")
    a = p.parse_args()

    print("(A) algebra: reconstructed vs simulated transmittance")
    err = check_algebra()
    print(f"  max abs error over 200 random rays x 64 slots: {err:.3e}")
    assert err < 1e-5, "telescoping identity is wrong"

    print("\n(A2) truncation keeps a contiguous prefix")
    T, k = check_truncation_semantics()
    print(f"  T = {[round(t,3) for t in T]}")
    print(f"  keep = {k}")

    if not a.skip_real:
        print(f"\n(B) real exported view from {a.scene}/{a.variant}")
        check_real_view(a.scene, a.variant, a.tau)
