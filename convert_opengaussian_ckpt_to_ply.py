"""Convert OpenGaussian's raw chkpnt30000.pth (graphdeco-inria capture()-tuple format, with an
extra `ins_feat` element OpenGaussian inserts at index 7) into a .ply file that
splat-distiller's GaussianPrimitive._load_ply() can load directly, so distill.py's real,
unmodified CLIP-feature-lifting code can run against it -- reusing their tested PLY loader
rather than writing a new checkpoint loader (this project's "no reimplemented math on either
side" principle).

Verified tuple layout by direct inspection (torch.load(..., weights_only=False) on
scene0000_00/chkpnt30000.pth): ckpt = (capture_tuple, iteration). capture_tuple has 14 elements:
  0 active_sh_degree (int, =3)
  1 xyz                    (N, 3)
  2 features_dc            (N, 1, 3)
  3 features_rest          (N, 15, 3)   -- 45 f_rest values/point, matches _load_ply's
                                            hard assert len(extra_f_names) == 3*(sh_degree+1)^2-3
                                            for sh_degree=3
  4 scaling (raw/log-space)(N, 3)
  5 rotation (raw, unnormalized quat) (N, 4)
  6 opacity (raw/logit-space)         (N, 1)
  7 ins_feat -- OpenGaussian's own low-dim instance-clustering embedding, NOT a CLIP feature.
               SKIPPED here; this is exactly the caveat documented in Experiment-F-scannet.md --
               the real per-primitive CLIP feature has to come from our own SAM+CLIP distill,
               not from this checkpoint.
  8 max_radii2D (empty at this stage)
  9-11 xyz_gradient_accum / denom-like optimizer bookkeeping tensors (unused here)
  12 optimizer state_dict (unused here)
  13 spatial_lr_scale (unused here)

_load_ply() expects RAW (pre-activation) values for opacity/scaling/rotation -- it applies
sigmoid/exp/normalize itself -- so we write chkpnt30000.pth's raw tensors straight through,
no re-derivation.

Run in the splat-distiller env (needs plyfile, numpy, torch):
    python convert_opengaussian_ckpt_to_ply.py --checkpoint <chkpnt30000.pth> --output <out.ply>
"""
import argparse

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def convert(checkpoint_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    capture_tuple, iteration = ckpt
    active_sh_degree = capture_tuple[0]
    xyz = capture_tuple[1].detach().cpu().numpy()
    features_dc = capture_tuple[2].detach().cpu().numpy()      # (N, 1, 3)
    features_rest = capture_tuple[3].detach().cpu().numpy()    # (N, 15, 3)
    scaling = capture_tuple[4].detach().cpu().numpy()           # (N, 3), raw/log-space
    rotation = capture_tuple[5].detach().cpu().numpy()          # (N, 4), raw/unnormalized
    opacity = capture_tuple[6].detach().cpu().numpy()           # (N, 1), raw/logit-space
    n = xyz.shape[0]
    print(f"Loaded {n} Gaussians, active_sh_degree={active_sh_degree}, iteration={iteration}")
    assert features_rest.shape[1] == 15, (
        f"expected 15 f_rest coeffs/channel (sh_degree=3), got {features_rest.shape[1]} -- "
        "_load_ply's assert will fail otherwise, don't silently truncate/pad"
    )

    # f_dc: (N, 1, 3) -> (N, 3), one column per channel (matches _load_ply's f_dc_0/1/2 read-back,
    # which indexes plydata[..., channel, 0] conceptually -- i.e. just the 3 DC values per point).
    f_dc = features_dc.reshape(n, 3)
    # f_rest: (N, 15, 3) -> flatten to (N, 45) in the same channel-major order _load_ply expects
    # (it reshapes back to (N, 3, 15) after reading f_rest_0..44 in that flattened order, then
    # transposes -- so write in (channel, coeff) flattened order to round-trip correctly).
    f_rest = features_rest.transpose(0, 2, 1).reshape(n, 45)

    dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("mask", "f4"), ("opacity", "f4")]
    for i in range(3):
        dtype_full.append((f"f_dc_{i}", "f4"))
    for i in range(45):
        dtype_full.append((f"f_rest_{i}", "f4"))
    for i in range(scaling.shape[1]):
        dtype_full.append((f"scale_{i}", "f4"))
    for i in range(rotation.shape[1]):
        dtype_full.append((f"rot_{i}", "f4"))

    elements = np.empty(n, dtype=dtype_full)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["mask"] = np.ones(n, dtype=np.float32)  # _load_ply requires this field; all-valid
    elements["opacity"] = opacity[:, 0]
    for i in range(3):
        elements[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(45):
        elements[f"f_rest_{i}"] = f_rest[:, i]
    for i in range(scaling.shape[1]):
        elements[f"scale_{i}"] = scaling[:, i]
    for i in range(rotation.shape[1]):
        elements[f"rot_{i}"] = rotation[:, i]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)
    print(f"Wrote {output_path} ({n} points)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    convert(args.checkpoint, args.output)
