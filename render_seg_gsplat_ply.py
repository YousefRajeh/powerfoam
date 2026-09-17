"""Segmentation render of a 3DGS scene from a bundle PLY, using gsplat's own rasteriser.

COLOURS ARE FINAL RGB, NOT SH. gsplat.rasterization(..., colors=(N,3), sh_degree=None) treats the
colour tensor as already-evaluated RGB and skips SH evaluation entirely, so a flat class hue stays
flat from every view. Passing sh_degree would make the hue view-dependent and shimmer between
columns of the figure.

BACKGROUND KEEPS ITS REAL APPEARANCE, desaturated to luma, so the room stays readable instead of
becoming a grey void. It uses the SH DC term only (SH_C0 * f_dc + 0.5), which is view-independent
for the same reason.

ACTIVATIONS ARE APPLIED ON LOAD: exp(scales), sigmoid(opacity), normalised quats. The bundle PLYs
store raw log-scale and logits despite what load_splats' docstring implies; rendering them unactivated
produces a plausible-looking but wrong image (everything tiny and transparent).

The class labelling itself lives in seg_palette so both arms share one implementation.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SH_C0 = 0.28209479177387814


def load_ply_splats(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    g = lambda k: np.asarray(v[k])
    means = np.stack([g("x"), g("y"), g("z")], 1).astype(np.float32)
    scales = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1)).astype(np.float32)
    quats = np.stack([g(f"rot_{i}") for i in range(4)], 1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
    opac = (1.0 / (1.0 + np.exp(-g("opacity")))).astype(np.float32)
    dc = np.stack([g(f"f_dc_{i}") for i in range(3)], 1).astype(np.float32)
    rgb = np.clip(SH_C0 * dc + 0.5, 0.0, 1.0)
    return means, scales, quats, opac, rgb


def load_cameras(path):
    cams = json.load(open(path))
    cams = sorted(cams, key=lambda c: c.get("img_name", ""))
    return cams


def cam_to_viewmat(c):
    R = np.asarray(c["rotation"], dtype=np.float64)
    t = np.asarray(c["position"], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=None)
    # gsplat-format checkpoint: {means, scales, quats, opacities, sh0, shN}. The unfrozen 3DGS
    # baseline is stored this way (scannet_full/gs_unfrozen/<scene>/recon.pt), not as a .ply --
    # the OpenGaussian .ply outputs are all FROZEN (one gaussian per GT point).
    ap.add_argument("--recon-pt", default=None,
                    help="load geometry from a gsplat recon.pt instead of a .ply")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--features", required=True, help="solved_*.pt with primitive_features/valid_mask")
    ap.add_argument("--class-names", required=True, help="comma-separated; ScanNet++ has no fixed set")
    ap.add_argument("--view", type=int, required=True, help="index into sorted(image names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-json", default=None)
    ap.add_argument("--no-opacity-mask", action="store_true",
                    help="disable OpenGaussian's sigmoid(opacity)<0.1 masking")
    ap.add_argument("--per-primitive", action="store_true",
                    help="argmax instead of the pooled harness readout (for comparison only)")
    a = ap.parse_args()

    import seg_palette as P
    P.enable_determinism()
    from gsplat import rasterization

    dev = "cuda"
    if a.recon_pt:
        d = torch.load(a.recon_pt, map_location="cpu", weights_only=False)
        sp = d.get("splats", d)
        means = sp["means"].float().numpy()
        # Same activations the .ply loader applies: scales are log-space, opacity is a logit, and
        # quaternions are stored unnormalised.
        scales = np.exp(sp["scales"].float().numpy())
        q = sp["quats"].float().numpy()
        quats = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        opac = 1.0 / (1.0 + np.exp(-sp["opacities"].float().numpy().reshape(-1)))
        SH_C0 = 0.28209479177387814
        rgb = np.clip(SH_C0 * sp["sh0"].float().numpy().reshape(-1, 3) + 0.5, 0.0, 1.0)
    elif a.ply:
        means, scales, quats, opac, rgb = load_ply_splats(a.ply)
    else:
        raise SystemExit("need --ply or --recon-pt")
    sv = torch.load(a.features, map_location="cpu", weights_only=True)
    feats, vm = sv["primitive_features"], sv["valid_mask"].numpy().astype(bool)
    if feats.shape[0] != means.shape[0]:
        raise SystemExit("row mismatch: %d features vs %d gaussians -- wrong solve for this ply"
                         % (feats.shape[0], means.shape[0]))

    names = [s.strip() for s in a.class_names.split(",") if s.strip()]
    # OpenGaussian's rule: sigmoid(opacity) < 0.1 -> grey, not argmax'd into a class.
    # opac already has sigmoid applied at load time.
    col, cls, meta = P.primitive_colours(feats, vm, names, device=dev,
                                         pooled=not a.per_primitive,
                                         opacity=None if a.no_opacity_mask else opac)
    col = col.numpy()

    # Compositing is seg_palette.composite_three_case -- the SAME call the foam arm makes, so the
    # two renderers cannot drift apart. Each previously had its own copy and they diverged twice.
    shade = P.to_shade(rgb)                                  # shared with the foam arm
    col = P.composite_three_case(col, cls.numpy() if hasattr(cls, "numpy") else cls, meta, shade)

    cams = load_cameras(a.cameras)
    if not (0 <= a.view < len(cams)):
        raise SystemExit("view %d out of range (%d cameras)" % (a.view, len(cams)))
    c = cams[a.view]
    W, H = int(c["width"]), int(c["height"])
    K = torch.tensor([[c["fx"], 0, W / 2.0], [0, c["fy"], H / 2.0], [0, 0, 1]],
                     dtype=torch.float32, device=dev)[None]
    viewmat = torch.tensor(cam_to_viewmat(c), dtype=torch.float32, device=dev)[None]

    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    img, _, _ = rasterization(
        means=t(means), quats=t(quats), scales=t(scales), opacities=t(opac),
        colors=t(col), viewmats=viewmat, Ks=K, width=W, height=H,
        sh_degree=None,                       # colours are FINAL RGB; do not evaluate SH
        backgrounds=torch.ones(1, 3, device=dev),
    )
    arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    from PIL import Image
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(arr).save(a.out)
    meta.update({"arm": "3dgs", "view": a.view, "image": c.get("img_name"), "size": [W, H],
                 "features": os.path.basename(a.features)})
    P.write_labels_json(a.labels_json or (os.path.splitext(a.out)[0] + "_labels.json"), meta)
    print("wrote %s  (%dx%d, view %d = %s)" % (a.out, W, H, a.view, c.get("img_name")))
    print("  classes with primitives:", ", ".join(meta["classes_with_primitives"]) or "(none)")


if __name__ == "__main__":
    main()
