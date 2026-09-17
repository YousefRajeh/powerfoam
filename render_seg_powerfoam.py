"""Segmentation render of a PowerFoam scene, using PowerFoam's own rasteriser.

MIRRORS PowerfoamScene.forward AND OVERRIDES ONLY texel_rgb. Everything geometric -- normals,
tangents, radii, texel-site offsets -- is computed exactly as forward() does, so the image differs
from an ordinary render only in colour. Re-deriving the geometry here would risk a silent mismatch
with the real renderer, which is the whole thing a qualitative figure must not have.

VIEW-INDEPENDENT WITHOUT TOUCHING THE SPHERICAL-LOBE BASIS. forward() evaluates a view-dependent
appearance model per texel site, then the rasteriser returns a site-weighted average. Writing the
SAME flat hue into all num_texel_sites sites makes that average return the hue from any direction,
so the class colour is stable across views without disabling or special-casing the lobe basis.

BACKGROUND KEEPS ITS REAL APPEARANCE, desaturated to luma, so the room stays readable rather than
collapsing to a grey void.

The class labelling lives in seg_palette, shared with the 3DGS renderer, so the two arms cannot
drift apart in vocabulary, hue slots or readout.

--split all IS REQUIRED for view indices to line up with the 3DGS arm: COLMAPDataset's train split
drops every 8th image, so the same index would silently name a different frame in each arm.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_scene(ckpt_dir, split="all"):
    """Shared foam setup. VisOptions is a warp struct and ZERO-initialises, so every field it uses
    must be set explicitly -- a bare VisOptions() renders black, which cost a day previously."""
    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
    pr = configargparse.ArgParser()
    add_group(pr, Params)
    pr.add_argument("-c", "--config", is_config_file=True)
    args = pr.parse_args(["-c", os.path.join(ckpt_dir, "config.yaml")])
    dh = DataHandler(args)
    dh.reload(split, downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(os.path.join(ckpt_dir, "model.pt"))
    return model, dh, args


def render_view(model, camera, colours, bg=(1.0, 1.0, 1.0)):
    """PowerfoamScene.forward with texel_rgb replaced.

    colours: either (N,3) -- one flat colour per primitive, broadcast to every texel site -- or
    (N,S,3), already per-site. The per-site form matters: averaging the appearance model over the
    S sites before rasterising discards all within-cell detail, which turns the background into
    per-primitive blobs while the shipped renderer stays sharp on the same view.
    """
    # Argument list copied from PowerfoamScene.forward (scene.py:412-426) rather than reconstructed:
    # the rasteriser takes camera, depth_quantiles, points, radii, density, normals, texel_sites,
    # texel_rgb, texel_height, adjacency, adjacency_offsets, ray_gt, return_point_err -- in that
    # order. Guessing a shorter signature raised "missing 5 required positional arguments".
    normals = model.get_normals()
    tangents, bitangent = model.get_tangents()
    radii = model.get_radii()

    # model.texel_sites is (N, S, 2) UV in the tangent frame, NOT world positions. forward() lifts
    # it to 3-D before handing it to the rasteriser; passing the raw UV array straight through gets
    # "Could not convert array interface with shape (N, 8, 2) ... ensure inner shape is (3,)".
    offsets = model.texel_sites * radii[:, None, None]
    offsets = (offsets[..., 0:1] * tangents[:, None, :]
               + offsets[..., 1:2] * bitangent[:, None, :])
    texel_sites = model.points[:, None, :] + offsets

    texel_height = model.texel_height * radii[:, None]

    n_sites = model.args.num_texel_sites
    if colours.dim() == 3:
        texel_rgb = colours.contiguous()
    else:
        texel_rgb = colours[:, None, :].expand(-1, n_sites, -1).contiguous()

    # visualize() rather than forward(): it exposes vis_options, and therefore bkgd_color.
    # THIS MATTERS FOR THE FROZEN ARM. truefrozen has exactly one primitive per GT point, so
    # wherever the ScanNet scan has a hole there is no geometry at all and rays composite against
    # the background. With the default black that reads as a rendering artefact rather than as
    # missing data -- confirmed by rendering the same view through unmodified model.forward(), which
    # shows the identical black patch. White makes a hole look like a hole.
    # forward() returns only color_out (no alpha), so compositing after the fact is not an option.
    import warp as wp
    from powerfoam.rasterize import VisOptions
    vis = VisOptions()
    vis.transmittance_threshold = 1e-3
    vis.max_intersections = 1024
    vis.depth_quantile = 0.5
    vis.bkgd_color = wp.vec3f(float(bg[0]), float(bg[1]), float(bg[2]))

    return model.rasterizer.visualize(
        camera,
        model.points,
        radii,
        model.get_density(),
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        model.adjacency,
        model.adjacency_offsets,
        vis_options=vis,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir with model.pt + config.yaml")
    ap.add_argument("--features", required=True)
    ap.add_argument("--class-names", required=True)
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-json", default=None)
    ap.add_argument("--split", default="all",
                    help="MUST be 'all' to match the 3DGS arm's frame order")
    ap.add_argument("--no-opacity-mask", action="store_true",
                    help="disable the alpha<0.1 rule (alpha = 1-exp(-sigma*2r))")
    # PER-PRIMITIVE IS THE DEFAULT: it is the readout of the reported arm
    # (percell-argmax+opacitymask@0.1), so the figure and the table describe the same method.
    # Pooling is a codebook borrowed from OpenGaussian, which needs it because Gaussians overlap;
    # foam rays are near one-hot (1.01 primitives/ray vs 7.8), so its cells need no such smoothing.
    ap.add_argument("--bg", default="1,1,1",
                    help="background r,g,b in 0-1; frozen arms have real holes where the scan does, and black reads as an artefact")
    ap.add_argument("--pooled", action="store_true",
                    help="use the kmeans320 pooled readout instead of per-primitive argmax")
    a = ap.parse_args()

    import seg_palette as P
    P.enable_determinism()

    model, dh, args = build_scene(a.ckpt, a.split)
    sv = torch.load(a.features, map_location="cpu", weights_only=True)
    feats, vm = sv["primitive_features"], sv["valid_mask"].numpy().astype(bool)
    n = model.points.shape[0]
    if feats.shape[0] != n:
        raise SystemExit("row mismatch: %d features vs %d primitives -- wrong solve for this ckpt"
                         % (feats.shape[0], n))

    # ROW COUNT IS NOT ENOUGH. Every frozen-style arm has exactly one primitive per GT point, so
    # `frozen` and `truefrozen` both report 72,007 on scene0097_00 while being entirely different
    # reconstructions (mean per-point displacement 2.07 in a room a few metres across). Pairing a
    # truefrozen solve with a frozen checkpoint passes the count check and renders as dense
    # confetti, which is easy to misread as a property of the arm rather than a loading error.
    # There is no provenance inside the solve file, so guard on the naming convention.
    _ck = os.path.basename(os.path.normpath(a.ckpt)).lower()
    _ft = os.path.basename(a.features).lower()
    for _arm in ("truefrozen", "nonfrozen", "unfroz", "tfroz"):
        if _arm in _ft and _arm not in _ck:
            print("[warn] features look like '%s' but the checkpoint is '%s' -- if these are "
                  "different reconstructions the labels are applied to the wrong cells."
                  % (_arm, _ck))
            break

    # FOAM'S OPACITY RULE IS NOT sigmoid(logit). run_purity_miou.py:87-88 derives it as the optical
    # depth through the cell:
    #     sigma = softplus(density, beta=100);  alpha = 1 - exp(-max(sigma,0) * 2 * radius)
    # so a large low-density cell can pass the 0.1 threshold while a small dense one fails. Using the
    # Gaussian form here would mask the wrong cells, and foam radii vary enough for that to matter.
    alpha = None
    if not a.no_opacity_mask:
        sd = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu", weights_only=False)
        sigma = torch.nn.functional.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
        radii_np = model.get_radii().detach().cpu().numpy().reshape(-1)
        alpha = 1.0 - np.exp(-np.maximum(sigma, 0.0) * 2.0 * radii_np)

    names = [s.strip() for s in a.class_names.split(",") if s.strip()]
    col, cls, meta = P.primitive_colours(feats, vm, names, device="cuda",
                                         pooled=a.pooled, opacity=alpha)

    if not (0 <= a.view < len(dh.cameras)):
        raise SystemExit("view %d out of range (%d cameras in split=%s)"
                         % (a.view, len(dh.cameras), a.split))
    cam = dh.cameras[a.view]

    # BACKGROUND KEEPS FOAM'S REAL APPEARANCE, mirroring the 3DGS side (which uses the SH DC term).
    #
    # There is no get_base_rgb on PowerfoamScene -- an earlier version of this file guessed that name
    # and silently fell back to a flat 0.6 grey, which made the whole room uniform while the 3DGS
    # panel kept its shading. That asymmetry made foam look far worse than its labels justify.
    #
    # Foam's appearance is the spherical-lobe model forward() evaluates to build texel_rgb:
    #     sv.forward(texel_sites, camera, *get_att_sv())
    # Evaluated once at the render camera and averaged over the texel sites, it is the foam analogue
    # of the DC term: a per-primitive base colour. Desaturated to luma exactly as the 3DGS arm is.
    col = col.cuda()
    n_sites = model.args.num_texel_sites
    with torch.no_grad():
        tangents, bitangent = model.get_tangents()
        radii = model.get_radii()
        off = model.texel_sites * radii[:, None, None]
        off = off[..., 0:1] * tangents[:, None, :] + off[..., 1:2] * bitangent[:, None, :]
        sites = model.points[:, None, :] + off
        att_sites, att_values, att_temps = model.get_att_sv()
        # PER TEXEL SITE, exactly as forward() builds texel_rgb. An earlier version averaged this
        # over the S sites to get one colour per primitive; that average is what made the background
        # render as per-primitive blobs while the shipped renderer stayed sharp on the same view.
        # The appearance detail within a cell lives entirely in the site-to-site variation.
        base = model.sv.forward(sites.view(-1, 3).detach(), cam,
                                att_sites, att_values, att_temps)
        base = base.view(model.points.shape[0], n_sites, 3).clamp(0, 1)
    shade = P.to_shade(base)                                 # real detail, desaturated (shared)

    # Compositing is seg_palette.composite_three_case -- shared with the 3DGS arm so the two cannot
    # drift apart. They already did twice, each time making one arm look worse for reasons unrelated
    # to the method.
    texel_rgb = P.composite_three_case(col, cls, meta, shade)

    with torch.no_grad():
        bg = tuple(float(x) for x in a.bg.split(","))
        out = render_view(model, cam, texel_rgb.contiguous(), bg=bg)
    img = out[0] if isinstance(out, (tuple, list)) else out
    arr = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))[..., :3]

    from PIL import Image
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(arr).save(a.out)
    meta.update({"arm": "powerfoam", "view": a.view, "split": a.split,
                 "size": [int(cam.width), int(cam.height)],
                 "features": os.path.basename(a.features)})
    P.write_labels_json(a.labels_json or (os.path.splitext(a.out)[0] + "_labels.json"), meta)
    print("wrote %s  (view %d of %d, split=%s)" % (a.out, a.view, len(dh.cameras), a.split))
    print("  classes with primitives:", ", ".join(meta["classes_with_primitives"]) or "(none)")


if __name__ == "__main__":
    main()
