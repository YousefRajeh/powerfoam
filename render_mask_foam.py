"""Per-class mask renders through PowerFoam's OWN rasteriser.

WHY NOT THE ORTHOGRAPHIC SPLAT. `render_label_compare.py` projects GT points with a painter's
algorithm. That is fine for a layout overview but it is not what the method produces: it ignores
the foam's actual geometry, opacity and view-dependent shading, and a top-down projection is
dominated by the floor. This drives the shipped rasteriser (`PowerfoamScene.forward_visualization`
-> `Rasterizer.visualize`) with the real cameras, so what is shown is what the representation
renders.

COLOURING. The rasteriser consumes `texel_rgb`, a per-primitive per-texel colour, so the class
overlay is applied there rather than as a post-process:

  * primitives predicted as the QUERIED class  -> a saturated class colour
  * every other primitive                      -> its ORIGINAL colour, DESATURATED to its own
                                                  luminance

The second point is the whole trick. Painting non-class primitives a flat grey erases the room:
geometry, furniture and unlabelled structure all collapse into one slab and the mask floats in a
void with no context. Keeping each primitive's own luminance leaves the scene fully legible in
greyscale -- you still see the unlabelled things -- while colour is reserved entirely for the mask.
`--grey-mix` blends between flat grey (0) and full original luminance (1).

Opacity is untouched, so nothing is made to "cover" anything: the mask is occluded by geometry in
front of it exactly as it should be.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import configargparse
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from graphcut import multiclass_potts_icm, binary_graphcut
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import load_scannet_pointcept_gt

GT_ROOT = r"D:\Downloads\scannet_pointcept"
CLASS_RGB = (1.0, 0.15, 0.15)          # default overlay colour


def luminance(rgb):
    """Rec. 709 luma, kept per-primitive so structure survives the desaturation."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0097_00")
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--cls", required=True, help="class name to highlight")
    ap.add_argument("--feat", default=None)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--views", default="0", help="comma-separated camera indices")
    ap.add_argument("--grey-mix", type=float, default=1.0,
                    help="0 = flat grey (erases the room), 1 = full original luminance")
    ap.add_argument("--grey-gain", type=float, default=0.85)
    ap.add_argument("--color", default=None, help="r,g,b in 0-1 for the overlay")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cut", type=float, default=0.005,
                    help="Potts weight for the SINGLE-QUERY binary min-cut; 0 disables")
    ap.add_argument("--cut-t", type=float, default=0.21,
                    help="score threshold the cut is taken around")
    a = ap.parse_args()
    feat_file = a.feat or f"solved_geometric_median_{a.recon}_ogl3"
    dev = "cuda"
    wp.init()

    ckpt = f"output/scannet_{a.scene}_{a.recon}"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ckpt}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ckpt}/model.pt")
    model.update_vis_cache()

    d = torch.load(f"artifacts/scannet/{a.scene}/{feat_file}.pt", map_location=dev,
                   weights_only=True)
    feats = d["primitive_features"].to(dev).float()
    valid = d["valid_mask"].cpu().numpy()

    cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", a.scene)) if os.path.isdir(p)]
    _, raw_labels, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw_labels).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if name_to_id[n] in present]
    assert a.cls in kept, f"'{a.cls}' not present in {a.scene}; have {kept}"
    k = kept.index(a.cls)
    text = embed_class_names(kept, dev)
    sim = (F.normalize(feats, dim=-1) @ text.T).cpu().numpy()

    g = torch.load(f"artifacts/ablation_cache/{a.scene}_pf_"
                   f"{'tfroz' if a.recon == 'truefrozen' else 'nonfroz'}_delaunay.pt",
                   map_location="cpu", weights_only=False)
    indptr = g["offsets"].numpy().astype(np.int64)
    indices = g["adjacent"].numpy().astype(np.int64)

    # Three arms, all scored per-primitive on the SAME features:
    #   plain  -- argmax over the class set (the unary optimum)
    #   potts  -- multi-class smoothing, the variant comparable to the reported mIoU
    #   cut    -- the SINGLE-QUERY binary min-cut for THIS class alone, which is the arm that
    #             tied the best method on IoU while cutting fragmentation 9.4x. It is exact
    #             (submodular binary Potts) and needs no other class, so it is the one a
    #             one-query-at-a-time system would actually run.
    labs = {"plain": sim.argmax(1) == k,
            f"potts lam={a.lam}": multiclass_potts_icm(sim, indptr, indices, lam=a.lam,
                                                       live=valid.astype(bool)) == k}
    if a.cut > 0:
        labs[f"binary cut lam={a.cut}"] = binary_graphcut(
            sim[:, k], a.cut_t, indptr, indices, lam=a.cut, subset=valid.astype(bool))
    overlay = torch.tensor([float(x) for x in a.color.split(",")] if a.color else CLASS_RGB,
                           device=dev)

    # --- replicate forward_visualization, but substitute texel_rgb ---
    cache = model._vis_cache
    points, radii = cache["points"], cache["radii"]
    density, normals = cache["density"], cache["normals"]
    texel_sites, texel_height = cache["texel_sites"], cache["texel_height"]
    att_sites, att_values, att_temps = cache["att_sites"], cache["att_values"], cache["att_temps"]
    adjacency, adjacency_offsets = cache["adjacency"], cache["adjacency_offsets"]

    views = [int(v) for v in a.views.split(",")]
    rows = []
    for vi in views:
        cam = dh.cameras[vi]
        with torch.no_grad():
            rgb = model.sv.forward(texel_sites.view(-1, 3).detach(), cam,
                                   att_sites, att_values, att_temps)
            rgb = rgb.view(points.shape[0], model.args.num_texel_sites, 3)

            panels = []
            # panel 0: the scene as it actually renders, for reference
            panels.append(("render", rgb.clone()))
            for name, lab in labs.items():
                lum = luminance(rgb).unsqueeze(-1).expand_as(rgb)
                grey = a.grey_gain * (a.grey_mix * lum + (1.0 - a.grey_mix) * 0.5)
                out = grey.clone()
                sel = torch.from_numpy(lab & valid).to(dev)
                out[sel] = overlay.view(1, 1, 3).expand(int(sel.sum()),
                                                        model.args.num_texel_sites, 3)
                panels.append((f"{name}  ({int(sel.sum()):,} cells)", out))

            imgs = []
            for title, trgb in panels:
                col, *_ = model.rasterizer.visualize(
                    cam, points, radii, density, normals, texel_sites, trgb.contiguous(),
                    texel_height, adjacency, adjacency_offsets)
                imgs.append((title, col.clamp(0, 1).cpu().numpy()))
        rows.append((vi, imgs))

    n = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), n, figsize=(5.2 * n, 4.4 * len(rows)), squeeze=False)
    for r, (vi, imgs) in enumerate(rows):
        for c, (title, im) in enumerate(imgs):
            axes[r][c].imshow(im)
            axes[r][c].set_title(f"view {vi} | {title}", fontsize=10)
            axes[r][c].axis("off")
    fig.suptitle(f"{a.scene} / {a.recon} -- class '{a.cls}' via PowerFoam's rasteriser",
                 fontsize=13)
    fig.tight_layout()
    out = a.out or f"artifacts/scannet/{a.scene}_mask_{a.cls}_lam{a.lam}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
