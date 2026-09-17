"""Tests for the segmentation mask generator, plus a rasteriser-vs-raytracer cross-check.

THE QUESTION THESE ANSWER. The point-cloud diagnostic says 62% of GT points get the right class,
yet the rendered masks look considerably worse. Either the mask generator has a bug, or the two
views weight primitives differently. These tests separate those:

  T1-T5 are unit tests of the mask generator on synthetic data with a KNOWN answer -- they fail if
        the label->colour mapping, the readout, determinism, or the opacity rule is wrong.
  T6    renders the SAME labels through both shipped backends. The rasteriser and the raytracer are
        independent implementations, so if they agree the image is not a renderer artefact.
  T7    is the actual hypothesis test: it compares the class histogram weighted by PROJECTED IMAGE
        AREA against the one weighted by GT POINTS OWNED. A cell contributes to the picture in
        proportion to how much of the screen it covers, but to the metric in proportion to how many
        GT points fall inside it. If a small number of large, wrongly-labelled cells covers a large
        share of the screen, the render looks worse than the score -- with no bug anywhere.

Run:  python test_seg_masks.py                 (T1-T5, fast, no checkpoint needed)
      python test_seg_masks.py --scene-tests   (adds T6-T7, needs a checkpoint + solve)
"""
import argparse
import sys

import numpy as np
import torch

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


# --------------------------------------------------------------------------------------------
# Synthetic fixture: C classes, each with an unmistakable feature direction, so the correct
# per-primitive answer is known by construction and any deviation is a real bug.
# --------------------------------------------------------------------------------------------
def synthetic(n_per_class=50, dim=512, seed=0):
    import seg_palette as P
    names = ["wall", "floor", "cabinet", "sink"]
    g = torch.Generator().manual_seed(seed)
    txt = P.__dict__.get("_dummy_txt")
    basis = torch.zeros(len(names), dim)
    for i in range(len(names)):
        basis[i, i * 7] = 1.0
    feats, truth = [], []
    for i in range(len(names)):
        f = basis[i].repeat(n_per_class, 1) + 0.01 * torch.randn(n_per_class, dim, generator=g)
        feats.append(f)
        truth += [i] * n_per_class
    return names, torch.cat(feats), np.array(truth), basis


def t1_to_t5():
    import seg_palette as P
    P.enable_determinism()
    names, feats, truth, basis = synthetic()
    n = feats.shape[0]
    vm = np.ones(n, dtype=bool)

    # Text embeddings come from CLIP inside primitive_colours, which will NOT align with our
    # synthetic basis. So T1/T2 assert INTERNAL CONSISTENCY -- that the colour shown equals the
    # label returned, and that the label equals an independently recomputed argmax -- rather than
    # asserting a particular class. A generator bug breaks those regardless of what CLIP says.
    col, cls, meta = P.primitive_colours(feats, vm, names, device="cuda", pooled=False)
    col, cls = col.numpy(), cls.numpy()

    # T1: every primitive carrying class c shows exactly the hue registered for c, and nothing else.
    ok = True
    for c, nm in enumerate(names):
        m = cls == c
        if not m.any():
            continue
        want = meta["hue_slots"].get(nm) or meta.get("background_hues", {}).get(nm)
        if want is None:
            ok = False
            break
        ok &= np.allclose(col[m], np.asarray(want, dtype=np.float32), atol=1e-5)
    check("T1 label -> colour mapping is exact and injective", ok)

    # T2: grey appears ONLY where there is no prediction.
    grey = np.asarray(P.GREY, dtype=np.float32)
    is_grey = np.all(np.isclose(col, grey, atol=1e-5), axis=1)
    check("T2 grey iff no prediction", bool(np.array_equal(is_grey, cls < 0)),
          "grey=%d, unpredicted=%d" % (is_grey.sum(), (cls < 0).sum()))

    # T3: the per-primitive readout is exactly argmax of cosine against the text embeddings.
    from evaluate_point_cloud_miou import embed_class_names
    txt = torch.nn.functional.normalize(embed_class_names(names, "cuda").float(), dim=-1)
    unit = torch.nn.functional.normalize(feats.float().cuda(), dim=-1)
    ref = (unit @ txt.T).argmax(-1).cpu().numpy()
    check("T3 readout == independent cosine argmax", bool(np.array_equal(cls, ref)),
          "%d/%d differ" % (int((cls != ref).sum()), n))

    # T4: determinism -- identical inputs must give identical labels.
    col2, cls2, _ = P.primitive_colours(feats, vm, names, device="cuda", pooled=False)
    check("T4 deterministic across calls", bool(np.array_equal(cls, cls2.numpy())))

    # T5: the opacity rule. Primitives below threshold must lose their prediction, and those above
    # must be untouched -- this is the rule that was applied with the wrong semantics once.
    op = np.ones(n, dtype=np.float32)
    op[: n // 2] = 0.01
    _, cls_op, meta_op = P.primitive_colours(feats, vm, names, device="cuda", pooled=False,
                                             opacity=op, opacity_threshold=0.1)
    cls_op = cls_op.numpy()
    lo_cleared = bool(np.all(cls_op[: n // 2] == -1))
    hi_kept = bool(np.array_equal(cls_op[n // 2:], cls[n // 2:]))
    check("T5 opacity < threshold clears prediction, above is untouched",
          lo_cleared and hi_kept,
          "cleared=%s kept=%s reported=%d" % (lo_cleared, hi_kept, meta_op["n_opacity_masked"]))


# --------------------------------------------------------------------------------------------
def scene_tests(ckpt, features, view, class_names, gt_dir):
    import seg_palette as P
    import warp as wp
    from point_cloud_query import assign_points_to_power_cells
    from powerfoam.rasterize import VisOptions
    from render_seg_powerfoam import build_scene
    P.enable_determinism()

    names = [s.strip() for s in class_names.split(",") if s.strip()]
    model, dh, args = build_scene(ckpt, "all")
    sv = torch.load(features, map_location="cpu", weights_only=True)
    feats, vm = sv["primitive_features"], sv["valid_mask"].numpy().astype(bool)
    col, cls, meta = P.primitive_colours(feats, vm, names, device="cuda", pooled=False)
    cls_np = cls.numpy()
    cam = dh.cameras[view]

    # Shared geometry, built exactly as PowerfoamScene.forward does.
    normals = model.get_normals()
    tangents, bitangent = model.get_tangents()
    radii = model.get_radii()
    off = model.texel_sites * radii[:, None, None]
    off = off[..., 0:1] * tangents[:, None, :] + off[..., 1:2] * bitangent[:, None, :]
    sites = model.points[:, None, :] + off
    texel_height = model.texel_height * radii[:, None]
    n_sites = model.args.num_texel_sites

    # ONE FLAT COLOUR PER CLASS so a rendered pixel can be decoded back to a class id. Using an
    # identifiable palette (not the figure hues) makes the decode unambiguous.
    n_cls = len(names)
    key = torch.zeros(n_cls + 1, 3, device="cuda")
    for i in range(n_cls):
        key[i] = torch.tensor([(i + 1) / (n_cls + 1), 1.0 - (i + 1) / (n_cls + 1), 0.5])
    key[n_cls] = torch.tensor([0.0, 0.0, 0.0])
    lab = torch.as_tensor(np.where(cls_np < 0, n_cls, cls_np), device="cuda", dtype=torch.long)
    texel_rgb = key[lab][:, None, :].expand(-1, n_sites, -1).contiguous()

    vis = VisOptions()
    vis.transmittance_threshold = 1e-3
    vis.max_intersections = 256
    vis.depth_quantile = 0.5
    vis.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    imgs = {}
    for mode, renderer in (("rasterize", model.rasterizer), ("raytrace", model.raytracer)):
        try:
            with torch.no_grad():
                out = renderer.visualize(cam, model.points, radii, model.get_density(), normals,
                                         sites, texel_rgb, texel_height, model.adjacency,
                                         model.adjacency_offsets, vis_options=vis)
            im = out[0] if isinstance(out, (tuple, list)) else out
            a = im.detach().float().clamp(0, 1).cpu().numpy()
            if a.ndim == 3 and a.shape[0] in (3, 4):
                a = np.transpose(a, (1, 2, 0))
            imgs[mode] = a[..., :3]
        except Exception as e:
            # NOTE (verified 2026-09-12): RayTracer in this checkout defines only __init__ and
            # benchmark -- it has no visualize/forward, so scene.forward_visualization's
            # render_mode="raytrace" branch is dead code and the raytracer cannot serve as an
            # independent cross-check. Left in place so this re-tests automatically if that lands.
            print("  [warn] %s backend unavailable: %s: %s" % (mode, type(e).__name__, e))

    def decode(img):
        k = key.cpu().numpy()[None, None, :, :]
        d = np.linalg.norm(img[:, :, None, :] - k, axis=-1)
        lab = d.argmin(-1)
        lab[d.min(-1) > 0.25] = n_cls          # blended/background pixels are not a class
        return lab

    # T6: two independent renderer implementations, same labels -> same picture.
    if len(imgs) == 2:
        la, lb = decode(imgs["rasterize"]), decode(imgs["raytrace"])
        both = (la < n_cls) & (lb < n_cls)
        agree = float((la[both] == lb[both]).mean()) if both.any() else float("nan")
        check("T6 rasteriser and raytracer agree on class per pixel", agree > 0.90,
              "agreement %.3f over %d classified pixels" % (agree, int(both.sum())))
    else:
        print("  [skip] T6 needs both backends")

    # T7: THE HYPOTHESIS. Screen area per class vs GT points owned per class.
    if "rasterize" in imgs and gt_dir:
        lab_img = decode(imgs["rasterize"])
        gt_pts = np.load(gt_dir + "/coord.npy").astype(np.float64)
        centers = model.points.detach().cpu().numpy()
        rad = radii.detach().cpu().numpy()
        assigned = assign_points_to_power_cells(gt_pts, centers, rad, valid=vm, k=64)
        owned = assigned >= 0
        pt_cls = np.full(len(gt_pts), -1, dtype=np.int64)
        pt_cls[owned] = cls_np[assigned[owned]]

        print("\n  class            screen-area%%   GT-point%%   ratio")
        rows = []
        for i, nm in enumerate(names):
            area = float((lab_img == i).mean()) * 100.0
            pts = float((pt_cls == i).mean()) * 100.0
            rows.append((nm, area, pts))
            print("  %-14s %10.2f %11.2f %8s" % (nm, area, pts,
                  ("%.2fx" % (area / pts)) if pts > 0.05 else "-"))
        skew = max((a / p) for _, a, p in rows if p > 0.05) if rows else 1.0
        check("T7 screen area and GT-point share are within 3x for every class", skew < 3.0,
              "worst class is %.1fx over-represented on screen" % skew)
        print("\n  -> A large skew means the render and the metric weight primitives differently:")
        print("     the image is area-weighted, the score is point-weighted. That is not a bug.")


def t8_compositing():
    """The two renderers must composite identically. They each had a private copy once and drifted
    twice -- first a flat grey vs the real appearance, then background classes tinted in foam but
    painted flat in 3DGS, which cost 3DGS all its wall/floor texture in a side-by-side figure."""
    import numpy as _np
    import seg_palette as P

    names = ["wall", "floor", "cabinet", "sink"]
    meta = {"background_class_idx": [0, 1]}          # wall, floor
    cls = _np.array([-1, 0, 1, 2, 3])
    col = _np.array([P.GREY, (0.72, 0.70, 0.62), (0.62, 0.68, 0.72),
                     (0.9, 0.2, 0.2), (0.2, 0.4, 0.9)], dtype=_np.float32)
    shade_np = _np.full((5, 3), 0.5, dtype=_np.float32)

    out_np = P.composite_three_case(col, cls, meta, shade_np)
    check("T8a no-prediction keeps the shading untouched",
          bool(_np.allclose(out_np[0], shade_np[0])))
    check("T8b foreground classes are the flat hue",
          bool(_np.allclose(out_np[3], col[3]) and _np.allclose(out_np[4], col[4])))
    bg_expected = _np.clip(shade_np[1] * col[1] * P.BG_TINT_GAIN, 0, 1)
    check("T8c background classes are shading tinted by the hue, not flat",
          bool(_np.allclose(out_np[1], bg_expected) and not _np.allclose(out_np[1], col[1])))

    # The (N,S,3) torch path the foam arm uses must agree with the (N,3) numpy path the 3DGS arm
    # uses, given the same inputs -- that agreement IS the anti-drift guarantee.
    t_shade = torch.as_tensor(shade_np)[:, None, :].expand(-1, 8, -1).contiguous()
    out_t = P.composite_three_case(torch.as_tensor(col), torch.as_tensor(cls), meta, t_shade)
    same = _np.allclose(out_t[:, 0, :].cpu().numpy(), out_np, atol=1e-6)
    check("T8d torch (N,S,3) path == numpy (N,3) path", bool(same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-tests", action="store_true")
    ap.add_argument("--ckpt", default="output/scannet_scene0000_00_nonfrozen")
    ap.add_argument("--features", default="artifacts/scannet_figs/solved_scene0000_00.pt")
    ap.add_argument("--gt-dir", default="D:/Downloads/scannet_pointcept/train/scene0000_00")
    ap.add_argument("--view", type=int, default=140)
    ap.add_argument("--class-names", default="wall,floor,cabinet,curtain,sofa,table,bed,desk")
    a = ap.parse_args()

    print("== mask generator unit tests ==")
    t1_to_t5()
    print("\n== shared compositing (anti-drift) ==")
    t8_compositing()
    if a.scene_tests:
        print("\n== renderer cross-check and area-vs-point weighting ==")
        scene_tests(a.ckpt, a.features, a.view, a.class_names, a.gt_dir)

    print("\n%d failure(s)%s" % (len(FAILS), (": " + ", ".join(FAILS)) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
