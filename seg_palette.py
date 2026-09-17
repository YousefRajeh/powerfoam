"""Shared colour/label logic for the qualitative segmentation figures.

BOTH RENDERERS CALL THIS so the two arms cannot drift apart. The colour a primitive gets is decided
once, here, from the solved CLIP features; the renderers only rasterise it with their own engine.

THE LABELLING PATH IS THE mIoU HARNESS PATH, not a second implementation:
    embed_class_names        -> CLIP text embeddings (evaluate_point_cloud_miou)
    spherical_kmeans(K=320)  -> feature-space groups (diagnose_scannet_miou, the feat_kmeans320 arm)
    pool_classify_broadcast  -> one class per group, broadcast to members (run_cluster_classify_eval)
Pooled, NOT per-primitive. Per-primitive argmax is measurably the weakest readout (A_base scores
below every pooled variant on ScanNet++) and produces visibly wrong renders -- a room that is ~48%
wall had its walls labelled "toilet" per-primitive.

GT IS USED ONLY TO CHOOSE THE VOCABULARY, never to colour anything. Classes come from the scene's
own ScanNet++ annotation ranked by GT SURFACE AREA, not vertex count: vertex density reflects scan
resolution, so counting vertices over-ranks finely-tessellated small objects.

valid_mask == False primitives are zeroed so they fall to grey rather than being argmax'd into a
class they have no evidence for.

DETERMINISM: index_add_ in the pooling step is nondeterministic on CUDA and shifted ~2% of labels
between runs, so callers must enable torch.use_deterministic_algorithms(True) with
CUBLAS_WORKSPACE_CONFIG=:4096:8 before calling in. enable_determinism() here does that.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CVD-validated categorical hues, in fixed slot order so a class keeps its colour across scenes and
# across the two arms. Eight foreground slots; anything past that falls to grey.
_HUES = [
    (0.902, 0.624, 0.000),   # amber
    (0.337, 0.706, 0.914),   # sky
    (0.000, 0.620, 0.451),   # green
    (0.941, 0.894, 0.259),   # yellow
    (0.000, 0.447, 0.698),   # blue
    (0.835, 0.369, 0.000),   # vermillion
    (0.800, 0.475, 0.655),   # mauve
    (0.580, 0.404, 0.741),   # purple
    # --- extension past the 8 CVD-validated Okabe-Ito hues -------------------------------------
    # A ScanNet++ scene has 20+ GT classes. With only 8 slots every class past the 8th fell through
    # to GREY, i.e. silently rendered as "no prediction" -- the figure would have understated
    # coverage rather than shown it. These 12 keep maximal spacing in hue while alternating
    # lightness so neighbouring slots stay distinguishable side by side; they are NOT claimed to be
    # colour-blind-safe the way the first eight are, so keep the most important classes in slots
    # 0-7 (class_names_by_gt_area already orders by GT area, which does exactly that).
    # Chosen by farthest-point sampling in RGB against the eight above, restricted to chromatic
    # colours (std > 0.12) with luma in [0.22, 0.82] and >0.32 from GREY. Hand-picking these first
    # produced pairs 0.018 apart -- visually identical, so two classes would have been
    # indistinguishable in the figure. Minimum pairwise distance across all 20 is now 0.247.
    (0.333, 1.000, 0.000),
    (1.000, 0.000, 1.000),
    (0.250, 0.333, 0.000),
    (1.000, 0.000, 0.417),
    (0.500, 0.917, 0.500),
    (0.250, 0.167, 1.000),
    (0.000, 1.000, 0.750),
    (0.583, 0.167, 0.333),
    (1.000, 0.667, 1.000),
    (0.000, 0.667, 0.000),
    (0.000, 1.000, 0.250),
    (0.500, 0.583, 0.250),
]
GREY = (0.60, 0.60, 0.585)

# Deliberately classified, then routed to grey: they dominate the vocabulary by area and would
# otherwise consume most of the hue slots, leaving the objects the figure is about uncoloured.
BACKGROUND = {"wall", "floor", "ceiling"}

# Background classes get MUTED HUES, not grey. Previously wall/floor were painted with GREY, which
# made a correct `wall` prediction pixel-identical to an invalid or opacity-masked primitive -- so
# the figure could not distinguish "predicted background" from "no prediction at all", and read as
# cleaner than the labels actually are. Grey is now reserved exclusively for "no prediction".
# These are desaturated on purpose: the renderer TINTS the real shading with them rather than
# painting flat, so the room keeps its appearance while its predictions stay legible.
_BG_HUES = {
    "wall":    (0.72, 0.70, 0.62),
    "floor":   (0.62, 0.68, 0.72),
    "ceiling": (0.70, 0.66, 0.72),
}

# NOT OBJECTS. ScanNet++ annotations carry segmentation markers whose labels rank high by area --
# on 09c1414f1b, "SPLIT" (35.55 m2) and "split" (17.05 m2) are the two largest foreground labels,
# above carpet and sofa. They would take two of the eight hue slots AND be embedded as CLIP text
# prompts, where they carry no visual meaning and attract arbitrary primitives. The scored mIoU
# harness never sees them because it maps through ScanNet++'s top100 benchmark list; only an
# area-ranked vocabulary built straight from segments_anno.json picks them up.
NON_OBJECT = {"split", "remove", "removed", "unknown", "undefined", "invalid", "none", ""}


def enable_determinism(seed: int = 0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)


def class_names_by_gt_area(scene, gt_root, top_k=8):
    """Top-k BENCHMARK classes present in this scene, ranked by GT surface area.

    VOCABULARY COMES FROM ScanNet++'s OWN BENCHMARK LIST, not from raw annotation text:
        metadata/semantic_benchmark/top100.txt        the official class set, frequency-ordered
        metadata/semantic_benchmark/map_benchmark.csv semantic_map_to: open vocabulary -> benchmark
    This is the same pair run_spp_eval.py::benchmark_map() reads, so the figure now shows exactly the
    classes the scored mIoU is computed over. Mining segments_anno.json directly instead put raw
    labels in the vocabulary -- on 09c1414f1b the two largest "foreground" labels by area were
    "SPLIT" (35.55 m2) and "split" (17.05 m2), annotation markers rather than objects, which would
    have taken two of the eight hue slots and been embedded as meaningless CLIP prompts.

    AREA, NOT VERTEX COUNT: ScanNet++ meshes are non-uniformly tessellated, so vertex count measures
    scan resolution as much as object size. Triangle area is summed per class by assigning each face
    the label of its majority vertex.

    Ranking is still per scene -- a scene only advertises classes it actually contains -- but every
    name is guaranteed to be a benchmark class.
    """
    from plyfile import PlyData
    d = os.path.join(gt_root, scene, "scans")
    ply = PlyData.read(os.path.join(d, "mesh_aligned_0.05.ply"))
    v = ply["vertex"]
    pts = np.stack([np.asarray(v[k]) for k in ("x", "y", "z")], 1).astype(np.float64)
    faces = np.stack(ply["face"].data["vertex_indices"])

    # ScanNet++'s own benchmark vocabulary and its open-vocabulary -> benchmark folding
    import csv as _csv
    meta = os.path.join(gt_root, "metadata", "semantic_benchmark")
    top = [l.strip() for l in open(os.path.join(meta, "top100.txt")) if l.strip()]
    top_set = set(top)
    raw2bench = {}
    with open(os.path.join(meta, "map_benchmark.csv")) as fh:
        for row in _csv.DictReader(fh):
            tgt = (row.get("semantic_map_to") or "").strip()
            if tgt:
                raw2bench[row["class"].strip()] = tgt

    seg = json.load(open(os.path.join(d, "segments.json")))
    anno = json.load(open(os.path.join(d, "segments_anno.json")))
    vert_seg = np.asarray(seg["segIndices"], dtype=np.int64)
    seg2label = {}
    for g in anno["segGroups"]:
        lab = g["label"].strip()
        lab = raw2bench.get(lab, lab)            # fold onto the benchmark vocabulary
        if lab not in top_set:                   # anything off-benchmark (incl. SPLIT) is dropped
            lab = ""
        for s in g["segments"]:
            seg2label[int(s)] = lab
    vert_label = np.array([seg2label.get(int(s), "") for s in vert_seg], dtype=object)

    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    # majority vertex label per face
    fl = vert_label[faces]
    face_label = np.where(fl[:, 0] == fl[:, 1], fl[:, 0],
                          np.where(fl[:, 0] == fl[:, 2], fl[:, 0], fl[:, 1]))

    tot = {}
    for lab, ar in zip(face_label, area):
        if lab:
            tot[lab] = tot.get(lab, 0.0) + float(ar)
    ranked = [k for k, _ in sorted(tot.items(), key=lambda kv: -kv[1])]
    dropped = [k for k in ranked if k.strip().lower() in NON_OBJECT]
    if dropped:
        print("  [vocab] dropping non-object labels: %s" % ", ".join(dropped))
    ranked = [k for k in ranked if k.strip().lower() not in NON_OBJECT]
    fg = [k for k in ranked if k not in BACKGROUND][:top_k]
    present_bg = [k for k in ("wall", "floor", "ceiling") if k in tot]
    return present_bg + fg, tot


OPACITY_THRESHOLD = 0.1   # OpenGaussian's rule; do not change


def primitive_colours(features, valid_mask, class_names, device="cuda", pooled=True, k_groups=320,
                      seed=0, opacity=None, opacity_threshold=OPACITY_THRESHOLD):
    """-> (colours (N,3) float, labels (N,) int with -1 for grey, meta dict).

    pooled=True is the mIoU-harness readout. pooled=False is the per-primitive argmax and exists
    only so the difference can be shown; it is not what the figure should use.
    """
    from evaluate_point_cloud_miou import embed_class_names
    from run_cluster_classify_eval import pool_classify_broadcast
    from diagnose_scannet_miou import spherical_kmeans

    X = features.float().to(device)
    vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    n = X.shape[0]

    # OpenGaussian's low-opacity rule (eval_scannet.py:127-129, sigmoid(opacity) < 0.1). Applied
    # here so the figure matches the documented protocol: a near-transparent primitive contributes
    # almost nothing to the image but still OWNS its share of surface, so colouring it by class
    # makes a render look noisier than the underlying labels are. Masked primitives fall to grey
    # exactly like valid_mask == False ones -- they are not argmax'd into a class.
    n_opacity_masked = 0
    if opacity is not None:
        op = torch.as_tensor(opacity, dtype=torch.float32, device=device).reshape(-1)
        if op.numel() != n:
            raise ValueError("opacity has %d entries, expected %d" % (op.numel(), n))
        low = op < opacity_threshold
        n_opacity_masked = int((low & vm).sum())
        vm = vm & ~low

    unit = torch.zeros_like(X)
    unit[vm] = torch.nn.functional.normalize(X[vm], dim=-1)   # invalid rows stay exactly zero

    txt = torch.nn.functional.normalize(embed_class_names(list(class_names), device).float(), dim=-1)

    if pooled:
        # init="fps": farthest-point sampling in cosine distance. The default "randperm" draws seeds
        # uniformly over primitives, so seeds follow DENSITY -- with wall+floor at ~55% of points in
        # these scenes, most of the 320 seeds land in background modes and a rare class (sink 1.1%,
        # counter 2.5%) may get none. A class with no seed is absorbed into a dominant cluster and
        # INHERITS its label, which is exactly how a small object disappears from the figure. FPS
        # spreads seeds across feature modes, so a rare-but-distinct class still gets a cluster.
        groups, _ = spherical_kmeans(unit[vm], k_groups, seed=seed, init="fps")
        full = torch.full((n,), -1, dtype=torch.long, device=device)
        full[vm] = groups
        cls = torch.full((n,), -1, dtype=torch.long, device=device)
        sub = pool_classify_broadcast(groups, unit[vm], int(groups.max()) + 1, txt)
        cls[vm] = sub if sub.ndim == 1 else sub.argmax(-1)
    else:
        cls = torch.full((n,), -1, dtype=torch.long, device=device)
        cls[vm] = (unit[vm] @ txt.T).argmax(-1)

    # hue slots: background classes and overflow past 8 hues route to grey
    fg = [i for i, nm in enumerate(class_names) if nm not in BACKGROUND][:len(_HUES)]
    slot = {c: i for i, c in enumerate(fg)}

    col = torch.tensor(GREY, device=device, dtype=torch.float32).repeat(n, 1)

    def _fill(mask, rgb):
        # Explicit expand to (k,3). Assigning a bare (3,) into a boolean-masked (N,3) slice trips
        # deterministic indexing: "number of flattened indices did not match number of elements".
        k = int(mask.sum())
        if k:
            col[mask] = torch.tensor(rgb, device=device, dtype=torch.float32).unsqueeze(0).expand(k, 3)

    for c, i in slot.items():
        _fill(cls == c, _HUES[i])
    # Background-class predictions are painted with their muted hue so they are visible as
    # predictions. Only primitives with NO prediction stay GREY.
    bg_idx = {i: nm for i, nm in enumerate(class_names) if nm in BACKGROUND}
    for c, nm in bg_idx.items():
        _fill(cls == c, _BG_HUES.get(nm, GREY))
    _fill(~vm, GREY)

    meta = {
        "class_names": list(class_names),
        "hue_slots": {class_names[c]: _HUES[i] for c, i in slot.items()},
        "background": sorted(BACKGROUND & set(class_names)),
        "background_hues": {nm: _BG_HUES[nm] for nm in sorted(BACKGROUND & set(class_names))
                            if nm in _BG_HUES},
        "background_class_idx": sorted(i for i, nm in enumerate(class_names) if nm in BACKGROUND),
        "kmeans_init": "fps" if pooled else None,
        "n_no_prediction": int((~vm).sum()),
        "grey": list(GREY),
        "pooled": bool(pooled),
        "k_groups": int(k_groups) if pooled else None,
        "n_primitives": int(n),
        "n_valid": int(vm.sum()),
        "opacity_threshold": float(opacity_threshold) if opacity is not None else None,
        "n_opacity_masked": int(n_opacity_masked),
        "classes_with_primitives": sorted(
            {class_names[c] for c in torch.unique(cls).tolist() if c >= 0}),
    }
    return col.cpu(), cls.cpu(), meta


BG_TINT_GAIN = 1.35
SHADE_GAIN, SHADE_LIFT = 0.85, 0.10


def to_shade(rgb):
    """Real appearance -> the desaturated backdrop both arms draw on.

    Shared for the same reason as composite_three_case: the gain and lift are what set how dark the
    unlabelled room reads, and if the two arms used different constants one representation would
    look dimmer than the other for no methodological reason. Each arm still SOURCES its own rgb
    (foam evaluates the spherical-lobe model per texel site; 3DGS uses the SH DC term) -- only the
    conversion is shared. Accepts torch or numpy, (N,3) or (N,S,3); returns the same type/shape.
    """
    import numpy as _np
    import torch as _torch
    w = (0.2126, 0.7152, 0.0722)
    if isinstance(rgb, _np.ndarray):
        lum = (rgb * _np.asarray(w, dtype=rgb.dtype)).sum(-1, keepdims=True)
        return _np.repeat(lum, 3, axis=-1) * SHADE_GAIN + SHADE_LIFT
    lum = (rgb * _torch.tensor(w, device=rgb.device, dtype=rgb.dtype)).sum(-1, keepdim=True)
    return lum.expand_as(rgb) * SHADE_GAIN + SHADE_LIFT


def composite_three_case(col, cls, meta, shade, bg_gain=BG_TINT_GAIN):
    """THE single implementation of how class colours combine with the rendered appearance.

    Both renderers MUST call this. They previously each had their own copy, and they drifted twice:
    once when the foam arm evaluated the appearance model while the 3DGS arm used a flat grey, and
    again when background classes gained muted hues and only the foam arm learned to tint them --
    leaving 3DGS with flat, textureless walls and floors for reasons that had nothing to do with
    3DGS. An unfair comparison produced by a figure script is worse than no figure.

    Three cases:
      no prediction (cls < 0) -> `shade` unchanged; the ONLY neutral grey in the image
      background class        -> `shade` TINTED by the muted hue, so texture and depth survive
      foreground class        -> the flat class hue, stable from any view direction

    Accepts torch tensors or numpy arrays, and `shade` of shape (N,3) or (N,S,3); `col` is (N,3).
    Returns the same type and shape as `shade`.
    """
    import numpy as _np
    import torch as _torch

    was_numpy = isinstance(shade, _np.ndarray)
    t_shade = _torch.as_tensor(shade) if was_numpy else shade
    dev = t_shade.device
    t_col = _torch.as_tensor(col).to(dev) if isinstance(col, _np.ndarray) else col.to(dev)
    t_cls = _torch.as_tensor(_np.asarray(cls) if isinstance(cls, _np.ndarray) else cls).to(dev)
    t_shade = t_shade.float()
    t_col = t_col.float()

    if t_shade.dim() == 3:                      # (N, S, 3) per-texel-site
        col_b = t_col[:, None, :].expand_as(t_shade)
        sel = lambda m: m[:, None, None]        # noqa: E731
    else:                                       # (N, 3) per-primitive
        col_b = t_col
        sel = lambda m: m[:, None]              # noqa: E731

    bg_idx = list(meta.get("background_class_idx", []))
    is_none = t_cls < 0
    is_bg = (_torch.isin(t_cls, _torch.as_tensor(bg_idx, device=dev))
             if bg_idx else _torch.zeros_like(is_none))

    tinted = (t_shade * col_b * bg_gain).clamp(0, 1)
    out = _torch.where(sel(is_bg), tinted, _torch.where(sel(is_none), t_shade, col_b))
    return out.cpu().numpy() if was_numpy else out


def write_labels_json(path, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(meta, open(path, "w"), indent=1)
