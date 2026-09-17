"""Determine, from the stored artifacts alone, which mask-fill / crop-pad colours produced a
SAM+CLIP feature set.

WHY THIS IS NEEDED. `FILL_VALUE`/`PAD_VALUE` in splat-distiller's pre_processing.py are read from
the environment at import time and are recorded NOWHERE in the outputs -- not in the feature
manifest, not beside the .npy files. So "which protocol were these extracted under" cannot be
answered from provenance, only from the numbers. File dates and current defaults are inference, not
evidence: the default itself changed during this project (white fill + black pad -> black + black,
to match LangSplat/OpenGaussian, which is what their published numbers come from).

HOW IT IS ANSWERED WITHOUT RE-RUNNING SAM. `_s.npy` stores the per-level mask-id map, and
`mask2segmap` writes `seg_map[mask_i.segmentation] = i` -- so the pixel set of mask i is recoverable
as `s[level] == i`, and row i of that level's block in `_f.npy` is its CLIP embedding. Rebuilding the
crop through the identical path (fill outside mask inside bbox, square pad, cv2.resize to 224,
/255, CLIP normalize, encode, L2) under each candidate colour pair and correlating against the
stored vector identifies the protocol. No SAM inference is involved.

WHY THE TEST IS READ COMPARATIVELY, NOT AS AN ABSOLUTE MATCH. `seg_map` is written in mask order, so
a mask overlapped by a later one loses those pixels from the map, and its recovered bounding box can
be slightly tighter than the `mask['bbox']` SAM actually passed to the cropper. That perturbs every
candidate equally, so the verdict is which colour pair wins on cosine, aggregated over many masks --
not whether any single cosine reaches exactly 1.0. A fill-colour difference moves the embedding far
more than a few pixels of bbox slack, and the margin below makes that visible rather than assumed.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\splat-distiller")

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def level_blocks(s, n_rows):
    """Confirm the mask ids in `_s.npy` are GLOBAL row indices into `_f.npy`, not per-level ones.

    Measured on figurines/frame_00001: the per-level id ranges run 0-66, 0-189, 0-247, 0-273 with
    64/121/56/26 distinct ids, i.e. each level's ids continue where the previous level left off and
    the last level ends exactly at n_rows-1. So a level's map indexes `_f.npy` directly and no
    per-level offset applies -- adding one (the natural first guess) pushes every row out of bounds.
    """
    counts = [int((np.unique(s[L]) >= 0).sum()) for L in range(s.shape[0])]
    hi = int(s.max())
    if hi != n_rows - 1:
        print(f"[warn] max mask id {hi} but _f.npy has {n_rows} rows -- ids may not be global",
              flush=True)
    return [0] * s.shape[0], counts


def build_crop(img, seg, fill, pad):
    """The splat-distiller crop path, reproduced exactly: fill, tight bbox, square pad, 224."""
    import cv2

    ys, xs = np.where(seg)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = img[y0:y1, x0:x1].copy()
    crop[~seg[y0:y1, x0:x1]] = fill
    h, w, _ = crop.shape
    l = max(h, w)
    sq = np.full((l, l, 3), pad, dtype=np.uint8)
    if h > w:
        sq[:, (h - w) // 2:(h - w) // 2 + w, :] = crop
    else:
        sq[(w - h) // 2:(w - h) // 2 + h, :, :] = crop
    return cv2.resize(sq, (224, 224))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir holding <stem>_f.npy / <stem>_s.npy")
    ap.add_argument("--images", required=True)
    ap.add_argument("--stem", required=True, help="e.g. frame_00001")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-masks", type=int, default=24, help="largest-area masks to test")
    a = ap.parse_args()

    import cv2
    import open_clip

    f = np.load(os.path.join(a.features, f"{a.stem}_f.npy")).astype(np.float32)
    s = np.load(os.path.join(a.features, f"{a.stem}_s.npy"))
    img = cv2.cvtColor(cv2.imread(os.path.join(a.images, a.stem + a.ext)), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != s.shape[1:]:
        img = cv2.resize(img, (s.shape[2], s.shape[1]))
    print(f"{a.stem}: image {img.shape}, seg {s.shape}, feats {f.shape}", flush=True)

    offs, counts = level_blocks(s, f.shape[0])
    L, off = a.level, offs[a.level]
    ids, areas = np.unique(s[L][s[L] >= 0], return_counts=True)
    order = ids[np.argsort(-areas)][:a.n_masks]
    print(f"level {L}: {counts[L]} masks, block offset {off}; testing {len(order)} largest",
          flush=True)

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16")
    model = model.eval().to("cuda")
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    combos = [("black fill + black pad  (LangSplat/OpenGaussian)", 0, 0),
              ("white fill + black pad  (splat-distiller default)", 255, 0),
              ("white fill + white pad", 255, 255),
              ("black fill + white pad", 0, 255)]
    scores = {name: [] for name, _, _ in combos}
    for k in order:
        row = off + int(k)
        if row >= f.shape[0]:
            continue
        ref = torch.tensor(f[row], device="cuda")
        ref = ref / ref.norm()
        seg = s[L] == k
        for name, fill, pad in combos:
            crop = build_crop(img, seg, fill, pad)
            if crop is None:
                continue
            t = torch.from_numpy(crop.astype("float32")).permute(2, 0, 1)[None].cuda() / 255.0
            t = ((t - mean) / std).half()
            with torch.no_grad():
                e = model.encode_image(t).float()[0]
            scores[name].append(float((e / e.norm()) @ ref))

    print(f"\ncosine to the STORED embedding, over {len(scores[combos[0][0]])} masks:", flush=True)
    ranked = sorted(combos, key=lambda c: -np.mean(scores[c[0]]))
    for name, _, _ in ranked:
        v = np.array(scores[name])
        print(f"  {name:52s} mean {v.mean():.4f}  median {np.median(v):.4f}  min {v.min():.4f}")
    best, second = ranked[0][0], ranked[1][0]
    wins = int((np.array(scores[best]) > np.array(scores[second])).sum())
    n = len(scores[best])
    print(f"\nVERDICT: {best}\n  beats runner-up on {wins}/{n} masks, "
          f"mean margin {np.mean(scores[best]) - np.mean(scores[second]):+.4f}")


if __name__ == "__main__":
    main()
