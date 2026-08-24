"""Measured A/B test: LangSplat BLACK mask-surround fill vs splat-distiller WHITE fill.

Reuses the REAL SAM masks already extracted for scene0347_00 (reconstructed from the
saved `*_s.npy` seg maps, level 3 = 'l' = the large level OpenGaussian uses), rebuilds
each crop both ways through the exact same get_seg_img/pad_img/resize code path, and
embeds both with OpenCLIP ViT-B-16 / laion2b_s34b_b88k on CPU.

CAVEAT: seg maps are written with `seg_map[mask['segmentation']] = i` in a loop, so a
mask that is later overwritten by an overlapping mask is recovered slightly ERODED.
Geometry is real SAM geometry; boundaries where masks overlap are approximate.
"""
import argparse, os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open_clip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, OPENGAUSSIAN_NAME_OVERRIDES, CLIP_MODEL, CLIP_PRETRAINED,
)

FEAT_DIR_T = r"D:\Downloads\powerfoam\artifacts\scannet\{s}\openclip_features_sam"
IMG_DIR_T = r"D:\Downloads\powerfoam\data\scannet\{s}_colmap\images"


# ---- verbatim copies of the two implementations (do NOT import; repo must stay untouched)
def get_seg_img(seg_bool, bbox, image, fill):
    image = image.copy()
    image[seg_bool == 0] = np.array(fill, dtype=np.uint8)
    x, y, w, h = np.int32(bbox)
    return image[y:y + h, x:x + w, ...]


def pad_img(img, pad_value=0):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.full((l, l, 3), pad_value, dtype=np.uint8)   # np.zeros when pad_value=0
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def build_tile(seg_bool, bbox, image, fill, pad_value):
    seg = get_seg_img(seg_bool, bbox, image, fill)
    if seg.size == 0:
        return None
    return cv2.resize(pad_img(seg, pad_value), (224, 224))


def level_masks(seg_maps, level=3):
    """Recover (bool_mask, bbox) for every mask id present in the given level."""
    sm = seg_maps[level].astype(np.int32)
    ids = np.unique(sm)
    ids = ids[ids >= 0]
    out = []
    for i in ids:
        m = sm == i
        ys, xs = np.nonzero(m)
        if ys.size < 64:            # skip degenerate leftovers from the overwrite erosion
            continue
        x, y = xs.min(), ys.min()
        w, h = xs.max() - x + 1, ys.max() - y + 1
        out.append((m, (x, y, w, h)))
    return out


def embed(tiles, model, bs=32):
    feats = []
    for i in range(0, len(tiles), bs):
        arr = np.stack(tiles[i:i + bs], 0).astype("float32")
        t = torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0   # exactly the repo path
        with torch.no_grad():
            f = model.encode_image(t).float()
        feats.append(F.normalize(f, dim=-1))
    return torch.cat(feats, 0)


def report(name, sim, classes):
    top2 = sim.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])
    mean_per_class = sim.mean(0)
    std_per_class = sim.std(0)
    win = sim.argmax(1)
    print(f"\n===== {name}  (N={sim.shape[0]} crops) =====")
    print(f"  cos-to-text overall: min {sim.min():.4f}  mean {sim.mean():.4f}  max {sim.max():.4f}"
          f"   [range of per-crop MAX: {sim.max(1).values.min():.4f}..{sim.max(1).values.max():.4f}]")
    print(f"  top1-top2 margin: mean {margin.mean():.4f}  median {margin.median():.4f}"
          f"  frac<0.01 {(margin < 0.01).float().mean() * 100:.1f}%  frac<0.02 {(margin < 0.02).float().mean() * 100:.1f}%")
    order = torch.argsort(mean_per_class, descending=True)
    print("  per-class  mean / std / win%   (sorted by mean):")
    for k in order.tolist():
        wp = (win == k).float().mean() * 100
        print(f"    {classes[k]:<16s} {mean_per_class[k]:.4f}  {std_per_class[k]:.4f}  {wp:5.2f}%")
    # attractor score: how often the highest-mean class also wins
    top_mean_cls = order[0].item()
    print(f"  ATTRACTOR CHECK: highest-mean class = '{classes[top_mean_cls]}'"
          f" ({mean_per_class[top_mean_cls]:.4f}); its win share = {(win == top_mean_cls).float().mean() * 100:.2f}%")
    print(f"  spread of per-class means: {mean_per_class.min():.4f}..{mean_per_class.max():.4f}"
          f" (span {mean_per_class.max() - mean_per_class.min():.4f})")
    return margin, mean_per_class, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--scene", default="scene0347_00")
    args = ap.parse_args()

    torch.set_num_threads(4)
    global FEAT_DIR, IMG_DIR
    FEAT_DIR = FEAT_DIR_T.format(s=args.scene)
    IMG_DIR = IMG_DIR_T.format(s=args.scene)
    print("scene:", args.scene)
    classes = [OPENGAUSSIAN_NAME_OVERRIDES.get(n, n)
               for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"]]
    print("classes:", classes)

    print(f"loading {CLIP_MODEL} / {CLIP_PRETRAINED} on CPU ...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.eval()
    tok = open_clip.get_tokenizer(CLIP_MODEL)
    with torch.no_grad():
        text = F.normalize(model.encode_text(tok(classes)).float(), dim=-1)

    stems = sorted({f.rsplit("_", 1)[0] for f in os.listdir(FEAT_DIR) if f.endswith("_s.npy")},
                   key=lambda s: int(s))[:args.num_images]
    tiles_b, tiles_w = [], []
    for stem in stems:
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            print("  missing image", stem); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sm = np.load(os.path.join(FEAT_DIR, stem + "_s.npy"))
        for m, bb in level_masks(sm, args.level):
            tb = build_tile(m, bb, img, (0, 0, 0), 0)          # LangSplat: black fill, black pad
            tw = build_tile(m, bb, img, (255, 255, 255), 0)    # splat-distiller: WHITE fill, black pad
            if tb is None or tw is None:
                continue
            tiles_b.append(tb); tiles_w.append(tw)
    print(f"built {len(tiles_b)} crop pairs from {len(stems)} images (level {args.level})")

    fb = embed(tiles_b, model)
    fw = embed(tiles_w, model)

    pair_cos = (fb * fw).sum(1)
    print(f"\n### BLACK-fill vs WHITE-fill embedding cosine, SAME mask (N={len(pair_cos)})")
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        print(f"    q{q:<5.2f} {torch.quantile(pair_cos, q):.4f}")
    print(f"    mean {pair_cos.mean():.4f}  std {pair_cos.std():.4f}")

    mb, _, wb = report("BLACK fill (LangSplat)", fb @ text.T, classes)
    mw, _, ww = report("WHITE fill (splat-distiller)", fw @ text.T, classes)
    print(f"\nargmax class agreement black vs white: {(wb == ww).float().mean() * 100:.1f}%")

    # extra control: white fill AND white padding (removes the two-background boundary)
    tiles_ww = []
    for stem in stems:
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sm = np.load(os.path.join(FEAT_DIR, stem + "_s.npy"))
        for m, bb in level_masks(sm, args.level):
            t = build_tile(m, bb, img, (255, 255, 255), 255)
            if t is not None: tiles_ww.append(t)
    fww = embed(tiles_ww, model)
    report("WHITE fill + WHITE pad (control)", fww @ text.T, classes)
    print(f"\ncosine white/blackpad vs white/whitepad: mean {(fw * fww).sum(1).mean():.4f}")


if __name__ == "__main__":
    main()
