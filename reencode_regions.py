"""NEW naming evidence for 3D regions: project each foam region into the views that see it, crop
the actual projected footprint, and re-encode it with CLIP.

WHY THIS AND NOT MORE AGGREGATION. Our diagnostics say the dominant error is NAMING, not grouping:
83% of errors are interior cells of coherent regions, SAM regions are 92.6% pure while CLIP names
them correctly only 49.2% of the time. That is a per-region BIAS -- every view of the region
contributes the same wrong embedding -- and no robust average over biased evidence recovers the
right name. Geometric median, ROFA, trimming and shrinkage have all been measured on this and all
sit within noise, which is exactly what a bias predicts. The only fix is new evidence.

WHAT IS NEW HERE. A region's existing feature is a weighted average of SAM-MASK embeddings: the 2D
masks were chosen by SAM, so a region straddling two masks inherits a blend of both. Here the crop
is the region's OWN projected footprint -- the 3D grouping decides the 2D extent, not SAM -- so CLIP
sees the object the region actually represents. For a region that is geometrically right but was
named from a mask covering the wrong thing, that is genuinely independent evidence.

THE CONTROL IS EXACT. Scoring the SAME regions with their existing pooled features is
new-evidence vs averaged-old-evidence over an identical partition, so any difference is
attributable to the evidence rather than to the grouping.

CROP PROTOCOL matches the extraction it is compared against (verify_crop_padding.py established it
from the artifacts, since FILL/PAD are read from the environment and recorded nowhere): fill
outside the mask inside the bbox, pad to square, resize to 224, /255, CLIP-normalize, encode, L2.
Default is black fill + black pad, which is what LangSplat/OpenGaussian report under.
"""
import argparse
import os

import configargparse
import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

import sys
import warp as wp

sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")
sys.path.insert(0, "D:/Downloads/powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def make_crop(img, mask, fill, pad):
    """Crop `img` to `mask`'s bbox under the extraction's fill/pad protocol; None if mask empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = img[y0:y1, x0:x1].astype(np.float32).copy()
    m = mask[y0:y1, x0:x1]
    sub[~m] = fill
    h, w = sub.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), pad, np.float32)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = sub
    out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
    return (out - MEAN) / STD


def render_region_map(m, dh, c, rgbz, vi, lab, alpha_eps):
    """Per-pixel region id for view `vi` (-1 where no primitive is in front at alpha >= eps)."""
    with torch.no_grad():
        out = m.rasterizer.visualize(dh.cameras[vi], c["points"], c["radii"], c["density"],
                                     c["normals"], c["texel_sites"], rgbz, c["texel_height"],
                                     c["adjacency"], c["adjacency_offsets"])
    alpha, fpi = out[3], out[7].long()
    H, W = alpha.shape[-2], alpha.shape[-1]
    alpha = alpha.reshape(H, W)
    fpi = fpi.reshape(H, W)
    ok = (alpha >= alpha_eps) & (fpi >= 0)
    rl = torch.full((H, W), -1, dtype=torch.long, device=alpha.device)
    if bool(ok.any()):
        rl[ok] = lab[fpi[ok]]
    return rl, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--recon", default="nonfrozen")
    ap.add_argument("--regions", required=True, help="solve .pt carrying `labels` (the partition)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-px", type=int, default=64, help="skip a region in a view below this")
    ap.add_argument("--max-views", type=int, default=8, help="top-N views per region by pixel count")
    ap.add_argument("--alpha-eps", type=float, default=0.5)
    ap.add_argument("--fill", default="black", choices=["black", "white"])
    ap.add_argument("--pad", default="black", choices=["black", "white"])
    ap.add_argument("--batch", type=int, default=64)
    # CROP-QUALITY GATES. Measured on the reg=0.03 partition: crops had median fill ratio 0.325,
    # median 8 connected components, 88% fragmented, median 0.37% of the frame -- scattered confetti,
    # not objects, because that partition is 4770 fragments (median size 1 primitive) over 244k
    # primitives. CLIP cannot name that, and re-encoding scored 0.2517 vs a 0.4044 control.
    # A crop only carries independent naming evidence if it looks like a thing, so gate on it and
    # fall back to the existing pooled feature wherever the gate fails.
    ap.add_argument("--min-fill", type=float, default=0.5, help="mask pixels / bbox area")
    ap.add_argument("--max-comps", type=int, default=2, help="connected components allowed")
    ap.add_argument("--min-frac", type=float, default=0.002, help="min fraction of the frame")
    a = ap.parse_args()
    dev = "cuda"
    wp.init()
    fill = 0.0 if a.fill == "black" else 255.0
    pad = 0.0 if a.pad == "black" else 255.0

    ck = "output/scannet_%s_%s" % (a.scene, a.recon)
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", "%s/config.yaml" % ck])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args)
    m.initialize_from_dataset(dh, device=dev)
    m.load_pt("%s/model.pt" % ck)
    m.update_vis_cache()

    sv = torch.load(a.regions, map_location="cpu", weights_only=True)
    if "labels" not in sv:
        raise SystemExit("--regions must carry `labels` (use a Cut Pursuit output)")
    lab = sv["labels"].to(dev)
    K = int(lab.max()) + 1
    print("[regions] %d regions over %d primitives" % (K, lab.numel()), flush=True)

    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to(dev).eval()

    names = sorted(os.listdir("data/scannet/%s_colmap/images" % a.scene))
    c = m._vis_cache
    rgbz = torch.zeros(c["points"].shape[0], m.args.num_texel_sites, 3, device=dev)

    # pass 1 -- which (region, view) pairs are worth cropping, ranked by projected pixel count
    cand = {}
    for vi in range(len(names)):
        rl, _, _ = render_region_map(m, dh, c, rgbz, vi, lab, a.alpha_eps)
        vis = rl[rl >= 0]
        if vis.numel() == 0:
            continue
        ids, cnt = torch.unique(vis, return_counts=True)
        for r, n in zip(ids.tolist(), cnt.tolist()):
            if n >= a.min_px:
                cand.setdefault(r, []).append((n, vi))
        if vi % 40 == 0:
            print("  scan view %d/%d" % (vi, len(names)), flush=True)

    todo = {}
    for r, lst in cand.items():
        for n, vi in sorted(lst, reverse=True)[:a.max_views]:
            todo.setdefault(vi, []).append(r)
    print("[plan] %d/%d regions visible, %d crops"
          % (len(cand), K, sum(len(v) for v in todo.values())), flush=True)

    acc = torch.zeros(K, 512, device=dev)
    wsum = torch.zeros(K, device=dev)
    buf_img, buf_r = [], []
    n_rej = [0]

    def flush():
        if not buf_img:
            return
        x = torch.from_numpy(np.stack(buf_img)).permute(0, 3, 1, 2).float().to(dev)
        with torch.no_grad():
            e = F.normalize(clip.encode_image(x).float(), dim=-1)
        idx = torch.tensor(buf_r, device=dev)
        acc.index_add_(0, idx, e)
        wsum.index_add_(0, idx, torch.ones(len(buf_r), device=dev))
        buf_img.clear()
        buf_r.clear()

    for k, vi in enumerate(sorted(todo)):
        img = np.asarray(Image.open("data/scannet/%s_colmap/images/%s"
                                    % (a.scene, names[vi])).convert("RGB"))
        rl, H, W = render_region_map(m, dh, c, rgbz, vi, lab, a.alpha_eps)
        rl_np = rl.cpu().numpy()
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        for r in todo[vi]:
            mk = rl_np == r
            ys, xs = np.nonzero(mk)
            if ys.size == 0:
                continue
            bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            ncomp = cv2.connectedComponents(mk.astype(np.uint8))[0] - 1
            if (ys.size / bb) < a.min_fill or ncomp > a.max_comps or (ys.size / (H * W)) < a.min_frac:
                n_rej[0] += 1
                continue
            cr = make_crop(img, mk, fill, pad)
            if cr is None:
                continue
            buf_img.append(cr)
            buf_r.append(r)
            if len(buf_img) >= a.batch:
                flush()
        if k % 20 == 0:
            print("  crop view %d/%d" % (k, len(todo)), flush=True)
    flush()

    seen = wsum > 0
    reg_feat = torch.zeros(K, 512, device=dev)
    reg_feat[seen] = F.normalize(acc[seen] / wsum[seen].unsqueeze(1), dim=-1)
    prim = reg_feat[lab]
    prim[~seen[lab]] = 0
    torch.save({"primitive_features": prim.cpu(), "valid_mask": seen[lab].cpu(),
                "region_features": reg_feat.cpu(), "region_seen": seen.cpu(),
                "labels": lab.cpu(), "num_segments": K, "views_per_region": wsum.cpu()},
               a.output)
    print("[done] %d/%d regions re-encoded, mean %.1f views each, %d crops rejected by gates -> %s"
          % (int(seen.sum()), K, float(wsum[seen].mean()) if bool(seen.any()) else 0.0,
             n_rej[0], a.output))


if __name__ == "__main__":
    main()
