"""Compose the qualitative segmentation figure: rows = representation, columns = scene x view.

The legend is built from each renderer's labels.json and FILTERED TO CLASSES THAT ACTUALLY EARNED
PRIMITIVES, so the figure never advertises a class it does not show -- a legend entry with nothing
coloured reads as a failure of the method rather than an absence of the object. A single
"background / unassigned" grey swatch is appended, covering wall/floor/ceiling, hue overflow past
eight slots, and primitives whose features were unusable (valid_mask == False).

The two arms render at different resolutions (3DGS 1752x1168 from cameras.json, foam 1600x1066 from
max_image_width). Panels are resized to a common height here purely for layout; nothing is cropped,
so the same content is visible in both rows.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_panel(path):
    img = Image.open(path).convert("RGB")
    meta_p = os.path.splitext(path)[0] + "_labels.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    return img, meta


def _font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True,
                    help="NAME=img1.png,img2.png[,...]  (repeat; one per representation)")
    ap.add_argument("--col-labels", default=None, help="comma-separated column headers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=520, help="per-panel height in the figure")
    ap.add_argument("--pad", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for spec in a.row:
        name, _, files = spec.partition("=")
        paths = [p for p in files.split(",") if p]
        panels = [load_panel(p) for p in paths]
        rows.append((name, panels))

    ncol = max(len(p) for _, p in rows)
    H = a.height
    scaled = []
    for name, panels in rows:
        imgs = []
        for img, meta in panels:
            w = max(1, int(round(img.width * H / img.height)))
            imgs.append((img.resize((w, H), Image.LANCZOS), meta))
        scaled.append((name, imgs))

    colw = [0] * ncol
    for _, imgs in scaled:
        for j, (im, _) in enumerate(imgs):
            colw[j] = max(colw[j], im.width)

    f_row, f_col, f_leg = _font(20), _font(18), _font(17)
    left = 150
    top = 34 if a.col_labels else 0

    # legend: union over every panel of classes that actually earned primitives
    hue_of, seen = {}, []
    for _, imgs in scaled:
        for _, meta in imgs:
            for nm in meta.get("classes_with_primitives", []):
                h = meta.get("hue_slots", {}).get(nm)
                if h and nm not in hue_of:
                    hue_of[nm] = h
                    seen.append(nm)
    leg_h = 40 + 26 * ((len(seen) + 1 + 3) // 4)

    Wt = left + sum(colw) + a.pad * (ncol + 1)
    Ht = top + len(scaled) * (H + a.pad) + a.pad + leg_h
    canvas = Image.new("RGB", (Wt, Ht), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    if a.col_labels:
        for j, lab in enumerate(a.col_labels.split(",")[:ncol]):
            x = left + a.pad + sum(colw[:j]) + a.pad * j
            d.text((x, 8), lab.strip(), fill=(20, 20, 20), font=f_col)

    for i, (name, imgs) in enumerate(scaled):
        y = top + a.pad + i * (H + a.pad)
        d.text((10, y + H // 2 - 10), name, fill=(20, 20, 20), font=f_row)
        for j, (im, _) in enumerate(imgs):
            x = left + a.pad + sum(colw[:j]) + a.pad * j
            canvas.paste(im, (x, y))

    ly = top + len(scaled) * (H + a.pad) + a.pad + 8
    x, col = left, 0
    for nm in seen + ["background / unassigned"]:
        rgb = hue_of.get(nm, (0.60, 0.60, 0.585))
        d.rectangle([x, ly + 3, x + 20, ly + 23],
                    fill=tuple(int(255 * c) for c in rgb), outline=(90, 90, 90))
        d.text((x + 27, ly + 4), nm, fill=(20, 20, 20), font=f_leg)
        col += 1
        x += (Wt - left) // 4
        if col % 4 == 0:
            x, ly = left, ly + 26

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    canvas.save(a.out)
    print("wrote %s  (%d rows x %d cols, %d legend entries)" % (a.out, len(scaled), ncol, len(seen) + 1))


if __name__ == "__main__":
    main()
