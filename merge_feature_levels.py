"""Stack the three single-level ScanNet++ extractions into one multi-level LangSplat-format set.

WHY. LaGa and LangSplat both expect ONE language_features directory whose <img>_s.npy is (L, H, W)
and whose values are GLOBAL row indices into a single <img>_f.npy. Our extraction produced three
SEPARATE single-level directories (openclip_features_sam_{s,m,l3}), each with a (1, H, W) map
indexing its own _f.npy. LaGa's `sam_masks[lvl]` therefore hits "index 1 is out of bounds for
dimension 0 with size 1".

Stacking them is not an approximation: concatenating the per-level embedding tables and offsetting
each level's segment ids by the cumulative row count of the preceding levels reproduces exactly what
a single multi-level run would have written. The pixel-to-embedding correspondence is preserved
per level; only the row numbering changes.

LEVEL ORDER is s, m, l -- coarse-to-fine as LangSplat orders its granularities, so a consumer
indexing lvl=0,1,2 walks the same direction the upstream format implies.

NEGATIVE IDS (-1 = no mask at this pixel) are preserved as -1 and never offset, since they are a
sentinel rather than an index.
"""
import argparse
import os

import numpy as np

LEVELS = ["s", "m", "l3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\Downloads\spp_data_1600")
    ap.add_argument("--out-name", default="language_features_multi")
    ap.add_argument("--out-root", default=None,
                    help="write under <out-root>/<scene>/<out-name> instead of beside the inputs. "
                         "Use the datawaha mount: 995's local disk has been filled by this kind of "
                         "output three times, and the merged set is larger than its inputs.")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images per scene")
    a = ap.parse_args()

    scenes = a.scenes or sorted(d for d in os.listdir(a.root)
                                if os.path.isdir(os.path.join(a.root, d)))
    for scene in scenes:
        sd = os.path.join(a.root, scene)
        dirs = [os.path.join(sd, "openclip_features_sam_" + lv) for lv in LEVELS]
        if not all(os.path.isdir(d) for d in dirs):
            print("[miss] {}: not all levels present".format(scene))
            continue
        out = (os.path.join(a.out_root, scene, a.out_name) if a.out_root
               else os.path.join(sd, a.out_name))
        os.makedirs(out, exist_ok=True)

        stems = sorted(f[:-6] for f in os.listdir(dirs[0]) if f.endswith("_f.npy"))
        if a.limit:
            stems = stems[:a.limit]
        done = skipped = 0
        for stem in stems:
            fo = os.path.join(out, stem + "_f.npy")
            so = os.path.join(out, stem + "_s.npy")
            if os.path.exists(fo) and os.path.exists(so):
                skipped += 1
                continue
            feats, maps, offset, ok = [], [], 0, True
            for d in dirs:
                fp, sp = os.path.join(d, stem + "_f.npy"), os.path.join(d, stem + "_s.npy")
                if not (os.path.exists(fp) and os.path.exists(sp)):
                    ok = False
                    break
                f = np.load(fp)
                s = np.load(sp)
                if s.ndim == 3 and s.shape[0] == 1:
                    s = s[0]
                # Match the SOURCE dtype (int32). Defaulting to int64 doubled every map, which on
                # top of stacking 3 levels made the merged set 6x its inputs and filled the disk.
                # Ids are per-image segment counts (max seen: 146), so int32 is ample.
                s = s.astype(np.int32)
                shifted = np.where(s >= 0, s + offset, -1).astype(np.int32)  # -1 sentinel, never offset
                feats.append(f)
                maps.append(shifted)
                offset += f.shape[0]
            if not ok:
                continue
            np.save(fo, np.concatenate(feats, 0))
            np.save(so, np.stack(maps, 0))
            done += 1
        print("[ok] {}: wrote {}, already-had {}, levels={} -> {}".format(
            scene, done, skipped, len(LEVELS), a.out_name), flush=True)


if __name__ == "__main__":
    main()
