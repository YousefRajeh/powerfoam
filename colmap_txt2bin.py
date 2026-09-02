"""COLMAP sparse model: text -> binary.

WHY. splat-distiller reaches pycolmap's `scene_manager._load_images_txt`, which does
`np.array(map(float, data[1:5]))` -- a Python-2 idiom that under py3 hands `Quaternion` a 0-d map
object and raises "Input quaternion should be a 3- or 4-vector". ScanNet never hit it because its
`sparse/0` is BINARY; ScanNet++ ships TEXT, so the broken reader is the one that runs.

Converting the model is preferable to patching site-packages: it leaves the environment intact,
it is verifiable against the text source, and every other consumer (the RadFoam lift, PowerFoam's
loader) already reads either format, so nothing else changes. Binary layout per COLMAP's
`src/colmap/scene/reconstruction.cc`.
"""
import argparse
import os
import struct


def _w(f, fmt, *v):
    f.write(struct.pack("<" + fmt, *v))


def cameras(src, dst):
    rows = [l.split() for l in open(src) if l.strip() and not l.startswith("#")]
    MODEL = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
             "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6, "FOV": 7,
             "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10}
    with open(dst, "wb") as f:
        _w(f, "Q", len(rows))
        for r in rows:
            _w(f, "i", int(r[0])); _w(f, "i", MODEL[r[1]])
            _w(f, "Q", int(r[2])); _w(f, "Q", int(r[3]))
            for p in r[4:]:
                _w(f, "d", float(p))
    return len(rows)


def images(src, dst):
    lines = [l.rstrip("\n") for l in open(src) if l.strip() and not l.startswith("#")]
    # two lines per image: pose, then the 2D point list (which may legitimately be empty)
    recs = [(lines[i].split(), lines[i + 1].split() if i + 1 < len(lines) else [])
            for i in range(0, len(lines), 2)]
    with open(dst, "wb") as f:
        _w(f, "Q", len(recs))
        for pose, pts in recs:
            _w(f, "i", int(pose[0]))
            for q in pose[1:5]:
                _w(f, "d", float(q))
            for t in pose[5:8]:
                _w(f, "d", float(t))
            _w(f, "i", int(pose[8]))
            f.write(" ".join(pose[9:]).encode() + b"\x00")   # names may contain spaces
            n = len(pts) // 3
            _w(f, "Q", n)
            for k in range(n):
                _w(f, "d", float(pts[3 * k])); _w(f, "d", float(pts[3 * k + 1]))
                _w(f, "q", int(pts[3 * k + 2]))
    return len(recs)


def points3d(src, dst, valid_images=None):
    """valid_images: ids present in images.txt. ScanNet++'s refbench release keeps a SUBSET of the
    capture's images, but points3D tracks still reference the full set, so gsplat_ext's
    `image_id_to_name[image_id]` raises KeyError on the dangling ones. Those track entries are
    dropped; the 3D points themselves are untouched."""
    rows = [l.split() for l in open(src) if l.strip() and not l.startswith("#")]
    with open(dst, "wb") as f:
        _w(f, "Q", len(rows))
        for r in rows:
            _w(f, "Q", int(r[0]))
            for x in r[1:4]:
                _w(f, "d", float(x))
            for c in r[4:7]:
                _w(f, "B", int(c))
            _w(f, "d", float(r[7]))
            tr = r[8:]
            pairs = [(int(tr[2 * k]), int(tr[2 * k + 1])) for k in range(len(tr) // 2)]
            if valid_images is not None:
                pairs = [(a, b) for a, b in pairs if a in valid_images]
            _w(f, "Q", len(pairs))
            for a, b in pairs:
                _w(f, "i", a); _w(f, "i", b)
    return len(rows)


def convert(d):
    out = []
    ids = set()
    ip = os.path.join(d, "images.txt")
    if os.path.exists(ip):
        ls = [l for l in open(ip) if l.strip() and not l.startswith("#")]
        ids = {int(ls[i].split()[0]) for i in range(0, len(ls), 2)}
    for name, fn in (("cameras", cameras), ("images", images), ("points3D", points3d)):
        s, b = os.path.join(d, name + ".txt"), os.path.join(d, name + ".bin")
        if os.path.exists(b) or not os.path.exists(s):
            out.append((name, "skip")); continue
        out.append((name, fn(s, b, ids) if name == "points3D" else fn(s, b)))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    for d in p.parse_args().dirs:
        print(d, convert(d), flush=True)
