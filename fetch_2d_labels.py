"""Fetch ONLY the 2D label masks we actually use, by partial-reading the remote zip.

ScanNet ships one zip per scene containing EVERY frame -- 756 to 5,578 of them -- but our COLMAP
subset is 37 to 279. Downloading the whole archive wastes ~95% of the transfer (scene0000 is 113 MB
for 5,578 frames when 279 are needed).

The server sends `Accept-Ranges: bytes` and answers 206, so the zip can be read partially:

  1. range-read the tail to find the End Of Central Directory record,
  2. range-read the central directory and parse the member table,
  3. range-read ONLY the local headers + compressed bytes of the wanted members,
  4. inflate each in memory and write it out.

No third-party dependency (`remotezip` is not installed); this uses stdlib `zlib` and `urllib`.
Expected saving across the ten scenes is roughly 20x.

Correctness is not taken on trust: `--verify` re-downloads one full archive and compares every
extracted file byte-for-byte against the partial-fetch result.
"""
from __future__ import annotations
import argparse
import os
import struct
import sys
import urllib.request
import zlib

# Canonical host per ScanNet's current terms-of-use email; kaldir.vc.in.tum.de is an
# alias that also resolves, but this is the one they document.
BASE = "https://kaldir.vc.cit.tum.de/scannet/v2/scans"
EOCD_SIG = b"PK\x05\x06"
CEN_SIG = 0x02014B50
LOC_SIG = 0x04034B50


def _get(url, start=None, end=None, timeout=120):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("server does not advertise byte ranges")
        return int(r.headers["Content-Length"])


def central_directory(url, size, tail=1 << 16):
    """-> {name: (local_header_offset, compressed_size, method)}"""
    blob, _ = _get(url, max(0, size - tail), size - 1)
    i = blob.rfind(EOCD_SIG)
    if i < 0:
        raise RuntimeError("EOCD not found in tail; zip64 or tail too small")
    cd_size, cd_off = struct.unpack("<II", blob[i + 12:i + 20])
    cd, _ = _get(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p + 46 <= len(cd):
        if struct.unpack("<I", cd[p:p + 4])[0] != CEN_SIG:
            break
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize = struct.unpack("<I", cd[p + 20:p + 24])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        out[name] = (lho, csize, method)
        p += 46 + nlen + elen + clen
    return out


def fetch_member(url, lho, csize, method):
    # the local header repeats the name/extra lengths, which may DIFFER from the central directory's
    head, _ = _get(url, lho, lho + 29)
    if struct.unpack("<I", head[:4])[0] != LOC_SIG:
        raise RuntimeError("bad local header signature")
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = lho + 30 + nlen + elen
    data, _ = _get(url, start, start + csize - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported compression method {method}")


def wanted_stems(scene, images_root):
    d = os.path.join(images_root, f"{scene}_colmap", "images")
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(d))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--kind", default="2d-label", choices=["2d-label", "2d-label-filt"])
    ap.add_argument("--out", default=r"D:\Downloads\scannet_2dlabels")
    ap.add_argument("--images-root", default="data/scannet")
    ap.add_argument("--verify", action="store_true",
                    help="also download the FULL zip for the first scene and byte-compare")
    a = ap.parse_args()
    sub = "label" if a.kind == "2d-label" else "label-filt"
    grand_full = grand_got = 0
    for si, scene in enumerate(a.scenes.split(",")):
        url = f"{BASE}/{scene}/{scene}_{a.kind}.zip"
        dst = os.path.join(a.out, scene, sub)
        os.makedirs(dst, exist_ok=True)
        try:
            size = remote_size(url)
            cdir = central_directory(url, size)
        except Exception as e:
            print(f"[{scene}] SKIP {type(e).__name__}: {e}", flush=True)
            continue
        stems = wanted_stems(scene, a.images_root)
        got = miss = 0
        nbytes = 0
        for st in stems:
            key = f"{sub}/{st}.png"
            if key not in cdir:
                key = next((k for k in cdir if k.endswith(f"/{st}.png")), None)
            if key is None:
                miss += 1
                continue
            fp = os.path.join(dst, f"{st}.png")
            if os.path.exists(fp):
                got += 1
                continue
            lho, csize, method = cdir[key]
            blob = fetch_member(url, lho, csize, method)
            with open(fp, "wb") as f:
                f.write(blob)
            nbytes += csize
            got += 1
        grand_full += size
        grand_got += nbytes
        print(f"[{scene}] {got}/{len(stems)} masks ({miss} missing)  "
              f"fetched {nbytes/1e6:6.1f} MB of {size/1e6:7.1f} MB "
              f"({nbytes/max(size,1):.1%})", flush=True)

        if a.verify and si == 0:
            import io, zipfile
            print(f"[{scene}] VERIFY: downloading the full archive to byte-compare...", flush=True)
            full, _ = _get(url)
            zf = zipfile.ZipFile(io.BytesIO(full))
            bad = 0
            for st in stems:
                key = next((k for k in zf.namelist() if k.endswith(f"/{st}.png")), None)
                if key is None:
                    continue
                ref = zf.read(key)
                with open(os.path.join(dst, f"{st}.png"), "rb") as f:
                    if f.read() != ref:
                        bad += 1
            print(f"[{scene}] VERIFY: {bad} mismatches out of {len(stems)} "
                  f"-> {'IDENTICAL' if bad == 0 else '*** DIFFERS ***'}", flush=True)
    if grand_full:
        print(f"\nTOTAL fetched {grand_got/1e6:.1f} MB of {grand_full/1e6:.1f} MB "
              f"({grand_got/grand_full:.1%}) -- {grand_full/max(grand_got,1):.0f}x saving")


if __name__ == "__main__":
    main()
