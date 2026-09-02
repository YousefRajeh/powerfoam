"""Archive one-off probes and the garden scene to datawaha (Z:), freeing local disk.

COPY-VERIFY-DELETE, never a bare move. Z: is a network mount; a `move` that fails part-way leaves
the source already unlinked and the destination truncated, and these are the only copies. Each item
is copied, its size compared byte-for-byte, and only then removed locally. Anything that fails
verification is left on BOTH sides and reported.

WHAT COUNTS AS A ONE-OFF: a `solved_*` variant present on FEWER THAN 10 of the 10 ScanNet scenes,
i.e. an exploratory probe rather than a full arm. The full-coverage arms that paper numbers rest on
-- geometric_median/weighted x nonfrozen/truefrozen x _ogl3, and the two gs_*_ogl3 arms -- all have
coverage 10 and are therefore NEVER selected by this rule. Verified before running: the four foam
_ogl3 arms and both gs arms report coverage 10.

Also archived: single-scene budget/voronoi reconstruction dirs (`output/*bud*`, `output/*voro*`) and
the garden scene, none of which feed a ScanNet or ScanNet++ result.
"""
import json
import os
import shutil
import sys

DEST = r"Z:\users\rajehyl\powerfoam-archive"
MANIFEST = os.environ.get("MANIFEST", "artifacts/archive_manifest.json")


def dir_size(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def main():
    m = json.load(open(MANIFEST))
    os.makedirs(DEST, exist_ok=True)
    moved = failed = 0
    freed = 0

    for f in m["files"]:
        if not os.path.exists(f):
            continue
        rel = f.replace("/", os.sep)
        dst = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src_sz = os.path.getsize(f)
        try:
            shutil.copy2(f, dst)
            if os.path.getsize(dst) != src_sz:
                print(f"  [SIZE MISMATCH] {f} -- kept on both sides", flush=True)
                failed += 1
                continue
            os.remove(f)
            moved += 1
            freed += src_sz
        except Exception as e:
            print(f"  [FAIL] {f}: {type(e).__name__} {e}", flush=True)
            failed += 1
        if moved % 20 == 0 and moved:
            print(f"  ... {moved} files, {freed/2**30:.1f} GB freed", flush=True)

    for d in m["dirs"] + m.get("zips", []):
        if not os.path.exists(d):
            continue
        dst = os.path.join(DEST, d.replace("/", os.sep))
        try:
            if os.path.isdir(d):
                src_sz = dir_size(d)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(d, dst)
                ok = dir_size(dst) == src_sz
            else:
                src_sz = os.path.getsize(d)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(d, dst)
                ok = os.path.getsize(dst) == src_sz
            if not ok:
                print(f"  [SIZE MISMATCH] {d} -- kept on both sides", flush=True)
                failed += 1
                continue
            shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
            moved += 1
            freed += src_sz
            print(f"  [ok] {d}  {src_sz/2**30:.2f} GB", flush=True)
        except Exception as e:
            print(f"  [FAIL] {d}: {type(e).__name__} {e}", flush=True)
            failed += 1

    print(f"\narchived {moved} items to {DEST}, freed {freed/2**30:.1f} GB, {failed} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
